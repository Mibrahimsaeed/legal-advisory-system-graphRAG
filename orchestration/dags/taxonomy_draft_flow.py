"""Stage 1.2b: draft a legal-domain taxonomy from a reviewed clustering.

    (resume Phase 4's clustering) -> review summaries -> eligibility gates
        -> LLM labeling of eligible clusters -> draft taxonomy + audit

Runs against an existing ``run_id``: because it reuses
:func:`orchestration.dags.domain_discovery_flow._run_sample_embed_cluster`,
pointing it at the run_id a cluster review already used resumes from that
run's checkpoint -- same sample, same embeddings, same cluster ids -- so
the taxonomy describes exactly the clustering that was reviewed, and
nothing is re-embedded.

Output, all of it draft:

* ``<discovery.taxonomy_output_dir>/<run_id>.json`` -- the taxonomy card:
  domains (id, name, description, inclusion/exclusion criteria,
  keywords, representative cases), the Other/Uncertain bucket, and the
  audit of every cluster that did not become a domain.
* ``domain_candidates`` rows, ``status='draft'``.

What it never does: write ``config/domains.yaml``, set any status other
than ``draft``, or promote a candidate. Freezing a taxonomy is a human
decision taken after reading this output -- see the report this flow
logs at the end.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from orchestration.dags.domain_discovery_flow import (
    DEFAULT_SCRATCH_ROOT,
    _artifact_dir,
    _run_sample_embed_cluster,
)
from src.clustering.cluster import Clusterer
from src.clustering.cluster_assignments import persist_cluster_assignments
from src.clustering.cluster_summary import ClusterReview, summarize_clusters
from src.clustering.label_clusters import build_keyword_corpus
from src.clustering.taxonomy_card import (
    DomainDraft,
    TaxonomyAudit,
    TaxonomyCard,
    build_taxonomy_card,
    new_run_id,
    persist_taxonomy_card,
    write_taxonomy_card_json,
)
from src.clustering.taxonomy_draft import (
    OUTCOME_UNMAPPED,
    build_draft_taxonomy,
)
from src.common.checkpoint import CheckpointManager
from src.common.config import get_settings
from src.common.db import DEFAULT_DB_PATH, init_schema
from src.common.llm_client import LLMClient, get_llm_client
from src.common.logging_utils import current_run_id, get_logger, log_context
from src.common.metrics import MetricsStore
from src.embedding.embed_model import EmbeddingModel

logger = get_logger(__name__)


@dataclass(frozen=True)
class TaxonomyDraftResult:
    run_id: str
    card: TaxonomyCard
    review: ClusterReview
    card_path: Path | None = None
    domains: list[DomainDraft] = field(default_factory=list)

    @property
    def audit(self) -> TaxonomyAudit | None:
        return self.card.audit

    @property
    def is_frozen(self) -> bool:
        """Always False. Freezing is a human step; see the module docstring."""

        return False


def _log_draft(result: TaxonomyDraftResult) -> None:
    card, audit = result.card, result.card.audit
    logger.info(
        "DRAFT taxonomy for run %s: %d domain(s), %d doc(s) in Other/Uncertain",
        result.run_id,
        len(card.domains),
        card.other_bucket.doc_count if card.other_bucket else 0,
    )
    for domain in card.domains:
        logger.info(
            "  %-28s | %4d docs | confidence=%-4s%s | %s",
            domain.domain_id,
            domain.doc_count,
            domain.confidence,
            " REVIEW" if domain.review_required else "",
            domain.name,
        )
    if audit:
        for decision in audit.decisions:
            if decision.outcome == OUTCOME_UNMAPPED:
                logger.info(
                    "  unmapped cluster %3d | %4d docs | %s",
                    decision.cluster_id,
                    decision.doc_count,
                    ", ".join(decision.reasons),
                )
    logger.warning(
        "Taxonomy for run %s is a DRAFT and has NOT been frozen. "
        "config/domains.yaml is untouched; accepting these domains is a "
        "human decision.",
        result.run_id,
    )


def run_taxonomy_draft(
    run_id: str | None = None,
    db_path: str | Path = DEFAULT_DB_PATH,
    checkpoint_dir: str | Path | None = None,
    metrics_db_path: str | Path | None = None,
    scratch_root: str | Path | None = None,
    output_dir: str | Path | None = None,
    sample_size: int | None = None,
    embedder: EmbeddingModel | None = None,
    clusterer: Clusterer | None = None,
    llm_client: LLMClient | None = None,
) -> TaxonomyDraftResult:
    """Produce (but do not freeze) a draft domain taxonomy for ``run_id``."""

    settings = get_settings()
    discovery = settings.discovery
    checkpoint_dir = checkpoint_dir or settings.pipeline.checkpoint_dir
    metrics_db_path = metrics_db_path or settings.metrics.db_path
    scratch_root = Path(scratch_root or settings.scratch.root or DEFAULT_SCRATCH_ROOT)
    output_dir = output_dir or discovery.taxonomy_output_dir
    if sample_size is None:
        sample_size = discovery.review_sample_size

    run_id = run_id or new_run_id()

    init_schema(db_path=db_path, schema_file=discovery.domain_registry_schema_file)
    if discovery.corpus_source == "representations":
        init_schema(
            db_path=db_path, schema_file=settings.caselaw.representation_schema_file
        )

    metrics = MetricsStore(metrics_db_path)
    metrics.init_schema()
    metrics_run_id = current_run_id()

    checkpoint = CheckpointManager(
        checkpoint_dir=checkpoint_dir,
        run_key=run_id,
        phases=["sample", "embed", "cluster", "label", "persist"],
    )
    artifact_dir = _artifact_dir(scratch_root, run_id)
    embeddings_path = artifact_dir / "embeddings.npz"

    with log_context(batch_id=run_id, phase="draft_taxonomy"):
        # Resumes the reviewed run: same sample, same cluster ids.
        from orchestration.dags.cluster_review_flow import _sampler_for

        sample, doc_ids, vectors, labels, probabilities, total_documents = (
            _run_sample_embed_cluster(
                settings, discovery, checkpoint, metrics, metrics_run_id, run_id,
                db_path, embeddings_path, embedder, clusterer,
                sampler=_sampler_for(sample_size),
            )
        )

        if not sample:
            logger.warning("Run %s: no documents; nothing to draft", run_id)
            card = build_taxonomy_card(
                run_id=run_id,
                sample_size=0,
                total_signatures=total_documents,
                embedding_model=discovery.embedding_model_name,
                domains=[],
                other_bucket=None,
                audit=TaxonomyAudit(notes=["No documents available"]),
            )
            return TaxonomyDraftResult(run_id=run_id, card=card, review=ClusterReview(
                run_id=run_id, total_documents=0, clustered_documents=0,
                noise_documents=0, n_clusters=0,
            ))

        persist_cluster_assignments(
            run_id, doc_ids, labels, probabilities, db_path=db_path
        )

        documents_by_id = {d.doc_id: d for d in sample}
        keyword_corpus = build_keyword_corpus(sample)

        review = summarize_clusters(
            run_id=run_id,
            doc_ids=doc_ids,
            labels=labels,
            probabilities=probabilities,
            vectors=vectors,
            documents_by_id=documents_by_id,
            keyword_corpus=keyword_corpus,
            representative_docs_per_cluster=discovery.review_representative_docs_per_cluster,
            keywords_per_cluster=discovery.keywords_per_cluster,
            max_sample_doc_ids=discovery.review_max_sample_doc_ids,
            major_min_share=discovery.review_major_min_share,
            mixed_max_mean_probability=discovery.review_mixed_max_mean_probability,
            mixed_max_cohesion=discovery.review_mixed_max_cohesion,
            mixed_max_share=discovery.review_mixed_max_share,
            court_dominance_threshold=discovery.review_court_dominance_threshold,
            relative_confidence_ratio=discovery.review_relative_confidence_ratio,
        )

        doc_ids_by_cluster: dict[int, list[str]] = {}
        for doc_id, label in zip(doc_ids, labels.tolist()):
            doc_ids_by_cluster.setdefault(int(label), []).append(doc_id)

        vectors_by_doc = (
            {d: v for d, v in zip(doc_ids, vectors)}
            if getattr(vectors, "ndim", 0) == 2 and vectors.shape[0] == len(doc_ids)
            else {}
        )
        probabilities_by_doc = (
            {d: float(p) for d, p in zip(doc_ids, probabilities.tolist())}
            if probabilities is not None
            else None
        )

        # -- label (one LLM call per eligible cluster) ------------------
        with metrics.record_phase(
            run_id=metrics_run_id, phase="label", batch_id=run_id
        ) as label_metric:
            active_llm_client = llm_client or get_llm_client(
                discovery.llm_model, max_tokens=discovery.llm_max_tokens
            )
            drafted = build_draft_taxonomy(
                review=review,
                doc_ids_by_cluster=doc_ids_by_cluster,
                documents_by_id=documents_by_id,
                keyword_corpus=keyword_corpus,
                llm_client=active_llm_client,
                vectors_by_doc=vectors_by_doc,
                probabilities_by_doc=probabilities_by_doc,
                min_domain_docs=discovery.taxonomy_min_domain_docs,
                min_domain_share=discovery.taxonomy_min_domain_share,
                max_domains=discovery.top_n_domains,
                representative_docs_per_cluster=discovery.representative_docs_per_cluster,
                keywords_per_cluster=discovery.keywords_per_cluster,
                uncertain_membership_probability=discovery.taxonomy_uncertain_membership_probability,
            )
            label_metric.processed = len(drafted.domains)

        card = build_taxonomy_card(
            run_id=run_id,
            sample_size=len(sample),
            total_signatures=total_documents,
            embedding_model=discovery.embedding_model_name,
            domains=drafted.domains,
            other_bucket=drafted.other_bucket,
            notes=[
                "DRAFT taxonomy: not frozen, not applied to any document. "
                "config/domains.yaml is untouched.",
            ],
            audit=drafted.audit,
        )

        card_path = write_taxonomy_card_json(card, output_dir)
        persist_taxonomy_card(card, db_path=db_path)

        result = TaxonomyDraftResult(
            run_id=run_id,
            card=card,
            review=review,
            card_path=Path(card_path),
            domains=drafted.domains,
        )
        _log_draft(result)
        return result
