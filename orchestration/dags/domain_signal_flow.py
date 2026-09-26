"""Phase 3: gather domain evidence for every case that survived Phase 2.

    document_representations (dropped documents already excluded)
        -> compact case representation
        -> embedding            (deterministic)
        -> UMAP + HDBSCAN       (deterministic, corpus-level)
        -> keyword profiles     (deterministic, per case)
        -> broad LLM reading    (the one stochastic step, per case)
        -> document_domain_signals

Three signals, deliberately independent, stored side by side. **No domain
is decided here.** A cluster id is a discovery signal, not a label --
nothing in this module maps a cluster to a legal domain, and Phase 4 is
where the three are weighed into a verdict.

(Domain ids appear nowhere in this package's Python: they come from the
frozen registry in ``config/domains.yaml``, and a test enforces that.)

Structure of the run: the corpus-level steps (embed, reduce, cluster) need
every vector at once and happen first; the per-case steps (keywords, LLM)
then run in batches that are persisted as they complete. A crash costs at
most one batch, and re-invoking with the same ``run_id`` skips documents
that already have evidence, so no LLM call is paid for twice.

Documents Phase 2 dropped never appear: the corpus is loaded through
:func:`src.extraction.representation_store.list_representations`, which
excludes ``dropped_*`` documents by default. That is the short-circuit,
and it is asserted by tests rather than assumed here.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from src.classification.case_representation import (
    SIGNATURE_VERSION,
    CaseRepresentation,
    build_case_representation,
)
from src.classification.domain_assessment import (
    ASSESSMENT_VERSION,
    DomainAssessment,
    assess_domain,
    render_domain_definitions,
)
from src.classification.keyword_signals import (
    KeywordSignals,
    compile_profiles,
    detect_keyword_signals,
)
from src.classification.signature_store import (
    get_case_signatures,
    upsert_case_signatures,
)
from src.classification.signal_store import (
    build_signal_record,
    get_signalled_doc_ids,
    persist_domain_signals,
    signal_stats,
)
from src.classification.taxonomy_registry import FrozenTaxonomy, load_frozen_taxonomy
from src.clustering.cluster import Clusterer, NOISE_LABEL, hdbscan_clusterer
from src.clustering.cluster_assignments import persist_cluster_assignments
from src.clustering.reduce import reduce_dimensions
from src.common.checkpoint import CheckpointManager
from src.common.config import get_settings
from src.common.db import DEFAULT_DB_PATH, init_schema
from src.common.llm_client import LLMClient, get_llm_client
from src.common.logging_utils import current_run_id, get_logger, log_context
from src.common.metrics import MetricsStore
from src.embedding.doc_pooling import embed_documents
from src.embedding.embed_model import EmbeddingModel, get_embedder
from src.extraction.representation_store import list_representations

logger = get_logger(__name__)


@dataclass(frozen=True)
class DomainSignalResult:
    run_id: str
    signal_version: str
    corpus_size: int = 0
    processed: int = 0
    skipped_already_done: int = 0
    n_clusters: int = 0
    noise_documents: int = 0
    representations: list[CaseRepresentation] = field(default_factory=list)
    stats: dict = field(default_factory=dict)

    @property
    def noise_share(self) -> float:
        total = len(self.representations)
        return self.noise_documents / total if total else 0.0


def _build_or_reuse_signatures(
    corpus: list,
    db_path,
    max_text_chars: int,
    max_headings: int,
) -> tuple[list[CaseRepresentation], dict[str, int]]:
    """Get one signature per case, reusing the stored one where it still holds.

    A stored signature is reused when its ``signature_version`` and the
    source document's ``content_hash`` both match -- at that point the
    reduction is correct by construction, so regenerating it could only
    produce the same bytes. Anything else (re-scraped text, a new builder
    version, no stored row) is built fresh and written.

    Returns the representations plus the write outcomes, so a rerun over an
    unchanged corpus visibly reports "unchanged" rather than silently
    re-stamping every row.
    """

    stored = get_case_signatures([document.doc_id for document in corpus], db_path=db_path)
    source_hashes = {d.doc_id: getattr(d, "content_hash", None) for d in corpus}

    representations: list[CaseRepresentation] = []
    fresh: list[CaseRepresentation] = []
    reused = 0
    empty = 0

    for document in corpus:
        signature = stored.get(document.doc_id)
        if signature is not None and signature.is_current(
            source_hashes.get(document.doc_id), version=SIGNATURE_VERSION
        ):
            representations.append(signature.to_representation())
            reused += 1
            continue

        representation = build_case_representation(
            document, max_text_chars=max_text_chars, max_headings=max_headings
        )
        # Same guard build_case_representations() applies: a document with
        # nothing to embed would become a near-zero vector and cluster with
        # every other empty one, so it is counted and dropped rather than
        # stored or passed on.
        if not representation.signal_text.strip():
            empty += 1
            continue
        representations.append(representation)
        fresh.append(representation)

    outcomes = upsert_case_signatures(fresh, source_hashes, db_path=db_path)
    outcomes["reused"] = reused
    if empty:
        logger.warning("Skipped %d document(s) with no representable text", empty)
    return representations, outcomes


def _cluster_corpus(
    representations: list[CaseRepresentation],
    discovery,
    embedder: EmbeddingModel | None,
    clusterer: Clusterer | None,
) -> tuple[dict[str, int], dict[str, float], str, int]:
    """Embed and cluster the whole corpus.

    Returns ``(cluster_by_doc, confidence_by_doc, embedding_model,
    n_clusters)``. Clustering is a *signal*: the caller stores the ids and
    never interprets them as domains.
    """

    if not representations:
        return {}, {}, discovery.embedding_model_name, 0

    active_embedder = embedder or get_embedder(
        discovery.embedding_model_name, batch_size=discovery.embedding_batch_size
    )
    pooled = embed_documents(
        representations,
        active_embedder,
        title_weight=discovery.title_weight,
        toc_weight=discovery.toc_weight,
        body_weight=discovery.body_weight,
        body_chunk_chars=discovery.body_chunk_chars,
        max_body_chunks=discovery.max_body_chunks,
        doc_batch_size=discovery.embedding_doc_batch_size,
    )
    if not pooled:
        return {}, {}, discovery.embedding_model_name, 0

    doc_ids = list(pooled)
    vectors = np.stack([pooled[d] for d in doc_ids])

    reduced = reduce_dimensions(
        vectors,
        n_components=discovery.umap_n_components,
        n_neighbors=discovery.umap_n_neighbors,
        min_dist=discovery.umap_min_dist,
        metric=discovery.umap_metric,
        min_docs=discovery.umap_min_docs,
    )
    active_clusterer = clusterer or hdbscan_clusterer(
        min_cluster_size=discovery.hdbscan_min_cluster_size,
        min_samples=discovery.hdbscan_min_samples,
        metric=discovery.hdbscan_metric,
    )
    result = active_clusterer(reduced)

    cluster_by_doc = {d: int(l) for d, l in zip(doc_ids, result.labels.tolist())}
    confidence_by_doc = (
        {d: float(p) for d, p in zip(doc_ids, result.probabilities.tolist())}
        if result.probabilities is not None
        else {}
    )
    return (
        cluster_by_doc,
        confidence_by_doc,
        discovery.embedding_model_name,
        result.n_clusters,
    )


def _batches(items: list, size: int):
    for start in range(0, len(items), max(size, 1)):
        yield start // max(size, 1), items[start : start + max(size, 1)]


def run_domain_signals(
    run_id: str,
    db_path: str | Path = DEFAULT_DB_PATH,
    limit: int | None = None,
    batch_size: int | None = None,
    checkpoint_dir: str | Path | None = None,
    metrics_db_path: str | Path | None = None,
    embedder: EmbeddingModel | None = None,
    clusterer: Clusterer | None = None,
    llm_client: LLMClient | None = None,
    taxonomy: FrozenTaxonomy | None = None,
    llm_enabled: bool | None = None,
) -> DomainSignalResult:
    """Gather Phase 3 evidence for the corpus under one ``run_id``.

    ``run_id`` is the version boundary: reuse it to resume, choose a new
    one whenever the embedding model, keyword profiles or prompt change so
    that earlier evidence stays intact and comparable.
    """

    settings = get_settings()
    signals_config = settings.domain_signals
    discovery = settings.discovery
    batch_size = batch_size or signals_config.batch_size
    checkpoint_dir = checkpoint_dir or settings.pipeline.checkpoint_dir
    metrics_db_path = metrics_db_path or settings.metrics.db_path
    if llm_enabled is None:
        llm_enabled = signals_config.llm_enabled

    taxonomy = taxonomy or load_frozen_taxonomy(settings.classification.taxonomy_file)
    compiled_profiles = compile_profiles(signals_config.profiles, taxonomy)

    init_schema(db_path=db_path, schema_file=settings.caselaw.representation_schema_file)
    init_schema(db_path=db_path, schema_file=signals_config.schema_file)
    init_schema(db_path=db_path, schema_file=signals_config.signature_schema_file)
    # cluster_assignments lives in the domain-registry schema.
    init_schema(db_path=db_path, schema_file=discovery.domain_registry_schema_file)

    metrics = MetricsStore(metrics_db_path)
    metrics.init_schema()
    metrics_run_id = current_run_id()

    checkpoint = CheckpointManager(
        checkpoint_dir=checkpoint_dir,
        run_key=f"signals_{run_id}",
        phases=["represent", "cluster", "assess"],
    )

    with log_context(batch_id=run_id, phase="domain_signals"):
        # -- corpus: Phase 2 drops are already excluded here -------------
        corpus = sorted(list_representations(db_path=db_path), key=lambda r: r.doc_id)
        representations, signature_outcomes = _build_or_reuse_signatures(
            corpus,
            db_path=db_path,
            max_text_chars=signals_config.max_text_chars,
            max_headings=signals_config.max_headings,
        )
        if limit is not None:
            representations = representations[:limit]

        logger.info(
            "Signal run %s: %d document(s) in corpus, %d representable "
            "(signatures: %s)",
            run_id, len(corpus), len(representations), signature_outcomes,
        )
        checkpoint.complete_phase(
            "represent", state={"corpus": len(corpus), "represented": len(representations)}
        )

        if not representations:
            return DomainSignalResult(
                run_id=run_id,
                signal_version=ASSESSMENT_VERSION,
                corpus_size=len(corpus),
                stats=signal_stats(run_id, db_path=db_path),
            )

        # -- signal 1: embedding + HDBSCAN (corpus-level, deterministic) --
        with metrics.record_phase(
            run_id=metrics_run_id, phase="cluster", batch_id=run_id
        ) as cluster_metric:
            cluster_by_doc, confidence_by_doc, embedding_model, n_clusters = (
                _cluster_corpus(representations, discovery, embedder, clusterer)
            )
            cluster_metric.processed = len(cluster_by_doc)

        if cluster_by_doc:
            doc_ids = list(cluster_by_doc)
            persist_cluster_assignments(
                run_id,
                doc_ids,
                np.array([cluster_by_doc[d] for d in doc_ids], dtype=int),
                (
                    np.array([confidence_by_doc[d] for d in doc_ids])
                    if confidence_by_doc
                    else None
                ),
                db_path=db_path,
            )
        noise_documents = sum(1 for c in cluster_by_doc.values() if c == NOISE_LABEL)
        logger.info(
            "Clustering: %d cluster(s) over %d document(s), %d noise "
            "(a discovery signal only -- no cluster means a domain)",
            n_clusters, len(cluster_by_doc), noise_documents,
        )
        checkpoint.complete_phase(
            "cluster", state={"n_clusters": n_clusters, "noise": noise_documents}
        )

        # -- signals 2 and 3: per case, batched and resumable -------------
        already_done = get_signalled_doc_ids(run_id, db_path=db_path)
        pending = [r for r in representations if r.doc_id not in already_done]

        active_llm_client = None
        domain_block = None
        if llm_enabled and pending:
            active_llm_client = llm_client or get_llm_client(
                signals_config.llm_model, max_tokens=signals_config.llm_max_tokens
            )
            domain_block = render_domain_definitions(taxonomy)

        processed = 0
        with metrics.record_phase(
            run_id=metrics_run_id, phase="assess", batch_id=run_id
        ) as assess_metric:
            for index, batch in _batches(pending, batch_size):
                batch_id = f"{run_id}_b{index:04d}"
                records = []

                for representation in batch:
                    keyword_signals: KeywordSignals = detect_keyword_signals(
                        representation.doc_id,
                        representation.signal_text,
                        compiled_profiles,
                        min_matches=signals_config.min_keyword_matches,
                    )

                    assessment: DomainAssessment | None = None
                    if active_llm_client is not None:
                        # No keyword_signals argument: the LLM must not see
                        # the keyword verdict, or the two stop being the
                        # independent readings Phase 5 weights them as (CI-7).
                        assessment = assess_domain(
                            representation,
                            taxonomy,
                            active_llm_client,
                            domain_block=domain_block,
                            prompt_chars=signals_config.llm_prompt_chars,
                            model_name=signals_config.llm_model,
                        )

                    records.append(
                        build_signal_record(
                            representation,
                            keyword_signals,
                            assessment,
                            cluster_id=cluster_by_doc.get(representation.doc_id),
                            cluster_confidence=confidence_by_doc.get(representation.doc_id),
                            embedding_model=embedding_model,
                        )
                    )

                persist_domain_signals(
                    run_id,
                    records,
                    signal_version=ASSESSMENT_VERSION,
                    batch_id=batch_id,
                    db_path=db_path,
                )
                processed += len(records)
                checkpoint.set_state(
                    "assess", {"last_batch_index": index, "processed": processed}
                )
                logger.info(
                    "Batch %s: %d case(s) signalled (%d/%d this run)",
                    batch_id, len(records), processed, len(pending),
                )

            assess_metric.processed = processed

        checkpoint.complete_phase("assess", state={"processed": processed})
        stats = signal_stats(run_id, db_path=db_path)
        logger.info(
            "Signal run %s complete: %d processed, %d already done. "
            "keyword/LLM agreement %.1f%%",
            run_id, processed, len(already_done),
            stats.get("keyword_llm_agreement_rate", 0.0) * 100,
        )

    return DomainSignalResult(
        run_id=run_id,
        signal_version=ASSESSMENT_VERSION,
        corpus_size=len(corpus),
        processed=processed,
        skipped_already_done=len(already_done),
        n_clusters=n_clusters,
        noise_documents=noise_documents,
        representations=representations,
        stats=stats,
    )
