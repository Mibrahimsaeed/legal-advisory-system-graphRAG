"""Stage 1.2: Domain Discovery (full-corpus, unsupervised).

    load -> embed (+pool) -> reduce -> cluster -> rank -> label -> persist

The corpus this loads is selected by ``discovery.corpus_source``:
``"representations"`` (the default) reads ``document_representations``,
produced from case folders by
:mod:`orchestration.dags.case_ingest_flow`; ``"signatures"`` reads the
legacy ``document_signatures`` table produced by the PDF/book stage
(:mod:`orchestration.dags.feature_extraction_flow`). Everything after the
load is identical for both -- the same embedding, UMAP, HDBSCAN, ranking
and labeling -- because the stage only ever needed four attributes from a
document record (see
:class:`src.extraction.doc_representation.EmbeddableDocument`).

Runs exactly once against the *entire corpus* of Stage 1 documents
(~10k+ docs currently -- no sampling), embeds them purely for clustering
purposes, reduces dimensionality with UMAP, clusters with HDBSCAN, ranks
clusters by size, LLM-labels the top-N as draft domains, buckets
everything else into "Other / Uncertain", and persists a draft taxonomy
card. Nothing here writes to the frozen domain registry
(``config/domains.yaml``) -- see ``src/clustering/taxonomy_card.py`` for
why ``domain_candidates`` rows are explicitly ``status='draft'``.

This stage is intended to run exactly once: the draft taxonomy it
produces is manually reviewed and frozen afterward (see
``orchestration.dags.cluster_review_flow`` for the no-LLM inspection
pass), and subsequent pipeline runs are not expected to invoke domain
discovery again. Nothing in this module enforces that -- it's an
operational/orchestration decision, not a code-level gate -- but it's why
there is deliberately no re-sampling or incremental-corpus logic here.

Unlike Stage 0/1, this stage isn't keyed by ``batch_id`` -- it operates on
the whole corpus per invocation, so its checkpoint/resume unit is its own
``run_id`` (:func:`~src.clustering.taxonomy_card.new_run_id`). Checkpointed
phases: ``sample`` (loads every document record from SQLite and fixes the
doc_id list, so a resume doesn't need to re-query -- named ``"sample"``
for continuity with the checkpoint/state-key conventions below rather than
because anything is actually sampled), ``embed`` (the expensive
model-inference step), ``cluster`` (reduce+HDBSCAN+rank -- cheap enough to
not need its own artifact, but checkpointed anyway for clean resume
semantics), ``label`` (one LLM call per top cluster, checkpointed
incrementally so a crash after cluster 2/3 doesn't re-spend an LLM call
on cluster 1), and ``persist``.

Embeddings for the corpus are the one artifact too large for the JSON
checkpoint state file, so they're saved to
``<scratch_root>/discovery/<run_id>/embeddings.npz`` -- ephemeral, same
retention posture as Stage 0/1's scratch (see docs/data_retention_policy.md):
purged once the run completes. The ``cluster`` phase's checkpoint state
DOES durably keep the small (doc_id, label, probability) triples for the
corpus, though -- see :func:`_run_sample_embed_cluster` -- specifically so
that a read-only cluster inspection
(:mod:`orchestration.dags.cluster_review_flow`) never needs to re-embed
just to look at cluster assignments that were already computed, even
after the embeddings artifact itself has been purged.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import numpy as np

from src.clustering.cluster import Clusterer, hdbscan_clusterer
from src.clustering.cluster_assignments import persist_cluster_assignments
from src.clustering.label_clusters import (
    build_keyword_corpus,
    extract_cluster_keywords,
    generate_domain_draft,
    select_representative_docs,
)
from src.clustering.reduce import reduce_dimensions
from src.clustering.taxonomy_card import (
    DomainDraft,
    build_other_bucket_draft,
    build_taxonomy_card,
    get_candidates_for_run,
    new_run_id,
    persist_taxonomy_card,
    write_taxonomy_card_json,
)
from src.common.checkpoint import CheckpointManager
from src.common.config import get_settings
from src.common.db import DEFAULT_DB_PATH, init_schema
from src.common.exceptions import ClusteringError, ConfigurationError
from src.common.llm_client import LLMClient, get_llm_client
from src.common.logging_utils import current_run_id, get_logger, log_context
from src.common.metrics import MetricsStore
from src.embedding.doc_pooling import embed_documents
from src.embedding.embed_model import EmbeddingModel, get_embedder
from src.extraction.doc_representation import EmbeddableDocument
from src.extraction.representation_store import list_representations
from src.ranking.rank_and_select import select_top_n
from src.ranking.volume_aggregator import aggregate_cluster_volumes

logger = get_logger(__name__)

DEFAULT_SCRATCH_ROOT = Path("var/scratch")


def _load_corpus(
    corpus_source: str, db_path: str | Path
) -> list[EmbeddableDocument]:
    """Load every usable document record for the configured corpus source.

    The legacy signature store is imported lazily, inside its own branch,
    so a case-law run never imports the PDF/signature modules at all.
    """

    if corpus_source == "representations":
        return list(list_representations(db_path=db_path))

    if corpus_source == "signatures":
        from src.extraction.signature_store import list_signatures

        return list(list_signatures(db_path=db_path))

    raise ConfigurationError(
        f"Unknown discovery.corpus_source={corpus_source!r}; expected "
        "'representations' or 'signatures'"
    )


@dataclass(frozen=True)
class DomainDiscoveryResult:
    run_id: str
    sample_size: int
    # Total document records in the corpus, whatever the corpus source.
    # Kept under its original name: it is also a taxonomy-card field
    # (src/clustering/taxonomy_card.py) and a checkpoint state key, and
    # renaming it would invalidate every taxonomy card already written.
    total_signatures: int
    domains: list[DomainDraft]
    other_bucket: DomainDraft | None
    taxonomy_card_path: Path | None = None
    notes: list[str] = field(default_factory=list)


def _artifact_dir(scratch_root: Path, run_id: str) -> Path:
    return scratch_root / "discovery" / run_id


def _save_embeddings(
    path: Path, doc_ids: list[str], vectors: np.ndarray, model_name: str
) -> None:
    """Write the run's embeddings plus enough provenance to trust them later.

    ``doc_ids[i]`` identifies ``vectors[i]`` -- that positional pairing is
    the document-level identity the whole stage rests on, so it is stored
    in one file with the model that produced it rather than being
    reconstructed from the corpus on resume.
    """

    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        path,
        doc_ids=np.array(doc_ids, dtype=object),
        vectors=vectors,
        model_name=np.array(model_name),
        dimension=np.array(int(vectors.shape[1]) if vectors.ndim == 2 else 0),
        created_at=np.array(datetime.now(timezone.utc).isoformat()),
    )


def _load_embeddings(path: Path) -> tuple[list[str], np.ndarray, str | None]:
    """Read back ``(doc_ids, vectors, model_name)``.

    ``model_name`` is ``None`` for artifacts written before it was
    recorded; callers treat that as "unknown" and reuse the vectors.
    """

    with np.load(path, allow_pickle=True) as data:
        model_name = str(data["model_name"]) if "model_name" in data.files else None
        return list(data["doc_ids"]), data["vectors"], model_name


def _domain_draft_from_row(row) -> DomainDraft:
    return DomainDraft(
        cluster_id=row["cluster_id"] if row["cluster_id"] is not None else -1,
        name=row["name"] or "",
        description=row["description"] or "",
        inclusion_criteria=json.loads(row["inclusion_criteria"] or "[]"),
        exclusion_criteria=json.loads(row["exclusion_criteria"] or "[]"),
        keywords=json.loads(row["keywords_json"] or "[]"),
        representative_doc_ids=json.loads(row["representative_doc_ids_json"] or "[]"),
        doc_count=row["doc_count"],
        sample_size=row["sample_size"],
        is_other_bucket=bool(row["is_other_bucket"]),
    )


def _result_from_completed_run(
    run_id: str,
    db_path: str | Path,
    checkpoint: CheckpointManager,
) -> DomainDiscoveryResult:
    """Rebuild a :class:`DomainDiscoveryResult` for a run whose ``persist``
    phase already completed, straight from durable storage (SQLite +
    checkpoint state) -- without touching the embedder, clusterer, or LLM
    client at all.

    This is what makes re-invoking :func:`run_stage1_2` with a ``run_id``
    that already finished successfully cheap: without this, even a fully
    completed run would re-embed the whole sample on every resume, since
    the embeddings scratch artifact is deleted once a run succeeds (see
    the module docstring / docs/data_retention_policy.md) -- there'd be
    nothing on disk for the ``embed`` phase's own resume check to find.
    """

    rows = get_candidates_for_run(run_id, db_path=db_path)
    domains: list[DomainDraft] = []
    other_bucket: DomainDraft | None = None
    for row in rows:
        draft = _domain_draft_from_row(row)
        if draft.is_other_bucket:
            other_bucket = draft
        else:
            domains.append(draft)
    domains.sort(key=lambda d: -d.doc_count)

    sample_state = checkpoint.get_state("sample")
    persist_state = checkpoint.get_state("persist")
    card_path = persist_state.get("card_path")

    return DomainDiscoveryResult(
        run_id=run_id,
        sample_size=len(sample_state.get("doc_ids", [])),
        total_signatures=sample_state.get("total_signatures", 0),
        domains=domains,
        other_bucket=other_bucket,
        taxonomy_card_path=Path(card_path) if card_path else None,
    )


def _run_sample_embed_cluster(
    settings,
    discovery,
    checkpoint: CheckpointManager,
    metrics: MetricsStore,
    metrics_run_id: str,
    run_id: str,
    db_path: str | Path,
    embeddings_path: Path,
    embedder: EmbeddingModel | None,
    clusterer: Clusterer | None,
    sampler: Callable[[list[EmbeddableDocument]], list[EmbeddableDocument]] | None = None,
) -> tuple[
    list[EmbeddableDocument], list[str], np.ndarray, np.ndarray, np.ndarray | None, int
]:
    """Run (or resume, via ``checkpoint``) sample -> embed -> cluster.

    Shared by :func:`run_stage1_2` (full domain discovery: goes on to rank,
    LLM-label, and persist) and
    :func:`orchestration.dags.cluster_review_flow.run_cluster_review`
    (read-only cluster inspection: stops right here, no LLM, no
    ``domain_candidates`` writes). Keeping this logic in one place means
    the two flows can never drift apart on what "the sample" or "the
    clustering" actually was for a given ``run_id`` -- reviewing a run and
    later promoting it to a full domain-discovery run (or vice versa) both
    read/write the exact same checkpoint phases.

    Returns ``(sample, doc_ids, vectors, labels, probabilities,
    total_signatures)``. ``vectors`` is a ``(0, 0)`` array and ``doc_ids``
    matches ``sample`` is empty when there was nothing to sample; callers
    should check for that themselves (this helper already marks ``embed``
    and ``cluster`` complete with empty state in that case, but does not
    know about, and does not touch, any phases beyond ``cluster``).
    """

    # -- sample (loads the full corpus; no sampling happens -- see the
    #    module docstring for why this phase is still named "sample") ----
    with metrics.record_phase(
        run_id=metrics_run_id, phase="sample", batch_id=run_id
    ) as sample_metric:
        if checkpoint.is_completed("sample"):
            cached_doc_ids = checkpoint.get_state("sample")["doc_ids"]
            total_signatures = checkpoint.get_state("sample")["total_signatures"]
            logger.info(
                "Resuming run %s: corpus already loaded (%d docs)",
                run_id,
                len(cached_doc_ids),
            )
            all_documents = {
                d.doc_id: d for d in _load_corpus(discovery.corpus_source, db_path)
            }
            sample = [all_documents[d] for d in cached_doc_ids if d in all_documents]
        else:
            checkpoint.start_phase("sample")
            corpus = _load_corpus(discovery.corpus_source, db_path)
            total_signatures = len(corpus)
            # Stage 1.2 proper passes no sampler and clusters the whole
            # corpus (see the module docstring). The review flow uses this
            # hook to cluster a controlled sample first; either way the
            # doc_ids that survive here are what the checkpoint records,
            # so a resume reproduces the same sample exactly.
            sample = sampler(corpus) if sampler is not None else corpus
            checkpoint.complete_phase(
                "sample",
                state={
                    "doc_ids": [s.doc_id for s in sample],
                    "total_signatures": total_signatures,
                },
            )
        sample_metric.processed = len(sample)

    if not sample:
        logger.warning("Run %s: no documents available in the corpus; nothing to do", run_id)
        checkpoint.complete_phase("embed", state={})
        checkpoint.complete_phase(
            "cluster",
            state={"doc_ids": [], "labels": [], "probabilities": None, "n_clusters": 0},
        )
        return [], [], np.zeros((0, 0)), np.zeros(0, dtype=int), None, total_signatures

    by_doc_id: dict[str, EmbeddableDocument] = {s.doc_id: s for s in sample}

    # Checked up front (not just inside the "cluster" block below) so the
    # "embed" block itself can skip real re-embedding when clustering is
    # already done and its doc_ids survived in the checkpoint -- checking
    # this only *after* running embed (as an earlier version of this
    # function did) meant the "skip re-embedding" win never actually
    # materialized: embed would already have paid the full re-embedding
    # cost before cluster's cached doc_ids got a chance to short-circuit
    # anything.
    cluster_already_done = checkpoint.is_completed("cluster")
    cached_cluster_state = checkpoint.get_state("cluster") if cluster_already_done else {}
    cached_doc_ids = cached_cluster_state.get("doc_ids") if cluster_already_done else None

    # -- embed ----------------------------------------------------------
    with metrics.record_phase(
        run_id=metrics_run_id, phase="embed", batch_id=run_id
    ) as embed_metric:
        reusable_embeddings: tuple[list[str], np.ndarray] | None = None
        if checkpoint.is_completed("embed") and embeddings_path.exists():
            artifact_doc_ids, artifact_vectors, artifact_model = _load_embeddings(
                embeddings_path
            )
            if artifact_model and artifact_model != discovery.embedding_model_name:
                # Reusing these would mix vectors from two different models
                # in one clustering pass -- the resulting "domains" would be
                # an artifact of the model switch, not of the corpus.
                logger.warning(
                    "Run %s: cached embeddings were produced by %r but the "
                    "configured model is now %r; re-embedding the corpus",
                    run_id,
                    artifact_model,
                    discovery.embedding_model_name,
                )
            else:
                reusable_embeddings = (artifact_doc_ids, artifact_vectors)

        if reusable_embeddings is not None:
            doc_ids, vectors = reusable_embeddings
            logger.info("Resuming run %s: loaded %d cached embeddings", run_id, len(doc_ids))
        elif cluster_already_done and cached_doc_ids:
            # Clustering is already known and its doc_ids survived in the
            # checkpoint -- there is nothing left for embedding to
            # produce that clustering needs, so skip calling the embedder
            # entirely. `vectors` is intentionally empty; representative
            # document selection falls back gracefully to
            # probability-based (or, failing that, arbitrary) choice when
            # vectors aren't available -- see
            # src.clustering.label_clusters.select_representative_docs.
            doc_ids = cached_doc_ids
            vectors = np.zeros((0, 0))
            if not checkpoint.is_completed("embed"):
                checkpoint.complete_phase(
                    "embed",
                    state={"doc_count": len(doc_ids), "note": "skipped: reused cached cluster doc_ids"},
                )
            logger.info(
                "Resuming run %s: cluster already computed, skipping re-embedding "
                "of %d document(s) entirely",
                run_id, len(doc_ids),
            )
        else:
            checkpoint.start_phase("embed")
            active_embedder = embedder or get_embedder(
                discovery.embedding_model_name,
                batch_size=discovery.embedding_batch_size,
            )
            pooled = embed_documents(
                sample,
                active_embedder,
                title_weight=discovery.title_weight,
                toc_weight=discovery.toc_weight,
                body_weight=discovery.body_weight,
                body_chunk_chars=discovery.body_chunk_chars,
                max_body_chunks=discovery.max_body_chunks,
                doc_batch_size=discovery.embedding_doc_batch_size,
            )
            doc_ids = list(pooled.keys())
            vectors = np.stack(list(pooled.values())) if doc_ids else np.zeros((0, 0))
            _save_embeddings(
                embeddings_path, doc_ids, vectors, discovery.embedding_model_name
            )
            checkpoint.complete_phase(
                "embed",
                state={
                    "doc_count": len(doc_ids),
                    "embedding_model": discovery.embedding_model_name,
                },
            )
        embed_metric.processed = len(doc_ids)

    skipped = [d for d in by_doc_id if d not in set(doc_ids)]
    if skipped:
        logger.warning(
            "%d sampled document(s) had no embeddable content and were "
            "dropped before clustering",
            len(skipped),
        )

    # -- cluster (reduce + HDBSCAN) ---------------------------------------
    with metrics.record_phase(
        run_id=metrics_run_id, phase="cluster", batch_id=run_id
    ) as cluster_metric:
        if cluster_already_done:
            cluster_state = cached_cluster_state
            if cached_doc_ids:
                doc_ids = cached_doc_ids
            labels = np.array(cluster_state["labels"], dtype=int)
            probabilities = (
                np.array(cluster_state["probabilities"])
                if cluster_state.get("probabilities") is not None
                else None
            )
            logger.info("Resuming run %s: cluster assignment already computed", run_id)
        else:
            checkpoint.start_phase("cluster")
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
            try:
                result = active_clusterer(reduced)
            except ClusteringError:
                checkpoint.fail_phase("cluster", "clustering backend unavailable/failed")
                raise

            labels = result.labels
            probabilities = result.probabilities
            checkpoint.complete_phase(
                "cluster",
                state={
                    "doc_ids": doc_ids,
                    "labels": labels.tolist(),
                    "probabilities": (
                        probabilities.tolist() if probabilities is not None else None
                    ),
                    "n_clusters": result.n_clusters,
                },
            )
        cluster_metric.processed = len(labels)

    return sample, doc_ids, vectors, labels, probabilities, total_signatures


def run_stage1_2(
    run_id: str | None = None,
    db_path: str | Path = DEFAULT_DB_PATH,
    checkpoint_dir: str | Path | None = None,
    metrics_db_path: str | Path | None = None,
    scratch_root: str | Path | None = None,
    embedder: EmbeddingModel | None = None,
    clusterer: Clusterer | None = None,
    llm_client: LLMClient | None = None,
) -> DomainDiscoveryResult:
    """Run Stage 1.2 (Domain Discovery) end to end, with checkpoint/resume.

    ``embedder`` / ``clusterer`` / ``llm_client`` are injectable purely so
    tests (and any environment missing the heavy ML deps) can supply
    fakes; production runs leave them ``None`` and get the real
    sentence-transformers / HDBSCAN / Anthropic backends built from
    ``settings.discovery``.
    """

    settings = get_settings()
    discovery = settings.discovery
    checkpoint_dir = checkpoint_dir or settings.pipeline.checkpoint_dir
    metrics_db_path = metrics_db_path or settings.metrics.db_path
    scratch_root = Path(scratch_root or settings.scratch.root)

    run_id = run_id or new_run_id()

    init_schema(db_path=db_path, schema_file=settings.database.schema_file)
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

    with log_context(batch_id=run_id, phase="discover_domains"):
        if checkpoint.is_completed("persist"):
            # The whole run already finished successfully on a previous
            # invocation -- reconstruct from SQLite/checkpoint state rather
            # than re-deriving anything (there's nothing left to compute,
            # and the embeddings artifact this run produced is long gone;
            # see docs/data_retention_policy.md).
            logger.info(
                "Run %s already fully persisted; returning the stored result",
                run_id,
            )
            return _result_from_completed_run(run_id, db_path, checkpoint)

        sample, doc_ids, vectors, labels, probabilities, total_signatures = (
            _run_sample_embed_cluster(
                settings, discovery, checkpoint, metrics, metrics_run_id, run_id,
                db_path, embeddings_path, embedder, clusterer,
            )
        )

        if not sample:
            checkpoint.complete_phase("label", state={})
            checkpoint.complete_phase("persist", state={})
            return DomainDiscoveryResult(
                run_id=run_id,
                sample_size=0,
                total_signatures=total_signatures,
                domains=[],
                other_bucket=None,
                notes=["No documents available"],
            )

        by_doc_id: dict[str, EmbeddableDocument] = {s.doc_id: s for s in sample}

        # Persisted immediately after clustering, before ranking/labeling
        # touch anything -- every sampled doc's raw HDBSCAN label
        # (including noise) is durable from here on, independent of which
        # clusters later get ranked/labeled into domain_candidates.
        persist_cluster_assignments(
            run_id, doc_ids, labels, probabilities, db_path=db_path
        )

        volumes = aggregate_cluster_volumes(doc_ids, labels, probabilities)
        selection = select_top_n(volumes, top_n=discovery.top_n_domains)

        # -- label (LLM call per top cluster, checkpointed incrementally) ----
        with metrics.record_phase(
            run_id=metrics_run_id, phase="label", batch_id=run_id
        ) as label_metric:
            if checkpoint.is_completed("label"):
                labeled_state = checkpoint.get_state("label")
                domains = [DomainDraft(**d) for d in labeled_state["domains"]]
                logger.info("Resuming run %s: %d domain(s) already labeled", run_id, len(domains))
            else:
                checkpoint.start_phase("label")
                partial_state = checkpoint.get_state("label")
                already_labeled = {
                    d["cluster_id"]: DomainDraft(**d)
                    for d in partial_state.get("domains", [])
                }

                keyword_corpus = build_keyword_corpus(sample)
                probabilities_by_doc = (
                    {d: p for d, p in zip(doc_ids, probabilities.tolist())}
                    if probabilities is not None
                    else None
                )
                vectors_by_doc = {d: v for d, v in zip(doc_ids, vectors)}

                active_llm_client = llm_client or get_llm_client(
                    discovery.llm_model, max_tokens=discovery.llm_max_tokens
                )

                domains = []
                for volume in selection.top_clusters:
                    if volume.cluster_id in already_labeled:
                        domains.append(already_labeled[volume.cluster_id])
                        continue

                    cluster_documents = [
                        by_doc_id[d] for d in volume.doc_ids if d in by_doc_id
                    ]
                    keywords = extract_cluster_keywords(
                        cluster_documents,
                        keyword_corpus,
                        top_k=discovery.keywords_per_cluster,
                    )
                    cluster_probs = (
                        {d: probabilities_by_doc.get(d, 0.0) for d in volume.doc_ids}
                        if probabilities_by_doc
                        else None
                    )
                    representative_ids = select_representative_docs(
                        volume.doc_ids,
                        vectors_by_doc,
                        probabilities=cluster_probs,
                        top_k=discovery.representative_docs_per_cluster,
                    )
                    draft = generate_domain_draft(
                        volume.cluster_id,
                        cluster_documents,
                        representative_ids,
                        keywords,
                        sample_size=len(sample),
                        llm_client=active_llm_client,
                    )
                    domains.append(draft)

                    # Incremental checkpoint: a crash after labeling cluster 1
                    # of 3 shouldn't re-spend an LLM call re-labeling it on resume.
                    checkpoint.set_state(
                        "label", {"domains": [vars(d) for d in domains]}
                    )

                checkpoint.complete_phase(
                    "label", state={"domains": [vars(d) for d in domains]}
                )
            label_metric.processed = len(domains)

        other_bucket = build_other_bucket_draft(
            selection.other_doc_ids, sample_size=len(sample)
        )

        # -- persist -------------------------------------------------------
        with metrics.record_phase(
            run_id=metrics_run_id, phase="persist", batch_id=run_id
        ) as persist_metric:
            if checkpoint.is_completed("persist"):
                card_path = checkpoint.get_state("persist").get("card_path")
                logger.info("Resuming run %s: already persisted", run_id)
            else:
                checkpoint.start_phase("persist")
                card = build_taxonomy_card(
                    run_id=run_id,
                    sample_size=len(sample),
                    total_signatures=total_signatures,
                    embedding_model=discovery.embedding_model_name,
                    domains=domains,
                    other_bucket=other_bucket,
                )
                card_path = str(write_taxonomy_card_json(card, discovery.taxonomy_output_dir))
                persist_taxonomy_card(card, db_path=db_path)
                checkpoint.complete_phase("persist", state={"card_path": card_path})
            persist_metric.processed = len(domains) + 1

        # Ephemeral by design, same posture as Stage 0/1 scratch: the only
        # thing worth keeping past this run is the taxonomy card + the
        # domain_candidates rows, both already persisted above.
        if artifact_dir.exists():
            import shutil

            shutil.rmtree(artifact_dir, ignore_errors=True)

        logger.info(
            "Stage 1.2 complete for run %s: %d domain(s) discovered, "
            "%d doc(s) in Other/Uncertain, sample=%d/%d",
            run_id,
            len(domains),
            other_bucket.doc_count,
            len(sample),
            total_signatures,
        )

        return DomainDiscoveryResult(
            run_id=run_id,
            sample_size=len(sample),
            total_signatures=total_signatures,
            domains=domains,
            other_bucket=other_bucket,
            taxonomy_card_path=Path(card_path) if card_path else None,
        )