"""Stage 1.2: Domain Discovery (sample-based, unsupervised).

    sample -> embed (+pool) -> reduce -> cluster -> rank -> label -> persist

Runs once against a *stratified sample* of Stage 1 signatures (~1,500-2,000
docs, not the full corpus), embeds them purely for clustering purposes,
reduces dimensionality with UMAP, clusters with HDBSCAN, ranks clusters by
size, LLM-labels the top-N as draft domains, buckets everything else into
"Other / Uncertain", and persists a draft taxonomy card. Nothing here
writes to the frozen domain registry (``config/domains.yaml``) -- see
``src/clustering/taxonomy_card.py`` for why ``domain_candidates`` rows are
explicitly ``status='draft'``.

Unlike Stage 0/1, this stage isn't keyed by ``batch_id`` -- it operates on
a single corpus-wide sample per invocation, so its checkpoint/resume unit
is its own ``run_id`` (:func:`~src.clustering.taxonomy_card.new_run_id`).
Checkpointed phases: ``sample`` (fixes the random sample so a resume
doesn't redraw a different one), ``embed`` (the expensive model-inference
step), ``cluster`` (reduce+HDBSCAN+rank -- cheap enough to not need its
own artifact, but checkpointed anyway for clean resume semantics),
``label`` (one LLM call per top cluster, checkpointed incrementally so a
crash after cluster 2/3 doesn't re-spend an LLM call on cluster 1), and
``persist``.

Embeddings for the sample are the one artifact too large for the JSON
checkpoint state file, so they're saved to
``<scratch_root>/discovery/<run_id>/embeddings.npz`` -- ephemeral, same
retention posture as Stage 0/1's scratch (see docs/data_retention_policy.md):
purged once the run completes.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from src.clustering.cluster import Clusterer, hdbscan_clusterer
from src.clustering.label_clusters import (
    build_keyword_corpus,
    extract_cluster_keywords,
    generate_domain_draft,
    select_representative_docs,
)
from src.clustering.reduce import reduce_dimensions
from src.clustering.sampling import stratified_sample
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
from src.common.exceptions import ClusteringError
from src.common.llm_client import LLMClient, get_llm_client
from src.common.logging_utils import current_run_id, get_logger, log_context
from src.common.metrics import MetricsStore
from src.embedding.doc_pooling import embed_signatures
from src.embedding.embed_model import EmbeddingModel, get_embedder
from src.extraction.signature import DocumentSignature
from src.extraction.signature_store import list_signatures
from src.ranking.rank_and_select import select_top_n
from src.ranking.volume_aggregator import aggregate_cluster_volumes

logger = get_logger(__name__)

DEFAULT_SCRATCH_ROOT = Path("var/scratch")


@dataclass(frozen=True)
class DomainDiscoveryResult:
    run_id: str
    sample_size: int
    total_signatures: int
    domains: list[DomainDraft]
    other_bucket: DomainDraft | None
    taxonomy_card_path: Path | None = None
    notes: list[str] = field(default_factory=list)


def _artifact_dir(scratch_root: Path, run_id: str) -> Path:
    return scratch_root / "discovery" / run_id


def _save_embeddings(path: Path, doc_ids: list[str], vectors: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, doc_ids=np.array(doc_ids, dtype=object), vectors=vectors)


def _load_embeddings(path: Path) -> tuple[list[str], np.ndarray]:
    with np.load(path, allow_pickle=True) as data:
        return list(data["doc_ids"]), data["vectors"]


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

        # -- sample -------------------------------------------------------
        with metrics.record_phase(
            run_id=metrics_run_id, phase="sample", batch_id=run_id
        ) as sample_metric:
            if checkpoint.is_completed("sample"):
                sampled_doc_ids = checkpoint.get_state("sample")["doc_ids"]
                total_signatures = checkpoint.get_state("sample")["total_signatures"]
                logger.info(
                    "Resuming run %s: sample already fixed (%d docs)",
                    run_id,
                    len(sampled_doc_ids),
                )
                all_signatures = {
                    s.doc_id: s for s in list_signatures(db_path=db_path)
                }
                sample = [
                    all_signatures[d] for d in sampled_doc_ids if d in all_signatures
                ]
            else:
                checkpoint.start_phase("sample")
                all_signatures_list = list_signatures(db_path=db_path)
                total_signatures = len(all_signatures_list)
                sample = stratified_sample(
                    all_signatures_list,
                    sample_min=discovery.sample_min,
                    sample_max=discovery.sample_max,
                    seed=discovery.sample_seed,
                )
                checkpoint.complete_phase(
                    "sample",
                    state={
                        "doc_ids": [s.doc_id for s in sample],
                        "total_signatures": total_signatures,
                    },
                )
            sample_metric.processed = len(sample)

        if not sample:
            logger.warning("Run %s: no signatures available to sample; nothing to do", run_id)
            checkpoint.complete_phase("embed", state={})
            checkpoint.complete_phase("cluster", state={})
            checkpoint.complete_phase("label", state={})
            checkpoint.complete_phase("persist", state={})
            return DomainDiscoveryResult(
                run_id=run_id,
                sample_size=0,
                total_signatures=total_signatures,
                domains=[],
                other_bucket=None,
                notes=["No signatures available"],
            )

        by_doc_id: dict[str, DocumentSignature] = {s.doc_id: s for s in sample}

        # -- embed ----------------------------------------------------------
        with metrics.record_phase(
            run_id=metrics_run_id, phase="embed", batch_id=run_id
        ) as embed_metric:
            if checkpoint.is_completed("embed") and embeddings_path.exists():
                doc_ids, vectors = _load_embeddings(embeddings_path)
                logger.info("Resuming run %s: loaded %d cached embeddings", run_id, len(doc_ids))
            else:
                checkpoint.start_phase("embed")
                active_embedder = embedder or get_embedder(
                    discovery.embedding_model_name,
                    batch_size=discovery.embedding_batch_size,
                )
                pooled = embed_signatures(
                    sample,
                    active_embedder,
                    title_weight=discovery.title_weight,
                    toc_weight=discovery.toc_weight,
                    body_weight=discovery.body_weight,
                    body_chunk_chars=discovery.body_chunk_chars,
                    max_body_chunks=discovery.max_body_chunks,
                )
                doc_ids = list(pooled.keys())
                vectors = np.stack(list(pooled.values())) if doc_ids else np.zeros((0, 0))
                _save_embeddings(embeddings_path, doc_ids, vectors)
                checkpoint.complete_phase("embed", state={"doc_count": len(doc_ids)})
            embed_metric.processed = len(doc_ids)

        skipped = [d for d in by_doc_id if d not in set(doc_ids)]
        if skipped:
            logger.warning(
                "%d sampled document(s) had no embeddable content and were "
                "dropped before clustering",
                len(skipped),
            )

        # -- cluster (reduce + HDBSCAN + rank) --------------------------------
        with metrics.record_phase(
            run_id=metrics_run_id, phase="cluster", batch_id=run_id
        ) as cluster_metric:
            if checkpoint.is_completed("cluster"):
                cluster_state = checkpoint.get_state("cluster")
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
                        "labels": labels.tolist(),
                        "probabilities": (
                            probabilities.tolist() if probabilities is not None else None
                        ),
                        "n_clusters": result.n_clusters,
                    },
                )
            cluster_metric.processed = len(labels)

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

                    cluster_signatures = [
                        by_doc_id[d] for d in volume.doc_ids if d in by_doc_id
                    ]
                    keywords = extract_cluster_keywords(
                        cluster_signatures,
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
                        cluster_signatures,
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