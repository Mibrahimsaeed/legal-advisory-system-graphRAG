"""Stage 1.2a: read-only cluster review (no LLM, no taxonomy).

    load -> (optional stratified sample) -> embed -> reduce -> HDBSCAN
         -> per-cluster summaries -> JSON report

Same corpus, same embeddings, same UMAP, same HDBSCAN as
:mod:`orchestration.dags.domain_discovery_flow` -- it literally calls
that module's ``_run_sample_embed_cluster``, so a review and a discovery
run of the same ``run_id`` share checkpoint phases and can never disagree
about what "the clustering" was. What this flow does *not* do is the
second half of Stage 1.2: no LLM labeling, no ``domain_candidates`` rows,
no taxonomy card. Naming domains is a later decision; this stage exists
to look at the clusters first.

Two things are written:

* ``cluster_assignments`` in SQLite -- the durable (run_id, doc_id,
  cluster_id, confidence) mapping, including noise (``-1``). Same table
  and same idempotent-per-run behavior as discovery uses.
* ``<discovery.review_output_dir>/<run_id>.json`` -- the human-readable
  report: parameters, sizes, noise share, keywords, representative cases
  and quality flags.

Run it against a sample first (``sample_size=...``, or
``discovery.review_sample_size``): if the clustering is wrong, a
well-stratified 200-document sample shows it as clearly as the full
corpus, for a fraction of the embedding cost.
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

from orchestration.dags.domain_discovery_flow import (
    DEFAULT_SCRATCH_ROOT,
    _artifact_dir,
    _load_corpus,
    _run_sample_embed_cluster,
)
from src.clustering.cluster import Clusterer
from src.clustering.cluster_assignments import persist_cluster_assignments
from src.clustering.cluster_summary import ClusterReview, summarize_clusters
from src.clustering.label_clusters import build_keyword_corpus
from src.clustering.sampling import stratified_sample
from src.clustering.taxonomy_card import new_run_id
from src.common.checkpoint import CheckpointManager
from src.common.config import get_settings
from src.common.db import DEFAULT_DB_PATH, init_schema
from src.common.logging_utils import current_run_id, get_logger, log_context
from src.common.metrics import MetricsStore
from src.embedding.embed_model import EmbeddingModel
from src.extraction.doc_representation import EmbeddableDocument

logger = get_logger(__name__)


def _sampler_for(sample_size: int | None, seed: int = 42):
    """A stratified sampler capped at ``sample_size``, or ``None`` for all."""

    if not sample_size:
        return None

    def _sample(corpus: list[EmbeddableDocument]) -> list[EmbeddableDocument]:
        return stratified_sample(
            corpus, sample_min=sample_size, sample_max=sample_size, seed=seed
        )

    return _sample


def write_review_json(review: ClusterReview, output_dir: str | Path) -> Path:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"{review.run_id}.json"

    payload = dataclasses.asdict(review)
    payload["noise_share"] = review.noise_share
    payload["suspicious_cluster_ids"] = [s.cluster_id for s in review.suspicious_clusters]
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def _log_review(review: ClusterReview) -> None:
    logger.info(
        "Run %s: %d cluster(s) over %d document(s); %d noise (%.1f%%)",
        review.run_id,
        review.n_clusters,
        review.total_documents,
        review.noise_documents,
        review.noise_share * 100,
    )
    for summary in review.summaries:
        logger.info(
            "  cluster %3d | %-6s | %4d docs (%5.1f%%) | cohesion=%s | mean_prob=%s | %s%s",
            summary.cluster_id,
            summary.kind,
            summary.doc_count,
            summary.share * 100,
            f"{summary.cohesion:.2f}" if summary.cohesion is not None else "n/a",
            f"{summary.mean_probability:.2f}" if summary.mean_probability is not None else "n/a",
            ", ".join(summary.keywords[:6]) or "(no keywords)",
            f" | FLAGS: {','.join(summary.flags)}" if summary.flags else "",
        )


def run_cluster_review(
    run_id: str | None = None,
    db_path: str | Path = DEFAULT_DB_PATH,
    checkpoint_dir: str | Path | None = None,
    metrics_db_path: str | Path | None = None,
    scratch_root: str | Path | None = None,
    output_dir: str | Path | None = None,
    sample_size: int | None = None,
    embedder: EmbeddingModel | None = None,
    clusterer: Clusterer | None = None,
    keep_artifacts: bool = True,
) -> ClusterReview:
    """Cluster the corpus (or a sample of it) and describe what came out.

    ``embedder`` / ``clusterer`` are injectable for tests exactly as in
    Stage 1.2; production leaves them ``None`` and gets the configured
    sentence-transformers model and real HDBSCAN.

    ``keep_artifacts`` defaults to ``True`` -- unlike a completed
    discovery run, a review is usually the first of several passes, and
    keeping ``embeddings.npz`` lets the next parameter tweak re-cluster
    without paying to re-embed. Set it to ``False`` to purge on exit.
    """

    settings = get_settings()
    discovery = settings.discovery
    checkpoint_dir = checkpoint_dir or settings.pipeline.checkpoint_dir
    metrics_db_path = metrics_db_path or settings.metrics.db_path
    scratch_root = Path(scratch_root or settings.scratch.root or DEFAULT_SCRATCH_ROOT)
    output_dir = output_dir or discovery.review_output_dir
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

    with log_context(batch_id=run_id, phase="review_clusters"):
        sample, doc_ids, vectors, labels, probabilities, total_documents = (
            _run_sample_embed_cluster(
                settings, discovery, checkpoint, metrics, metrics_run_id, run_id,
                db_path, embeddings_path, embedder, clusterer,
                sampler=_sampler_for(sample_size),
            )
        )

        parameters = {
            "corpus_source": discovery.corpus_source,
            "corpus_size": total_documents,
            "sample_size": len(sample),
            "requested_sample_size": sample_size,
            "embedding_model": discovery.embedding_model_name,
            "title_weight": discovery.title_weight,
            "toc_weight": discovery.toc_weight,
            "body_weight": discovery.body_weight,
            "body_chunk_chars": discovery.body_chunk_chars,
            "max_body_chunks": discovery.max_body_chunks,
            "umap_n_components": discovery.umap_n_components,
            "umap_n_neighbors": discovery.umap_n_neighbors,
            "umap_min_dist": discovery.umap_min_dist,
            "umap_metric": discovery.umap_metric,
            "umap_min_docs": discovery.umap_min_docs,
            "hdbscan_min_cluster_size": discovery.hdbscan_min_cluster_size,
            "hdbscan_min_samples": discovery.hdbscan_min_samples,
            "hdbscan_metric": discovery.hdbscan_metric,
        }

        if not sample:
            logger.warning("Run %s: nothing to review; corpus is empty", run_id)
            review = ClusterReview(
                run_id=run_id,
                total_documents=0,
                clustered_documents=0,
                noise_documents=0,
                n_clusters=0,
                parameters=parameters,
            )
            write_review_json(review, output_dir)
            return review

        # Durable, and identical to what Stage 1.2 would persist for this
        # run: every document's raw cluster id, noise included.
        persist_cluster_assignments(
            run_id, doc_ids, labels, probabilities, db_path=db_path
        )

        documents_by_id = {d.doc_id: d for d in sample}
        review = summarize_clusters(
            run_id=run_id,
            doc_ids=doc_ids,
            labels=labels,
            probabilities=probabilities,
            vectors=vectors,
            documents_by_id=documents_by_id,
            keyword_corpus=build_keyword_corpus(sample),
            parameters=parameters,
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

        _log_review(review)
        report_path = write_review_json(review, output_dir)
        logger.info("Cluster review for run %s written to %s", run_id, report_path)

        if not keep_artifacts and artifact_dir.exists():
            import shutil

            shutil.rmtree(artifact_dir, ignore_errors=True)

        return review


__all__ = ["run_cluster_review", "write_review_json", "_load_corpus"]
