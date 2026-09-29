"""Phase 4: blind cluster validation.

    Phase 3 representations -> embeddings -> [ UMAP -> HDBSCAN ] x sweep
        -> score each run against source-folder labels (afterwards only)
        -> JSON report + a verdict on whether cluster membership is useful

The clustering is blind by construction: :data:`src.clustering.cluster.Clusterer`
is ``(vectors) -> ClusterResult``, so no label can reach it. Labels are
read from ``source_relpath`` *after* every clustering run completes, and
only inside :mod:`src.clustering.cluster_validation`.

Why a parameter sweep rather than a single run: "HDBSCAN does not carry a
domain signal" and "these particular hyperparameters do not" are
different findings, and only the first justifies dropping the signal from
the classifier. The sweep is configured, small, and reported in full --
the best result is highlighted, and every result is kept so a lucky
setting cannot be mistaken for a robust one.

This flow decides nothing about any document: it writes no
``document_classifications``, no ``domain_candidates`` and no
``document_domain_signals``. Its output is a report and a verdict for a
human (and for Phase 5's design) to act on.
"""

from __future__ import annotations

import dataclasses
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from src.classification.case_representation import (
    CaseRepresentation,
    build_case_representations,
)
from src.clustering.cluster import Clusterer, hdbscan_clusterer
from src.clustering.cluster_validation import (
    VERDICT_NOT_EVALUABLE,
    ValidationReport,
    evaluate_clustering,
    labels_from_source_folder,
)
from src.clustering.reduce import reduce_dimensions
from src.common.config import get_settings
from src.common.db import DEFAULT_DB_PATH, init_schema
from src.common.logging_utils import current_run_id, get_logger, log_context
from src.common.metrics import MetricsStore
from src.embedding.doc_pooling import embed_documents
from src.embedding.embed_model import EmbeddingModel, get_embedder
from src.extraction.representation_store import list_representations

logger = get_logger(__name__)


@dataclass(frozen=True)
class ClusterValidationResult:
    run_id: str
    n_documents: int
    reports: list[ValidationReport] = field(default_factory=list)
    best: ValidationReport | None = None
    report_path: Path | None = None

    @property
    def verdict(self) -> str:
        return self.best.verdict if self.best else VERDICT_NOT_EVALUABLE

    @property
    def cluster_is_a_useful_signal(self) -> bool:
        """Whether Phase 5 may use cluster membership as domain evidence."""

        return bool(self.best and self.best.cluster_is_a_useful_signal)


def _sweep_settings(validation) -> list[dict[str, Any]]:
    """Expand the configured sweep into concrete parameter sets."""

    combos: list[dict[str, Any]] = []
    for min_cluster_size in validation.sweep_min_cluster_sizes:
        for min_samples in validation.sweep_min_samples:
            combos.append(
                {
                    "hdbscan_min_cluster_size": min_cluster_size,
                    # YAML cannot express None inside a list cleanly; 0 means
                    # "let HDBSCAN default min_samples to min_cluster_size".
                    "hdbscan_min_samples": min_samples or None,
                }
            )
    return combos or [{"hdbscan_min_cluster_size": None, "hdbscan_min_samples": None}]


def _report_to_dict(report: ValidationReport) -> dict:
    payload = dataclasses.asdict(report)
    payload["noise_share"] = report.noise_share
    payload["coverage"] = report.coverage
    payload["mixed_cluster_ids"] = [c.cluster_id for c in report.mixed_clusters]
    payload["cluster_is_a_useful_signal"] = report.cluster_is_a_useful_signal
    return payload


def write_validation_report(
    result: ClusterValidationResult, output_dir: str | Path
) -> Path:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"{result.run_id}.json"

    payload = {
        "run_id": result.run_id,
        "n_documents": result.n_documents,
        "verdict": result.verdict,
        "cluster_is_a_useful_signal": result.cluster_is_a_useful_signal,
        "best": _report_to_dict(result.best) if result.best else None,
        "sweep": [_report_to_dict(r) for r in result.reports],
    }
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def _log_report(report: ValidationReport) -> None:
    logger.info(
        "  params=%s -> %d cluster(s), coverage %.2f, noise %.1f%%, "
        "purity %.2f (baseline %.2f, lift %+.2f), ARI=%s -> %s",
        {
            "min_cluster_size": report.parameters.get("hdbscan_min_cluster_size"),
            "min_samples": report.parameters.get("hdbscan_min_samples"),
        },
        report.n_clusters,
        report.coverage,
        report.noise_share * 100,
        report.weighted_purity,
        report.baseline_majority_share,
        report.purity_lift,
        f"{report.adjusted_rand_index:.2f}" if report.adjusted_rand_index is not None else "n/a",
        report.verdict,
    )


def run_cluster_validation(
    run_id: str,
    db_path: str | Path = DEFAULT_DB_PATH,
    output_dir: str | Path | None = None,
    limit: int | None = None,
    metrics_db_path: str | Path | None = None,
    embedder: EmbeddingModel | None = None,
    clusterer: Clusterer | None = None,
    label_depth: int | None = None,
) -> ClusterValidationResult:
    """Cluster the Phase 3 corpus blind, then score it against labels.

    ``clusterer``, when given, replaces the HDBSCAN factory for every
    sweep point -- used by tests. Production leaves it ``None`` and gets
    real HDBSCAN at each configured parameter setting.
    """

    settings = get_settings()
    validation = settings.cluster_validation
    discovery = settings.discovery
    signals_config = settings.domain_signals
    output_dir = output_dir or validation.output_dir
    metrics_db_path = metrics_db_path or settings.metrics.db_path
    label_depth = label_depth if label_depth is not None else validation.label_folder_depth

    init_schema(db_path=db_path, schema_file=settings.caselaw.representation_schema_file)

    metrics = MetricsStore(metrics_db_path)
    metrics.init_schema()
    metrics_run_id = current_run_id()

    with log_context(batch_id=run_id, phase="validate_clusters"):
        # Phase 2 drops are excluded by list_representations; Phase 3's
        # representation builder is reused verbatim so the vectors scored
        # here are the vectors the pipeline actually uses.
        corpus = sorted(list_representations(db_path=db_path), key=lambda r: r.doc_id)
        representations: list[CaseRepresentation] = build_case_representations(
            corpus,
            max_text_chars=signals_config.max_text_chars,
            max_headings=signals_config.max_headings,
        )
        if limit is not None:
            representations = representations[:limit]

        if not representations:
            logger.warning("Run %s: no representable documents to validate", run_id)
            result = ClusterValidationResult(run_id=run_id, n_documents=0)
            return dataclasses.replace(
                result, report_path=write_validation_report(result, output_dir)
            )

        # -- embed once; the sweep only varies clustering ---------------
        with metrics.record_phase(
            run_id=metrics_run_id, phase="embed", batch_id=run_id
        ) as embed_metric:
            active_embedder = embedder or get_embedder(
                discovery.embedding_model_name,
                batch_size=discovery.embedding_batch_size,
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
            embed_metric.processed = len(pooled)

        doc_ids = list(pooled)
        vectors = np.stack([pooled[d] for d in doc_ids]) if doc_ids else np.zeros((0, 0))

        reduced = reduce_dimensions(
            vectors,
            n_components=discovery.umap_n_components,
            n_neighbors=discovery.umap_n_neighbors,
            min_dist=discovery.umap_min_dist,
            metric=discovery.umap_metric,
            min_docs=discovery.umap_min_docs,
        )

        # Labels are built here but handed only to the evaluator, never to
        # a clusterer.
        ground_truth = labels_from_source_folder(representations, depth=label_depth)
        logger.info(
            "Run %s: %d document(s), %d with source-folder labels (%d distinct)",
            run_id, len(doc_ids), len(ground_truth), len(set(ground_truth.values())),
        )

        reports: list[ValidationReport] = []
        with metrics.record_phase(
            run_id=metrics_run_id, phase="cluster", batch_id=run_id
        ) as cluster_metric:
            for combo in _sweep_settings(validation):
                min_cluster_size = (
                    combo["hdbscan_min_cluster_size"]
                    or discovery.hdbscan_min_cluster_size
                )
                active_clusterer = clusterer or hdbscan_clusterer(
                    min_cluster_size=min_cluster_size,
                    min_samples=combo["hdbscan_min_samples"],
                    metric=discovery.hdbscan_metric,
                )
                outcome = active_clusterer(reduced)

                parameters = {
                    "embedding_model": discovery.embedding_model_name,
                    "umap_n_components": discovery.umap_n_components,
                    "umap_n_neighbors": discovery.umap_n_neighbors,
                    "umap_metric": discovery.umap_metric,
                    "umap_min_docs": discovery.umap_min_docs,
                    "umap_applied": reduced.shape[1] != vectors.shape[1],
                    "hdbscan_min_cluster_size": min_cluster_size,
                    "hdbscan_min_samples": combo["hdbscan_min_samples"],
                    "hdbscan_metric": discovery.hdbscan_metric,
                }
                report = evaluate_clustering(
                    run_id=run_id,
                    doc_ids=doc_ids,
                    labels=outcome.labels,
                    ground_truth=ground_truth,
                    probabilities=outcome.probabilities,
                    parameters=parameters,
                    label_source=f"source_folder(depth={label_depth})",
                    min_coverage=validation.min_coverage,
                    min_purity_lift=validation.min_purity_lift,
                    min_ari=validation.min_adjusted_rand,
                    min_label_recall=validation.min_label_recall,
                )
                reports.append(report)
                _log_report(report)
            cluster_metric.processed = len(reports)

        # "Best" = strongest evidence that the signal exists at all:
        # agreement first, then lift, then coverage. It is a summary of the
        # sweep, not a recommendation to adopt those parameters.
        best = max(
            reports,
            key=lambda r: (
                r.adjusted_rand_index if r.adjusted_rand_index is not None else -1.0,
                r.purity_lift,
                r.coverage,
            ),
            default=None,
        )

        result = ClusterValidationResult(
            run_id=run_id,
            n_documents=len(doc_ids),
            reports=reports,
            best=best,
        )
        report_path = write_validation_report(result, output_dir)

        if best is not None:
            logger.info(
                "Run %s verdict: %s -- %s",
                run_id, best.verdict, "; ".join(best.verdict_reasons),
            )
            if not best.cluster_is_a_useful_signal:
                logger.warning(
                    "Cluster membership is NOT a reliable domain signal for this "
                    "corpus (verdict=%s). Phase 5 should not weight it as one.",
                    best.verdict,
                )
        logger.info("Validation report written to %s", report_path)

    return dataclasses.replace(result, report_path=report_path)
