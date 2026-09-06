"""Stage 2: classify the case-law corpus against the frozen taxonomy.

    frozen taxonomy (config/domains.yaml)
        + document_representations
        + cluster_assignments (optional, for provenance)
        -> one LLM call per document, validated
        -> document_classifications

Batching, resumability and the review queue are the whole point of this
module; the per-document decision lives in
:mod:`src.classification.domain_classifier`.

* **Batches.** Documents are processed in ``classification.batch_size``
  chunks, in deterministic ``doc_id`` order, and each batch is persisted
  before the next starts. A crash costs at most one batch.
* **Resumability.** A run skips documents it has already decided
  (:func:`~src.classification.classification_store.get_classified_doc_ids`),
  so re-invoking with the same ``run_id`` continues where it stopped
  without re-spending LLM calls. Progress is also checkpointed per batch.
* **Pilot first.** ``pilot_size`` classifies a bounded slice and returns
  the same statistics as a full run -- see ``scripts/run_pipeline.py
  classify --pilot N``. Failed documents are NOT skipped on resume by
  default (``retry_failed=True``): a transient LLM error should be
  retried, while a document already classified is never re-billed.

Nothing here writes the taxonomy, mutates another run's rows, or touches
retrieval/RAG.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from src.classification.classification_store import (
    DEFAULT_CLASSIFICATION_SCHEMA_FILE,
    classification_stats,
    get_classifications_for_run,
    persist_classifications,
)
from src.classification.domain_classifier import (
    CLASSIFIER_VERSION,
    STATUS_FAILED,
    STATUS_NEEDS_REVIEW,
    ClassificationResult,
    classify_document,
    render_taxonomy_prompt,
)
from src.classification.taxonomy_registry import (
    FrozenTaxonomy,
    load_frozen_taxonomy,
)
from src.common.checkpoint import CheckpointManager
from src.common.config import get_settings
from src.common.db import DEFAULT_DB_PATH, connection_scope, init_schema
from src.common.llm_client import LLMClient, get_llm_client
from src.common.logging_utils import current_run_id, get_logger, log_context
from src.common.metrics import MetricsStore
from src.extraction.representation_store import list_representations

logger = get_logger(__name__)


@dataclass(frozen=True)
class ClassificationRunResult:
    run_id: str
    taxonomy_version: str
    classifier_version: str
    model_name: str
    processed: int = 0
    skipped_already_done: int = 0
    results: list[ClassificationResult] = field(default_factory=list)
    stats: dict = field(default_factory=dict)
    is_pilot: bool = False

    @property
    def needs_review(self) -> list[ClassificationResult]:
        return [r for r in self.results if r.status == STATUS_NEEDS_REVIEW]

    @property
    def failed(self) -> list[ClassificationResult]:
        return [r for r in self.results if r.status == STATUS_FAILED]


def _cluster_ids_for(
    cluster_run_id: str | None, db_path: str | Path
) -> dict[str, int]:
    """doc_id -> cluster_id from a discovery/review run, if one is given."""

    if not cluster_run_id:
        return {}
    with connection_scope(db_path) as conn:
        rows = conn.execute(
            "SELECT doc_id, cluster_id FROM cluster_assignments WHERE run_id = ?",
            (cluster_run_id,),
        ).fetchall()
    return {r["doc_id"]: r["cluster_id"] for r in rows}


def _batches(items: list, size: int):
    for start in range(0, len(items), max(size, 1)):
        yield start // max(size, 1), items[start : start + max(size, 1)]


def run_classification(
    run_id: str,
    db_path: str | Path = DEFAULT_DB_PATH,
    taxonomy_path: str | Path | None = None,
    cluster_run_id: str | None = None,
    batch_size: int | None = None,
    pilot_size: int | None = None,
    checkpoint_dir: str | Path | None = None,
    metrics_db_path: str | Path | None = None,
    llm_client: LLMClient | None = None,
    taxonomy: FrozenTaxonomy | None = None,
    retry_failed: bool = True,
) -> ClassificationRunResult:
    """Classify the corpus (or a pilot slice) under one ``run_id``.

    ``run_id`` is the version boundary: reuse it to resume, and choose a
    new one whenever the taxonomy or classifier version changes, so the
    previous verdicts stay intact and comparable.
    """

    settings = get_settings()
    classification = settings.classification
    checkpoint_dir = checkpoint_dir or settings.pipeline.checkpoint_dir
    metrics_db_path = metrics_db_path or settings.metrics.db_path
    batch_size = batch_size or classification.batch_size
    taxonomy_path = taxonomy_path or classification.taxonomy_file

    # Fails loudly when the taxonomy is missing/empty -- classifying a
    # corpus against a taxonomy that was never frozen is the one outcome
    # worse than not classifying it.
    taxonomy = taxonomy or load_frozen_taxonomy(taxonomy_path)

    init_schema(db_path=db_path, schema_file=settings.caselaw.representation_schema_file)
    init_schema(db_path=db_path, schema_file=classification.schema_file)
    # cluster_assignments lives in the domain-registry schema; without it a
    # --cluster-run-id lookup raises "no such table" on a fresh database.
    init_schema(db_path=db_path, schema_file=settings.discovery.domain_registry_schema_file)

    metrics = MetricsStore(metrics_db_path)
    metrics.init_schema()
    metrics_run_id = current_run_id()

    corpus = sorted(list_representations(db_path=db_path), key=lambda r: r.doc_id)

    done_by_status = {
        row["doc_id"]: row["status"] for row in get_classifications_for_run(run_id, db_path=db_path)
    }
    already_done = {
        doc_id
        for doc_id, status in done_by_status.items()
        if not (retry_failed and status == STATUS_FAILED)
    }
    pending = [d for d in corpus if d.doc_id not in already_done]

    is_pilot = bool(pilot_size)
    if is_pilot:
        pending = pending[:pilot_size]

    checkpoint = CheckpointManager(
        checkpoint_dir=checkpoint_dir,
        run_key=f"classify_{run_id}",
        phases=["classify"],
    )

    cluster_ids = _cluster_ids_for(cluster_run_id, db_path)
    active_llm_client = llm_client or get_llm_client(
        classification.llm_model, max_tokens=classification.llm_max_tokens
    )
    taxonomy_block = render_taxonomy_prompt(taxonomy)

    all_results: list[ClassificationResult] = []

    with log_context(batch_id=run_id, phase="classify"):
        logger.info(
            "Classification run %s: %d document(s) pending of %d in corpus "
            "(%d already done)%s; taxonomy=%s classifier=%s model=%s",
            run_id, len(pending), len(corpus), len(already_done),
            f"; PILOT limited to {pilot_size}" if is_pilot else "",
            taxonomy.version, CLASSIFIER_VERSION, classification.llm_model,
        )

        if not pending:
            stats = classification_stats(run_id, db_path=db_path)
            return ClassificationRunResult(
                run_id=run_id,
                taxonomy_version=taxonomy.version,
                classifier_version=CLASSIFIER_VERSION,
                model_name=classification.llm_model,
                processed=0,
                skipped_already_done=len(already_done),
                stats=stats,
                is_pilot=is_pilot,
            )

        checkpoint.start_phase("classify")

        with metrics.record_phase(
            run_id=metrics_run_id, phase="classify", batch_id=run_id
        ) as metric:
            for index, batch in _batches(pending, batch_size):
                batch_id = f"{run_id}_b{index:04d}"
                batch_results: list[ClassificationResult] = []

                for document in batch:
                    result = classify_document(
                        document,
                        taxonomy,
                        active_llm_client,
                        taxonomy_block=taxonomy_block,
                        min_confidence=classification.min_confidence,
                        body_chars=classification.body_chars,
                        review_multi_domain=classification.review_multi_domain,
                        review_other_bucket=classification.review_other_bucket,
                    )
                    result.cluster_id = cluster_ids.get(document.doc_id)
                    batch_results.append(result)

                # Persisted per batch: a crash costs this batch, not the run.
                persist_classifications(
                    run_id,
                    batch_results,
                    taxonomy_version=taxonomy.version,
                    classifier_version=CLASSIFIER_VERSION,
                    model_name=classification.llm_model,
                    batch_id=batch_id,
                    db_path=db_path,
                )
                all_results.extend(batch_results)

                checkpoint.set_state(
                    "classify",
                    {
                        "last_batch_index": index,
                        "processed": len(all_results),
                        "taxonomy_version": taxonomy.version,
                        "classifier_version": CLASSIFIER_VERSION,
                    },
                )
                logger.info(
                    "Batch %s: %d classified, %d need review, %d failed "
                    "(%d/%d done this run)",
                    batch_id,
                    sum(1 for r in batch_results if r.status not in {STATUS_FAILED, STATUS_NEEDS_REVIEW}),
                    sum(1 for r in batch_results if r.status == STATUS_NEEDS_REVIEW),
                    sum(1 for r in batch_results if r.status == STATUS_FAILED),
                    len(all_results), len(pending),
                )

            metric.processed = sum(1 for r in all_results if r.status != STATUS_FAILED)
            metric.failed = sum(1 for r in all_results if r.status == STATUS_FAILED)

        checkpoint.complete_phase(
            "classify",
            state={
                "processed": len(all_results),
                "taxonomy_version": taxonomy.version,
                "classifier_version": CLASSIFIER_VERSION,
                "pilot": is_pilot,
            },
        )

        stats = classification_stats(run_id, db_path=db_path)
        logger.info(
            "Classification run %s complete: %d processed, needs_review=%.1f%%, "
            "failures=%.1f%%",
            run_id, len(all_results),
            stats["needs_review_rate"] * 100, stats["failure_rate"] * 100,
        )

    return ClassificationRunResult(
        run_id=run_id,
        taxonomy_version=taxonomy.version,
        classifier_version=CLASSIFIER_VERSION,
        model_name=classification.llm_model,
        processed=len(all_results),
        skipped_already_done=len(already_done),
        results=all_results,
        stats=stats,
        is_pilot=is_pilot,
    )
