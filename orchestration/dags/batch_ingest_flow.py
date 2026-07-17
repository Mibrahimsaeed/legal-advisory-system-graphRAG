from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from src.common.checkpoint import CheckpointManager
from src.common.config import get_settings
from src.common.db import DEFAULT_DB_PATH, init_schema
from src.common.exceptions import IngestionError
from src.common.logging_utils import current_run_id, get_logger, log_context
from src.common.metrics import MetricsStore
from src.ingestion import manifest as manifest_db
from src.ingestion.batch_puller import PullResult, pull_batch
from src.ingestion.scratch_manager import (
    DEFAULT_SCRATCH_ROOT,
    ScratchWorkspace,
)

logger = get_logger(__name__)


@dataclass(frozen=True)
class Stage0Result:
    batch_id: str
    workspace: ScratchWorkspace
    pull_results: list[PullResult]

    @property
    def succeeded_doc_ids(self) -> list[str]:
        return [
            r.doc_id
            for r in self.pull_results
            if r.success
        ]

    @property
    def failed_doc_ids(self) -> list[str]:
        return [
            r.doc_id
            for r in self.pull_results
            if not r.success
        ]


def run_stage0(
    batch_size: int | None = None,
    db_path: str | Path = DEFAULT_DB_PATH,
    scratch_root: str | Path = DEFAULT_SCRATCH_ROOT,
    batch_id: str | None = None,
    checkpoint_dir: str | Path | None = None,
    metrics_db_path: str | Path | None = None,
) -> Stage0Result:
    """Run Stage 0 (claim + pull) with checkpoint/resume and metrics support.

    ``batch_size`` / ``checkpoint_dir`` / ``metrics_db_path`` default to the
    values in :func:`~src.common.config.get_settings` when not supplied
    explicitly. If a batch previously failed partway through, re-running
    with the same ``batch_id`` skips phases already marked completed in the
    checkpoint.
    """

    settings = get_settings()
    batch_size = batch_size or settings.pipeline.batch_size
    checkpoint_dir = checkpoint_dir or settings.pipeline.checkpoint_dir
    metrics_db_path = metrics_db_path or settings.metrics.db_path

    init_schema(db_path=db_path)

    metrics = MetricsStore(metrics_db_path)
    metrics.init_schema()
    run_id = current_run_id()

    with log_context(phase="stage0"):
        with metrics.record_phase(run_id=run_id, phase="claim", batch_id=batch_id) as claim_metric:
            claimed_batch_id, entries = manifest_db.claim_batch(
                batch_size=batch_size,
                db_path=db_path,
                batch_id=batch_id,
            )
            claim_metric.processed = len(entries)

        checkpoint = CheckpointManager(
            checkpoint_dir=checkpoint_dir,
            run_key=claimed_batch_id,
            phases=["claim", "pull"],
        )
        checkpoint.complete_phase("claim", state={"doc_count": len(entries)})

        with log_context(batch_id=claimed_batch_id):
            if not entries:
                logger.info(
                    "No pending documents available; batch %s is empty",
                    claimed_batch_id,
                )

                manifest_db.set_batch_status(
                    claimed_batch_id,
                    "pulled",
                    db_path=db_path,
                    notes="empty batch",
                    finished=True,
                )
                checkpoint.complete_phase("pull", state={"succeeded": 0, "failed": 0})

                empty_workspace = ScratchWorkspace(
                    batch_id=claimed_batch_id,
                    root=Path(scratch_root),
                )

                return Stage0Result(
                    batch_id=claimed_batch_id,
                    workspace=empty_workspace,
                    pull_results=[],
                )

            if checkpoint.is_completed("pull"):
                logger.info(
                    "Batch %s already pulled per checkpoint; skipping re-pull",
                    claimed_batch_id,
                )
                workspace = ScratchWorkspace(
                    batch_id=claimed_batch_id,
                    root=Path(scratch_root),
                )
                return Stage0Result(
                    batch_id=claimed_batch_id,
                    workspace=workspace,
                    pull_results=[],
                )

            manifest_db.set_batch_status(
                claimed_batch_id,
                "pulling",
                db_path=db_path,
            )

            workspace = ScratchWorkspace(
                batch_id=claimed_batch_id,
                root=Path(scratch_root),
            )

            workspace._create()

            with log_context(phase="pull"), metrics.record_phase(
                run_id=run_id, phase="pull", batch_id=claimed_batch_id
            ) as pull_metric:
                try:
                    pull_results = pull_batch(
                        entries,
                        workspace,
                        db_path=db_path,
                    )

                except Exception as exc:
                    logger.exception(
                        "Stage 0 pull_batch raised unexpectedly for batch %s",
                        claimed_batch_id,
                    )

                    manifest_db.set_batch_status(
                        claimed_batch_id,
                        "failed",
                        db_path=db_path,
                        notes="pull_batch raised",
                        finished=True,
                    )
                    checkpoint.fail_phase("pull", f"{type(exc).__name__}: {exc}")

                    workspace.purge()
                    raise IngestionError(
                        "pull_batch raised unexpectedly",
                        phase="pull",
                        batch_id=claimed_batch_id,
                        cause=exc,
                    ) from exc

                succeeded = sum(1 for r in pull_results if r.success)
                failed = len(pull_results) - succeeded
                pull_metric.processed = succeeded
                pull_metric.failed = failed

            status = (
                "pulled"
                if failed == 0
                else ("failed" if succeeded == 0 else "pulled")
            )

            manifest_db.set_batch_status(
                claimed_batch_id,
                status,
                db_path=db_path,
                notes=f"{succeeded} succeeded, {failed} failed",
            )
            checkpoint.complete_phase("pull", state={"succeeded": succeeded, "failed": failed})

            logger.info(
                "Stage 0 complete for batch %s: %d succeeded, %d failed",
                claimed_batch_id,
                succeeded,
                failed,
            )

            return Stage0Result(
                batch_id=claimed_batch_id,
                workspace=workspace,
                pull_results=pull_results,
            )