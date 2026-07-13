from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from src.common.db import DEFAULT_DB_PATH, init_schema
from src.common.logging_utils import get_logger
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
    batch_size: int = 500,
    db_path: str | Path = DEFAULT_DB_PATH,
    scratch_root: str | Path = DEFAULT_SCRATCH_ROOT,
    batch_id: str | None = None,
) -> Stage0Result:
    init_schema(db_path=db_path)

    claimed_batch_id, entries = manifest_db.claim_batch(
        batch_size=batch_size,
        db_path=db_path,
        batch_id=batch_id,
    )

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

        empty_workspace = ScratchWorkspace(
            batch_id=claimed_batch_id,
            root=Path(scratch_root),
        )

        return Stage0Result(
            batch_id=claimed_batch_id,
            workspace=empty_workspace,
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

    try:
        pull_results = pull_batch(
            entries,
            workspace,
            db_path=db_path,
        )

    except Exception:
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

        workspace.purge()
        raise

    succeeded = sum(
        1
        for r in pull_results
        if r.success
    )

    failed = len(pull_results) - succeeded

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