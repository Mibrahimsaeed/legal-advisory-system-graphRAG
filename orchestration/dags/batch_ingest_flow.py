"""Stage 0: claim a batch of pending manifest entries, then "pull" them.

Runs after :func:`src.ingestion.discovery.discover_local_documents` (or an
equivalent registration step) has populated ``document_manifest``, and
before :func:`orchestration.dags.feature_extraction_flow.run_stage1`.

What "pull" means depends on ``settings.storage.backend``:

* ``"local"`` -- the manifest's ``source_uri`` is *already* a path on a
  mounted disk (internal, external, or network share) that Stage 1 can
  open directly. Pulling is therefore just a readability check: confirm
  the file still exists and is openable. **No byte copy happens** -- not
  into ``var/scratch``, not anywhere else in the project. This is the
  zero-copy path for "ingest straight off an external disk."
* ``"s3"`` -- a PDF library can't open an S3 object directly, so bytes
  genuinely have to land on a local filesystem somewhere first. This path
  downloads each object into the batch's ephemeral scratch workspace
  (purged once Stage 1 finishes with it, per
  ``docs/data_retention_policy.md``). Requires ``boto3``; raises a clear
  :class:`~src.common.exceptions.ConfigurationError` up front if it's
  missing rather than failing midway through a batch.

Checkpointed phases: ``claim`` (fixes which manifest rows this batch
owns, so a resume doesn't claim a second, different batch) and ``pull``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from src.common.checkpoint import CheckpointManager
from src.common.config import get_settings
from src.common.db import DEFAULT_DB_PATH, init_schema
from src.common.exceptions import ConfigurationError
from src.common.logging_utils import current_run_id, get_logger, log_context
from src.common.metrics import MetricsStore
from src.ingestion import manifest as manifest_db
from src.ingestion.manifest import ManifestEntry
from src.ingestion.scratch_manager import DEFAULT_SCRATCH_ROOT, ScratchWorkspace

logger = get_logger(__name__)



@dataclass(frozen=True)
class Stage0Result:
    batch_id: str
    entries: list[ManifestEntry]
    # How many manifest rows this batch actually claimed, *before* any pull
    # attempt -- deliberately separate from len(entries) (which is only the
    # subset that were *successfully* pulled). A caller looping batch after
    # batch needs this to tell "this batch claimed 0 because there's nothing
    # left pending" apart from "this batch claimed some but pulling all of
    # them failed" -- the latter should not be treated as "done".
    claimed_count: int = 0

    @property
    def pulled_doc_ids(self) -> list[str]:
        return [e.doc_id for e in self.entries]

def _pull_local(entry: ManifestEntry) -> str | None:
    """Zero-copy pull for a locally-mounted source.

    Verifies the file is actually there and openable, but never reads or
    copies its bytes -- Stage 1 will open ``entry.source_uri`` itself.
    Returns an error string on failure, ``None`` on success.
    """

    path = Path(entry.source_uri)
    if not path.exists():
        return f"Source file not found at {path} (disk unmounted or moved?)"
    if not path.is_file():
        return f"Source path is not a file: {path}"
    try:
        with path.open("rb"):
            pass
    except OSError as exc:
        return f"Source file exists but is not readable: {exc}"
    return None


def _pull_s3(entry: ManifestEntry, workspace: ScratchWorkspace, bucket: str) -> str | None:
    """Download one object into scratch. Only path that ever copies bytes."""

    try:
        import boto3
    except ImportError as exc:
        raise ConfigurationError(
            "storage.backend='s3' but boto3 is not installed"
        ) from exc

    client = boto3.client("s3")
    suffix = Path(entry.source_uri).suffix or ".pdf"
    dest = workspace.path_for_doc(entry.doc_id, suffix=suffix)
    dest.parent.mkdir(parents=True, exist_ok=True)

    key = entry.source_uri
    prefix = f"s3://{bucket}/"
    if key.startswith(prefix):
        key = key[len(prefix):]

    try:
        client.download_file(bucket, key, str(dest))
    except Exception as exc:  # noqa: BLE001 - surfaced as a pull failure, not raised
        return f"{type(exc).__name__}: {exc}"
    return None


def run_stage0(
    batch_size: int | None = None,
    db_path: str | Path = DEFAULT_DB_PATH,
    scratch_root: str | Path = DEFAULT_SCRATCH_ROOT,
    checkpoint_dir: str | Path | None = None,
    metrics_db_path: str | Path | None = None,
    batch_id: str | None = None,
) -> Stage0Result:
    """Claim a batch of pending documents and pull them per ``storage.backend``.

    Pass ``batch_id`` back in to resume a specific in-flight batch (e.g.
    after a crash between ``claim`` and ``pull``); omit it to claim a
    fresh batch.
    """

    settings = get_settings()
    batch_size = batch_size or settings.pipeline.batch_size
    checkpoint_dir = checkpoint_dir or settings.pipeline.checkpoint_dir
    metrics_db_path = metrics_db_path or settings.metrics.db_path
    storage = settings.storage

    init_schema(db_path=db_path, schema_file=settings.database.schema_file)

    metrics = MetricsStore(metrics_db_path)
    metrics.init_schema()
    run_id = current_run_id()

    checkpoint = CheckpointManager(
        checkpoint_dir=checkpoint_dir,
        run_key=batch_id or manifest_db.new_batch_id(),
        phases=["claim", "pull"],
    )
    batch_id = checkpoint.run_key
    workspace = ScratchWorkspace(batch_id=batch_id, root=Path(scratch_root))

    with log_context(batch_id=batch_id, phase="pull"):
        # -- claim ------------------------------------------------------
        with metrics.record_phase(run_id=run_id, phase="claim", batch_id=batch_id) as claim_metric:
            if checkpoint.is_completed("claim"):
                entries = manifest_db.get_batch_entries(batch_id, db_path=db_path)
                logger.info(
                    "Resuming batch %s: %d entries already claimed", batch_id, len(entries),
                )
            else:
                checkpoint.start_phase("claim")
                _, entries = manifest_db.claim_batch(
                    batch_size, db_path=db_path, batch_id=batch_id
                )
                checkpoint.complete_phase("claim", state={"doc_count": len(entries)})
            claim_metric.processed = len(entries)

        if not entries:
            logger.info("No pending documents to claim for batch %s; nothing to pull", batch_id)
            checkpoint.complete_phase("pull", state={"pulled": 0, "failed": 0})
            return Stage0Result(batch_id=batch_id, entries=[])

        # -- pull ---------------------------------------------------------
        with metrics.record_phase(run_id=run_id, phase="pull", batch_id=batch_id) as pull_metric:
            if checkpoint.is_completed("pull"):
                pulled_entries = [
                    e for e in manifest_db.get_batch_entries(batch_id, db_path=db_path)
                    if e.ingest_status == "pulled"
                ]
                logger.info("Resuming batch %s: pull already completed", batch_id)
                return Stage0Result(batch_id=batch_id, entries=pulled_entries)

            checkpoint.start_phase("pull")
            pulled: list[str] = []
            failed: list[tuple[str, str]] = []

            for entry in entries:
                with log_context(doc_id=entry.doc_id):
                    if storage.backend == "local":
                        error = _pull_local(entry)
                    elif storage.backend == "s3":
                        if not storage.bucket:
                            error = "storage.backend='s3' but storage.bucket is not configured"
                        else:
                            error = _pull_s3(entry, workspace, storage.bucket)
                    else:
                        error = f"Unsupported storage backend: {storage.backend!r}"

                    if error is None:
                        pulled.append(entry.doc_id)
                    else:
                        logger.error("Pull failed for %s: %s", entry.doc_id, error)
                        failed.append((entry.doc_id, error))

            if pulled:
                manifest_db.mark_status(pulled, "pulled", db_path=db_path)
            for doc_id, error in failed:
                manifest_db.mark_status(
                    [doc_id], "pull_failed", db_path=db_path,
                    error=error, increment_attempts=True,
                )

            pull_metric.processed = len(pulled)
            pull_metric.failed = len(failed)
            checkpoint.complete_phase(
                "pull", state={"pulled": len(pulled), "failed": len(failed)}
            )

        manifest_db.set_batch_status(
            batch_id,
            "pulled" if not failed else ("failed" if not pulled else "pulled"),
            db_path=db_path,
            notes=f"{len(pulled)} pulled, {len(failed)} failed",
        )

        logger.info(
            "Stage 0 complete for batch %s: %d pulled, %d failed",
            batch_id, len(pulled), len(failed),
        )

        final_entries = [
            e for e in manifest_db.get_batch_entries(batch_id, db_path=db_path)
            if e.ingest_status == "pulled"
        ]
        return Stage0Result(batch_id=batch_id, entries=final_entries)