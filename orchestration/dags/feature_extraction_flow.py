"""Stage 1: Feature Extraction (document signatures).

Consumes documents marked as ``pulled`` by Stage 0.

Supports:
- storage.backend="local":
    Reads PDFs directly from the mounted disk.
    No copying. No deletion.
- storage.backend="s3":
    Reads PDFs downloaded into scratch workspace.
    Scratch files are deleted after extraction.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from src.common.checkpoint import CheckpointManager
from src.common.config import get_settings
from src.common.db import DEFAULT_DB_PATH, init_schema
from src.common.exceptions import ExtractionError
from src.common.logging_utils import current_run_id, get_logger, log_context
from src.common.metrics import MetricsStore

from src.extraction.signature import DocumentSignature
from src.extraction.signature_builder import build_signature
from src.extraction.signature_store import (
    get_signatures_for_batch,
    upsert_signatures,
)

from src.ingestion import manifest as manifest_db
from src.ingestion.scratch_manager import (
    DEFAULT_SCRATCH_ROOT,
    ScratchWorkspace,
)

logger = get_logger(__name__)


@dataclass(frozen=True)
class Stage1Result:
    batch_id: str
    signatures: list[DocumentSignature]

    @property
    def succeeded_doc_ids(self) -> list[str]:
        return [
            s.doc_id
            for s in self.signatures
            if s.extraction_status != "failed"
        ]

    @property
    def failed_doc_ids(self) -> list[str]:
        return [
            s.doc_id
            for s in self.signatures
            if s.extraction_status == "failed"
        ]


def _find_pulled_file(
    workspace: ScratchWorkspace,
    doc_id: str,
) -> Path | None:
    """
    Locate file downloaded by Stage 0 for S3 backend.
    """

    safe_name = doc_id.replace("/", "_")

    matches = sorted(
        workspace.raw_dir.glob(f"{safe_name}.*")
    ) or sorted(
        workspace.raw_dir.glob(safe_name)
    )

    return matches[0] if matches else None


def _resolve_source_path(
    entry: manifest_db.ManifestEntry,
    workspace: ScratchWorkspace,
    backend: str,
) -> tuple[Path | None, bool]:
    """
    Resolve document path.

    Returns:
        (path, is_scratch_copy)

    is_scratch_copy:
        True  -> safe to delete after extraction
        False -> original source file, never delete
    """

    if backend == "local":
        path = Path(entry.source_uri)

        if path.exists():
            return path, False

        return None, False


    # S3 path
    path = _find_pulled_file(
        workspace,
        entry.doc_id,
    )

    return path, True



def run_stage1(
    batch_id: str,
    db_path: str | Path = DEFAULT_DB_PATH,
    scratch_root: str | Path = DEFAULT_SCRATCH_ROOT,
    checkpoint_dir: str | Path | None = None,
    metrics_db_path: str | Path | None = None,
) -> Stage1Result:


    settings = get_settings()

    checkpoint_dir = (
        checkpoint_dir
        or settings.pipeline.checkpoint_dir
    )

    metrics_db_path = (
        metrics_db_path
        or settings.metrics.db_path
    )

    extraction = settings.extraction
    storage = settings.storage


    init_schema(
        db_path=db_path,
        schema_file=settings.database.schema_file,
    )

    init_schema(
        db_path=db_path,
        schema_file=extraction.signature_schema_file,
    )


    metrics = MetricsStore(metrics_db_path)
    metrics.init_schema()

    run_id = current_run_id()


    checkpoint = CheckpointManager(
        checkpoint_dir=checkpoint_dir,
        run_key=batch_id,
        phases=[
            "claim",
            "pull",
            "extract",
        ],
    )


    workspace = ScratchWorkspace(
        batch_id=batch_id,
        root=Path(scratch_root),
    )


    with log_context(
        batch_id=batch_id,
        phase="extract",
    ):


        if checkpoint.is_completed("extract"):

            logger.info(
                "Batch %s already extracted",
                batch_id,
            )

            return Stage1Result(
                batch_id=batch_id,
                signatures=get_signatures_for_batch(
                    batch_id,
                    db_path=db_path,
                ),
            )



        entries = [
            e
            for e in manifest_db.get_batch_entries(
                batch_id,
                db_path=db_path,
            )
            if e.ingest_status == "pulled"
        ]


        if not entries:

            logger.info(
                "No pulled documents for batch %s",
                batch_id,
            )

            checkpoint.complete_phase(
                "extract",
                state={
                    "succeeded":0,
                    "failed":0,
                },
            )

            return Stage1Result(
                batch_id=batch_id,
                signatures=[],
            )



        checkpoint.start_phase("extract")


        manifest_db.set_batch_status(
            batch_id,
            "extracting",
            db_path=db_path,
        )


        signatures: list[DocumentSignature] = []


        try:

            with metrics.record_phase(
                run_id=run_id,
                phase="extract",
                batch_id=batch_id,
            ) as extract_metric:


                for entry in entries:


                    with log_context(
                        doc_id=entry.doc_id
                    ):


                        path, is_scratch_copy = _resolve_source_path(
                            entry,
                            workspace,
                            storage.backend,
                        )



                        if path is None:

                            logger.error(
                                "Source missing for %s",
                                entry.doc_id,
                            )


                            signature = DocumentSignature(
                                doc_id=entry.doc_id,
                                source_uri=entry.source_uri,
                                signature_hash="",
                                is_scanned=False,
                                extraction_status="failed",
                                extractor_used="none",
                                batch_id=batch_id,
                                error="Source file unavailable",
                            )


                        else:

                            signature = build_signature(
                                entry.doc_id,
                                entry.source_uri,
                                path,

                                max_pages=extraction.max_pages,
                                min_chars_per_page=extraction.min_chars_per_page,
                                min_alpha_ratio=extraction.min_alpha_ratio,

                                ocr_dpi=extraction.ocr_dpi,
                                ocr_lang=extraction.ocr_lang,

                                body_preview_char_limit=(
                                    extraction.body_preview_char_limit
                                ),

                                batch_id=batch_id,
                            )



                            # ONLY delete S3 scratch copies
                            if is_scratch_copy:
                                path.unlink(
                                    missing_ok=True
                                )


                        signatures.append(signature)



                succeeded = [
                    s
                    for s in signatures
                    if s.extraction_status != "failed"
                ]


                failed = [
                    s
                    for s in signatures
                    if s.extraction_status == "failed"
                ]


                extract_metric.processed = len(succeeded)
                extract_metric.failed = len(failed)



        except Exception as exc:

            logger.exception(
                "Stage 1 failed for batch %s",
                batch_id,
            )

            checkpoint.fail_phase(
                "extract",
                f"{type(exc).__name__}: {exc}",
            )

            raise ExtractionError(
                "Stage 1 extraction failed",
                phase="extract",
                batch_id=batch_id,
                cause=exc,
            ) from exc



        upsert_signatures(
            signatures,
            db_path=db_path,
        )


        succeeded = [
            s
            for s in signatures
            if s.extraction_status != "failed"
        ]

        failed = [
            s
            for s in signatures
            if s.extraction_status == "failed"
        ]


        if succeeded:
            manifest_db.mark_status(
                [
                    s.doc_id
                    for s in succeeded
                ],
                "extracted",
                db_path=db_path,
            )


        if failed:

            for s in failed:

                manifest_db.mark_status(
                    [s.doc_id],
                    "extraction_failed",
                    db_path=db_path,
                    error=s.error,
                    increment_attempts=True,
                )


        workspace.purge()


        manifest_db.set_batch_status(
            batch_id,
            "extracted"
            if not failed
            else "failed",

            db_path=db_path,

            notes=(
                f"{len(succeeded)} succeeded, "
                f"{len(failed)} failed"
            ),

            finished=True,
        )


        checkpoint.complete_phase(
            "extract",
            state={
                "succeeded":len(succeeded),
                "failed":len(failed),
            },
        )


        logger.info(
            "Stage 1 complete for batch %s: %d succeeded, %d failed",
            batch_id,
            len(succeeded),
            len(failed),
        )


        return Stage1Result(
            batch_id=batch_id,
            signatures=signatures,
        )