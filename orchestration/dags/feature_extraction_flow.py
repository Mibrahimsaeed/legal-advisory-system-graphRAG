"""Stage 1: Feature Extraction (document signatures).

Consumes the scratch workspace :func:`orchestration.dags.batch_ingest_flow.run_stage0`
populated -- i.e. this runs *after* a batch has been claimed and pulled. For
every document still marked ``pulled`` in the manifest, this builds a
:class:`~src.extraction.signature.DocumentSignature` (text-layer extraction,
OCR fallback only for low-quality pages, cleaning/normalization), persists
it, and updates the manifest status.

Raw PDF bytes are purged from scratch per document immediately after that
document's signature has been built (success or failure) -- and the whole
batch workspace is purged again at the end as a safety net -- per
``docs/data_retention_policy.md``: only the lightweight signature record is
ever kept; full document text/binary never lands in long-term storage.
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
from src.extraction.signature_store import upsert_signatures
from src.ingestion import manifest as manifest_db
from src.ingestion.scratch_manager import DEFAULT_SCRATCH_ROOT, ScratchWorkspace

logger = get_logger(__name__)


@dataclass(frozen=True)
class Stage1Result:
    batch_id: str
    signatures: list[DocumentSignature]

    @property
    def succeeded_doc_ids(self) -> list[str]:
        return [s.doc_id for s in self.signatures if s.extraction_status != "failed"]

    @property
    def failed_doc_ids(self) -> list[str]:
        return [s.doc_id for s in self.signatures if s.extraction_status == "failed"]


def _find_pulled_file(workspace: ScratchWorkspace, doc_id: str) -> Path | None:
    """Locate the raw file Stage 0 pulled for ``doc_id`` in scratch.

    :meth:`ScratchWorkspace.path_for_doc` needs the original suffix to
    reconstruct the exact path, which this stage doesn't have on hand --
    so instead of re-deriving it from ``source_uri``, we just glob for
    whatever Stage 0 actually wrote.
    """

    safe_name = doc_id.replace("/", "_")
    matches = sorted(workspace.raw_dir.glob(f"{safe_name}.*")) or sorted(
        workspace.raw_dir.glob(f"{safe_name}")
    )
    return matches[0] if matches else None
def _resolve_source_path(
    entry: "manifest_db.ManifestEntry",
    workspace: ScratchWorkspace,
    backend: str,
) -> tuple[Path | None, bool]:
    """Resolve the readable path for one document, per ``storage.backend``.

    Returns ``(path, is_scratch_copy)``. ``is_scratch_copy`` tells the
    caller whether it's safe to delete ``path`` after extraction:

    * ``backend="local"`` -- ``path`` is ``entry.source_uri`` itself,
      i.e. the original file on the mounted (possibly external) disk.
      ``is_scratch_copy=False`` -- deleting it would delete the user's
      source data, which this pipeline must never do.
    * ``backend="s3"`` -- ``path`` is Stage 0's scratch download.
      ``is_scratch_copy=True`` -- safe, and intended, to delete once
      extraction is done with it (docs/data_retention_policy.md).
    """

    if backend == "local":
        path = Path(entry.source_uri)
        return (path, False) if path.exists() else (None, False)

    # s3 (and any other backend that requires a scratch download)
    return _find_pulled_file(workspace, entry.doc_id), True


def run_stage1(
    batch_id: str,
    db_path: str | Path = DEFAULT_DB_PATH,
    scratch_root: str | Path = DEFAULT_SCRATCH_ROOT,
    checkpoint_dir: str | Path | None = None,
    metrics_db_path: str | Path | None = None,
) -> Stage1Result:
    """Run Stage 1 (feature extraction) for a batch already pulled by Stage 0.

    Idempotent/resumable the same way :func:`~orchestration.dags.batch_ingest_flow.run_stage0`
    is: re-running with the same ``batch_id`` after a crash skips the
    ``extract`` phase entirely if the checkpoint already marked it
    completed, returning the signatures already persisted instead of
    rebuilding them.
    """

    settings = get_settings()
    checkpoint_dir = checkpoint_dir or settings.pipeline.checkpoint_dir
    metrics_db_path = metrics_db_path or settings.metrics.db_path
    extraction = settings.extraction

    init_schema(db_path=db_path, schema_file=settings.database.schema_file)
    init_schema(db_path=db_path, schema_file=extraction.signature_schema_file)

    metrics = MetricsStore(metrics_db_path)
    metrics.init_schema()
    run_id = current_run_id()

    checkpoint = CheckpointManager(
        checkpoint_dir=checkpoint_dir,
        run_key=batch_id,
        phases=["claim", "pull", "extract"],
    )

    workspace = ScratchWorkspace(batch_id=batch_id, root=Path(scratch_root))

    with log_context(batch_id=batch_id, phase="extract"):
        if checkpoint.is_completed("extract"):
            logger.info(
                "Batch %s already extracted per checkpoint; skipping re-extraction",
                batch_id,
            )
            from src.extraction.signature_store import get_signatures_for_batch

            return Stage1Result(
                batch_id=batch_id,
                signatures=get_signatures_for_batch(batch_id, db_path=db_path),
            )

        entries = [
            e
            for e in manifest_db.get_batch_entries(batch_id, db_path=db_path)
            if e.ingest_status == "pulled"
        ]

        if not entries:
            logger.info(
                "No 'pulled' documents to extract for batch %s; nothing to do",
                batch_id,
            )
            checkpoint.complete_phase("extract", state={"succeeded": 0, "failed": 0})
            return Stage1Result(batch_id=batch_id, signatures=[])

        checkpoint.start_phase("extract")
        manifest_db.set_batch_status(batch_id, "extracting", db_path=db_path)

        signatures: list[DocumentSignature] = []

        try:
            with metrics.record_phase(
                run_id=run_id, phase="extract", batch_id=batch_id
            ) as extract_metric:
                for entry in entries:
                    with log_context(doc_id=entry.doc_id):
                        local_path = _find_pulled_file(workspace, entry.doc_id)

                        if local_path is None:
                            logger.error(
                                "Pulled file missing from scratch for %s; "
                                "was it purged out from under Stage 1?",
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
                                error="Pulled file not found in scratch workspace",
                            )
                        else:
                            signature = build_signature(
                                entry.doc_id,
                                entry.source_uri,
                                local_path,
                                max_pages=extraction.max_pages,
                                min_chars_per_page=extraction.min_chars_per_page,
                                min_alpha_ratio=extraction.min_alpha_ratio,
                                ocr_dpi=extraction.ocr_dpi,
                                ocr_lang=extraction.ocr_lang,
                                body_preview_char_limit=extraction.body_preview_char_limit,
                                batch_id=batch_id,
                            )
                            # Ephemeral by design: drop the raw bytes the moment
                            # we've extracted whatever signal we're going to get
                            # out of them, success or not (docs/data_retention_policy.md).
                            local_path.unlink(missing_ok=True)

                        signatures.append(signature)

                succeeded = [s for s in signatures if s.extraction_status != "failed"]
                failed = [s for s in signatures if s.extraction_status == "failed"]
                extract_metric.processed = len(succeeded)
                extract_metric.failed = len(failed)

        except Exception as exc:
            logger.exception(
                "Stage 1 extraction raised unexpectedly for batch %s", batch_id
            )
            checkpoint.fail_phase("extract", f"{type(exc).__name__}: {exc}")
            manifest_db.set_batch_status(
                batch_id,
                "failed",
                db_path=db_path,
                notes="extract phase raised",
                finished=True,
            )
            workspace.purge()
            raise ExtractionError(
                "Stage 1 extraction raised unexpectedly",
                phase="extract",
                batch_id=batch_id,
                cause=exc,
            ) from exc

        upsert_signatures(signatures, db_path=db_path)

        if succeeded:
            manifest_db.mark_status(
                [s.doc_id for s in succeeded], "extracted", db_path=db_path
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

        # Safety-net purge: even though each raw file was removed as it was
        # processed above, sweep the whole batch workspace (including tmp/)
        # now that every document has been handled.
        workspace.purge()

        status = "extracted" if not failed else ("failed" if not succeeded else "extracted")
        manifest_db.set_batch_status(
            batch_id,
            status,
            db_path=db_path,
            notes=f"{len(succeeded)} succeeded, {len(failed)} failed",
            finished=True,
        )
        checkpoint.complete_phase(
            "extract", state={"succeeded": len(succeeded), "failed": len(failed)}
        )

        logger.info(
            "Stage 1 complete for batch %s: %d succeeded, %d failed",
            batch_id,
            len(succeeded),
            len(failed),
        )

        return Stage1Result(batch_id=batch_id, signatures=signatures)
    

