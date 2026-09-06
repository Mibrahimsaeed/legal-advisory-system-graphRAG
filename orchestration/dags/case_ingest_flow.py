"""Stage 1 (case law): case folders -> document representations.

Replaces :mod:`orchestration.dags.feature_extraction_flow` (PDF ->
document signature) as the active pipeline's first stage:

    <corpus_root>/**/case.html + metadata.json
        -> src.extraction.case_loader.load_case_folder
        -> document_representations (SQLite)

and that table is what Stage 1.2 (``domain_discovery_flow``) embeds and
clusters. No signature is built, stored or required anywhere in this
path.

Deliberately simpler than the PDF Stage 1: no Stage 0 manifest claim/pull
(case folders are read straight off disk, nothing is copied into scratch,
so there is nothing to claim, stage or purge) and no checkpoint phases
(the upsert is keyed by ``doc_id``, so re-running a scan is idempotent
and simply refreshes rows). Both are noted here rather than assumed --
if case ingest later needs batching/metrics, it should follow
``feature_extraction_flow``'s pattern.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

from src.common.config import get_settings
from src.common.db import DEFAULT_DB_PATH, init_schema
from src.common.exceptions import ConfigurationError
from src.common.logging_utils import get_logger, log_context
from src.extraction.case_loader import iter_case_folders, load_case_folder
from src.extraction.doc_representation import DocumentRepresentation
from src.extraction.representation_store import upsert_representations

logger = get_logger(__name__)

DEFAULT_UPSERT_CHUNK_SIZE = 500


def _counts_by_code(representations: list[DocumentRepresentation]) -> dict[str, int]:
    """Failure counts keyed by the ERROR_* code (the first token of ``error``)."""

    counts: Counter[str] = Counter()
    for rep in representations:
        counts[(rep.error or "unknown").split(":", 1)[0].strip()] += 1
    return dict(counts.most_common())


def _warning_counts(representations: list[DocumentRepresentation]) -> dict[str, int]:
    """Warning counts keyed by WARNING_* code, ignoring the ``:detail`` suffix."""

    counts: Counter[str] = Counter()
    for rep in representations:
        for warning in rep.warnings:
            counts[warning.split(":", 1)[0].strip()] += 1
    return dict(counts.most_common())


@dataclass(frozen=True)
class CaseIngestResult:
    root: Path
    representations: list[DocumentRepresentation] = field(default_factory=list)
    batch_id: str | None = None

    @property
    def succeeded_doc_ids(self) -> list[str]:
        return [r.doc_id for r in self.representations if r.status != "failed"]

    @property
    def failed_doc_ids(self) -> list[str]:
        return [r.doc_id for r in self.representations if r.status == "failed"]

    @property
    def failures_by_code(self) -> dict[str, int]:
        """How many cases failed, per ERROR_* code."""

        return _counts_by_code(
            [r for r in self.representations if r.status == "failed"]
        )

    @property
    def warnings_by_code(self) -> dict[str, int]:
        """How many stored cases carry each WARNING_* code."""

        return _warning_counts(
            [r for r in self.representations if r.status != "failed"]
        )


def run_case_ingest(
    root: str | Path | None = None,
    db_path: str | Path = DEFAULT_DB_PATH,
    batch_id: str | None = None,
    limit: int | None = None,
    chunk_size: int = DEFAULT_UPSERT_CHUNK_SIZE,
) -> CaseIngestResult:
    """Scan ``root`` for case folders and upsert one representation each.

    Args:
        root: Corpus root containing case folders (recursively). Defaults
            to ``caselaw.corpus_root`` from config.
        db_path: SQLite database holding ``document_representations``.
        batch_id: Optional tag stored on every row produced by this run.
        limit: Stop after this many case folders (useful for a smoke run
            against a large corpus).
        chunk_size: How many representations to upsert per transaction.

    Raises:
        ConfigurationError: if no corpus root is configured, or it does
            not exist / is not a directory.
    """

    settings = get_settings()
    caselaw = settings.caselaw
    document = settings.document

    root = root if root is not None else caselaw.corpus_root
    if root is None:
        raise ConfigurationError(
            "run_case_ingest() requires a corpus root; set caselaw.corpus_root "
            "in config to the directory holding the case folders"
        )

    root = Path(root)
    if not root.exists():
        raise ConfigurationError(f"Case corpus root does not exist: {root}")
    if not root.is_dir():
        raise ConfigurationError(f"Case corpus root is not a directory: {root}")

    init_schema(db_path=db_path, schema_file=caselaw.representation_schema_file)

    representations: list[DocumentRepresentation] = []
    pending: list[DocumentRepresentation] = []

    def _flush() -> None:
        nonlocal pending
        if pending:
            upsert_representations(pending, db_path=db_path)
            pending = []

    with log_context(batch_id=batch_id or "case_ingest", phase="represent"):
        for case_folder in iter_case_folders(
            root, case_html_filename=caselaw.case_html_filename
        ):
            if limit is not None and len(representations) >= limit:
                break

            representation = load_case_folder(
                case_folder,
                root=root,
                case_html_filename=caselaw.case_html_filename,
                metadata_filename=caselaw.metadata_filename,
                body_preview_char_limit=caselaw.body_preview_char_limit,
                max_headings=caselaw.max_headings,
                min_characters=document.min_characters,
                batch_id=batch_id,
            )
            representations.append(representation)
            pending.append(representation)

            if len(pending) >= chunk_size:
                _flush()

        _flush()

        succeeded = [r for r in representations if r.status != "failed"]
        failed = [r for r in representations if r.status == "failed"]

        logger.info(
            "Case ingest complete for %s: %d representation(s) stored, %d failed",
            root,
            len(succeeded),
            len(failed),
        )

        # Aggregated so a 10k-case scan surfaces *which* problems occurred
        # and how often, instead of only a per-case warning buried in the
        # log. Codes are the ERROR_*/WARNING_* vocabulary in case_loader.
        if failed:
            logger.warning("Case ingest failures by code: %s", _counts_by_code(failed))
        warning_counts = _warning_counts(succeeded)
        if warning_counts:
            logger.warning("Case ingest warnings by code: %s", warning_counts)

    return CaseIngestResult(
        root=root, representations=representations, batch_id=batch_id
    )
