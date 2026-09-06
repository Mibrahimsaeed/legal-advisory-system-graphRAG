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
from src.extraction.doc_representation import (
    CLASSIFICATION_STATUS_AUTO_ACCEPTED,
    CLASSIFICATION_STATUS_DROPPED_PROCEDURAL,
    CLASSIFICATION_STATUS_NEEDS_REVIEW,
    CLASSIFICATION_STATUS_PENDING,
    DocumentRepresentation,
)
from src.extraction.representation_store import (
    get_representation,
    upsert_representations,
)
from src.extraction.structural_filter import (
    REASON_INCOMPLETE_SCRAPE,
    StructuralDecision,
    validate_structure,
)

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


# A document a later phase has already ruled on is not re-decided by a
# re-scan: re-running ingest must not silently reset a human-reviewed or
# accepted case back to pending.
_DOWNSTREAM_DECIDED = frozenset(
    {CLASSIFICATION_STATUS_AUTO_ACCEPTED, CLASSIFICATION_STATUS_NEEDS_REVIEW}
)


def _apply_structural_filter(
    representation: DocumentRepresentation,
    existing: DocumentRepresentation | None,
    document,
) -> StructuralDecision | None:
    """Phase 2: set the classification state on one freshly loaded case.

    Mutates ``representation`` in place and returns the decision, or
    ``None`` when the document was left alone because a later phase had
    already ruled on it.

    Three outcomes:

    * **extraction failed** -- no text exists to judge, so the document is
      dropped as an incomplete scrape. ``status='failed'`` already keeps it
      out of the corpus; the drop reason makes *why* answerable in one
      column alongside every other exclusion.
    * **structurally substantive** -- ``cleaned_text`` is populated from the
      text the loader already extracted and the case continues as
      ``pending``.
    * **structural noise** -- ``dropped_procedural`` plus the specific
      ``drop_reason``. ``cleaned_text`` is left NULL: the bounded
      ``body_preview`` is retained, so the drop stays auditable without
      storing text nothing will read.
    """

    if existing is not None and existing.classification_status in _DOWNSTREAM_DECIDED:
        representation.classification_status = existing.classification_status
        representation.drop_reason = existing.drop_reason
        representation.primary_domain = existing.primary_domain
        representation.secondary_domain = existing.secondary_domain
        representation.domain_confidence = existing.domain_confidence
        representation.cleaned_text = existing.cleaned_text
        return None

    if representation.status == "failed":
        representation.classification_status = CLASSIFICATION_STATUS_DROPPED_PROCEDURAL
        representation.drop_reason = REASON_INCOMPLETE_SCRAPE
        return None

    decision = validate_structure(
        representation.full_text,
        title=representation.title,
        min_characters=document.min_characters,
        min_words=document.min_words,
        incomplete_scrape_max_characters=document.incomplete_scrape_max_characters,
        procedural_max_characters=document.procedural_max_characters,
        cause_list_min_case_numbers=document.cause_list_min_case_numbers,
        cause_list_min_list_ratio=document.cause_list_min_list_ratio,
        substantive_min_markers=document.substantive_min_markers,
    )

    if decision.passed:
        representation.classification_status = CLASSIFICATION_STATUS_PENDING
        representation.drop_reason = None
        representation.cleaned_text = representation.full_text
    else:
        representation.classification_status = CLASSIFICATION_STATUS_DROPPED_PROCEDURAL
        representation.drop_reason = decision.drop_reason
        representation.cleaned_text = None

    return decision


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
    def dropped_doc_ids(self) -> list[str]:
        """Documents the structural filter withheld from downstream stages."""

        return [
            r.doc_id
            for r in self.representations
            if r.classification_status == CLASSIFICATION_STATUS_DROPPED_PROCEDURAL
        ]

    @property
    def pending_doc_ids(self) -> list[str]:
        """Documents that passed structural validation and continue."""

        return [
            r.doc_id
            for r in self.representations
            if r.classification_status == CLASSIFICATION_STATUS_PENDING
        ]

    @property
    def drops_by_reason(self) -> dict[str, int]:
        """How many documents each structural rule withheld."""

        counts: Counter[str] = Counter()
        for rep in self.representations:
            if rep.classification_status == CLASSIFICATION_STATUS_DROPPED_PROCEDURAL:
                counts[rep.drop_reason or "unknown"] += 1
        return dict(counts.most_common())

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
            existing = get_representation(representation.doc_id, db_path=db_path)
            decision = _apply_structural_filter(representation, existing, document)

            # One line per document: enough to audit a 10k-case run without
            # logging any judgment text.
            logger.debug(
                "doc_id=%s source=%s result=%s status=%s drop_reason=%s rule=%s",
                representation.doc_id,
                representation.source_relpath or representation.source_uri,
                "PASS" if representation.drop_reason is None else "DROP",
                representation.classification_status,
                representation.drop_reason,
                decision.triggered_rule if decision else "preserved",
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

        dropped = [
            r for r in representations
            if r.classification_status == CLASSIFICATION_STATUS_DROPPED_PROCEDURAL
        ]
        kept = [
            r for r in representations
            if r.classification_status == CLASSIFICATION_STATUS_PENDING
        ]
        logger.info(
            "Structural filter: %d document(s) continue as pending, "
            "%d dropped before embedding",
            len(kept), len(dropped),
        )
        if dropped:
            counts: Counter[str] = Counter(r.drop_reason or "unknown" for r in dropped)
            logger.warning("Structural drops by reason: %s", dict(counts.most_common()))

    return CaseIngestResult(
        root=root, representations=representations, batch_id=batch_id
    )
