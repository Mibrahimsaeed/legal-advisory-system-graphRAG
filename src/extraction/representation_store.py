"""SQLite persistence for :class:`~src.extraction.doc_representation.DocumentRepresentation`.

Mirrors :mod:`src.extraction.signature_store` exactly -- thin functions
over :func:`src.common.db.connection_scope`, one row per ``doc_id``,
upserts via ``ON CONFLICT`` -- but against ``document_representations``
(``schemas/document_representation_schema.sql``) instead of
``document_signatures``.

:func:`list_representations` is the read path Stage 1.2 draws from when
``discovery.corpus_source='representations'`` (the default), and is a
drop-in replacement for :func:`src.extraction.signature_store.list_signatures`.
"""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

from src.common.db import DEFAULT_DB_PATH, connection_scope
from src.common.logging_utils import get_logger
from src.extraction.doc_representation import (
    CLASSIFICATION_STATUS_DROPPED_OFF_DOMAIN,
    CLASSIFICATION_STATUS_DROPPED_PROCEDURAL,
    DocumentRepresentation,
)

logger = get_logger(__name__)

DEFAULT_REPRESENTATION_SCHEMA_FILE = Path("schemas/document_representation_schema.sql")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _json_or(raw: str | None, default):
    return json.loads(raw) if raw else default


# Columns added in Phase 1 (cleaned text, legal-metadata storage and the
# current classification state). Applied with ALTER TABLE as well as in
# the schema file because `CREATE TABLE IF NOT EXISTS` silently skips a
# table that already exists, which would leave an earlier database
# missing them. Mirrors taxonomy_card.ensure_domain_candidate_columns().
#
# NOTE: the classification_status CHECK constraint lives in the schema
# file only -- SQLite cannot attach one via ALTER TABLE ADD COLUMN. A
# database created before Phase 1 therefore gets the column and its
# default but not the constraint; recreate the table to gain it.
_ADDED_REPRESENTATION_COLUMNS = {
    "cleaned_text": "TEXT",
    "statute_citations_json": "TEXT NOT NULL DEFAULT '[]'",
    "court_metadata_json": "TEXT NOT NULL DEFAULT '{}'",
    "primary_domain": "TEXT",
    "secondary_domain": "TEXT",
    "domain_confidence": "REAL",
    "classification_status": "TEXT NOT NULL DEFAULT 'pending'",
    "drop_reason": "TEXT",
}


def ensure_representation_columns(conn: sqlite3.Connection) -> list[str]:
    """Add any missing ``document_representations`` columns. Idempotent."""

    existing = {
        row["name"] for row in conn.execute("PRAGMA table_info(document_representations)")
    }
    if not existing:  # table not created yet; the schema file will do it
        return []

    added = []
    for column, ddl in _ADDED_REPRESENTATION_COLUMNS.items():
        if column not in existing:
            conn.execute(
                f"ALTER TABLE document_representations ADD COLUMN {column} {ddl}"
            )
            added.append(column)
    if added:
        logger.info("Added document_representations column(s): %s", ", ".join(added))
    return added


def _row_to_representation(row: sqlite3.Row) -> DocumentRepresentation:
    """Rebuild a representation from a row.

    ``full_text`` is intentionally left empty: it is not a column (see
    the schema file). Re-derive it from ``source_file`` via
    :func:`src.extraction.case_loader.load_case_folder` if a stage needs
    the whole judgment.
    """

    return DocumentRepresentation(
        doc_id=row["doc_id"],
        source_uri=row["source_uri"],
        source_type=row["source_type"],
        source_file=row["source_file"],
        source_relpath=row["source_relpath"],
        content_hash=row["content_hash"],
        title=row["title"],
        headings=_json_or(row["headings_json"], []),
        body_preview=row["body_preview"] or "",
        char_count=row["char_count"],
        court=row["court"],
        decision_date=row["decision_date"],
        citation=row["citation"],
        judges=_json_or(row["judges_json"], []),
        case_number=row["case_number"],
        cleaned_text=row["cleaned_text"],
        statute_citations=_json_or(row["statute_citations_json"], []),
        court_metadata=_json_or(row["court_metadata_json"], {}),
        primary_domain=row["primary_domain"],
        secondary_domain=row["secondary_domain"],
        domain_confidence=row["domain_confidence"],
        classification_status=row["classification_status"],
        drop_reason=row["drop_reason"],
        status=row["status"],
        warnings=_json_or(row["warnings_json"], []),
        error=row["error"],
        metadata=_json_or(row["metadata_json"], {}),
        batch_id=row["batch_id"],
    )


def upsert_representations(
    representations: list[DocumentRepresentation],
    db_path: str | Path = DEFAULT_DB_PATH,
) -> int:
    """Insert or update ``document_representations`` rows, keyed by ``doc_id``.

    Safe to call repeatedly with the same representations -- re-scanning a
    corpus root simply overwrites the row for a given ``doc_id``.
    """

    if not representations:
        return 0

    with connection_scope(db_path) as conn:
        ensure_representation_columns(conn)
        for rep in representations:
            conn.execute(
                """
                INSERT INTO document_representations (
                    doc_id, source_uri, source_file, source_relpath,
                    content_hash, source_type, status, title, headings_json,
                    body_preview, char_count, court, decision_date, citation,
                    judges_json, case_number, metadata_json, warnings_json,
                    error, batch_id, cleaned_text, statute_citations_json,
                    court_metadata_json, primary_domain, secondary_domain,
                    domain_confidence, classification_status, drop_reason,
                    created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                          ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(doc_id) DO UPDATE SET
                    source_uri     = excluded.source_uri,
                    source_file    = excluded.source_file,
                    source_relpath = excluded.source_relpath,
                    content_hash   = excluded.content_hash,
                    source_type    = excluded.source_type,
                    status         = excluded.status,
                    title          = excluded.title,
                    headings_json  = excluded.headings_json,
                    body_preview   = excluded.body_preview,
                    char_count     = excluded.char_count,
                    court          = excluded.court,
                    decision_date  = excluded.decision_date,
                    citation       = excluded.citation,
                    judges_json    = excluded.judges_json,
                    case_number    = excluded.case_number,
                    metadata_json  = excluded.metadata_json,
                    warnings_json  = excluded.warnings_json,
                    error          = excluded.error,
                    batch_id       = excluded.batch_id,
                    cleaned_text   = excluded.cleaned_text,
                    statute_citations_json = excluded.statute_citations_json,
                    court_metadata_json    = excluded.court_metadata_json,
                    primary_domain         = excluded.primary_domain,
                    secondary_domain       = excluded.secondary_domain,
                    domain_confidence      = excluded.domain_confidence,
                    classification_status  = excluded.classification_status,
                    drop_reason            = excluded.drop_reason,
                    updated_at     = excluded.updated_at
                """,
                (
                    rep.doc_id,
                    rep.source_uri,
                    rep.source_file,
                    rep.source_relpath,
                    rep.content_hash,
                    rep.source_type,
                    rep.status,
                    rep.title,
                    json.dumps(rep.headings, ensure_ascii=False),
                    rep.body_preview,
                    rep.char_count,
                    rep.court,
                    rep.decision_date,
                    rep.citation,
                    json.dumps(rep.judges, ensure_ascii=False),
                    rep.case_number,
                    json.dumps(rep.metadata, ensure_ascii=False),
                    json.dumps(rep.warnings, ensure_ascii=False),
                    rep.error,
                    rep.batch_id,
                    rep.cleaned_text,
                    json.dumps(rep.statute_citations, ensure_ascii=False),
                    json.dumps(rep.court_metadata, ensure_ascii=False),
                    rep.primary_domain,
                    rep.secondary_domain,
                    rep.domain_confidence,
                    rep.classification_status,
                    rep.drop_reason,
                    _now(),
                    _now(),
                ),
            )

    logger.info("Upserted %d document representation(s)", len(representations))
    return len(representations)


# Outcomes of a classification-state write. Reported rather than
# silently swallowed, because "nothing changed" and "refused to change a
# reviewed document" are different facts about a rerun.
WRITE_UPDATED = "updated"
WRITE_UNCHANGED = "unchanged"
WRITE_PROTECTED = "protected"
WRITE_MISSING = "missing"
# The row is in a state no classifier verdict may overwrite (a structural
# drop), or it moved after the caller's evidence was gathered.
WRITE_INELIGIBLE = "ineligible"
WRITE_STALE = "stale_evidence"

_CLASSIFICATION_FIELDS = (
    "classification_status",
    "primary_domain",
    "secondary_domain",
    "domain_confidence",
    "drop_reason",
)


def update_classification_state(
    doc_id: str,
    classification_status: str,
    primary_domain: str | None = None,
    secondary_domain: str | None = None,
    domain_confidence: float | None = None,
    drop_reason: str | None = None,
    protect_statuses: tuple[str, ...] = (),
    skip_if_unchanged: bool = True,
    ineligible_statuses: tuple[str, ...] = (),
    evidence_created_at: str | None = None,
    evidence_run_id: str | None = None,
    state_evidence_run_id: str | None = None,
    db_path: str | Path = DEFAULT_DB_PATH,
) -> str:
    """Write a classification verdict onto Phase 1's current-state fields.

    A targeted UPDATE of those five columns only: a classifier decides a
    document's domain and must never touch its extracted text, metadata,
    provenance or Phase 2 outcome.

    Returns one of ``WRITE_UPDATED``, ``WRITE_UNCHANGED``,
    ``WRITE_PROTECTED`` or ``WRITE_MISSING``.

    Two guards, both serving the same rule -- a rerun must not undo
    settled work:

    * ``protect_statuses`` refuses to overwrite a row already in one of
      those states. Phase 6 passes the statuses a human decided, so
      re-running the classifier cannot quietly revert a review.
    * ``skip_if_unchanged`` (default) makes a write that would change
      nothing a no-op, leaving ``updated_at`` alone. Without it every
      rerun would re-stamp the whole corpus and destroy the one signal
      that says when a document's classification actually last moved.
    * ``ineligible_statuses`` refuses outright -- used for a structural
      drop, which no classifier verdict may revive.
    * ``evidence_created_at`` + ``evidence_run_id`` refuse a **stale**
      write: if the row moved after the evidence was gathered, and not by
      a run using that same evidence (``state_evidence_run_id``), the
      caller is holding an out-of-date verdict.

    All four checks run against a row read **inside this function's own
    transaction**, immediately before the UPDATE. A caller that checked a
    snapshot taken earlier in the run would leave a window in which the
    row could move between the check and the write; here there is none.
    """

    proposed = {
        "classification_status": classification_status,
        "primary_domain": primary_domain,
        "secondary_domain": secondary_domain,
        "domain_confidence": domain_confidence,
        "drop_reason": drop_reason,
    }

    with connection_scope(db_path) as conn:
        ensure_representation_columns(conn)
        # updated_at is read here too: the staleness comparison must use the
        # row as it is now, not as it was when the run started.
        current = conn.execute(
            "SELECT classification_status, primary_domain, secondary_domain, "
            "domain_confidence, drop_reason, updated_at "
            "FROM document_representations WHERE doc_id = ?",
            (doc_id,),
        ).fetchone()

        if current is None:
            return WRITE_MISSING
        if current["classification_status"] in ineligible_statuses:
            return WRITE_INELIGIBLE
        if current["classification_status"] in protect_statuses:
            return WRITE_PROTECTED
        if _evidence_is_stale(
            evidence_created_at,
            current["updated_at"],
            state_evidence_run_id,
            evidence_run_id,
        ):
            return WRITE_STALE
        if skip_if_unchanged and _same_state(current, proposed):
            return WRITE_UNCHANGED

        conn.execute(
            """
            UPDATE document_representations
               SET classification_status = ?,
                   primary_domain        = ?,
                   secondary_domain      = ?,
                   domain_confidence     = ?,
                   drop_reason           = ?,
                   updated_at            = ?
             WHERE doc_id = ?
            """,
            (
                classification_status,
                primary_domain,
                secondary_domain,
                domain_confidence,
                drop_reason,
                _now(),
                doc_id,
            ),
        )
        return WRITE_UPDATED


def _evidence_is_stale(
    evidence_created_at: str | None,
    state_updated_at: str | None,
    state_evidence_run_id: str | None,
    evidence_run_id: str | None,
) -> bool:
    """Delegate to the policy rule, skipping the check when unused.

    Imported lazily: this module is extraction-layer and must not import
    the classification package at module scope.
    """

    if evidence_created_at is None and evidence_run_id is None:
        return False

    from src.classification.review_policy import evidence_is_stale

    return evidence_is_stale(
        signal_created_at=evidence_created_at,
        state_updated_at=state_updated_at,
        state_signal_run_id=state_evidence_run_id,
        signal_run_id=evidence_run_id,
    )


def _same_state(current: sqlite3.Row, proposed: dict) -> bool:
    """Whether a proposed verdict differs from what is already stored.

    Confidence is compared at six decimal places: it round-trips through
    SQLite REAL, and a difference in the twelfth place is not a change to
    a document's classification.
    """

    for column in _CLASSIFICATION_FIELDS:
        before, after = current[column], proposed[column]
        if column == "domain_confidence":
            before = round(before, 6) if before is not None else None
            after = round(after, 6) if after is not None else None
        if before != after:
            return False
    return True


def get_classification_states(
    db_path: str | Path = DEFAULT_DB_PATH,
) -> dict[str, tuple[str, str | None]]:
    """``{doc_id: (classification_status, updated_at)}`` for every document.

    A read-only projection, deliberately not routed through
    :func:`list_representations`: a caller that only needs the current
    state should not have to load every document's text to get it, and
    ``updated_at`` is not carried on
    :class:`~src.extraction.doc_representation.DocumentRepresentation`.

    Every row is returned, dropped ones included -- a dropped document's
    status is precisely what a caller checking eligibility needs to see.
    """

    with connection_scope(db_path) as conn:
        ensure_representation_columns(conn)
        rows = conn.execute(
            "SELECT doc_id, classification_status, updated_at "
            "FROM document_representations"
        ).fetchall()
    return {r["doc_id"]: (r["classification_status"], r["updated_at"]) for r in rows}


def get_representation(
    doc_id: str, db_path: str | Path = DEFAULT_DB_PATH
) -> DocumentRepresentation | None:
    with connection_scope(db_path) as conn:
        row = conn.execute(
            "SELECT * FROM document_representations WHERE doc_id = ?", (doc_id,)
        ).fetchone()
    return _row_to_representation(row) if row else None


def get_representations_for_batch(
    batch_id: str, db_path: str | Path = DEFAULT_DB_PATH
) -> list[DocumentRepresentation]:
    with connection_scope(db_path) as conn:
        rows = conn.execute(
            "SELECT * FROM document_representations WHERE batch_id = ?", (batch_id,)
        ).fetchall()
    return [_row_to_representation(r) for r in rows]


def count_by_status(db_path: str | Path = DEFAULT_DB_PATH) -> dict[str, int]:
    """How many representations per status bucket (CLI/dashboard helper)."""

    with connection_scope(db_path) as conn:
        rows = conn.execute(
            """
            SELECT status, COUNT(*) AS n
            FROM document_representations
            GROUP BY status
            """
        ).fetchall()
    return {row["status"]: row["n"] for row in rows}


def list_representations(
    statuses: tuple[str, ...] = ("ok",),
    db_path: str | Path = DEFAULT_DB_PATH,
    exclude_dropped: bool = True,
) -> list[DocumentRepresentation]:
    """The corpus every downstream stage reads.

    Two exclusions, both deliberate:

    * ``status`` -- extraction outcome. ``failed`` rows have no
      title/headings/body to embed.
    * ``classification_status`` -- when ``exclude_dropped`` (the default),
      documents the Phase 2 structural filter dropped
      (``dropped_procedural``) and documents dropped as off-domain
      (``dropped_off_domain``) are withheld.

    This is the short-circuit that keeps structural noise out of embedding
    and clustering: both
    :func:`orchestration.dags.domain_discovery_flow._load_corpus` and
    :func:`orchestration.dags.classification_flow.run_classification` load
    their corpus through here, so a dropped document is never embedded,
    clustered or classified. Pass ``exclude_dropped=False`` to inspect or
    audit the dropped population.
    """

    if not statuses:
        return []

    placeholders = ",".join("?" for _ in statuses)
    sql = f"SELECT * FROM document_representations WHERE status IN ({placeholders})"
    params: list[str] = list(statuses)

    if exclude_dropped:
        dropped = [
            CLASSIFICATION_STATUS_DROPPED_PROCEDURAL,
            CLASSIFICATION_STATUS_DROPPED_OFF_DOMAIN,
        ]
        sql += " AND classification_status NOT IN ({})".format(
            ",".join("?" for _ in dropped)
        )
        params.extend(dropped)

    with connection_scope(db_path) as conn:
        rows = conn.execute(sql, params).fetchall()
    return [_row_to_representation(r) for r in rows]
