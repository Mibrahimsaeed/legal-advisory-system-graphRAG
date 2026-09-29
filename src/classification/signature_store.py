"""Durable storage for the Phase 3 case signature.

The signature is the compact case representation
(:class:`~src.classification.case_representation.CaseRepresentation`):
title + headings + the bounded substantive text that Phase 3 actually
reasons about. It used to live only in memory for the length of a run.
Persisting it does three things:

* **makes the derivation auditable.** "What text produced this verdict?"
  becomes a query rather than a re-derivation that may no longer match.
* **makes runs comparable.** Two signal runs over the same corpus can be
  shown to have analysed identical text -- or shown not to have.
* **lets Phase 3 reuse rather than regenerate.** When the source text and
  the builder version are both unchanged, the stored signature is still
  correct by construction, so it is returned as-is and the row is not
  rewritten (see :func:`upsert_case_signatures`' unchanged outcome).

Deliberately *not* here: the full judgment (it stays in
``document_representations.cleaned_text`` -- this table holds the bounded
derivative, a different and smaller thing) and embeddings (a separate
concern, to be persisted separately).
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

from src.classification.case_representation import (
    SIGNATURE_VERSION,
    CaseRepresentation,
)
from src.common.db import DEFAULT_DB_PATH, connection_scope
from src.common.logging_utils import get_logger

logger = get_logger(__name__)

DEFAULT_SIGNATURE_SCHEMA_FILE = Path("schemas/case_signature_schema.sql")

# Write outcomes, reported rather than swallowed: on a rerun over an
# unchanged corpus every row should come back "unchanged", and anything
# else is worth seeing in the log.
WRITE_INSERTED = "inserted"
WRITE_UPDATED = "updated"
WRITE_UNCHANGED = "unchanged"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass(frozen=True)
class CaseSignature:
    """One stored signature, with everything needed to rebuild it."""

    doc_id: str
    signature_text: str
    signature_version: str
    signature_hash: str
    title: str | None = None
    headings: list[str] | None = None
    source_content_hash: str | None = None
    text_source: str = "cleaned_text"
    char_count: int = 0
    word_count: int = 0
    created_at: str | None = None
    updated_at: str | None = None

    def is_current(
        self, source_content_hash: str | None, version: str = SIGNATURE_VERSION
    ) -> bool:
        """Whether this stored signature can be reused as-is.

        Both halves matter. A matching ``source_content_hash`` says the
        judgment's text has not changed; a matching ``signature_version``
        says the *builder* has not changed. Either one alone would let a
        stale signature through -- re-scraped text under an old hash, or
        the same text reduced by different rules.

        A row with no ``source_content_hash`` recorded is never reused:
        there is nothing to compare, and silently trusting it would defeat
        the point of the check.
        """

        if self.signature_version != version:
            return False
        if not self.source_content_hash or not source_content_hash:
            return False
        return self.source_content_hash == source_content_hash

    def to_representation(self) -> CaseRepresentation:
        """Rebuild the in-memory representation this row was made from.

        ``body_preview`` is recovered by removing the title/headings
        prefix that :attr:`CaseRepresentation.signal_text` puts in front of
        it. That is exact rather than heuristic, because ``signal_text`` is
        a pure function of the three parts -- but it is verified, and falls
        back to treating the whole signature as the body if a row was
        written by some other path.
        """

        headings = list(self.headings or [])
        body = _body_from_signature_text(self.signature_text, self.title, headings)
        return CaseRepresentation(
            doc_id=self.doc_id,
            title=self.title,
            headings=headings,
            body_preview=body,
            representation_hash=self.signature_hash,
            text_source=self.text_source,
            char_count=self.char_count,
            word_count=self.word_count,
        )


def _body_from_signature_text(
    signature_text: str, title: str | None, headings: list[str]
) -> str:
    """Strip the title/headings prefix ``signal_text`` prepended to the body."""

    prefix_parts = [p for p in [title or "", " ".join(headings)] if p]
    if not prefix_parts:
        return signature_text

    prefix = "\n".join(prefix_parts) + "\n"
    if signature_text.startswith(prefix):
        return signature_text[len(prefix) :]

    logger.warning(
        "Signature text for a stored row does not start with its "
        "title/headings prefix; treating the whole signature as the body"
    )
    return signature_text


def _row_to_signature(row: sqlite3.Row) -> CaseSignature:
    return CaseSignature(
        doc_id=row["doc_id"],
        signature_text=row["signature_text"],
        signature_version=row["signature_version"],
        signature_hash=row["signature_hash"],
        title=row["title"],
        headings=json.loads(row["headings_json"] or "[]"),
        source_content_hash=row["source_content_hash"],
        text_source=row["text_source"],
        char_count=row["char_count"],
        word_count=row["word_count"],
        created_at=row["created_at"],
        updated_at=row["updated_at"],
    )


def build_signature_record(
    representation: CaseRepresentation,
    source_content_hash: str | None = None,
) -> dict:
    """Flatten one representation into its storable row."""

    return {
        "doc_id": representation.doc_id,
        "signature_text": representation.signal_text,
        "title": representation.title,
        "headings": list(representation.headings),
        "signature_version": SIGNATURE_VERSION,
        "signature_hash": representation.representation_hash,
        "source_content_hash": source_content_hash,
        "text_source": representation.text_source,
        "char_count": representation.char_count,
        "word_count": representation.word_count,
    }


def upsert_case_signatures(
    representations: Iterable[CaseRepresentation],
    source_hashes: dict[str, str | None] | None = None,
    db_path: str | Path = DEFAULT_DB_PATH,
) -> dict[str, int]:
    """Store one signature per representation, keyed by ``doc_id``.

    Returns counts by outcome (``inserted`` / ``updated`` / ``unchanged``).

    A row whose signature, version and source hash all already match is
    left completely alone -- not even ``updated_at`` is touched. Without
    that, every rerun would re-stamp the whole corpus and destroy the one
    column that says when a document's signature actually last changed.
    """

    source_hashes = source_hashes or {}
    outcomes = {WRITE_INSERTED: 0, WRITE_UPDATED: 0, WRITE_UNCHANGED: 0}

    with connection_scope(db_path) as conn:
        for representation in representations:
            record = build_signature_record(
                representation, source_hashes.get(representation.doc_id)
            )
            existing = conn.execute(
                "SELECT signature_hash, signature_version, source_content_hash "
                "FROM case_signatures WHERE doc_id = ?",
                (record["doc_id"],),
            ).fetchone()

            if existing is not None and (
                existing["signature_hash"] == record["signature_hash"]
                and existing["signature_version"] == record["signature_version"]
                and existing["source_content_hash"] == record["source_content_hash"]
            ):
                outcomes[WRITE_UNCHANGED] += 1
                continue

            now = _now()
            conn.execute(
                """
                INSERT INTO case_signatures (
                    doc_id, signature_text, title, headings_json,
                    signature_version, signature_hash, source_content_hash,
                    text_source, char_count, word_count, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(doc_id) DO UPDATE SET
                    signature_text      = excluded.signature_text,
                    title               = excluded.title,
                    headings_json       = excluded.headings_json,
                    signature_version   = excluded.signature_version,
                    signature_hash      = excluded.signature_hash,
                    source_content_hash = excluded.source_content_hash,
                    text_source         = excluded.text_source,
                    char_count          = excluded.char_count,
                    word_count          = excluded.word_count,
                    updated_at          = excluded.updated_at
                """,
                (
                    record["doc_id"],
                    record["signature_text"],
                    record["title"],
                    json.dumps(record["headings"], ensure_ascii=False),
                    record["signature_version"],
                    record["signature_hash"],
                    record["source_content_hash"],
                    record["text_source"],
                    record["char_count"],
                    record["word_count"],
                    now,
                    now,
                ),
            )
            outcomes[WRITE_UPDATED if existing is not None else WRITE_INSERTED] += 1

    written = outcomes[WRITE_INSERTED] + outcomes[WRITE_UPDATED]
    if written or outcomes[WRITE_UNCHANGED]:
        logger.info(
            "Case signatures: %d inserted, %d updated, %d unchanged",
            outcomes[WRITE_INSERTED], outcomes[WRITE_UPDATED],
            outcomes[WRITE_UNCHANGED],
        )
    return outcomes


def get_case_signature(
    doc_id: str, db_path: str | Path = DEFAULT_DB_PATH
) -> CaseSignature | None:
    """The stored signature for one document, or None."""

    with connection_scope(db_path) as conn:
        row = conn.execute(
            "SELECT * FROM case_signatures WHERE doc_id = ?", (doc_id,)
        ).fetchone()
    return _row_to_signature(row) if row else None


def get_case_signatures(
    doc_ids: Iterable[str] | None = None,
    db_path: str | Path = DEFAULT_DB_PATH,
) -> dict[str, CaseSignature]:
    """Stored signatures keyed by ``doc_id``; all of them when ids are omitted.

    Chunked so a corpus-sized id list cannot exceed SQLite's variable
    limit (999 by default).
    """

    signatures: dict[str, CaseSignature] = {}
    with connection_scope(db_path) as conn:
        if doc_ids is None:
            rows = conn.execute("SELECT * FROM case_signatures").fetchall()
            return {r["doc_id"]: _row_to_signature(r) for r in rows}

        ids = list(doc_ids)
        for start in range(0, len(ids), 500):
            chunk = ids[start : start + 500]
            placeholders = ",".join("?" for _ in chunk)
            rows = conn.execute(
                f"SELECT * FROM case_signatures WHERE doc_id IN ({placeholders})",
                chunk,
            ).fetchall()
            signatures.update({r["doc_id"]: _row_to_signature(r) for r in rows})

    return signatures


def signature_stats(db_path: str | Path = DEFAULT_DB_PATH) -> dict:
    """Counts by version and text source, for a run summary."""

    with connection_scope(db_path) as conn:
        total = conn.execute(
            "SELECT COUNT(*) AS n FROM case_signatures"
        ).fetchone()["n"]
        by_version = {
            r["signature_version"]: r["n"]
            for r in conn.execute(
                "SELECT signature_version, COUNT(*) AS n FROM case_signatures "
                "GROUP BY signature_version"
            )
        }
        by_source = {
            r["text_source"]: r["n"]
            for r in conn.execute(
                "SELECT text_source, COUNT(*) AS n FROM case_signatures "
                "GROUP BY text_source"
            )
        }
        aggregate = conn.execute(
            "SELECT AVG(char_count) AS chars, AVG(word_count) AS words "
            "FROM case_signatures"
        ).fetchone()

    return {
        "signatures": total,
        "by_version": by_version,
        "by_text_source": by_source,
        "mean_char_count": aggregate["chars"],
        "mean_word_count": aggregate["words"],
    }
