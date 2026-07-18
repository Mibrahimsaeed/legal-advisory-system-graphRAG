"""SQLite persistence for :class:`~src.extraction.signature.DocumentSignature`.

Mirrors the conventions of :mod:`src.ingestion.manifest`: thin functions
over :func:`src.common.db.connection_scope`, one row per ``doc_id``,
upserts via ``ON CONFLICT``. Schema lives in ``schemas/signature_schema.sql``
and is loaded the same way ``schemas/manifest_schema.sql`` is -- via
:func:`src.common.db.init_schema(schema_file=...)`.
"""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

from src.common.db import DEFAULT_DB_PATH, connection_scope
from src.common.logging_utils import get_logger
from src.extraction.signature import DocumentSignature

logger = get_logger(__name__)

DEFAULT_SIGNATURE_SCHEMA_FILE = Path("schemas/signature_schema.sql")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _row_to_signature(row: sqlite3.Row) -> DocumentSignature:
    toc_raw = row["toc_json"]
    return DocumentSignature(
        doc_id=row["doc_id"],
        source_uri=row["source_uri"],
        signature_hash=row["signature_hash"],
        is_scanned=bool(row["is_scanned"]),
        extraction_status=row["extraction_status"],
        extractor_used=row["extractor_used"],
        title=row["title"],
        toc=json.loads(toc_raw) if toc_raw else [],
        body_preview=row["body_preview"] or "",
        pages_used=row["pages_used"],
        char_count=row["char_count"],
        quality_score=row["quality_score"],
        batch_id=row["batch_id"],
        error=row["error"],
    )


def upsert_signatures(
    signatures: list[DocumentSignature],
    db_path: str | Path = DEFAULT_DB_PATH,
) -> int:
    """Insert or update ``document_signatures`` rows, keyed by ``doc_id``.

    Safe to call repeatedly with the same signatures (e.g. on a checkpoint
    resume after a partial batch failure) -- later calls simply overwrite
    the row for a given ``doc_id``.
    """

    if not signatures:
        return 0

    with connection_scope(db_path) as conn:
        for sig in signatures:
            conn.execute(
                """
                INSERT INTO document_signatures (
                    doc_id, source_uri, signature_hash, is_scanned,
                    extraction_status, extractor_used, title, toc_json,
                    body_preview, pages_used, char_count, quality_score,
                    batch_id, error, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(doc_id) DO UPDATE SET
                    source_uri        = excluded.source_uri,
                    signature_hash    = excluded.signature_hash,
                    is_scanned        = excluded.is_scanned,
                    extraction_status = excluded.extraction_status,
                    extractor_used    = excluded.extractor_used,
                    title             = excluded.title,
                    toc_json          = excluded.toc_json,
                    body_preview      = excluded.body_preview,
                    pages_used        = excluded.pages_used,
                    char_count        = excluded.char_count,
                    quality_score     = excluded.quality_score,
                    batch_id          = excluded.batch_id,
                    error             = excluded.error,
                    updated_at        = excluded.updated_at
                """,
                (
                    sig.doc_id,
                    sig.source_uri,
                    sig.signature_hash,
                    int(sig.is_scanned),
                    sig.extraction_status,
                    sig.extractor_used,
                    sig.title,
                    json.dumps(sig.toc, ensure_ascii=False),
                    sig.body_preview,
                    sig.pages_used,
                    sig.char_count,
                    sig.quality_score,
                    sig.batch_id,
                    sig.error,
                    _now(),
                    _now(),
                ),
            )

    logger.info("Upserted %d signature record(s)", len(signatures))
    return len(signatures)


def get_signature(
    doc_id: str, db_path: str | Path = DEFAULT_DB_PATH
) -> DocumentSignature | None:
    with connection_scope(db_path) as conn:
        row = conn.execute(
            "SELECT * FROM document_signatures WHERE doc_id = ?", (doc_id,)
        ).fetchone()
    return _row_to_signature(row) if row else None


def get_signatures_for_batch(
    batch_id: str, db_path: str | Path = DEFAULT_DB_PATH
) -> list[DocumentSignature]:
    with connection_scope(db_path) as conn:
        rows = conn.execute(
            "SELECT * FROM document_signatures WHERE batch_id = ?", (batch_id,)
        ).fetchall()
    return [_row_to_signature(r) for r in rows]


def count_by_status(
    db_path: str | Path = DEFAULT_DB_PATH,
) -> dict[str, int]:
    """Handy for a CLI/dashboard: how many signatures per status bucket."""

    with connection_scope(db_path) as conn:
        rows = conn.execute(
            """
            SELECT extraction_status, COUNT(*) AS n
            FROM document_signatures
            GROUP BY extraction_status
            """
        ).fetchall()
    return {row["extraction_status"]: row["n"] for row in rows}