from __future__ import annotations

import sqlite3
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from src.common.db import DEFAULT_DB_PATH, connection_scope
from src.common.logging_utils import get_logger

logger = get_logger(__name__)


@dataclass(frozen=True)
class ManifestEntry:
    doc_id: str
    source_uri: str
    checksum: str
    checksum_algo: str
    byte_size: Optional[int]
    ingest_status: str
    batch_id: Optional[str]
    attempts: int
    last_error: Optional[str]
    created_at: str
    updated_at: str

    @classmethod
    def from_row(cls, row: sqlite3.Row) -> "ManifestEntry":
        return cls(
            doc_id=row["doc_id"],
            source_uri=row["source_uri"],
            checksum=row["checksum"],
            checksum_algo=row["checksum_algo"],
            byte_size=row["byte_size"],
            ingest_status=row["ingest_status"],
            batch_id=row["batch_id"],
            attempts=row["attempts"],
            last_error=row["last_error"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
        )


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def new_batch_id() -> str:
    return f"batch_{uuid.uuid4().hex[:12]}"


def register_documents(
    entries: list[dict],
    db_path: str | Path = DEFAULT_DB_PATH,
) -> int:
    inserted = 0

    with connection_scope(db_path) as conn:
        for e in entries:
            cur = conn.execute(
                """
                INSERT INTO document_manifest
                    (doc_id, source_uri, checksum, checksum_algo, byte_size,
                     ingest_status, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, 'pending', ?, ?)
                ON CONFLICT(doc_id) DO NOTHING
                """,
                (
                    e["doc_id"],
                    e["source_uri"],
                    e["checksum"],
                    e.get("checksum_algo", "sha256"),
                    e.get("byte_size"),
                    _now(),
                    _now(),
                ),
            )
            inserted += cur.rowcount if cur.rowcount and cur.rowcount > 0 else 0

    logger.info("Registered %d new document(s) in manifest", inserted)
    return inserted


def claim_batch(
    batch_size: int,
    db_path: str | Path = DEFAULT_DB_PATH,
    batch_id: Optional[str] = None,
) -> tuple[str, list[ManifestEntry]]:
    batch_id = batch_id or new_batch_id()

    with connection_scope(db_path) as conn:
        rows = conn.execute(
            """
            SELECT doc_id FROM document_manifest
            WHERE ingest_status = 'pending'
            ORDER BY created_at ASC
            LIMIT ?
            """,
            (batch_size,),
        ).fetchall()

        doc_ids = [r["doc_id"] for r in rows]

        if doc_ids:
            placeholders = ",".join("?" for _ in doc_ids)

            conn.execute(
                f"""
                UPDATE document_manifest
                SET ingest_status = 'claimed', batch_id = ?, updated_at = ?
                WHERE doc_id IN ({placeholders})
                """,
                (batch_id, _now(), *doc_ids),
            )

        conn.execute(
            """
            INSERT INTO batch_runs
            (batch_id, requested_size, actual_size, status, started_at)
            VALUES (?, ?, ?, 'open', ?)
            ON CONFLICT(batch_id)
            DO UPDATE SET actual_size = excluded.actual_size
            """,
            (batch_id, batch_size, len(doc_ids), _now()),
        )

        claimed_rows = conn.execute(
            "SELECT * FROM document_manifest WHERE batch_id = ?",
            (batch_id,),
        ).fetchall()

    entries = [ManifestEntry.from_row(r) for r in claimed_rows]

    logger.info(
        "Claimed batch %s with %d document(s)",
        batch_id,
        len(entries),
    )

    return batch_id, entries


def mark_status(
    doc_ids: list[str],
    status: str,
    db_path: str | Path = DEFAULT_DB_PATH,
    error: Optional[str] = None,
    increment_attempts: bool = False,
) -> None:
    if not doc_ids:
        return

    with connection_scope(db_path) as conn:
        placeholders = ",".join("?" for _ in doc_ids)

        attempts_clause = (
            "attempts = attempts + 1,"
            if increment_attempts
            else ""
        )

        conn.execute(
            f"""
            UPDATE document_manifest
            SET ingest_status = ?,
                {attempts_clause}
                last_error = ?,
                updated_at = ?
            WHERE doc_id IN ({placeholders})
            """,
            (status, error, _now(), *doc_ids),
        )

    logger.info(
        "Marked %d document(s) as '%s'%s",
        len(doc_ids),
        status,
        f" (error: {error})" if error else "",
    )


def set_batch_status(
    batch_id: str,
    status: str,
    db_path: str | Path = DEFAULT_DB_PATH,
    notes: Optional[str] = None,
    finished: bool = False,
) -> None:
    with connection_scope(db_path) as conn:
        if finished:
            conn.execute(
                """
                UPDATE batch_runs
                SET status = ?, notes = ?, finished_at = ?
                WHERE batch_id = ?
                """,
                (status, notes, _now(), batch_id),
            )
        else:
            conn.execute(
                """
                UPDATE batch_runs
                SET status = ?, notes = ?
                WHERE batch_id = ?
                """,
                (status, notes, batch_id),
            )

    logger.info("Batch %s status -> %s", batch_id, status)


def get_batch_entries(
    batch_id: str,
    db_path: str | Path = DEFAULT_DB_PATH,
) -> list[ManifestEntry]:
    with connection_scope(db_path) as conn:
        rows = conn.execute(
            "SELECT * FROM document_manifest WHERE batch_id = ?",
            (batch_id,),
        ).fetchall()

    return [ManifestEntry.from_row(r) for r in rows]


def requeue_failed(
    batch_id: str,
    max_attempts: int = 3,
    db_path: str | Path = DEFAULT_DB_PATH,
) -> int:
    with connection_scope(db_path) as conn:
        cur = conn.execute(
            """
            UPDATE document_manifest
            SET ingest_status = 'pending',
                batch_id = NULL,
                updated_at = ?
            WHERE batch_id = ?
              AND ingest_status = 'failed'
              AND attempts < ?
            """,
            (_now(), batch_id, max_attempts),
        )

        requeued = (
            cur.rowcount
            if cur.rowcount and cur.rowcount > 0
            else 0
        )

    logger.info(
        "Requeued %d failed document(s) from batch %s",
        requeued,
        batch_id,
    )

    return requeued