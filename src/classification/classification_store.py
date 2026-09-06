"""SQLite persistence for domain classifications.

Mirrors the conventions of :mod:`src.extraction.representation_store` and
:mod:`src.clustering.cluster_assignments`: thin functions over
:func:`src.common.db.connection_scope`, schema in
``schemas/classification_schema.sql``.

The one rule that differs, and the reason this module exists rather than
another upsert helper: **rows are never overwritten across runs**. An
upsert here is keyed by ``(run_id, doc_id)``, so re-processing a batch
after a crash refreshes that run's own rows and nothing else. Classifying
the corpus again -- new taxonomy version, new classifier version, a
re-run six months later -- uses a new ``run_id`` and therefore adds rows
alongside the old ones. History is queryable
(:func:`get_classification_history`), and "the current label" is a query
over the newest run (:func:`get_current_classifications`) rather than a
mutable field.
"""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

from src.classification.domain_classifier import ClassificationResult
from src.common.db import DEFAULT_DB_PATH, connection_scope
from src.common.logging_utils import get_logger

logger = get_logger(__name__)

DEFAULT_CLASSIFICATION_SCHEMA_FILE = Path("schemas/classification_schema.sql")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _row_to_dict(row: sqlite3.Row) -> dict:
    data = dict(row)
    data["secondary_domains"] = json.loads(data.pop("secondary_domains_json") or "[]")
    return data


def persist_classifications(
    run_id: str,
    results: list[ClassificationResult],
    taxonomy_version: str,
    classifier_version: str,
    model_name: str | None = None,
    batch_id: str | None = None,
    db_path: str | Path = DEFAULT_DB_PATH,
) -> int:
    """Write one row per result for ``run_id``.

    Idempotent *within* a run (a resumed batch overwrites only its own
    rows); additive *across* runs. Every row carries the taxonomy and
    classifier versions that produced it.
    """

    if not results:
        return 0

    rows = [
        (
            f"{run_id}:{r.doc_id}",
            run_id,
            r.doc_id,
            r.cluster_id,
            r.primary_domain,
            json.dumps(r.secondary_domains, ensure_ascii=False),
            r.confidence,
            r.justification,
            r.status,
            r.review_reason,
            r.error,
            taxonomy_version,
            classifier_version,
            model_name,
            batch_id,
            _now(),
        )
        for r in results
    ]

    with connection_scope(db_path) as conn:
        conn.executemany(
            """
            INSERT INTO document_classifications (
                classification_id, run_id, doc_id, cluster_id,
                primary_domain, secondary_domains_json, confidence,
                justification, status, review_reason, error,
                taxonomy_version, classifier_version, model_name,
                batch_id, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(classification_id) DO UPDATE SET
                cluster_id             = excluded.cluster_id,
                primary_domain         = excluded.primary_domain,
                secondary_domains_json = excluded.secondary_domains_json,
                confidence             = excluded.confidence,
                justification          = excluded.justification,
                status                 = excluded.status,
                review_reason          = excluded.review_reason,
                error                  = excluded.error,
                taxonomy_version       = excluded.taxonomy_version,
                classifier_version     = excluded.classifier_version,
                model_name             = excluded.model_name,
                batch_id               = excluded.batch_id,
                created_at             = excluded.created_at
            """,
            rows,
        )

    logger.info("Persisted %d classification(s) for run %s", len(rows), run_id)
    return len(rows)


def get_classified_doc_ids(
    run_id: str, db_path: str | Path = DEFAULT_DB_PATH
) -> set[str]:
    """Documents this run has already decided -- the basis of resumability."""

    with connection_scope(db_path) as conn:
        rows = conn.execute(
            "SELECT doc_id FROM document_classifications WHERE run_id = ?", (run_id,)
        ).fetchall()
    return {r["doc_id"] for r in rows}


def get_classifications_for_run(
    run_id: str, db_path: str | Path = DEFAULT_DB_PATH
) -> list[dict]:
    with connection_scope(db_path) as conn:
        rows = conn.execute(
            "SELECT * FROM document_classifications WHERE run_id = ? ORDER BY doc_id",
            (run_id,),
        ).fetchall()
    return [_row_to_dict(r) for r in rows]


def get_classification_history(
    doc_id: str, db_path: str | Path = DEFAULT_DB_PATH
) -> list[dict]:
    """Every verdict ever recorded for a document, newest first."""

    with connection_scope(db_path) as conn:
        rows = conn.execute(
            """
            SELECT * FROM document_classifications
            WHERE doc_id = ?
            ORDER BY created_at DESC
            """,
            (doc_id,),
        ).fetchall()
    return [_row_to_dict(r) for r in rows]


def get_current_classifications(
    db_path: str | Path = DEFAULT_DB_PATH,
    taxonomy_version: str | None = None,
) -> list[dict]:
    """The newest verdict per document (optionally within one taxonomy version).

    "Current" is derived, never stored: superseding a classification means
    writing a newer row, so this query -- not an UPDATE -- is what moves a
    document to a new domain.
    """

    sql = """
        SELECT c.* FROM document_classifications c
        JOIN (
            SELECT doc_id, MAX(created_at) AS newest
            FROM document_classifications
            {where}
            GROUP BY doc_id
        ) latest
          ON latest.doc_id = c.doc_id AND latest.newest = c.created_at
        {where}
        ORDER BY c.doc_id
    """.format(where="WHERE taxonomy_version = :version" if taxonomy_version else "")

    params = {"version": taxonomy_version} if taxonomy_version else {}
    with connection_scope(db_path) as conn:
        rows = conn.execute(sql, params).fetchall()
    return [_row_to_dict(r) for r in rows]


def classification_stats(
    run_id: str, db_path: str | Path = DEFAULT_DB_PATH
) -> dict:
    """Quality statistics for one classification run."""

    with connection_scope(db_path) as conn:
        by_status = {
            r["status"]: r["n"]
            for r in conn.execute(
                "SELECT status, COUNT(*) AS n FROM document_classifications "
                "WHERE run_id = ? GROUP BY status",
                (run_id,),
            )
        }
        by_domain = {
            (r["primary_domain"] or "(none)"): r["n"]
            for r in conn.execute(
                "SELECT primary_domain, COUNT(*) AS n FROM document_classifications "
                "WHERE run_id = ? AND status != 'failed' "
                "GROUP BY primary_domain ORDER BY n DESC",
                (run_id,),
            )
        }
        by_review_reason = {
            (r["review_reason"] or "(none)"): r["n"]
            for r in conn.execute(
                "SELECT review_reason, COUNT(*) AS n FROM document_classifications "
                "WHERE run_id = ? AND status = 'needs_review' GROUP BY review_reason",
                (run_id,),
            )
        }
        by_error = {
            (r["error"] or "(none)").split(":", 1)[0]: r["n"]
            for r in conn.execute(
                "SELECT error, COUNT(*) AS n FROM document_classifications "
                "WHERE run_id = ? AND status = 'failed' GROUP BY error",
                (run_id,),
            )
        }
        aggregate = conn.execute(
            """
            SELECT COUNT(*) AS total,
                   AVG(confidence) AS mean_confidence,
                   MIN(confidence) AS min_confidence,
                   MAX(confidence) AS max_confidence,
                   SUM(CASE WHEN secondary_domains_json != '[]' THEN 1 ELSE 0 END) AS with_secondary
            FROM document_classifications
            WHERE run_id = ? AND status != 'failed'
            """,
            (run_id,),
        ).fetchone()
        versions = conn.execute(
            "SELECT DISTINCT taxonomy_version, classifier_version, model_name "
            "FROM document_classifications WHERE run_id = ?",
            (run_id,),
        ).fetchall()

    total = sum(by_status.values())
    return {
        "run_id": run_id,
        "total": total,
        "by_status": by_status,
        "by_primary_domain": by_domain,
        "by_review_reason": by_review_reason,
        "by_error": by_error,
        "mean_confidence": aggregate["mean_confidence"],
        "min_confidence": aggregate["min_confidence"],
        "max_confidence": aggregate["max_confidence"],
        "with_secondary_domains": aggregate["with_secondary"] or 0,
        "needs_review_rate": (by_status.get("needs_review", 0) / total) if total else 0.0,
        "failure_rate": (by_status.get("failed", 0) / total) if total else 0.0,
        "versions": [dict(v) for v in versions],
    }
