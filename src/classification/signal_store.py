"""SQLite persistence for Phase 3 domain evidence.

Same conventions as :mod:`src.classification.classification_store`: thin
functions over :func:`src.common.db.connection_scope`, schema in
``schemas/domain_signals_schema.sql``, upsert keyed by ``(run_id,
doc_id)`` so a resumed batch refreshes only its own rows while a new
``run_id`` adds evidence alongside the old.

The read helpers exist because Phase 4 needs them: it must be able to load
every signal for a run in one pass, and a human must be able to ask "what
did we observe about this document, and when".
"""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

from src.classification.domain_assessment import DomainAssessment
from src.classification.keyword_signals import KeywordSignals
from src.common.db import DEFAULT_DB_PATH, connection_scope
from src.common.logging_utils import get_logger

logger = get_logger(__name__)

DEFAULT_SIGNALS_SCHEMA_FILE = Path("schemas/domain_signals_schema.sql")

# The only llm_status that represents a completed assessment. The
# schema's other values -- 'failed' and 'skipped' -- both mean the row
# exists without an LLM reading.
LLM_STATUS_OK = "ok"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _row_to_dict(row: sqlite3.Row) -> dict:
    data = dict(row)
    data["keyword_signals"] = json.loads(data.pop("keyword_signals_json") or "{}")
    return data


def persist_domain_signals(
    run_id: str,
    records: list[dict],
    signal_version: str,
    batch_id: str | None = None,
    db_path: str | Path = DEFAULT_DB_PATH,
) -> int:
    """Write one evidence row per record for ``run_id``.

    Each record is the flat dict produced by
    :func:`build_signal_record`; keeping the shape explicit here means the
    flow never has to know the column order.
    """

    if not records:
        return 0

    rows = [
        (
            f"{run_id}:{r['doc_id']}",
            run_id,
            r["doc_id"],
            r.get("representation_hash"),
            r.get("text_source"),
            r.get("char_count", 0),
            r.get("word_count", 0),
            r.get("cluster_id"),
            r.get("cluster_confidence"),
            r.get("embedding_model"),
            json.dumps(r.get("keyword_signals", {}), ensure_ascii=False),
            r.get("keyword_top_domain"),
            r.get("keyword_margin"),
            r.get("llm_domain"),
            r.get("llm_confidence"),
            r.get("llm_reason"),
            r.get("llm_model"),
            r.get("llm_status", "ok"),
            r.get("llm_error"),
            signal_version,
            batch_id,
            _now(),
        )
        for r in records
    ]

    with connection_scope(db_path) as conn:
        conn.executemany(
            """
            INSERT INTO document_domain_signals (
                signal_id, run_id, doc_id, representation_hash, text_source,
                char_count, word_count, cluster_id, cluster_confidence,
                embedding_model, keyword_signals_json, keyword_top_domain,
                keyword_margin, llm_domain, llm_confidence, llm_reason,
                llm_model, llm_status, llm_error, signal_version, batch_id,
                created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(signal_id) DO UPDATE SET
                representation_hash  = excluded.representation_hash,
                text_source          = excluded.text_source,
                char_count           = excluded.char_count,
                word_count           = excluded.word_count,
                cluster_id           = excluded.cluster_id,
                cluster_confidence   = excluded.cluster_confidence,
                embedding_model      = excluded.embedding_model,
                keyword_signals_json = excluded.keyword_signals_json,
                keyword_top_domain   = excluded.keyword_top_domain,
                keyword_margin       = excluded.keyword_margin,
                llm_domain           = excluded.llm_domain,
                llm_confidence       = excluded.llm_confidence,
                llm_reason           = excluded.llm_reason,
                llm_model            = excluded.llm_model,
                llm_status           = excluded.llm_status,
                llm_error            = excluded.llm_error,
                signal_version       = excluded.signal_version,
                batch_id             = excluded.batch_id,
                created_at           = excluded.created_at
            """,
            rows,
        )

    logger.info("Persisted %d domain signal record(s) for run %s", len(rows), run_id)
    return len(rows)


def build_signal_record(
    representation,
    keyword_signals: KeywordSignals | None,
    assessment: DomainAssessment | None,
    cluster_id: int | None = None,
    cluster_confidence: float | None = None,
    embedding_model: str | None = None,
) -> dict:
    """Flatten the three signals for one case into a storable record."""

    record: dict = {
        "doc_id": representation.doc_id,
        "representation_hash": representation.representation_hash,
        "text_source": representation.text_source,
        "char_count": representation.char_count,
        "word_count": representation.word_count,
        "cluster_id": cluster_id,
        "cluster_confidence": cluster_confidence,
        "embedding_model": embedding_model,
        "keyword_signals": keyword_signals.as_evidence() if keyword_signals else {},
        "keyword_top_domain": keyword_signals.top_domain if keyword_signals else None,
        "keyword_margin": keyword_signals.margin if keyword_signals else None,
    }

    if assessment is None:
        record.update(llm_status="skipped")
        return record

    record.update(
        llm_domain=assessment.domain,
        llm_confidence=assessment.confidence,
        llm_reason=assessment.reason,
        llm_model=assessment.model_name,
        llm_status=assessment.status,
        llm_error=assessment.error,
    )
    return record


def get_signals_for_run(
    run_id: str, db_path: str | Path = DEFAULT_DB_PATH
) -> list[dict]:
    """Every evidence row for a run -- Phase 4's input."""

    with connection_scope(db_path) as conn:
        rows = conn.execute(
            "SELECT * FROM document_domain_signals WHERE run_id = ? ORDER BY doc_id",
            (run_id,),
        ).fetchall()
    return [_row_to_dict(r) for r in rows]


def get_signalled_doc_ids(
    run_id: str,
    db_path: str | Path = DEFAULT_DB_PATH,
    require_llm_ok: bool = False,
) -> set[str]:
    """Documents this run has already gathered evidence for (resumability).

    ``require_llm_ok`` decides what "already gathered" means, because that
    depends on whether an LLM reading is part of the evidence being
    gathered:

    * **False** (a deterministic-only run, ``llm_enabled=False``): any row
      is complete. The LLM was never going to run, so re-deriving the same
      keyword and cluster signals would be pure waste.
    * **True** (the LLM is enabled): only ``llm_status='ok'`` counts.
      ``'failed'`` means the assessment did not happen -- the model was
      unreachable, or its output could not be parsed -- and ``'skipped'``
      means a previous run deliberately did without one. In both cases the
      row exists but the LLM reading is missing, so the document is
      offered for retry rather than treated as finished.

    Without this distinction a transient outage is permanent: the failed
    rows count as done, and no rerun can ever fill them in.
    """

    sql = "SELECT doc_id FROM document_domain_signals WHERE run_id = ?"
    params: list = [run_id]
    if require_llm_ok:
        sql += " AND llm_status = ?"
        params.append(LLM_STATUS_OK)

    with connection_scope(db_path) as conn:
        rows = conn.execute(sql, params).fetchall()
    return {r["doc_id"] for r in rows}


def get_signal_history(
    doc_id: str, db_path: str | Path = DEFAULT_DB_PATH
) -> list[dict]:
    """Every observation ever recorded about a document, newest first."""

    with connection_scope(db_path) as conn:
        rows = conn.execute(
            "SELECT * FROM document_domain_signals WHERE doc_id = ? "
            "ORDER BY created_at DESC",
            (doc_id,),
        ).fetchall()
    return [_row_to_dict(r) for r in rows]


def signal_stats(run_id: str, db_path: str | Path = DEFAULT_DB_PATH) -> dict:
    """Summary of one signal run, for the operator and for QA."""

    with connection_scope(db_path) as conn:
        by_llm = {
            (r["llm_domain"] or "(none)"): r["n"]
            for r in conn.execute(
                "SELECT llm_domain, COUNT(*) AS n FROM document_domain_signals "
                "WHERE run_id = ? GROUP BY llm_domain ORDER BY n DESC",
                (run_id,),
            )
        }
        by_keyword = {
            (r["keyword_top_domain"] or "(none)"): r["n"]
            for r in conn.execute(
                "SELECT keyword_top_domain, COUNT(*) AS n FROM document_domain_signals "
                "WHERE run_id = ? GROUP BY keyword_top_domain ORDER BY n DESC",
                (run_id,),
            )
        }
        by_llm_status = {
            r["llm_status"]: r["n"]
            for r in conn.execute(
                "SELECT llm_status, COUNT(*) AS n FROM document_domain_signals "
                "WHERE run_id = ? GROUP BY llm_status",
                (run_id,),
            )
        }
        clusters = conn.execute(
            "SELECT COUNT(DISTINCT cluster_id) AS n_clusters, "
            "SUM(CASE WHEN cluster_id = -1 THEN 1 ELSE 0 END) AS noise, "
            "COUNT(*) AS total, AVG(llm_confidence) AS mean_confidence "
            "FROM document_domain_signals WHERE run_id = ?",
            (run_id,),
        ).fetchone()
        agreement = conn.execute(
            "SELECT COUNT(*) AS n FROM document_domain_signals "
            "WHERE run_id = ? AND keyword_top_domain IS NOT NULL "
            "AND keyword_top_domain = llm_domain",
            (run_id,),
        ).fetchone()["n"]

    total = clusters["total"] or 0
    return {
        "run_id": run_id,
        "total": total,
        "by_llm_domain": by_llm,
        "by_keyword_domain": by_keyword,
        "by_llm_status": by_llm_status,
        "n_clusters": clusters["n_clusters"] or 0,
        "noise_documents": clusters["noise"] or 0,
        "mean_llm_confidence": clusters["mean_confidence"],
        # How often the two independent deterministic/LLM signals agree.
        # Not a correctness measure -- a disagreement rate is exactly the
        # kind of thing Phase 4 should be tuned against.
        "keyword_llm_agreement": agreement,
        "keyword_llm_agreement_rate": (agreement / total) if total else 0.0,
    }
