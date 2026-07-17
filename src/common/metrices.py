"""SQLite-backed run metrics store.

Records per-phase execution statistics (processed/failed counts, duration)
so runs can be audited and dashboards/alerts built on top of a plain
SQLite file, consistent with the rest of the pipeline's "no extra
infrastructure" philosophy.

Usage::

    from src.common.metrics import MetricsStore

    store = MetricsStore("var/metrics.db")
    store.init_schema()

    with store.record_phase(run_id="r1", batch_id="b1", phase="pull") as rec:
        ...  # do work
        rec.processed = 480
        rec.failed = 20
"""

from __future__ import annotations

import sqlite3
import time
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator, Optional

from src.common.exceptions import DatabaseError
from src.common.logging_utils import get_logger

logger = get_logger(__name__)

DEFAULT_METRICS_DB_PATH = Path("var/metrics.db")

_SCHEMA = """
CREATE TABLE IF NOT EXISTS phase_metrics (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id          TEXT NOT NULL,
    batch_id        TEXT,
    doc_id          TEXT,
    phase           TEXT NOT NULL,
    status          TEXT NOT NULL,
    processed       INTEGER NOT NULL DEFAULT 0,
    failed          INTEGER NOT NULL DEFAULT 0,
    duration_ms     REAL,
    error           TEXT,
    started_at      TIMESTAMP NOT NULL,
    finished_at     TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_phase_metrics_run
    ON phase_metrics (run_id, phase);

CREATE INDEX IF NOT EXISTS idx_phase_metrics_batch
    ON phase_metrics (batch_id);
"""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class PhaseMetric:
    """Mutable record yielded by :meth:`MetricsStore.record_phase`."""

    run_id: str
    phase: str
    batch_id: Optional[str] = None
    doc_id: Optional[str] = None
    processed: int = 0
    failed: int = 0
    error: Optional[str] = None


class MetricsStore:
    """Thin wrapper around a SQLite metrics database."""

    def __init__(self, db_path: str | Path = DEFAULT_METRICS_DB_PATH) -> None:
        self.db_path = Path(db_path)

    def _connect(self) -> sqlite3.Connection:
        try:
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
            conn = sqlite3.connect(str(self.db_path), timeout=30.0, isolation_level=None)
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA journal_mode = WAL;")
            return conn
        except sqlite3.Error as exc:
            raise DatabaseError(f"Failed to open metrics DB at {self.db_path}", cause=exc) from exc

    def init_schema(self) -> None:
        conn = self._connect()
        try:
            conn.executescript(_SCHEMA)
        except sqlite3.Error as exc:
            raise DatabaseError("Failed to initialize metrics schema", cause=exc) from exc
        finally:
            conn.close()

    def _insert(self, metric: PhaseMetric, status: str, duration_ms: float) -> None:
        conn = self._connect()
        try:
            conn.execute(
                """
                INSERT INTO phase_metrics
                    (run_id, batch_id, doc_id, phase, status, processed,
                     failed, duration_ms, error, started_at, finished_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    metric.run_id,
                    metric.batch_id,
                    metric.doc_id,
                    metric.phase,
                    status,
                    metric.processed,
                    metric.failed,
                    duration_ms,
                    metric.error,
                    _now(),
                    _now(),
                ),
            )
        except sqlite3.Error as exc:
            raise DatabaseError("Failed to write phase metric", cause=exc) from exc
        finally:
            conn.close()

    @contextmanager
    def record_phase(
        self,
        run_id: str,
        phase: str,
        batch_id: str | None = None,
        doc_id: str | None = None,
    ) -> Iterator[PhaseMetric]:
        """Time a block of work and persist a metrics row for it.

        Yields a mutable :class:`PhaseMetric` the caller can update
        (``processed``, ``failed``) before the block exits. If the block
        raises, the row is recorded with ``status="failed"`` and the
        exception is re-raised.
        """

        metric = PhaseMetric(run_id=run_id, phase=phase, batch_id=batch_id, doc_id=doc_id)
        start = time.monotonic()
        try:
            yield metric
        except Exception as exc:
            metric.error = f"{type(exc).__name__}: {exc}"
            duration_ms = (time.monotonic() - start) * 1000
            self._insert(metric, status="failed", duration_ms=duration_ms)
            raise
        else:
            duration_ms = (time.monotonic() - start) * 1000
            status = "failed" if metric.failed and not metric.processed else "completed"
            self._insert(metric, status=status, duration_ms=duration_ms)

    def summary_for_run(self, run_id: str) -> list[dict]:
        conn = self._connect()
        try:
            rows = conn.execute(
                """
                SELECT phase, status, SUM(processed) AS processed,
                       SUM(failed) AS failed, AVG(duration_ms) AS avg_duration_ms,
                       COUNT(*) AS runs
                FROM phase_metrics
                WHERE run_id = ?
                GROUP BY phase, status
                ORDER BY phase
                """,
                (run_id,),
            ).fetchall()
            return [dict(r) for r in rows]
        except sqlite3.Error as exc:
            raise DatabaseError("Failed to summarize run metrics", cause=exc) from exc
        finally:
            conn.close()