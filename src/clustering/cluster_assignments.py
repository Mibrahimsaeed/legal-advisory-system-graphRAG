"""Persistence for raw per-document HDBSCAN cluster assignments.

Stage 1.2 (``orchestration/dags/domain_discovery_flow.py``) only turns the
top-N ranked/labeled clusters into ``domain_candidates`` rows -- every
other document's raw HDBSCAN label (including noise, ``-1``) is otherwise
never written anywhere durable. This module persists the full (doc_id ->
cluster_id, confidence) mapping for every run into the
``cluster_assignments`` table (``schemas/domain_registry_schema.sql``), so
a cluster's complete membership can be inspected later without
re-running discovery.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from src.common.db import DEFAULT_DB_PATH, connection_scope
from src.common.logging_utils import get_logger

logger = get_logger(__name__)


def persist_cluster_assignments(
    run_id: str,
    doc_ids: list[str],
    labels: np.ndarray,
    probabilities: np.ndarray | None,
    db_path: str | Path = DEFAULT_DB_PATH,
) -> int:
    """Insert one ``cluster_assignments`` row per document for this run.

    Idempotent per ``run_id``: re-running discovery with the same run_id
    (e.g. a checkpoint resume that redoes work past the ``cluster`` phase)
    replaces rows rather than duplicating them -- mirrors
    :func:`src.clustering.taxonomy_card.persist_taxonomy_card`.

    ``probabilities`` is ``None`` when the clusterer backend didn't
    provide one; every row's ``confidence`` is then stored as ``NULL``.
    """

    if not doc_ids:
        return 0

    created_at = datetime.now(timezone.utc).isoformat()
    confidences = (
        probabilities.tolist() if probabilities is not None else [None] * len(doc_ids)
    )
    rows = [
        (run_id, doc_id, int(label), confidence, created_at)
        for doc_id, label, confidence in zip(doc_ids, labels.tolist(), confidences)
    ]

    with connection_scope(db_path) as conn:
        conn.execute("DELETE FROM cluster_assignments WHERE run_id = ?", (run_id,))
        conn.executemany(
            """
            INSERT INTO cluster_assignments (
                run_id, doc_id, cluster_id, confidence, created_at
            ) VALUES (?, ?, ?, ?, ?)
            """,
            rows,
        )

    logger.info(
        "Persisted %d cluster assignment(s) for run %s", len(rows), run_id
    )
    return len(rows)
