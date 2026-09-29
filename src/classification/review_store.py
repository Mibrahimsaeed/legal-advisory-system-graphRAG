"""Phase 6: the review queue and the human decision ledger.

Three jobs:

* **build the queue** -- every document the policy routed to
  ``needs_review``, with the doc_id, the proposed domain, the confidence,
  the evidence and the reason, exported in a form a person can actually
  work through (:func:`build_review_queue`, :func:`export_review_queue`).
* **record what they decided** -- append-only into
  ``document_review_decisions`` (:func:`persist_review_decisions`).
* **apply it** -- write the human verdict onto Phase 1's current-state
  fields, and thereafter protect it from being reset by a rerun
  (:func:`apply_review_decisions`).

Nothing here deletes a document. A rejection writes
``dropped_off_domain``, which withholds the row from the downstream
corpus and leaves its text, metadata and provenance untouched.
"""

from __future__ import annotations

import csv
import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

from src.classification import review_policy as policy
from src.common.db import DEFAULT_DB_PATH, connection_scope
from src.common.logging_utils import get_logger
from src.extraction.representation_store import update_classification_state

logger = get_logger(__name__)

DEFAULT_REVIEW_SCHEMA_FILE = Path("schemas/review_schema.sql")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _row_to_dict(row: sqlite3.Row) -> dict:
    return dict(row)


# ---------------------------------------------------------------------------
# The queue
# ---------------------------------------------------------------------------


def build_review_queue(
    run_id: str,
    db_path: str | Path = DEFAULT_DB_PATH,
    statuses: tuple[str, ...] = (policy.STATUS_NEEDS_REVIEW,),
    limit: int | None = None,
    include_reviewed: bool = False,
) -> list[dict]:
    """The documents awaiting a human, newest decision run first.

    One row per document carrying exactly what the audit requirement
    names -- ``doc_id``, the classification result, the
    confidence/evidence, the status and the reason -- plus the title and
    source path, because a reviewer needs to find the judgment itself.

    Already-reviewed documents are excluded by default: a queue that
    keeps re-serving decided work is a queue nobody finishes.
    """

    if not statuses:
        return []

    placeholders = ",".join("?" for _ in statuses)
    sql = f"""
        SELECT c.doc_id, c.primary_domain, c.confidence, c.status,
               c.review_reason, c.justification, c.cluster_id,
               c.signal_run_id, c.batch_id, c.created_at,
               r.title, r.source_relpath, r.source_uri, r.court,
               r.decision_date, r.char_count,
               r.classification_status AS current_status
          FROM document_classifications c
          LEFT JOIN document_representations r ON r.doc_id = c.doc_id
         WHERE c.run_id = ? AND c.status IN ({placeholders})
         ORDER BY c.confidence ASC, c.doc_id ASC
    """
    params: list = [run_id, *statuses]

    with connection_scope(db_path) as conn:
        rows = [_row_to_dict(r) for r in conn.execute(sql, params)]

    if not include_reviewed:
        reviewed = get_reviewed_doc_ids(db_path=db_path)
        rows = [r for r in rows if r["doc_id"] not in reviewed]

    for row in rows:
        row["evidence"] = _split_evidence(row.get("justification"))
        row["review_reasons"] = (row.get("review_reason") or "").split(",") if row.get(
            "review_reason"
        ) else []

    return rows[:limit] if limit else rows


def _split_evidence(justification: str | None) -> dict:
    """Recover the structured evidence the decision stored alongside its prose."""

    if not justification or "evidence=" not in justification:
        return {}
    try:
        return json.loads(justification.split("evidence=", 1)[1])
    except json.JSONDecodeError:
        return {}


def export_review_queue(
    run_id: str,
    output_dir: str | Path,
    db_path: str | Path = DEFAULT_DB_PATH,
    statuses: tuple[str, ...] = (policy.STATUS_NEEDS_REVIEW,),
    limit: int | None = None,
) -> dict:
    """Write the queue as CSV (to work through) and JSONL (full evidence).

    The CSV carries a ``decision`` and ``notes`` column left blank for the
    reviewer to fill in, so the file they receive is the file they return:
    :func:`load_review_decisions_csv` reads it straight back.
    """

    queue = build_review_queue(run_id, db_path=db_path, statuses=statuses, limit=limit)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_path = output_dir / f"review_queue_{run_id}.csv"
    jsonl_path = output_dir / f"review_queue_{run_id}.jsonl"

    fields = [
        "doc_id", "proposed_domain", "confidence", "band", "review_reason",
        "title", "court", "decision_date", "source_relpath",
        "decision", "decision_domain", "reviewer", "notes",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in queue:
            writer.writerow({
                "doc_id": row["doc_id"],
                "proposed_domain": row["primary_domain"],
                "confidence": (
                    f"{row['confidence']:.3f}" if row["confidence"] is not None else ""
                ),
                "band": row["evidence"].get("band", ""),
                "review_reason": row.get("review_reason") or "",
                "title": row.get("title") or "",
                "court": row.get("court") or "",
                "decision_date": row.get("decision_date") or "",
                "source_relpath": row.get("source_relpath") or "",
                "decision": "", "decision_domain": "", "reviewer": "", "notes": "",
            })

    with jsonl_path.open("w", encoding="utf-8") as handle:
        for row in queue:
            handle.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")

    logger.info(
        "Review queue for run %s: %d document(s) -> %s", run_id, len(queue), csv_path
    )
    return {"queued": len(queue), "csv": str(csv_path), "jsonl": str(jsonl_path)}


def load_review_decisions_csv(csv_path: str | Path) -> list[dict]:
    """Read back a reviewer-completed queue file, skipping blank rows.

    A row with no ``decision`` is simply not yet reviewed -- not an error.
    Unknown decision values *are* an error: silently discarding a
    reviewer's work would be the worst possible failure mode here.
    """

    decisions = []
    with Path(csv_path).open(newline="", encoding="utf-8") as handle:
        for line_number, row in enumerate(csv.DictReader(handle), start=2):
            decision = (row.get("decision") or "").strip()
            if not decision:
                continue
            if decision not in policy.HUMAN_DECISIONS:
                raise ValueError(
                    f"{csv_path}:{line_number}: unknown decision {decision!r} for "
                    f"{row.get('doc_id')!r}; expected one of "
                    f"{', '.join(policy.HUMAN_DECISIONS)}"
                )
            decisions.append({
                "doc_id": (row.get("doc_id") or "").strip(),
                "decision": decision,
                "decision_domain": (row.get("decision_domain") or "").strip() or None,
                "reviewer": (row.get("reviewer") or "").strip(),
                "notes": (row.get("notes") or "").strip() or None,
            })
    return decisions


# ---------------------------------------------------------------------------
# The ledger
# ---------------------------------------------------------------------------


def persist_review_decisions(
    decisions: list[dict],
    db_path: str | Path = DEFAULT_DB_PATH,
    decision_run_id: str | None = None,
    source: str = "queue",
) -> int:
    """Append review decisions to the ledger.

    Each decision needs ``doc_id``, ``decision`` and ``reviewer``; the
    machine context (``machine_domain``, ``machine_confidence``,
    ``machine_status``, ``machine_band``, ``signal_run_id``) is optional
    and filled from the classification row when omitted.
    """

    if not decisions:
        return 0

    rows = []
    with connection_scope(db_path) as conn:
        for decision in decisions:
            doc_id = decision.get("doc_id")
            verdict = decision.get("decision")
            reviewer = (decision.get("reviewer") or "").strip()
            if not doc_id or not verdict:
                raise ValueError(f"review decision needs doc_id and decision: {decision!r}")
            if verdict not in policy.HUMAN_DECISIONS:
                raise ValueError(
                    f"unknown review decision {verdict!r} for {doc_id!r}; expected "
                    f"one of {', '.join(policy.HUMAN_DECISIONS)}"
                )
            if not reviewer:
                # An unattributed review cannot be followed up or trusted.
                raise ValueError(f"review of {doc_id!r} has no reviewer")

            context = decision
            if decision_run_id and "machine_domain" not in decision:
                found = conn.execute(
                    "SELECT primary_domain, confidence, status, signal_run_id "
                    "FROM document_classifications WHERE run_id = ? AND doc_id = ?",
                    (decision_run_id, doc_id),
                ).fetchone()
                if found:
                    context = {
                        **decision,
                        "machine_domain": found["primary_domain"],
                        "machine_confidence": found["confidence"],
                        "machine_status": found["status"],
                        "signal_run_id": found["signal_run_id"],
                    }

            created_at = _now()
            rows.append((
                f"{doc_id}:{created_at}",
                doc_id,
                verdict,
                context.get("decision_domain"),
                reviewer,
                context.get("notes"),
                context.get("machine_domain"),
                context.get("machine_confidence"),
                context.get("machine_status"),
                context.get("machine_band"),
                decision_run_id,
                context.get("signal_run_id"),
                source,
                created_at,
            ))

        conn.executemany(
            """
            INSERT INTO document_review_decisions (
                review_id, doc_id, decision, decision_domain, reviewer, notes,
                machine_domain, machine_confidence, machine_status, machine_band,
                decision_run_id, signal_run_id, source, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            rows,
        )

    logger.info("Recorded %d review decision(s)", len(rows))
    return len(rows)


def get_reviewed_doc_ids(db_path: str | Path = DEFAULT_DB_PATH) -> set[str]:
    """Every document a human has decided -- the basis of rerun protection."""

    with connection_scope(db_path) as conn:
        rows = conn.execute(
            "SELECT DISTINCT doc_id FROM document_review_decisions"
        ).fetchall()
    return {r["doc_id"] for r in rows}


def get_current_reviews(db_path: str | Path = DEFAULT_DB_PATH) -> dict[str, dict]:
    """The newest review per document, keyed by doc_id.

    "Current" is derived, never stored: re-reviewing writes a new row, so
    this query -- not an UPDATE -- is what supersedes an earlier verdict.
    """

    with connection_scope(db_path) as conn:
        rows = conn.execute(
            """
            SELECT d.* FROM document_review_decisions d
            JOIN (
                SELECT doc_id, MAX(created_at) AS newest
                FROM document_review_decisions GROUP BY doc_id
            ) latest
              ON latest.doc_id = d.doc_id AND latest.newest = d.created_at
            ORDER BY d.doc_id
            """
        ).fetchall()
    return {r["doc_id"]: _row_to_dict(r) for r in rows}


def get_review_history(
    doc_id: str, db_path: str | Path = DEFAULT_DB_PATH
) -> list[dict]:
    """Every review of one document, oldest first."""

    with connection_scope(db_path) as conn:
        rows = conn.execute(
            "SELECT * FROM document_review_decisions WHERE doc_id = ? "
            "ORDER BY created_at",
            (doc_id,),
        ).fetchall()
    return [_row_to_dict(r) for r in rows]


def review_stats(db_path: str | Path = DEFAULT_DB_PATH) -> dict:
    """Counts by decision and reviewer, plus machine/human agreement.

    Agreement is measured only over reviews that named a machine domain,
    and counts a correction to a *different* domain as a disagreement --
    which is what Phase 7 reports as classifier accuracy.
    """

    current = get_current_reviews(db_path=db_path)
    by_decision: dict[str, int] = {}
    by_reviewer: dict[str, int] = {}
    comparable = agreed = 0

    for review in current.values():
        by_decision[review["decision"]] = by_decision.get(review["decision"], 0) + 1
        by_reviewer[review["reviewer"]] = by_reviewer.get(review["reviewer"], 0) + 1

        machine = review.get("machine_domain")
        human = human_domain(review)
        if machine and human:
            comparable += 1
            agreed += int(machine == human)

    return {
        "reviewed_documents": len(current),
        "by_decision": by_decision,
        "by_reviewer": by_reviewer,
        "comparable": comparable,
        "agreed": agreed,
        "agreement_rate": (agreed / comparable) if comparable else None,
    }


def human_domain(review: dict) -> str | None:
    """The domain a review settled on, whatever route it took.

    An acceptance endorses the machine's domain; a correction names its
    own; a rejection means the catch-all. An unresolved review names
    nothing -- and must not be read as agreement.
    """

    decision = review.get("decision")
    if decision == policy.REVIEW_ACCEPTED:
        return review.get("machine_domain")
    if decision == policy.REVIEW_CORRECTED:
        return review.get("decision_domain")
    if decision == policy.REVIEW_REJECTED:
        from src.classification.taxonomy_registry import OTHER_DOMAIN_ID

        return OTHER_DOMAIN_ID
    return None


# ---------------------------------------------------------------------------
# Applying reviews to the corpus state
# ---------------------------------------------------------------------------


def apply_review_decisions(
    decisions: list[dict],
    db_path: str | Path = DEFAULT_DB_PATH,
    decision_run_id: str | None = None,
    source: str = "queue",
) -> dict:
    """Record reviews, then write their outcome onto the current state.

    Ledger first, state second: if this dies in between, the decision is
    still on record and re-applying is idempotent. The reverse order could
    lose a reviewer's work.
    """

    recorded = persist_review_decisions(
        decisions, db_path=db_path, decision_run_id=decision_run_id, source=source
    )

    outcomes: dict[str, int] = {}
    current = get_current_reviews(db_path=db_path)
    for decision in decisions:
        doc_id = decision["doc_id"]
        review = current.get(doc_id, {})
        status, domain, drop_reason = policy.resolve_human_decision(
            decision["decision"],
            machine_domain=decision.get("machine_domain") or review.get("machine_domain"),
            corrected_domain=decision.get("decision_domain"),
        )
        outcome = update_classification_state(
            doc_id,
            classification_status=status,
            primary_domain=None if domain == _other_id() else domain,
            domain_confidence=decision.get("machine_confidence")
            or review.get("machine_confidence"),
            drop_reason=drop_reason,
            db_path=db_path,
        )
        outcomes[outcome] = outcomes.get(outcome, 0) + 1

    logger.info("Applied %d review decision(s): %s", recorded, outcomes)
    return {"recorded": recorded, "outcomes": outcomes}


def _other_id() -> str:
    from src.classification.taxonomy_registry import OTHER_DOMAIN_ID

    return OTHER_DOMAIN_ID
