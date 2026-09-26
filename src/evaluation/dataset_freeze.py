"""Phase 7: freeze the accepted corpus for GraphRAG ingestion.

A freeze is the moment the project stops saying "the classifier thinks"
and starts saying "this is the Family Law corpus". Three rules follow
from that:

* **Evaluation first.** :func:`freeze_corpus` refuses unless a readiness
  report permits it. An unevaluated freeze would let every later
  GraphRAG result rest on an unmeasured corpus, which is precisely the
  failure this phase exists to prevent.
* **A snapshot, not a move.** The documents stay in
  ``document_representations``. A freeze records *which* doc_ids were
  accepted and under which evaluation, so an index built on one freeze
  stays reproducible after a later one exists, and so nothing is ever
  deleted to make a corpus.
* **Provenance travels with it.** Each member carries its domain, its
  confidence and whether a *human* confirmed it -- a distinction GraphRAG
  may want to weight, and one that cannot be recovered later if it is
  dropped here.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from src.classification import review_policy as policy
from src.classification.review_store import get_reviewed_doc_ids
from src.common.db import DEFAULT_DB_PATH, connection_scope
from src.common.exceptions import ConfigurationError
from src.common.logging_utils import get_logger
from src.evaluation.readiness import VERDICT_READY_WITH_RESERVATIONS, ReadinessReport

logger = get_logger(__name__)

DEFAULT_FREEZE_SCHEMA_FILE = Path("schemas/corpus_freeze_schema.sql")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass(frozen=True)
class FrozenCorpus:
    """One frozen snapshot, as handed to GraphRAG."""

    freeze_id: str
    decision_run_id: str
    taxonomy_version: str
    readiness_verdict: str
    document_count: int = 0
    domain_counts: dict[str, int] = field(default_factory=dict)
    human_reviewed_count: int = 0
    created_at: str = ""

    @property
    def human_reviewed_share(self) -> float:
        return (
            self.human_reviewed_count / self.document_count
            if self.document_count
            else 0.0
        )

    def as_dict(self) -> dict:
        return {
            "freeze_id": self.freeze_id,
            "decision_run_id": self.decision_run_id,
            "taxonomy_version": self.taxonomy_version,
            "readiness_verdict": self.readiness_verdict,
            "document_count": self.document_count,
            "domain_counts": self.domain_counts,
            "human_reviewed_count": self.human_reviewed_count,
            "human_reviewed_share": self.human_reviewed_share,
            "created_at": self.created_at,
        }


def accepted_documents(
    domains: list[str], db_path: str | Path = DEFAULT_DB_PATH
) -> list[dict]:
    """The documents currently accepted into a target domain.

    Read from ``document_representations`` -- the current state, which is
    where a human review has already taken effect -- rather than from a
    classification run, so a freeze reflects reviewed reality and not the
    machine's last word.
    """

    if not domains:
        return []

    placeholders = ",".join("?" for _ in domains)
    with connection_scope(db_path) as conn:
        rows = conn.execute(
            f"""
            SELECT doc_id, primary_domain, domain_confidence, source_relpath,
                   title, court, decision_date
              FROM document_representations
             WHERE classification_status = ?
               AND primary_domain IN ({placeholders})
             ORDER BY primary_domain, doc_id
            """,
            (policy.STATUS_AUTO_ACCEPTED, *domains),
        ).fetchall()
    return [dict(r) for r in rows]


def freeze_corpus(
    freeze_id: str,
    decision_run_id: str,
    domains: list[str],
    taxonomy_version: str,
    readiness: ReadinessReport,
    db_path: str | Path = DEFAULT_DB_PATH,
    signal_run_id: str | None = None,
    notes: str | None = None,
    allow_reservations: bool = True,
) -> FrozenCorpus:
    """Snapshot the accepted corpus, if the evaluation permits it.

    Raises :class:`ConfigurationError` when the readiness verdict does not
    allow a freeze -- deliberately an error rather than a warning and an
    empty snapshot, because a silently empty corpus would be discovered
    somewhere in GraphRAG, far from its cause.
    """

    if not readiness.can_freeze:
        raise ConfigurationError(
            f"cannot freeze corpus {freeze_id!r}: readiness verdict is "
            f"{readiness.verdict!r}. {readiness.answer} "
            "Fix the blockers and re-run the evaluation; the freeze is the "
            "last step, not the first."
        )
    if readiness.verdict == VERDICT_READY_WITH_RESERVATIONS and not allow_reservations:
        raise ConfigurationError(
            f"cannot freeze corpus {freeze_id!r}: the evaluation passed only with "
            f"reservations ({'; '.join(readiness.reservations)}), and "
            "allow_reservations=False was requested"
        )

    documents = accepted_documents(domains, db_path=db_path)
    if not documents:
        raise ConfigurationError(
            f"cannot freeze corpus {freeze_id!r}: no accepted documents found. "
            "A freeze with no members would be indistinguishable downstream "
            "from a corpus that had not been built yet."
        )

    reviewed = get_reviewed_doc_ids(db_path=db_path)
    domain_counts: dict[str, int] = {}
    human_reviewed = 0
    members = []
    created_at = _now()

    for document in documents:
        domain = document["primary_domain"]
        domain_counts[domain] = domain_counts.get(domain, 0) + 1
        is_reviewed = int(document["doc_id"] in reviewed)
        human_reviewed += is_reviewed
        members.append((
            freeze_id,
            document["doc_id"],
            domain,
            document["domain_confidence"],
            is_reviewed,
        ))

    with connection_scope(db_path) as conn:
        existing = conn.execute(
            "SELECT 1 FROM corpus_freezes WHERE freeze_id = ?", (freeze_id,)
        ).fetchone()
        if existing:
            raise ConfigurationError(
                f"freeze {freeze_id!r} already exists. A freeze is immutable by "
                "design -- choose a new freeze_id rather than overwriting the "
                "corpus an index may already have been built on."
            )

        conn.execute(
            """
            INSERT INTO corpus_freezes (
                freeze_id, decision_run_id, signal_run_id, taxonomy_version,
                readiness_verdict, readiness_json, validation_size, macro_f1,
                document_count, domain_counts_json, notes, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                freeze_id,
                decision_run_id,
                signal_run_id,
                taxonomy_version,
                readiness.verdict,
                json.dumps(readiness.as_dict(), ensure_ascii=False, default=str),
                readiness.validation.get("size", 0),
                readiness.classification.get("macro_f1"),
                len(members),
                json.dumps(domain_counts, ensure_ascii=False),
                notes,
                created_at,
            ),
        )
        conn.executemany(
            """
            INSERT INTO corpus_freeze_members (
                freeze_id, doc_id, domain, confidence, human_reviewed
            ) VALUES (?, ?, ?, ?, ?)
            """,
            members,
        )

    frozen = FrozenCorpus(
        freeze_id=freeze_id,
        decision_run_id=decision_run_id,
        taxonomy_version=taxonomy_version,
        readiness_verdict=readiness.verdict,
        document_count=len(members),
        domain_counts=domain_counts,
        human_reviewed_count=human_reviewed,
        created_at=created_at,
    )
    logger.info(
        "Froze corpus %s: %d document(s) %s, %.1f%% human-reviewed, verdict %s",
        freeze_id, len(members), domain_counts,
        frozen.human_reviewed_share * 100, readiness.verdict,
    )
    return frozen


def get_freeze(
    freeze_id: str, db_path: str | Path = DEFAULT_DB_PATH
) -> FrozenCorpus | None:
    with connection_scope(db_path) as conn:
        row = conn.execute(
            "SELECT * FROM corpus_freezes WHERE freeze_id = ?", (freeze_id,)
        ).fetchone()
        if row is None:
            return None
        reviewed = conn.execute(
            "SELECT COUNT(*) AS n FROM corpus_freeze_members "
            "WHERE freeze_id = ? AND human_reviewed = 1",
            (freeze_id,),
        ).fetchone()["n"]

    return FrozenCorpus(
        freeze_id=row["freeze_id"],
        decision_run_id=row["decision_run_id"],
        taxonomy_version=row["taxonomy_version"],
        readiness_verdict=row["readiness_verdict"],
        document_count=row["document_count"],
        domain_counts=json.loads(row["domain_counts_json"] or "{}"),
        human_reviewed_count=reviewed,
        created_at=row["created_at"],
    )


def list_freezes(db_path: str | Path = DEFAULT_DB_PATH) -> list[dict]:
    """Every freeze, newest first."""

    with connection_scope(db_path) as conn:
        rows = conn.execute(
            "SELECT freeze_id, decision_run_id, taxonomy_version, "
            "readiness_verdict, document_count, domain_counts_json, created_at "
            "FROM corpus_freezes ORDER BY created_at DESC"
        ).fetchall()
    return [
        {**dict(r), "domain_counts": json.loads(r["domain_counts_json"] or "{}")}
        for r in rows
    ]


def get_frozen_doc_ids(
    freeze_id: str,
    db_path: str | Path = DEFAULT_DB_PATH,
    domain: str | None = None,
) -> list[str]:
    """The members of a freeze -- what GraphRAG ingests."""

    sql = "SELECT doc_id FROM corpus_freeze_members WHERE freeze_id = ?"
    params: list = [freeze_id]
    if domain:
        sql += " AND domain = ?"
        params.append(domain)
    sql += " ORDER BY doc_id"

    with connection_scope(db_path) as conn:
        return [r["doc_id"] for r in conn.execute(sql, params)]
