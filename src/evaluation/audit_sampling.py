"""Phase 7: decide what a human must look at before the corpus is trusted.

The audit plan has four parts, and they are not interchangeable:

* **100% of ``needs_review``** -- an obligation, not a sample. The policy
  routed these here precisely because the machine could not settle them,
  so an unreviewed backlog is unfinished work, and the readiness verdict
  says so.
* **a sample of auto-accepted documents in each target domain** -- the
  only way to catch a *confident* error. Auto-accepted documents never
  reach a reviewer on their own, so without spot-checks the pipeline's
  most dangerous failure mode is also its least visible one.
* **a sample of ``other_uncertain``** -- to find out whether the catch-all
  is collecting genuinely off-domain material or quietly swallowing
  Family and Criminal judgments the signals missed. That second case is
  invisible in precision and shows up only here and in recall.

Sampling is **seeded and therefore reproducible**: the same corpus and
seed always produce the same sample, so an audit can be re-run, handed to
a second reviewer, or checked months later. Documents already reviewed
are excluded, so a reviewer is never asked the same question twice.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from pathlib import Path

from src.classification import review_policy as policy
from src.classification.review_store import get_reviewed_doc_ids
from src.common.db import DEFAULT_DB_PATH, connection_scope
from src.common.logging_utils import get_logger

logger = get_logger(__name__)

STRATUM_NEEDS_REVIEW = "needs_review"
STRATUM_AUTO_ACCEPTED = "auto_accepted"
STRATUM_OTHER_UNCERTAIN = "other_uncertain"
STRATUM_DROPPED = "dropped_off_domain"


@dataclass(frozen=True)
class AuditStratum:
    """One slice of the audit, and how much of it must be looked at."""

    name: str
    population: int
    required: int          # how many a human should review
    already_reviewed: int
    doc_ids: list[str] = field(default_factory=list)
    is_census: bool = False   # True when 100% is required, not a sample

    @property
    def outstanding(self) -> int:
        return len(self.doc_ids)

    @property
    def is_complete(self) -> bool:
        return self.outstanding == 0

    def as_dict(self) -> dict:
        return {
            "stratum": self.name,
            "population": self.population,
            "required": self.required,
            "already_reviewed": self.already_reviewed,
            "outstanding": self.outstanding,
            "is_census": self.is_census,
            "doc_ids": self.doc_ids,
        }


@dataclass(frozen=True)
class AuditPlan:
    """What to review, by stratum, with the seed that produced it."""

    run_id: str
    seed: int
    strata: dict[str, AuditStratum] = field(default_factory=dict)

    @property
    def total_outstanding(self) -> int:
        return sum(s.outstanding for s in self.strata.values())

    @property
    def review_backlog_cleared(self) -> bool:
        """Whether every ``needs_review`` document has been seen by a human."""

        census = [s for s in self.strata.values() if s.is_census]
        return all(s.is_complete for s in census)

    def doc_ids(self) -> list[str]:
        seen: list[str] = []
        for stratum in self.strata.values():
            seen.extend(d for d in stratum.doc_ids if d not in seen)
        return seen

    def as_dict(self) -> dict:
        return {
            "run_id": self.run_id,
            "seed": self.seed,
            "total_outstanding": self.total_outstanding,
            "review_backlog_cleared": self.review_backlog_cleared,
            "strata": {name: s.as_dict() for name, s in self.strata.items()},
        }


def _classified_docs(
    run_id: str, db_path: str | Path
) -> list[tuple[str, str, str | None]]:
    """``(doc_id, status, primary_domain)`` for one decision run."""

    with connection_scope(db_path) as conn:
        rows = conn.execute(
            "SELECT doc_id, status, primary_domain FROM document_classifications "
            "WHERE run_id = ? ORDER BY doc_id",
            (run_id,),
        ).fetchall()
    return [(r["doc_id"], r["status"], r["primary_domain"]) for r in rows]


def _sample(
    candidates: list[str], size: int, rng: random.Random
) -> list[str]:
    """A reproducible sample, or everything when the population is smaller."""

    if size >= len(candidates):
        return sorted(candidates)
    return sorted(rng.sample(sorted(candidates), size))


def build_audit_plan(
    run_id: str,
    domains: list[str],
    db_path: str | Path = DEFAULT_DB_PATH,
    sample_per_domain: int = 30,
    sample_other_uncertain: int = 20,
    sample_dropped: int = 10,
    seed: int = 42,
) -> AuditPlan:
    """Work out what still needs a human eye for this decision run.

    ``sample_*`` sizes are **audit effort, not statistical power**. Thirty
    documents per domain will surface a systematic error; it will not give
    a confidence interval, and the readiness report does not pretend
    otherwise.
    """

    rng = random.Random(seed)
    reviewed = get_reviewed_doc_ids(db_path=db_path)
    rows = _classified_docs(run_id, db_path=db_path)

    needs_review = [d for d, s, _ in rows if s == policy.STATUS_NEEDS_REVIEW]
    dropped = [d for d, s, _ in rows if s == policy.STATUS_DROPPED_OFF_DOMAIN]
    accepted_by_domain: dict[str, list[str]] = {d: [] for d in domains}
    other_uncertain: list[str] = []

    for doc_id, status, domain in rows:
        if status != policy.STATUS_AUTO_ACCEPTED:
            continue
        if domain in accepted_by_domain:
            accepted_by_domain[domain].append(doc_id)
        else:
            other_uncertain.append(doc_id)

    strata: dict[str, AuditStratum] = {}

    # 1. The census: every unresolved document, no sampling.
    outstanding = [d for d in needs_review if d not in reviewed]
    strata[STRATUM_NEEDS_REVIEW] = AuditStratum(
        name=STRATUM_NEEDS_REVIEW,
        population=len(needs_review),
        required=len(needs_review),
        already_reviewed=len(needs_review) - len(outstanding),
        doc_ids=sorted(outstanding),
        is_census=True,
    )

    # 2. Spot-checks of confident answers, per domain.
    for domain in domains:
        population = accepted_by_domain[domain]
        unreviewed = [d for d in population if d not in reviewed]
        sampled = _sample(unreviewed, sample_per_domain, rng)
        strata[f"{STRATUM_AUTO_ACCEPTED}:{domain}"] = AuditStratum(
            name=f"{STRATUM_AUTO_ACCEPTED}:{domain}",
            population=len(population),
            required=min(sample_per_domain, len(population)),
            already_reviewed=len(population) - len(unreviewed),
            doc_ids=sampled,
        )

    # 3. Is the catch-all a bucket of genuine "neither", or a leak?
    uncertain_population = sorted(set(other_uncertain))
    unreviewed_uncertain = [d for d in uncertain_population if d not in reviewed]
    strata[STRATUM_OTHER_UNCERTAIN] = AuditStratum(
        name=STRATUM_OTHER_UNCERTAIN,
        population=len(uncertain_population),
        required=min(sample_other_uncertain, len(uncertain_population)),
        already_reviewed=len(uncertain_population) - len(unreviewed_uncertain),
        doc_ids=_sample(unreviewed_uncertain, sample_other_uncertain, rng),
    )

    # 4. Dropped documents: nothing was deleted, so a drop stays auditable.
    unreviewed_dropped = [d for d in dropped if d not in reviewed]
    strata[STRATUM_DROPPED] = AuditStratum(
        name=STRATUM_DROPPED,
        population=len(dropped),
        required=min(sample_dropped, len(dropped)),
        already_reviewed=len(dropped) - len(unreviewed_dropped),
        doc_ids=_sample(unreviewed_dropped, sample_dropped, rng),
    )

    plan = AuditPlan(run_id=run_id, seed=seed, strata=strata)
    logger.info(
        "Audit plan for run %s: %d document(s) outstanding across %d strata "
        "(review backlog %s)",
        run_id, plan.total_outstanding, len(strata),
        "cleared" if plan.review_backlog_cleared else "NOT cleared",
    )
    return plan
