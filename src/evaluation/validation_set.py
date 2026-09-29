"""Phase 7: assemble the manually reviewed validation set.

The gold labels come from people, and there are exactly two places they
can come from:

* **the Phase 6 review ledger** -- the normal path. Every human decision
  recorded through the review queue already names a domain (an acceptance
  endorses the machine's, a correction supplies its own, a rejection means
  the catch-all), and carries who decided it and when.
* **an external labels file** -- CSV or JSON, for labels produced outside
  the pipeline (a supervisor's spreadsheet, an inter-annotator exercise).

What is **not** a source of gold labels: source-folder names. They are a
weak proxy that Phase 4 used for after-the-fact cluster diagnostics, and
nothing more. Scoring the classifier against them would measure agreement
with a directory layout, then report it as accuracy -- so
:func:`load_validation_set` will not read them, and the readiness verdict
refuses to treat a folder-derived set as a validation set.

An unresolved review (``human_uncertain``) contributes **no** label. A
reviewer who could not decide has not produced ground truth, and counting
their non-answer either way would be fabrication.
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass, field
from pathlib import Path

from src.classification.review_store import get_current_reviews, human_domain
from src.common.db import DEFAULT_DB_PATH
from src.common.exceptions import ConfigurationError
from src.common.logging_utils import get_logger

logger = get_logger(__name__)

SOURCE_REVIEW_LEDGER = "review_ledger"
SOURCE_LABELS_FILE = "labels_file"


@dataclass(frozen=True)
class ValidationSet:
    """Human labels, with enough provenance to judge their weight."""

    labels: dict[str, str] = field(default_factory=dict)
    source: str = SOURCE_REVIEW_LEDGER
    reviewers: dict[str, int] = field(default_factory=dict)
    unresolved: list[str] = field(default_factory=list)
    by_domain: dict[str, int] = field(default_factory=dict)
    audit_sampled: int = 0
    queue_reviewed: int = 0

    def __len__(self) -> int:
        return len(self.labels)

    @property
    def reviewer_count(self) -> int:
        return len(self.reviewers)

    @property
    def is_single_reviewer(self) -> bool:
        """One reviewer means no inter-annotator check is possible.

        Not a defect -- it is normal for a student project -- but it caps
        how much the numbers can be trusted, so the readiness report says
        so out loud rather than letting the F1 speak unqualified.
        """

        return self.reviewer_count <= 1

    def smallest_domain(self) -> tuple[str, int] | None:
        return min(self.by_domain.items(), key=lambda kv: kv[1]) if self.by_domain else None

    def as_dict(self) -> dict:
        return {
            "source": self.source,
            "size": len(self.labels),
            "by_domain": self.by_domain,
            "reviewers": self.reviewers,
            "reviewer_count": self.reviewer_count,
            "unresolved": len(self.unresolved),
            "audit_sampled": self.audit_sampled,
            "queue_reviewed": self.queue_reviewed,
        }


def load_validation_set(
    db_path: str | Path = DEFAULT_DB_PATH,
    labels_file: str | Path | None = None,
    domains: list[str] | None = None,
) -> ValidationSet:
    """Build the validation set from the review ledger, or from a file."""

    if labels_file is not None:
        return _from_labels_file(labels_file, domains=domains)
    return _from_review_ledger(db_path=db_path, domains=domains)


def _from_review_ledger(
    db_path: str | Path, domains: list[str] | None
) -> ValidationSet:
    reviews = get_current_reviews(db_path=db_path)

    labels: dict[str, str] = {}
    reviewers: dict[str, int] = {}
    unresolved: list[str] = []
    audit_sampled = 0
    queue_reviewed = 0

    for doc_id, review in reviews.items():
        domain = human_domain(review)
        if domain is None:
            unresolved.append(doc_id)
            continue
        if domains and domain not in domains:
            logger.warning(
                "Review of %s names domain %r outside the taxonomy; excluded",
                doc_id, domain,
            )
            continue
        labels[doc_id] = domain
        reviewers[review["reviewer"]] = reviewers.get(review["reviewer"], 0) + 1
        if review.get("source") == "audit_sample":
            audit_sampled += 1
        else:
            queue_reviewed += 1

    return ValidationSet(
        labels=labels,
        source=SOURCE_REVIEW_LEDGER,
        reviewers=reviewers,
        unresolved=sorted(unresolved),
        by_domain=_count_by_domain(labels),
        audit_sampled=audit_sampled,
        queue_reviewed=queue_reviewed,
    )


def _from_labels_file(
    labels_file: str | Path, domains: list[str] | None
) -> ValidationSet:
    """Read gold labels from CSV (doc_id,domain[,reviewer]) or JSON."""

    path = Path(labels_file)
    if not path.exists():
        raise ConfigurationError(f"validation labels file not found: {path}")

    labels: dict[str, str] = {}
    reviewers: dict[str, int] = {}

    if path.suffix.lower() == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        rows = (
            payload
            if isinstance(payload, list)
            else [{"doc_id": k, "domain": v} for k, v in payload.items()]
        )
    else:
        with path.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))

    for line_number, row in enumerate(rows, start=2):
        doc_id = str(row.get("doc_id") or "").strip()
        domain = str(row.get("domain") or row.get("label") or "").strip()
        if not doc_id or not domain:
            continue
        if domains and domain not in domains:
            raise ConfigurationError(
                f"{path}:{line_number}: label {domain!r} for {doc_id!r} is not a "
                f"domain in the frozen taxonomy ({', '.join(domains)})"
            )
        labels[doc_id] = domain
        reviewer = str(row.get("reviewer") or "unattributed").strip()
        reviewers[reviewer] = reviewers.get(reviewer, 0) + 1

    logger.info("Loaded %d gold label(s) from %s", len(labels), path)
    return ValidationSet(
        labels=labels,
        source=SOURCE_LABELS_FILE,
        reviewers=reviewers,
        by_domain=_count_by_domain(labels),
        queue_reviewed=len(labels),
    )


def _count_by_domain(labels: dict[str, str]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for domain in labels.values():
        counts[domain] = counts.get(domain, 0) + 1
    return dict(sorted(counts.items()))
