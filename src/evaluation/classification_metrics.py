"""Phase 7: score the classifier against human-reviewed labels.

Precision, recall, F1 and a confusion matrix, per domain and in aggregate.
Implemented directly rather than pulled from sklearn so that three things
stay explicit and inspectable:

* **which class is which.** The matrix is keyed by domain id in both
  directions, so a transposed reading is impossible. Getting precision and
  recall the wrong way round is the classic silent error in a report like
  this one.
* **support.** Every per-domain number carries the count it was computed
  from, because an F1 of 1.00 over three documents is not a result.
* **the empty cases.** Precision with no predictions and recall with no
  gold examples are *undefined*, not zero, and are reported as ``None``.
  Averaging them as zeroes would quietly understate a classifier that
  simply never guessed a rare class -- and understating is still lying.

``other_uncertain`` is scored as a class like any other. It is a real
answer -- "this judgment is neither" -- and a classifier that dumps its
hard cases there should be visibly penalised on the other two domains'
recall, which only happens if the catch-all is in the matrix.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping

from src.common.logging_utils import get_logger

logger = get_logger(__name__)

METRICS_VERSION = "classification_metrics/1.0"


def _safe_divide(numerator: int, denominator: int) -> float | None:
    """Undefined, not zero, when there is nothing to divide by."""

    return (numerator / denominator) if denominator else None


@dataclass(frozen=True)
class DomainMetrics:
    """One domain's scores, each with the count behind it."""

    domain: str
    support: int          # gold documents in this domain
    predicted: int        # documents the classifier put here
    true_positives: int
    false_positives: int
    false_negatives: int
    precision: float | None
    recall: float | None
    f1: float | None

    @property
    def is_measurable(self) -> bool:
        """Whether this domain had any gold examples at all."""

        return self.support > 0

    def as_dict(self) -> dict:
        return {
            "domain": self.domain,
            "support": self.support,
            "predicted": self.predicted,
            "true_positives": self.true_positives,
            "false_positives": self.false_positives,
            "false_negatives": self.false_negatives,
            "precision": self.precision,
            "recall": self.recall,
            "f1": self.f1,
        }


@dataclass(frozen=True)
class ClassificationReport:
    """The full classification evaluation, self-describing."""

    n_documents: int
    domains: list[str] = field(default_factory=list)
    per_domain: dict[str, DomainMetrics] = field(default_factory=dict)
    confusion: dict[str, dict[str, int]] = field(default_factory=dict)
    accuracy: float | None = None
    macro_f1: float | None = None
    micro_f1: float | None = None
    weighted_f1: float | None = None
    unlabelled: list[str] = field(default_factory=list)
    metrics_version: str = METRICS_VERSION

    @property
    def measurable_domains(self) -> list[str]:
        return [d for d, m in self.per_domain.items() if m.is_measurable]

    def weakest_domain(self) -> DomainMetrics | None:
        """The measurable domain with the lowest F1 -- what to look at first."""

        scored = [
            m for m in self.per_domain.values()
            if m.is_measurable and m.f1 is not None
        ]
        return min(scored, key=lambda m: m.f1) if scored else None

    def confusion_rows(self) -> list[tuple[str, dict[str, int]]]:
        """The matrix as ordered (gold, {predicted: n}) rows, for printing."""

        return [(domain, self.confusion.get(domain, {})) for domain in self.domains]

    def as_dict(self) -> dict:
        return {
            "metrics_version": self.metrics_version,
            "n_documents": self.n_documents,
            "domains": self.domains,
            "per_domain": {d: m.as_dict() for d, m in self.per_domain.items()},
            "confusion": self.confusion,
            "accuracy": self.accuracy,
            "macro_f1": self.macro_f1,
            "micro_f1": self.micro_f1,
            "weighted_f1": self.weighted_f1,
            "unlabelled": self.unlabelled,
        }


def evaluate_classification(
    gold: Mapping[str, str],
    predicted: Mapping[str, str | None],
    domains: list[str],
) -> ClassificationReport:
    """Score predictions against human labels.

    Only documents present in ``gold`` are scored -- a validation set is
    what a human actually reviewed, never the whole corpus. A gold
    document the classifier never labelled is counted as a miss for its
    domain rather than dropped: refusing to answer is a failure mode that
    has to show up somewhere, and it belongs in recall.

    ``domains`` fixes the row/column order of the confusion matrix, so two
    reports are always comparable.
    """

    confusion = {g: {p: 0 for p in domains} for g in domains}
    unlabelled: list[str] = []
    scored = 0
    correct = 0

    for doc_id, gold_domain in sorted(gold.items()):
        if gold_domain not in domains:
            logger.warning(
                "Validation document %s has label %r outside the taxonomy; skipped",
                doc_id, gold_domain,
            )
            continue

        prediction = predicted.get(doc_id)
        if prediction is None:
            # Never classified, or classified as nothing. It still counts
            # against this domain's recall; it just cannot land in a
            # confusion cell, because there is no predicted class.
            unlabelled.append(doc_id)
            scored += 1
            continue
        if prediction not in domains:
            logger.warning(
                "Document %s predicted as %r outside the taxonomy; treated as unlabelled",
                doc_id, prediction,
            )
            unlabelled.append(doc_id)
            scored += 1
            continue

        confusion[gold_domain][prediction] += 1
        scored += 1
        correct += int(prediction == gold_domain)

    unlabelled_by_domain: dict[str, int] = {}
    for doc_id in unlabelled:
        unlabelled_by_domain[gold[doc_id]] = unlabelled_by_domain.get(gold[doc_id], 0) + 1

    per_domain: dict[str, DomainMetrics] = {}
    for domain in domains:
        true_positives = confusion[domain][domain]
        # Predicted as this domain but actually something else.
        false_positives = sum(
            confusion[other][domain] for other in domains if other != domain
        )
        # Actually this domain but predicted as something else -- including
        # the documents that got no prediction at all.
        false_negatives = (
            sum(confusion[domain][other] for other in domains if other != domain)
            + unlabelled_by_domain.get(domain, 0)
        )
        support = true_positives + false_negatives
        predicted_count = true_positives + false_positives

        precision = _safe_divide(true_positives, predicted_count)
        recall = _safe_divide(true_positives, support)
        f1 = (
            2 * precision * recall / (precision + recall)
            if precision is not None and recall is not None and (precision + recall) > 0
            else (0.0 if precision is not None and recall is not None else None)
        )

        per_domain[domain] = DomainMetrics(
            domain=domain,
            support=support,
            predicted=predicted_count,
            true_positives=true_positives,
            false_positives=false_positives,
            false_negatives=false_negatives,
            precision=precision,
            recall=recall,
            f1=f1,
        )

    # Macro averages over MEASURABLE domains only: a domain with no gold
    # examples has no score to average, and inventing a zero for it would
    # drag the headline number down for a reason that is about the
    # validation set, not the classifier.
    scored_f1 = [
        m.f1 for m in per_domain.values() if m.is_measurable and m.f1 is not None
    ]
    macro_f1 = sum(scored_f1) / len(scored_f1) if scored_f1 else None

    total_support = sum(m.support for m in per_domain.values())
    weighted_f1 = (
        sum(
            m.f1 * m.support
            for m in per_domain.values()
            if m.is_measurable and m.f1 is not None
        )
        / total_support
        if total_support
        else None
    )

    # Micro-F1 over a single-label problem equals accuracy; computed from
    # the totals rather than asserted, so the report stays honest if the
    # task ever stops being single-label.
    micro_tp = sum(m.true_positives for m in per_domain.values())
    micro_fp = sum(m.false_positives for m in per_domain.values())
    micro_fn = sum(m.false_negatives for m in per_domain.values())
    micro_precision = _safe_divide(micro_tp, micro_tp + micro_fp)
    micro_recall = _safe_divide(micro_tp, micro_tp + micro_fn)
    micro_f1 = (
        2 * micro_precision * micro_recall / (micro_precision + micro_recall)
        if micro_precision and micro_recall and (micro_precision + micro_recall) > 0
        else None
    )

    return ClassificationReport(
        n_documents=scored,
        domains=list(domains),
        per_domain=per_domain,
        confusion=confusion,
        accuracy=_safe_divide(correct, scored),
        macro_f1=macro_f1,
        micro_f1=micro_f1,
        weighted_f1=weighted_f1,
        unlabelled=unlabelled,
    )


def render_confusion_matrix(report: ClassificationReport) -> str:
    """The matrix as fixed-width text: gold down the side, predicted across."""

    if not report.domains:
        return "(no domains)"

    width = max(len(d) for d in report.domains) + 2
    header = "gold \\ predicted".ljust(width) + "".join(
        d[:width - 1].rjust(width) for d in report.domains
    )
    lines = [header, "-" * len(header)]
    for gold_domain, row in report.confusion_rows():
        cells = "".join(str(row.get(p, 0)).rjust(width) for p in report.domains)
        lines.append(gold_domain.ljust(width) + cells)

    if report.unlabelled:
        lines.append(
            f"({len(report.unlabelled)} gold document(s) received no label at all; "
            "counted against recall)"
        )
    return "\n".join(lines)
