"""Phase 7: can this corpus be handed to GraphRAG, or not?

One question, four possible answers, and the honest one is available:

* ``ready`` -- the classifier was measured against enough human labels to
  believe it, every target domain scores adequately, the review backlog is
  cleared and each domain has enough accepted documents to be worth
  indexing.
* ``ready_with_reservations`` -- the numbers clear the bar, but something
  qualifies them: a single annotator, a thin validation set, an unfinished
  audit. Usable, with the caveats stated.
* ``not_ready`` -- measured and found wanting. A domain scores too poorly,
  or too little of the corpus survived to index.
* ``not_evaluable`` -- **the most likely answer early on, and not a
  failure**. There is no validation set, or it is too small or too
  one-sided to support a claim. Nothing is wrong with the pipeline; there
  is simply no evidence yet, and reporting "ready" from no evidence would
  be the worst outcome of this whole project.

**Clustering never gates readiness.** Cluster count, noise share, purity,
ARI and NMI are all carried in the report as diagnostics -- they explain
*why* the classifier behaves as it does and whether the cluster signal
earns its weight -- but no threshold on them can change the verdict. The
reason is that they measure structure in an embedding space, not
correctness against law: a corpus can cluster beautifully and be
misclassified, or cluster into mush and be classified correctly by the
keyword and LLM signals. Gating on purity or noise would be gating on a
proxy. :func:`assess_readiness` therefore takes the clustering report for
reporting only, and a test asserts that varying it cannot move the
verdict.

Every threshold here is configurable and none is a standard. They are
stated as what they are -- the point at which *this project* is willing to
call a domain usable -- and every raw measurement travels with the verdict
so a reader can disagree with the thresholds and still use the report.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from src.common.logging_utils import get_logger
from src.evaluation.classification_metrics import ClassificationReport
from src.evaluation.validation_set import SOURCE_REVIEW_LEDGER, ValidationSet

logger = get_logger(__name__)

READINESS_VERSION = "corpus_readiness/1.0"

VERDICT_READY = "ready"
VERDICT_READY_WITH_RESERVATIONS = "ready_with_reservations"
VERDICT_NOT_READY = "not_ready"
VERDICT_NOT_EVALUABLE = "not_evaluable"


@dataclass(frozen=True)
class ReadinessThresholds:
    """Where this project draws its lines. Not standards -- choices."""

    min_validation_documents: int = 100
    min_validation_per_domain: int = 25
    min_domain_f1: float = 0.75
    min_accepted_per_domain: int = 200
    # Below this the validation set is usable but thin enough to qualify
    # the result rather than invalidate it.
    comfortable_validation_documents: int = 200
    min_llm_agreement: float = 0.70

    @classmethod
    def from_settings(cls, config) -> "ReadinessThresholds":
        return cls(
            min_validation_documents=config.min_validation_documents,
            min_validation_per_domain=config.min_validation_per_domain,
            min_domain_f1=config.min_domain_f1,
            min_accepted_per_domain=config.min_accepted_per_domain,
            comfortable_validation_documents=config.comfortable_validation_documents,
            min_llm_agreement=config.min_llm_agreement,
        )


@dataclass(frozen=True)
class ReadinessReport:
    """The verdict, everything it rests on, and everything it ignored."""

    verdict: str
    question: str = (
        "Can we confidently produce a usable Family Law and Criminal Law "
        "corpus for the later GraphRAG phases, given the available data?"
    )
    answer: str = ""
    blockers: list[str] = field(default_factory=list)
    reservations: list[str] = field(default_factory=list)
    accepted_by_domain: dict[str, int] = field(default_factory=dict)
    validation: dict[str, Any] = field(default_factory=dict)
    classification: dict[str, Any] = field(default_factory=dict)
    clustering_diagnostics: dict[str, Any] = field(default_factory=dict)
    llm_diagnostics: dict[str, Any] = field(default_factory=dict)
    audit: dict[str, Any] = field(default_factory=dict)
    thresholds: dict[str, Any] = field(default_factory=dict)
    readiness_version: str = READINESS_VERSION

    @property
    def can_freeze(self) -> bool:
        """Whether a dataset freeze may proceed at all."""

        return self.verdict in (VERDICT_READY, VERDICT_READY_WITH_RESERVATIONS)

    def as_dict(self) -> dict:
        return {
            "readiness_version": self.readiness_version,
            "verdict": self.verdict,
            "question": self.question,
            "answer": self.answer,
            "blockers": self.blockers,
            "reservations": self.reservations,
            "accepted_by_domain": self.accepted_by_domain,
            "validation": self.validation,
            "classification": self.classification,
            "clustering_diagnostics": self.clustering_diagnostics,
            "llm_diagnostics": self.llm_diagnostics,
            "audit": self.audit,
            "thresholds": self.thresholds,
        }


def assess_readiness(
    validation: ValidationSet,
    classification: ClassificationReport,
    accepted_by_domain: dict[str, int],
    domains: list[str],
    thresholds: ReadinessThresholds | None = None,
    audit_plan: Any = None,
    clustering_report: Any = None,
    llm_agreement: float | None = None,
) -> ReadinessReport:
    """Answer the corpus question from the evidence, or decline to.

    ``clustering_report`` is recorded and never consulted: see the module
    docstring on why cluster metrics must not gate acceptance.
    """

    thresholds = thresholds or ReadinessThresholds()
    blockers: list[str] = []
    reservations: list[str] = []

    # -- diagnostics, gathered first so they appear whatever the verdict --
    clustering_diagnostics = _clustering_diagnostics(clustering_report)
    llm_diagnostics = {"agreement_with_human_labels": llm_agreement}
    if llm_agreement is not None and llm_agreement < thresholds.min_llm_agreement:
        # A note, not a blocker: the LLM is one signal of several, and the
        # classifier is judged on its output, not on any one input.
        reservations.append(
            f"LLM agreement with human labels is {llm_agreement:.1%}, below the "
            f"{thresholds.min_llm_agreement:.0%} comfort level; the broad "
            "assessment prompt may need revisiting"
        )

    audit_summary = audit_plan.as_dict() if audit_plan is not None else {}

    # -- 1. is there enough evidence to say anything at all? -------------
    not_evaluable_reasons: list[str] = []
    if len(validation) == 0:
        not_evaluable_reasons.append(
            "no human-reviewed validation set exists; nothing has been measured"
        )
    elif len(validation) < thresholds.min_validation_documents:
        not_evaluable_reasons.append(
            f"the validation set holds {len(validation)} document(s), below the "
            f"{thresholds.min_validation_documents} needed to support a claim "
            "about classifier quality"
        )

    for domain in domains:
        gold = validation.by_domain.get(domain, 0)
        if gold < thresholds.min_validation_per_domain:
            not_evaluable_reasons.append(
                f"{domain} has {gold} reviewed example(s), below the "
                f"{thresholds.min_validation_per_domain} needed to measure it"
            )

    if not_evaluable_reasons:
        answer = (
            "No -- not because the corpus failed, but because it has not been "
            "measured. " + _join(not_evaluable_reasons) + ". Review a validation "
            "set through the Phase 6 queue and re-run this evaluation."
        )
        return _report(
            VERDICT_NOT_EVALUABLE, answer, not_evaluable_reasons, reservations,
            accepted_by_domain, validation, classification, clustering_diagnostics,
            llm_diagnostics, audit_summary, thresholds,
        )

    # -- 2. measured: is it good enough? ---------------------------------
    for domain in domains:
        metrics = classification.per_domain.get(domain)
        if metrics is None or metrics.f1 is None:
            blockers.append(f"{domain} could not be scored")
            continue
        if metrics.f1 < thresholds.min_domain_f1:
            blockers.append(
                f"{domain} scores F1 {metrics.f1:.2f} (precision "
                f"{_fmt(metrics.precision)}, recall {_fmt(metrics.recall)}) over "
                f"{metrics.support} reviewed document(s), below the required "
                f"{thresholds.min_domain_f1:.2f}"
            )

    for domain in domains:
        accepted = accepted_by_domain.get(domain, 0)
        if accepted < thresholds.min_accepted_per_domain:
            blockers.append(
                f"{domain} has only {accepted} accepted document(s), below the "
                f"{thresholds.min_accepted_per_domain} worth indexing"
            )

    # -- 3. qualifications that do not block -----------------------------
    if validation.is_single_reviewer:
        reservations.append(
            "every gold label came from a single reviewer, so no "
            "inter-annotator agreement could be measured; the scores below "
            "reflect consistency with one person's judgement"
        )
    if len(validation) < thresholds.comfortable_validation_documents:
        reservations.append(
            f"the validation set ({len(validation)} documents) is above the "
            "minimum but thin; per-domain scores carry wide uncertainty"
        )
    if validation.source != SOURCE_REVIEW_LEDGER:
        reservations.append(
            f"gold labels came from {validation.source!r} rather than the review "
            "ledger, so reviewer identity and the machine context are not recorded"
        )
    if validation.unresolved:
        reservations.append(
            f"{len(validation.unresolved)} reviewed document(s) were left "
            "undecided by the reviewer and contribute no label"
        )
    if audit_plan is not None and not audit_plan.review_backlog_cleared:
        census = audit_plan.strata.get("needs_review")
        reservations.append(
            f"{census.outstanding if census else 'some'} needs_review document(s) "
            "have not been through a human yet; the audit obligation is to clear "
            "all of them"
        )
    if audit_plan is not None and audit_plan.total_outstanding:
        reservations.append(
            f"{audit_plan.total_outstanding} document(s) across the audit strata "
            "remain unsampled or unreviewed"
        )
    weakest = classification.weakest_domain()
    if weakest is not None and weakest.recall is not None and weakest.recall < 0.8:
        reservations.append(
            f"{weakest.domain} recall is {weakest.recall:.1%}: the corpus for it "
            "will be incomplete even where what it contains is correct"
        )

    if blockers:
        answer = (
            "No, not yet. " + _join(blockers) + ". The pipeline works and the "
            "evidence is measurable; the corpus is not yet good enough to index."
        )
        return _report(
            VERDICT_NOT_READY, answer, blockers, reservations, accepted_by_domain,
            validation, classification, clustering_diagnostics, llm_diagnostics,
            audit_summary, thresholds,
        )

    total = sum(accepted_by_domain.get(d, 0) for d in domains)
    if reservations:
        answer = (
            f"Yes, with reservations. {total} accepted document(s) across "
            f"{len(domains)} domain(s) meet the quality bar, but: "
            + _join(reservations) + "."
        )
        return _report(
            VERDICT_READY_WITH_RESERVATIONS, answer, blockers, reservations,
            accepted_by_domain, validation, classification, clustering_diagnostics,
            llm_diagnostics, audit_summary, thresholds,
        )

    answer = (
        f"Yes. {total} accepted document(s) across {len(domains)} domain(s), each "
        f"scoring at or above F1 {thresholds.min_domain_f1:.2f} against "
        f"{len(validation)} human-reviewed labels, with the review backlog "
        "cleared. The corpus is ready for GraphRAG ingestion."
    )
    return _report(
        VERDICT_READY, answer, blockers, reservations, accepted_by_domain,
        validation, classification, clustering_diagnostics, llm_diagnostics,
        audit_summary, thresholds,
    )


def _clustering_diagnostics(clustering_report: Any) -> dict:
    """Pull the clustering numbers out for reporting. Never for gating."""

    if clustering_report is None:
        return {}
    return {
        "n_clusters": clustering_report.n_clusters,
        "noise_documents": clustering_report.noise_documents,
        "noise_share": clustering_report.noise_share,
        "coverage": clustering_report.coverage,
        "cluster_sizes": [
            {"cluster_id": c.cluster_id, "size": c.size, "purity": c.purity,
             "dominant_label": c.dominant_label}
            for c in clustering_report.per_cluster
        ],
        "weighted_purity": clustering_report.weighted_purity,
        "purity_lift": clustering_report.purity_lift,
        "adjusted_rand_index": clustering_report.adjusted_rand_index,
        "normalized_mutual_info": clustering_report.normalized_mutual_info,
        "per_label_recall": clustering_report.per_label_recall,
        "contingency": clustering_report.contingency,
        "verdict": clustering_report.verdict,
        "note": (
            "Diagnostic only. Cluster quality does not gate corpus acceptance: "
            "it measures structure in an embedding space, not correctness "
            "against law."
        ),
    }


def _report(
    verdict, answer, blockers, reservations, accepted_by_domain, validation,
    classification, clustering_diagnostics, llm_diagnostics, audit, thresholds,
) -> ReadinessReport:
    report = ReadinessReport(
        verdict=verdict,
        answer=answer,
        blockers=list(blockers),
        reservations=list(reservations),
        accepted_by_domain=dict(accepted_by_domain),
        validation=validation.as_dict(),
        classification=classification.as_dict(),
        clustering_diagnostics=clustering_diagnostics,
        llm_diagnostics=llm_diagnostics,
        audit=audit,
        thresholds=vars(thresholds).copy(),
    )
    logger.info("Corpus readiness: %s -- %s", verdict, answer)
    return report


def _join(items: list[str]) -> str:
    if not items:
        return ""
    if len(items) == 1:
        return items[0]
    return "; ".join(items[:-1]) + f"; and {items[-1]}"


def _fmt(value: float | None) -> str:
    return f"{value:.2f}" if value is not None else "n/a"
