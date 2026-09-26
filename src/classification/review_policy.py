"""Phase 6: the routing policy -- which verdicts stand, and which a human sees.

The policy in one sentence per band:

* **high** -- the evidence is strong, unanimous enough and broad enough:
  ``auto_accepted``.
* **medium** -- a domain is indicated but something is off (thin coverage,
  a near-miss confidence, signals pointing different ways):
  ``needs_review``. A label is *proposed*, never final.
* **low** -- nothing is indicated well enough to assign: the domain
  becomes the catch-all and the document goes to ``needs_review``.
* **off_domain** -- the evidence positively says "neither", confidently
  and without contradiction: ``dropped_off_domain``.

Two rules that matter more than the arithmetic:

**Nothing is deleted.** ``dropped_off_domain`` is a *status on a row that
stays*, with its text, metadata and provenance intact. It only means the
document is withheld from the downstream corpus by
:func:`src.extraction.representation_store.list_representations`. A drop
is reversible by writing a new status; a delete would not be, which is
why the pipeline has no delete path at all.

**A human decision outranks a rerun.** Re-running the pipeline
recomputes machine verdicts, but a document a person has reviewed is
protected: :func:`is_protected` is consulted before any write, and the
review ledger (``document_review_decisions``) is the authority on what a
person decided. Reruns are also *quiet* -- a write that would change
nothing is skipped rather than re-stamping ``updated_at``, so the row's
history stays meaningful.

This module is pure policy: no I/O, no SQL, no model calls. It is the one
place where thresholds become dispositions, so
:func:`src.classification.domain_decision.decide_domain` delegates here
rather than keeping a second copy of the rules.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone

from src.classification.taxonomy_registry import OTHER_DOMAIN_ID

# Dispositions, drawn from Phase 1's vocabulary (and its CHECK constraint)
# in src/extraction/doc_representation.py.
STATUS_AUTO_ACCEPTED = "auto_accepted"
STATUS_NEEDS_REVIEW = "needs_review"
STATUS_DROPPED_OFF_DOMAIN = "dropped_off_domain"

# Confidence bands, stored on every decision so the policy that produced a
# verdict is legible without re-deriving it.
BAND_HIGH = "high"
BAND_MEDIUM = "medium"
BAND_LOW = "low"
BAND_OFF_DOMAIN = "off_domain"

REVIEW_LOW_CONFIDENCE = "low_confidence"
REVIEW_NARROW_MARGIN = "narrow_margin"
REVIEW_SIGNAL_CONFLICT = "signal_conflict"
REVIEW_NO_EVIDENCE = "no_evidence"
# Signals fired but support no domain -- e.g. a confident "neither".
# Distinct from no_evidence, which means nothing was observable at all.
REVIEW_NO_DOMAIN_SUPPORT = "no_domain_support"
REVIEW_UNCERTAIN = "uncertain_domain"
REVIEW_LOW_COVERAGE = "low_coverage"

DROP_REASON_OFF_DOMAIN = "off_domain"

# Human review outcomes, recorded in the ledger. The *representation* row
# keeps Phase 1's machine vocabulary (a human acceptance lands as
# 'auto_accepted'); the ledger is what records that a person decided it,
# who they were and when. Keeping the two separate means Phase 6 needs no
# change to Phase 1's CHECK constraint -- which SQLite could not ALTER
# anyway -- and still answers "was this reviewed by a human?" exactly.
REVIEW_ACCEPTED = "human_accepted"
REVIEW_CORRECTED = "human_corrected"
REVIEW_REJECTED = "human_rejected"
REVIEW_STILL_UNCERTAIN = "human_uncertain"

HUMAN_DECISIONS = (
    REVIEW_ACCEPTED,
    REVIEW_CORRECTED,
    REVIEW_REJECTED,
    REVIEW_STILL_UNCERTAIN,
)


@dataclass(frozen=True)
class ReviewThresholds:
    """The tunable part of the policy. Defaults mirror ``domain_decision``."""

    min_domain_score: float = 0.35
    min_margin: float = 0.10
    auto_accept_threshold: float = 0.80
    review_threshold: float = 0.50
    off_domain_drop_confidence: float = 0.70
    min_coverage: float = 0.50

    @classmethod
    def from_settings(cls, config) -> "ReviewThresholds":
        """Build from a ``DomainDecisionSettings``-shaped object."""

        return cls(
            min_domain_score=config.min_domain_score,
            min_margin=config.min_margin,
            auto_accept_threshold=config.auto_accept_threshold,
            review_threshold=config.review_threshold,
            off_domain_drop_confidence=config.off_domain_drop_confidence,
            min_coverage=config.min_coverage,
        )


@dataclass(frozen=True)
class ScoreSummary:
    """What the scoring engine observed, reduced to what the policy needs.

    Deliberately not the contributions themselves: the policy should not
    be able to re-weigh evidence, only to route it.
    """

    top_domain: str
    top_score: float
    margin: float
    coverage: float
    any_signal_available: bool
    any_domain_support: bool
    keyword_supports_a_domain: bool
    llm_other_confidence: float
    signals_conflict: bool


@dataclass(frozen=True)
class Disposition:
    """Where one document lands, and why."""

    primary_domain: str
    confidence: float
    status: str
    band: str
    review_reasons: tuple[str, ...] = field(default_factory=tuple)
    drop_reason: str | None = None

    @property
    def is_uncertain(self) -> bool:
        return self.primary_domain == OTHER_DOMAIN_ID

    @property
    def needs_review(self) -> bool:
        return self.status == STATUS_NEEDS_REVIEW

    @property
    def withheld_from_corpus(self) -> bool:
        """True when the document is excluded downstream -- not deleted."""

        return self.status == STATUS_DROPPED_OFF_DOMAIN


def route(summary: ScoreSummary, thresholds: ReviewThresholds) -> Disposition:
    """Turn an observation into a disposition. Pure and total.

    Order matters, and is the same order the rules are documented in:
    assignment first (is any domain indicated at all?), then status (is
    the indication strong and broad enough to stand unreviewed?).
    """

    reasons: list[str] = []

    # -- 1. is a domain indicated at all? -------------------------------
    if not summary.any_domain_support:
        primary, confidence = OTHER_DOMAIN_ID, 0.0
        # "Nothing was observable" and "what was observed points at no
        # domain" are different findings, and a reviewer needs to tell
        # them apart: the first is a gap, the second is an answer.
        reasons.append(
            REVIEW_NO_DOMAIN_SUPPORT
            if summary.any_signal_available
            else REVIEW_NO_EVIDENCE
        )
    elif summary.top_score < thresholds.min_domain_score:
        # Weak support for everything is not support for something.
        primary, confidence = OTHER_DOMAIN_ID, summary.top_score
        reasons.append(REVIEW_UNCERTAIN)
    elif summary.margin < thresholds.min_margin:
        # The signals point in different directions; a human resolves
        # that, not a rounding rule.
        primary, confidence = OTHER_DOMAIN_ID, summary.top_score
        reasons.append(REVIEW_NARROW_MARGIN)
    else:
        primary, confidence = summary.top_domain, summary.top_score

    if summary.signals_conflict:
        reasons.append(REVIEW_SIGNAL_CONFLICT)

    # -- 2. what happens to it? -----------------------------------------
    if primary == OTHER_DOMAIN_ID:
        # A drop needs the deterministic signal to agree by staying
        # silent: the LLM alone must not discard a document whose keyword
        # profiles matched, and a conflicted document is never dropped.
        confidently_neither = (
            summary.llm_other_confidence >= thresholds.off_domain_drop_confidence
            and not summary.keyword_supports_a_domain
            and REVIEW_SIGNAL_CONFLICT not in reasons
        )
        if confidently_neither:
            return Disposition(
                primary_domain=OTHER_DOMAIN_ID,
                confidence=summary.llm_other_confidence,
                status=STATUS_DROPPED_OFF_DOMAIN,
                band=BAND_OFF_DOMAIN,
                review_reasons=tuple(dict.fromkeys(reasons)),
                drop_reason=DROP_REASON_OFF_DOMAIN,
            )
        return Disposition(
            primary_domain=OTHER_DOMAIN_ID,
            confidence=confidence,
            status=STATUS_NEEDS_REVIEW,
            band=BAND_LOW,
            review_reasons=tuple(dict.fromkeys(reasons)),
        )

    if REVIEW_SIGNAL_CONFLICT in reasons:
        band, status = BAND_MEDIUM, STATUS_NEEDS_REVIEW
    elif summary.coverage < thresholds.min_coverage:
        # Unanimity among two weak signals is not the same evidence as
        # unanimity among four, and the difference should stay visible.
        reasons.append(REVIEW_LOW_COVERAGE)
        band, status = BAND_MEDIUM, STATUS_NEEDS_REVIEW
    elif confidence >= thresholds.auto_accept_threshold:
        band, status = BAND_HIGH, STATUS_AUTO_ACCEPTED
    else:
        reasons.append(REVIEW_LOW_CONFIDENCE)
        if confidence < thresholds.review_threshold:
            # Below the review floor too: worth distinguishing in the
            # audit trail from a verdict that merely missed auto-accept.
            reasons.append(REVIEW_UNCERTAIN)
        band, status = BAND_MEDIUM, STATUS_NEEDS_REVIEW

    return Disposition(
        primary_domain=primary,
        confidence=confidence,
        status=status,
        band=band,
        review_reasons=tuple(dict.fromkeys(reasons)),
    )


# Statuses Phase 5 must never write over, regardless of configuration.
#
# `dropped_procedural` is Phase 2's structural verdict: the document was
# found to be a cause list, an office report, an adjournment slip or an
# incomplete scrape. Phase 3 evidence gathered *before* that verdict is
# stale by definition, and using it to write `auto_accepted` would put
# non-judgments into the corpus. This is an invariant rather than a
# setting: `review.protect_statuses` is a deliberate operator choice, but
# reviving a structural drop is never a choice worth offering.
PHASE_5_INELIGIBLE_STATUSES = frozenset({"dropped_procedural"})


def is_eligible_for_decision(current_status: str | None) -> bool:
    """Whether Phase 5 may write a decision onto a document in this state.

    ``None`` means the document is not in ``document_representations`` at
    all -- there is nothing to write to, so it is not eligible either.
    """

    if current_status is None:
        return False
    return current_status not in PHASE_5_INELIGIBLE_STATUSES


def _parse_timestamp(value: str | None) -> datetime | None:
    """Parse a stored timestamp tolerantly.

    Rows written by this codebase carry ``datetime.now(timezone.utc)
    .isoformat()``. A row created by SQLite's ``DEFAULT CURRENT_TIMESTAMP``
    instead carries ``YYYY-MM-DD HH:MM:SS`` with no zone, so both shapes
    are accepted rather than compared as strings -- a lexicographic
    comparison across the two formats would be silently wrong.
    """

    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(value.replace(" ", "T", 1))
    except ValueError:
        return None
    return parsed.replace(tzinfo=timezone.utc) if parsed.tzinfo is None else parsed


def evidence_is_stale(
    signal_created_at: str | None,
    state_updated_at: str | None,
    state_signal_run_id: str | None,
    signal_run_id: str | None,
) -> bool:
    """Whether the evidence predates the document's current state.

    Phase 3 gathers evidence at one moment; Phase 5 may run much later,
    and in between the document's state can move -- Phase 2 re-runs and
    drops it, a reviewer rejects it, a newer decision run re-classifies it.
    Writing an old verdict over a newer state would silently undo whichever
    of those happened.

    The comparison is not simply "is the state newer than the evidence",
    because Phase 5's *own* write makes the state newer than the evidence
    every time -- that alone would make a document undecidable a second
    time and break resuming a run. So a state newer than the evidence is
    stale **unless it was produced from this same evidence**, which
    ``document_classifications.signal_run_id`` records.

    Returns False when either timestamp is unavailable: refusing to write
    on missing metadata would block the ordinary first pass.
    """

    signal_at = _parse_timestamp(signal_created_at)
    state_at = _parse_timestamp(state_updated_at)
    if signal_at is None or state_at is None:
        return False
    if state_at <= signal_at:
        # The state predates the evidence: the evidence is the newer fact.
        return False
    # The state is newer. Only this run's own evidence may have written it.
    return state_signal_run_id != signal_run_id


def is_protected(
    doc_id: str,
    current_status: str | None,
    reviewed_doc_ids: frozenset[str] | set[str] = frozenset(),
    protect_reviewed: bool = True,
    protect_statuses: tuple[str, ...] = (),
) -> bool:
    """Whether a rerun must leave this document's current state alone.

    Two independent reasons to protect:

    * **a person decided it** (``protect_reviewed``) -- the default, and
      the important one. Human review is expensive and authoritative; a
      rerun that silently reverted it would make the review queue a
      treadmill.
    * **its machine status is frozen** (``protect_statuses``) -- opt-in,
      for pinning a slice of the corpus while iterating on thresholds.

    Protection is about the *current state*, never the audit trail: a
    protected document still gets a new ``document_classifications`` row,
    so what the pipeline *would* have said is always recorded even where
    it does not take effect.
    """

    if protect_reviewed and doc_id in reviewed_doc_ids:
        return True
    return current_status is not None and current_status in protect_statuses


def resolve_human_decision(
    decision: str,
    machine_domain: str | None,
    corrected_domain: str | None = None,
) -> tuple[str, str | None, str | None]:
    """Map a human review outcome onto Phase 1's current-state fields.

    Returns ``(classification_status, primary_domain, drop_reason)``.

    The representation row speaks Phase 1's machine vocabulary; the ledger
    records that a human is behind it (see the module docstring). A
    rejection is a *withholding*, not a deletion.
    """

    if decision == REVIEW_ACCEPTED:
        return STATUS_AUTO_ACCEPTED, machine_domain, None
    if decision == REVIEW_CORRECTED:
        if not corrected_domain:
            raise ValueError(
                f"{REVIEW_CORRECTED!r} requires corrected_domain: a correction "
                "that does not say what the right answer is cannot be applied"
            )
        if corrected_domain == OTHER_DOMAIN_ID:
            return STATUS_DROPPED_OFF_DOMAIN, OTHER_DOMAIN_ID, DROP_REASON_OFF_DOMAIN
        return STATUS_AUTO_ACCEPTED, corrected_domain, None
    if decision == REVIEW_REJECTED:
        return STATUS_DROPPED_OFF_DOMAIN, OTHER_DOMAIN_ID, DROP_REASON_OFF_DOMAIN
    if decision == REVIEW_STILL_UNCERTAIN:
        # A reviewer who could not decide leaves it in the queue rather
        # than forcing a label -- the honest outcome, and it keeps the
        # document out of the accepted corpus.
        return STATUS_NEEDS_REVIEW, OTHER_DOMAIN_ID, None
    raise ValueError(
        f"unknown review decision {decision!r}; expected one of {HUMAN_DECISIONS}"
    )
