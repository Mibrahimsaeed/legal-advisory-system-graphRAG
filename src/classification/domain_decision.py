"""Phase 5: weigh the Phase 3 evidence into one broad domain decision.

The decision is **deterministic**. Every input is already stored: the
keyword scores, the cluster id and the LLM's broad reading all come from
``document_domain_signals``, so this module makes no model calls of its
own and re-running it over the same evidence produces byte-identical
verdicts. The one stochastic step in the whole chain stayed in Phase 3,
where it is recorded once and reused.

Four signals, each weighted by configuration:

* **keyword** -- the deterministic family/criminal concept profiles.
* **llm** -- Phase 3's broad reading, contributing its confidence to the
  domain it named and nothing to the others.
* **cluster** -- what the *rest of* a document's cluster looks like. A
  cluster id means nothing by itself, so it is turned into a domain
  signal by aggregating the other members' keyword and LLM evidence
  (leave-one-out, so a document never votes for itself). This is the only
  legitimate way to read a cluster without labels, and it is gated: when
  Phase 4 reports that clustering carries no domain signal, the weight is
  configured to zero and the signal drops out.
* **title** -- the cause title alone, scanned with the same profiles.
  Lightweight by design; "X v. The State" is a real hint, not a finding.

Two things this deliberately does not do:

* **Source-folder labels are never read.** They are Phase 4's validation
  ground truth; using them here would make the evaluation circular and
  the classifier useless on an unlabelled corpus. Nothing in this module
  imports :mod:`src.clustering.cluster_validation`.
* **The old 45%-statute / 25%-cluster / 15%-court scheme is not
  implemented.** There is no statute extraction to weight, court is
  metadata rather than evidence (it predicts forum, not subject), and the
  weights here are a different, smaller set.

Every signal is expressed as a **share of its own evidence** across the
target domains, which is what makes a weighted sum meaningful: the raw
keyword score is the fraction of a 15-concept profile that matched, so
even an unmistakable family-law judgment scores about 0.67 on it, while
the LLM reports a subjective 0.0-1.0 confidence. Summing those two
directly would compare a profile-coverage ratio against a confidence and
put auto-acceptance out of reach for the entire corpus. Shares are
comparable; raw scores are not.

Strength of evidence is then tracked separately as *coverage* -- the
weight that actually fired. A document only two signals could see is not
penalised by silently dragging its score down; it is flagged by an
explicit ``min_coverage`` rule, so "thin evidence" and "conflicting
evidence" stay distinguishable in the audit trail.

Ambiguity resolves to ``other_uncertain``, and a genuine conflict between
signals is routed to human review rather than settled by arithmetic.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping

from src.classification import review_policy as policy
from src.classification.taxonomy_registry import OTHER_DOMAIN_ID, FrozenTaxonomy
from src.common.logging_utils import get_logger

logger = get_logger(__name__)

# Bump when the signals, weights semantics or decision rules change: it is
# stored on every row so verdicts made under different logic are never
# silently compared.
DECISION_VERSION = "domain_decision/1.0"

# Outcome statuses, review reasons and drop reasons all belong to the
# Phase 6 policy (src/classification/review_policy.py), which is the single
# place thresholds become dispositions. They are re-exported here so
# existing importers of this module keep working.
STATUS_AUTO_ACCEPTED = policy.STATUS_AUTO_ACCEPTED
STATUS_NEEDS_REVIEW = policy.STATUS_NEEDS_REVIEW
STATUS_DROPPED_OFF_DOMAIN = policy.STATUS_DROPPED_OFF_DOMAIN

REVIEW_LOW_CONFIDENCE = policy.REVIEW_LOW_CONFIDENCE
REVIEW_NARROW_MARGIN = policy.REVIEW_NARROW_MARGIN
REVIEW_SIGNAL_CONFLICT = policy.REVIEW_SIGNAL_CONFLICT
REVIEW_NO_EVIDENCE = policy.REVIEW_NO_EVIDENCE
REVIEW_NO_DOMAIN_SUPPORT = policy.REVIEW_NO_DOMAIN_SUPPORT
REVIEW_UNCERTAIN = policy.REVIEW_UNCERTAIN
REVIEW_LOW_COVERAGE = policy.REVIEW_LOW_COVERAGE

DROP_REASON_OFF_DOMAIN = policy.DROP_REASON_OFF_DOMAIN


@dataclass(frozen=True)
class SignalContribution:
    """One signal's view of one domain, and what it contributed."""

    signal: str
    domain_scores: dict[str, float] = field(default_factory=dict)
    weight: float = 0.0
    available: bool = True
    detail: str = ""

    def weighted(self, domain: str) -> float:
        return self.weight * self.domain_scores.get(domain, 0.0)


@dataclass(frozen=True)
class DomainDecision:
    """The verdict for one document, with every contribution kept."""

    doc_id: str
    primary_domain: str
    confidence: float
    margin: float
    status: str
    # Which policy band produced that status (see review_policy).
    band: str = policy.BAND_HIGH
    # Weight that actually fired, 0.0-1.0. Confidence is the winner's share
    # of THIS, not of a full slate of signals, so it means the same thing
    # whether four signals were available or two.
    coverage: float = 1.0
    scores: dict[str, float] = field(default_factory=dict)
    contributions: list[SignalContribution] = field(default_factory=list)
    review_reasons: list[str] = field(default_factory=list)
    drop_reason: str | None = None
    cluster_id: int | None = None
    signal_run_id: str | None = None
    decision_version: str = DECISION_VERSION

    @property
    def is_uncertain(self) -> bool:
        return self.primary_domain == OTHER_DOMAIN_ID

    @property
    def needs_review(self) -> bool:
        return self.status == STATUS_NEEDS_REVIEW

    def justification(self) -> str:
        """One auditable sentence per signal, plus the outcome."""

        parts = [
            f"{self.primary_domain} [{self.band}] (confidence {self.confidence:.2f}, "
            f"margin {self.margin:.2f}, coverage {self.coverage:.2f})."
        ]
        for contribution in self.contributions:
            if not contribution.available:
                parts.append(f"{contribution.signal}: unavailable.")
                continue
            best = max(
                contribution.domain_scores.items(), key=lambda kv: kv[1], default=None
            )
            if best is None or best[1] <= 0:
                parts.append(f"{contribution.signal}: no support (w={contribution.weight:.2f}).")
            else:
                parts.append(
                    f"{contribution.signal}: {best[0]} {best[1]:.2f} "
                    f"(w={contribution.weight:.2f})"
                    + (f" -- {contribution.detail}" if contribution.detail else "")
                )
        if self.review_reasons:
            parts.append("review: " + ", ".join(self.review_reasons) + ".")
        return " ".join(parts)

    def evidence(self) -> dict:
        """Structured record of the decision, for storage and QA."""

        return {
            "decision_version": self.decision_version,
            "signal_run_id": self.signal_run_id,
            "scores": {k: round(v, 4) for k, v in self.scores.items()},
            "confidence": round(self.confidence, 4),
            "margin": round(self.margin, 4),
            "coverage": round(self.coverage, 4),
            "cluster_id": self.cluster_id,
            "band": self.band,
            "signals": [
                {
                    "signal": c.signal,
                    "weight": c.weight,
                    "available": c.available,
                    "domain_scores": {k: round(v, 4) for k, v in c.domain_scores.items()},
                    "detail": c.detail,
                }
                for c in self.contributions
            ],
            "review_reasons": self.review_reasons,
        }


def _as_shares(scores: Mapping[str, float], domains: list[str]) -> dict[str, float]:
    """Rescale one signal's raw scores into shares that sum to 1.0.

    This is what puts every signal on a comparable footing (see the module
    docstring): a domain that holds all of a signal's evidence contributes
    1.0 regardless of that signal's native scale, and a signal split
    evenly between domains contributes 0.5 to each -- which is exactly the
    ambiguity the margin rule is there to catch. An all-zero signal stays
    all-zero rather than becoming a uniform vote for everything.
    """

    total = sum(max(scores.get(d, 0.0), 0.0) for d in domains)
    if total <= 0:
        return {d: 0.0 for d in domains}
    return {d: max(scores.get(d, 0.0), 0.0) / total for d in domains}


def _target_domains(taxonomy: FrozenTaxonomy) -> list[str]:
    """The real domains a document can be assigned to, excluding the catch-all."""

    return [d.domain_id for d in taxonomy.domains if not d.is_other]


def build_cluster_profiles(
    signal_rows: Iterable[Mapping[str, Any]],
    domains: list[str],
    min_cluster_members: int = 3,
) -> dict[int, Counter]:
    """Per-cluster tallies of what its members' *other* signals say.

    A cluster with fewer than ``min_cluster_members`` labelled-ish members
    is omitted: a two-document cluster voting on itself is noise dressed
    up as corroboration. Noise (``-1``) is never profiled -- it is not a
    cluster.
    """

    profiles: dict[int, Counter] = {}
    for row in signal_rows:
        cluster_id = row.get("cluster_id")
        if cluster_id is None or cluster_id == -1:
            continue
        tally = profiles.setdefault(int(cluster_id), Counter())
        for domain in (row.get("keyword_top_domain"), row.get("llm_domain")):
            if domain in domains:
                tally[domain] += 1
        tally["_members"] += 1

    return {
        cluster_id: tally
        for cluster_id, tally in profiles.items()
        if tally["_members"] >= min_cluster_members
    }


def _cluster_contribution(
    row: Mapping[str, Any],
    profiles: Mapping[int, Counter],
    domains: list[str],
    weight: float,
) -> SignalContribution:
    """The document's own cluster, scored leave-one-out."""

    cluster_id = row.get("cluster_id")
    if weight <= 0:
        return SignalContribution(
            "cluster", {}, weight, available=False,
            detail="disabled by configuration (see Phase 4 verdict)",
        )
    if cluster_id is None or cluster_id == -1:
        return SignalContribution(
            "cluster", {}, weight, available=False,
            detail="document is cluster noise" if cluster_id == -1 else "no cluster",
        )

    tally = profiles.get(int(cluster_id))
    if tally is None:
        return SignalContribution(
            "cluster", {}, weight, available=False, detail="cluster too small to profile"
        )

    # Remove this document's own votes so it cannot corroborate itself.
    own = Counter()
    for domain in (row.get("keyword_top_domain"), row.get("llm_domain")):
        if domain in domains:
            own[domain] += 1

    votes = {d: max(tally.get(d, 0) - own.get(d, 0), 0) for d in domains}
    total = sum(votes.values())
    if total == 0:
        return SignalContribution(
            "cluster", {}, weight, available=False,
            detail="no domain votes from other cluster members",
        )

    scores = {d: votes[d] / total for d in domains}
    return SignalContribution(
        "cluster", scores, weight,
        detail=f"cluster {cluster_id}, {tally['_members'] - 1} other member(s)",
    )


def _keyword_contribution(
    row: Mapping[str, Any], domains: list[str], weight: float
) -> SignalContribution:
    evidence = row.get("keyword_signals") or {}
    per_domain = evidence.get("domains") or {}
    raw = {d: float(per_domain.get(d, {}).get("score", 0.0) or 0.0) for d in domains}
    if not any(raw.values()):
        return SignalContribution(
            "keyword", {d: 0.0 for d in domains}, weight,
            available=False, detail="no profile matches",
        )

    total_matches = evidence.get("total_matches", 0)
    raw_detail = ", ".join(f"{d}={raw[d]:.2f}" for d in domains)
    return SignalContribution(
        "keyword", _as_shares(raw, domains), weight,
        detail=f"{total_matches} concept match(es); profile coverage {raw_detail}",
    )


def _llm_contribution(
    row: Mapping[str, Any], domains: list[str], weight: float
) -> SignalContribution:
    if row.get("llm_status") != "ok" or not row.get("llm_domain"):
        return SignalContribution(
            "llm", {}, weight, available=False,
            detail=f"assessment {row.get('llm_status') or 'missing'}",
        )

    domain = row["llm_domain"]
    confidence = float(row.get("llm_confidence") or 0.0)
    if domain == OTHER_DOMAIN_ID:
        # A confident "neither" is evidence *against* both domains, which
        # is expressed by contributing nothing to either -- while still
        # being available, so it is not mistaken for a missing signal.
        return SignalContribution(
            "llm", {d: 0.0 for d in domains}, weight,
            detail=f"assessed {OTHER_DOMAIN_ID} at {confidence:.2f}",
        )

    scores = {d: (confidence if d == domain else 0.0) for d in domains}
    return SignalContribution("llm", scores, weight, detail=f"{domain} at {confidence:.2f}")


def _title_contribution(
    title: str | None,
    compiled_profiles: Mapping[str, list],
    domains: list[str],
    weight: float,
    min_matches: int = 1,
) -> SignalContribution:
    """Lightweight metadata signal: the cause title, nothing else."""

    from src.classification.keyword_signals import detect_keyword_signals

    if not title:
        return SignalContribution("title", {}, weight, available=False, detail="no title")

    signals = detect_keyword_signals("title", title, compiled_profiles, min_matches=min_matches)
    raw = {d: signals.score_for(d) for d in domains}
    if not any(raw.values()):
        return SignalContribution(
            "title", {d: 0.0 for d in domains}, weight,
            available=False, detail="no match in title",
        )
    return SignalContribution(
        "title", _as_shares(raw, domains), weight,
        detail=f"title matched {signals.top_domain}",
    )


def decide_domain(
    row: Mapping[str, Any],
    taxonomy: FrozenTaxonomy,
    weights: Mapping[str, float],
    cluster_profiles: Mapping[int, Counter] | None = None,
    compiled_profiles: Mapping[str, list] | None = None,
    title: str | None = None,
    min_domain_score: float = 0.35,
    min_margin: float = 0.10,
    auto_accept_threshold: float = 0.80,
    review_threshold: float = 0.50,
    off_domain_drop_confidence: float = 0.70,
    min_coverage: float = 0.50,
    title_min_matches: int = 1,
    signal_run_id: str | None = None,
) -> DomainDecision:
    """Weigh one document's evidence into a broad domain decision.

    The weighing lives here; the routing lives in
    :func:`src.classification.review_policy.route`, which this delegates
    to. The rules it applies, all configurable, in order:

    1. **No usable evidence** -> ``other_uncertain``, ``needs_review``.
       Never a coin flip.
    2. **Winning confidence below ``min_domain_score``** ->
       ``other_uncertain``. Weak support for everything is not support for
       something.
    3. **Margin below ``min_margin``** -> ``other_uncertain`` and review:
       the signals point in different directions, which a human should
       resolve rather than a rounding rule.
    4. Otherwise the higher-scoring domain wins, and the *status* follows
       the confidence: at/above ``auto_accept_threshold`` accepted,
       otherwise review (additionally flagged uncertain below
       ``review_threshold``).
    5. **Coverage below ``min_coverage``** does two things: it scales the
       scores down in proportion to the shortfall (so one weak signal
       cannot report near-certainty), and it forces review even where the
       verdict still clears the threshold. Unanimity among two weak
       signals is not the same evidence as unanimity among four.
    6. A **confident** ``other_uncertain`` (LLM says neither at/above
       ``off_domain_drop_confidence`` with no keyword support) is dropped
       as off-domain; an unconfident one goes to review.
    7. **Signal conflict** (keyword and LLM name different domains) always
       forces review, whatever the arithmetic says.

    Confidence is a share of the evidence that fired, not a probability:
    0.9 means "nine tenths of what could be observed points here", which
    is a statement about the signals, not about the law.
    """

    domains = _target_domains(taxonomy)
    contributions = [
        _keyword_contribution(row, domains, float(weights.get("keyword", 0.0))),
        _llm_contribution(row, domains, float(weights.get("llm", 0.0))),
        _cluster_contribution(
            row, cluster_profiles or {}, domains, float(weights.get("cluster", 0.0))
        ),
        _title_contribution(
            title, compiled_profiles or {}, domains,
            float(weights.get("title", 0.0)), min_matches=title_min_matches,
        ),
    ]

    # Coverage is the weight that actually fired. Scores are normalised by
    # it so confidence means the same thing however many signals were
    # available -- but only up to the coverage floor: below it the whole
    # score is scaled down in proportion to how far the evidence falls
    # short, so a lone title match cannot report confidence 1.0. At or
    # above the floor the factor is exactly 1.0 and distorts nothing.
    coverage = sum(c.weight for c in contributions if c.available)
    scores = {
        domain: sum(c.weighted(domain) for c in contributions if c.available)
        for domain in domains
    }
    coverage_factor = min(coverage / min_coverage, 1.0) if min_coverage > 0 else 1.0
    if coverage > 0:
        scores = {d: (v / coverage) * coverage_factor for d, v in scores.items()}

    ranked = sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))
    top_domain, top_score = ranked[0] if ranked else (OTHER_DOMAIN_ID, 0.0)
    runner_up = ranked[1][1] if len(ranked) > 1 else 0.0
    margin = top_score - runner_up

    keyword_top = row.get("keyword_top_domain")
    llm_domain = row.get("llm_domain") if row.get("llm_status") == "ok" else None
    conflict = (
        keyword_top in domains
        and llm_domain in domains
        and keyword_top != llm_domain
    )

    usable = [c for c in contributions if c.available and any(c.domain_scores.values())]
    keyword = next(c for c in contributions if c.signal == "keyword")
    llm_other_confidence = (
        float(row.get("llm_confidence") or 0.0)
        if row.get("llm_domain") == OTHER_DOMAIN_ID and row.get("llm_status") == "ok"
        else 0.0
    )

    # Routing is Phase 6's policy, not this module's: everything above
    # measures evidence, and the call below decides what happens to it.
    disposition = policy.route(
        policy.ScoreSummary(
            top_domain=top_domain,
            top_score=top_score,
            margin=margin,
            coverage=coverage,
            any_signal_available=any(c.available for c in contributions),
            any_domain_support=bool(usable),
            keyword_supports_a_domain=any(keyword.domain_scores.values()),
            llm_other_confidence=llm_other_confidence,
            signals_conflict=conflict,
        ),
        policy.ReviewThresholds(
            min_domain_score=min_domain_score,
            min_margin=min_margin,
            auto_accept_threshold=auto_accept_threshold,
            review_threshold=review_threshold,
            off_domain_drop_confidence=off_domain_drop_confidence,
            min_coverage=min_coverage,
        ),
    )

    return DomainDecision(
        doc_id=str(row.get("doc_id", "")),
        primary_domain=disposition.primary_domain,
        confidence=disposition.confidence,
        margin=margin,
        status=disposition.status,
        band=disposition.band,
        coverage=coverage,
        scores=scores,
        contributions=contributions,
        review_reasons=list(disposition.review_reasons),
        drop_reason=disposition.drop_reason,
        cluster_id=row.get("cluster_id"),
        signal_run_id=signal_run_id,
    )
