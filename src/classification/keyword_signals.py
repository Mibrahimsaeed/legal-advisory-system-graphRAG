"""Broad keyword/concept signals for the target domains.

Phase 3 collects *evidence*, not verdicts. This module contributes the
cheap deterministic half: for each case, how strongly its text matches the
configured Family Law and Criminal Law concept profiles.

Deliberately coarse. The profiles are high-level concepts ("dower",
"bail", "custody of the minor"), not an exhaustive statute index -- Phase
3 must not turn into section-level extraction, and a broad profile
generalises across the drafting styles of different courts far better than
a long list of exact citations would.

Three properties that matter downstream:

* **Configurable.** Profiles live in ``config/base.yaml`` under
  ``domain_signals.profiles`` and are validated against the frozen
  taxonomy, so a profile can never name a domain the taxonomy does not
  define. Tuning them needs no code change.
* **Deterministic.** Same text, same counts, always.
* **Non-committal.** A score is returned for every domain, plus the margin
  between the best two. Nothing here decides a domain: a clear winner and
  a dead heat are both reported faithfully and left for Phase 4.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Iterable, Mapping

from src.classification.taxonomy_registry import OTHER_DOMAIN_ID, FrozenTaxonomy
from src.common.exceptions import ConfigurationError
from src.common.logging_utils import get_logger

logger = get_logger(__name__)

# A concept must appear at least this many times in total before the
# keyword signal is considered present at all. One passing mention of
# "bail" in a family judgment should not register as criminal evidence.
DEFAULT_MIN_MATCHES = 2


@dataclass(frozen=True)
class DomainKeywordScore:
    """How strongly one domain's profile matched a case."""

    domain_id: str
    score: float
    total_matches: int
    matched_terms: dict[str, int] = field(default_factory=dict)

    @property
    def distinct_terms(self) -> int:
        return len(self.matched_terms)


@dataclass(frozen=True)
class KeywordSignals:
    """The keyword evidence for one case, across every profiled domain."""

    doc_id: str
    scores: dict[str, DomainKeywordScore] = field(default_factory=dict)
    top_domain: str | None = None
    margin: float = 0.0
    total_matches: int = 0

    def score_for(self, domain_id: str) -> float:
        entry = self.scores.get(domain_id)
        return entry.score if entry else 0.0

    def as_evidence(self) -> dict:
        """A JSON-serialisable record of the evidence, for storage."""

        return {
            "top_domain": self.top_domain,
            "margin": round(self.margin, 4),
            "total_matches": self.total_matches,
            "domains": {
                domain_id: {
                    "score": round(entry.score, 4),
                    "total_matches": entry.total_matches,
                    "distinct_terms": entry.distinct_terms,
                    # Bounded: the strongest handful, so the record stays
                    # readable and the row stays small on a 10k corpus.
                    "matched_terms": dict(
                        sorted(entry.matched_terms.items(), key=lambda kv: -kv[1])[:10]
                    ),
                }
                for domain_id, entry in self.scores.items()
            },
        }


def _compile_term(term: str) -> re.Pattern[str]:
    """Whole-word, case-insensitive matcher for one concept.

    Multi-word concepts ("family court", "penal code") are matched as
    phrases with flexible whitespace, so a line break inside the phrase
    still counts.
    """

    parts = [re.escape(word) for word in term.split()]
    return re.compile(r"\b" + r"\s+".join(parts) + r"\b", re.IGNORECASE)


def compile_profiles(
    profiles: Mapping[str, Iterable[str]],
    taxonomy: FrozenTaxonomy | None = None,
) -> dict[str, list[tuple[str, re.Pattern[str]]]]:
    """Compile ``{domain_id: [term, ...]}`` into matchers.

    Raises:
        ConfigurationError: if a profile names a domain the frozen
            taxonomy does not define. A profile for a non-existent domain
            would generate evidence nothing could ever act on.
    """

    compiled: dict[str, list[tuple[str, re.Pattern[str]]]] = {}
    for domain_id, terms in profiles.items():
        if taxonomy is not None and domain_id not in taxonomy.assignable_ids:
            raise ConfigurationError(
                f"domain_signals profile names unknown domain {domain_id!r}; "
                f"the frozen taxonomy defines {sorted(taxonomy.assignable_ids)}"
            )
        if domain_id == OTHER_DOMAIN_ID:
            # The catch-all has no positive evidence by definition: it is
            # where a case lands when nothing else matches.
            continue
        cleaned = [t.strip() for t in terms if t and t.strip()]
        compiled[domain_id] = [(term, _compile_term(term)) for term in cleaned]
    return compiled


def detect_keyword_signals(
    doc_id: str,
    text: str,
    compiled_profiles: Mapping[str, list[tuple[str, re.Pattern[str]]]],
    min_matches: int = DEFAULT_MIN_MATCHES,
) -> KeywordSignals:
    """Score one case's text against every compiled profile.

    The score is normalised by profile size, so a domain with a longer
    keyword list does not win merely by having more chances to match:

        score = (distinct terms matched / terms in profile)
                * (1 + log-ish weight of total matches)

    is deliberately *not* used -- the simpler ``distinct/total`` pair below
    keeps the number interpretable, which matters more than a tuned
    formula for evidence a human will read.
    """

    text = text or ""
    scores: dict[str, DomainKeywordScore] = {}

    for domain_id, terms in compiled_profiles.items():
        matched: dict[str, int] = {}
        total = 0
        for term, pattern in terms:
            count = len(pattern.findall(text))
            if count:
                matched[term] = count
                total += count

        # Distinct concepts carry the signal; raw repetition only breaks
        # ties. A judgment citing "custody" ten times is not ten times more
        # a family case than one citing custody, dower and khula once each.
        distinct_share = len(matched) / len(terms) if terms else 0.0
        score = distinct_share if total >= min_matches else 0.0

        scores[domain_id] = DomainKeywordScore(
            domain_id=domain_id,
            score=score,
            total_matches=total,
            matched_terms=matched,
        )

    ranked = sorted(scores.values(), key=lambda s: (-s.score, s.domain_id))
    top = ranked[0] if ranked and ranked[0].score > 0 else None
    runner_up = ranked[1].score if len(ranked) > 1 else 0.0

    return KeywordSignals(
        doc_id=doc_id,
        scores=scores,
        top_domain=top.domain_id if top else None,
        margin=(top.score - runner_up) if top else 0.0,
        total_matches=sum(s.total_matches for s in scores.values()),
    )
