"""Decide which clusters may become draft domains, and label the ones that may.

Phase 4 produced clusters and a quality read on each
(:mod:`src.clustering.cluster_summary`). This module turns that into a
*draft* taxonomy, and the interesting part is what it refuses to do:

* **A cluster is not a domain just because it exists.** A cluster has to
  clear both an absolute floor (``min_domain_docs``) and a share floor
  (``min_domain_share``) before it is eligible. Handful-of-documents
  clusters are how a taxonomy quietly acquires a "domain" that is really
  one unusual case and its neighbours.
* **A flagged cluster is drafted, but never as a clean domain.** The
  Phase 4 review flags clusters that look mixed. Dropping them would lose
  a real domain (the run that motivated this had a rent-control cluster
  contaminated with ten unrelated one-off cases); promoting them silently
  would launder the contamination into the taxonomy. They are drafted
  with ``confidence='low'``, ``review_required=True`` and the flags
  attached.
* **Everything not promoted is accounted for**, in the "Other /
  Uncertain" bucket and in a :class:`~src.clustering.taxonomy_card.TaxonomyAudit`
  that records, per cluster, why it went where it went.

Labeling itself reuses the existing mechanism unchanged --
:func:`src.clustering.label_clusters.generate_domain_draft`, one LLM call
per eligible cluster, degrading to an "(unlabeled)" draft rather than
failing the run.

Nothing here freezes anything: every row is ``status='draft'``, and
``config/domains.yaml`` (the frozen registry) is never written by this
module or anything it calls.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from src.clustering.cluster import NOISE_LABEL
from src.clustering.cluster_summary import ClusterReview, ClusterSummary
from src.clustering.label_clusters import (
    KeywordCorpus,
    extract_cluster_keywords,
    generate_domain_draft,
    select_representative_docs,
)
from src.clustering.taxonomy_card import (
    CONFIDENCE_HIGH,
    CONFIDENCE_LOW,
    ClusterDecision,
    DomainDraft,
    TaxonomyAudit,
    build_other_bucket_draft,
)
from src.common.llm_client import LLMClient
from src.common.logging_utils import get_logger

logger = get_logger(__name__)

OUTCOME_PROMOTED = "promoted"
OUTCOME_PROMOTED_LOW_CONFIDENCE = "promoted_low_confidence"
OUTCOME_UNMAPPED = "unmapped"

REASON_NOISE = "hdbscan_noise"
REASON_BELOW_MIN_DOCS = "below_min_domain_docs"
REASON_BELOW_MIN_SHARE = "below_min_domain_share"
REASON_OUTSIDE_TOP_N = "outside_top_n"
REASON_FLAGGED = "flagged_by_cluster_review"
REASON_UNLABELED = "llm_labeling_failed"

DEFAULT_MIN_DOMAIN_DOCS = 15
DEFAULT_MIN_DOMAIN_SHARE = 0.02
DEFAULT_MAX_DOMAINS = 15
# HDBSCAN membership probability below which a document's place in its
# own cluster is weak enough to be worth counting as "uncertain".
DEFAULT_UNCERTAIN_MEMBERSHIP_PROBABILITY = 0.5

_SLUG_STRIP_RE = re.compile(r"[^a-z0-9]+")
_UNLABELED_MARKER = "(unlabeled)"


def slugify_domain_name(name: str, fallback: str = "domain") -> str:
    """``"Criminal Appeals & Bail"`` -> ``"criminal_appeals_bail"``."""

    slug = _SLUG_STRIP_RE.sub("_", (name or "").lower()).strip("_")
    return slug or fallback


def assign_domain_id(name: str, cluster_id: int, taken: set[str]) -> str:
    """A stable, unique slug for one domain within a taxonomy card.

    Collisions (two clusters the LLM named the same thing -- which is
    itself a signal they should probably be merged) are disambiguated by
    cluster id rather than a counter, so the id says which cluster it
    came from.
    """

    slug = slugify_domain_name(name, fallback=f"cluster_{cluster_id}")
    if slug in taken:
        slug = f"{slug}_c{cluster_id}"
    taken.add(slug)
    return slug


@dataclass
class DraftTaxonomyResult:
    domains: list[DomainDraft] = field(default_factory=list)
    other_bucket: DomainDraft | None = None
    audit: TaxonomyAudit = field(default_factory=TaxonomyAudit)


def assess_clusters(
    review: ClusterReview,
    min_domain_docs: int = DEFAULT_MIN_DOMAIN_DOCS,
    min_domain_share: float = DEFAULT_MIN_DOMAIN_SHARE,
    max_domains: int = DEFAULT_MAX_DOMAINS,
) -> list[ClusterDecision]:
    """Decide, per cluster, whether it may become a draft domain.

    Ordering is by size, so ``max_domains`` keeps the largest. Every
    cluster gets a decision -- including the ones that are rejected, and
    including noise -- because "which clusters did not make it, and why"
    is the part a reviewer needs and the part that is easiest to lose.
    """

    decisions: list[ClusterDecision] = []
    eligible_count = 0

    for summary in sorted(
        review.summaries, key=lambda s: (s.is_noise, -s.doc_count, s.cluster_id)
    ):
        reasons: list[str] = []

        if summary.is_noise:
            decisions.append(
                ClusterDecision(
                    cluster_id=summary.cluster_id,
                    doc_count=summary.doc_count,
                    share=summary.share,
                    outcome=OUTCOME_UNMAPPED,
                    reasons=[REASON_NOISE],
                    flags=list(summary.flags),
                    keywords=summary.keywords[:10],
                )
            )
            continue

        if summary.doc_count < min_domain_docs:
            reasons.append(f"{REASON_BELOW_MIN_DOCS}:{min_domain_docs}")
        if summary.share < min_domain_share:
            reasons.append(f"{REASON_BELOW_MIN_SHARE}:{min_domain_share}")
        if not reasons and eligible_count >= max_domains:
            reasons.append(f"{REASON_OUTSIDE_TOP_N}:{max_domains}")

        if reasons:
            outcome = OUTCOME_UNMAPPED
        else:
            eligible_count += 1
            outcome = (
                OUTCOME_PROMOTED_LOW_CONFIDENCE if summary.flags else OUTCOME_PROMOTED
            )
            if summary.flags:
                reasons.append(f"{REASON_FLAGGED}:{','.join(summary.flags)}")

        decisions.append(
            ClusterDecision(
                cluster_id=summary.cluster_id,
                doc_count=summary.doc_count,
                share=summary.share,
                outcome=outcome,
                reasons=reasons,
                flags=list(summary.flags),
                keywords=summary.keywords[:10],
            )
        )

    return decisions


def _summary_by_cluster(review: ClusterReview) -> dict[int, ClusterSummary]:
    return {s.cluster_id: s for s in review.summaries}


def _uncertain_membership_count(
    doc_ids_by_cluster: dict[int, list[str]],
    probabilities_by_doc: dict[str, float] | None,
    threshold: float,
) -> int:
    """Count every clustered document sitting weakly inside its own cluster.

    A case that is 'mostly rent control, partly civil procedure' does not
    get its own cluster -- it lands in one of them with a low membership
    probability. That makes this the closest thing the clustering gives us
    to a count of mixed-domain cases, so it is counted over the whole
    sample (not just representatives) and reported, never silently
    resolved.
    """

    if not probabilities_by_doc:
        return 0

    return sum(
        1
        for cluster_id, doc_ids in doc_ids_by_cluster.items()
        if cluster_id != NOISE_LABEL
        for doc_id in doc_ids
        if probabilities_by_doc.get(doc_id, 1.0) < threshold
    )


def build_draft_taxonomy(
    review: ClusterReview,
    doc_ids_by_cluster: dict[int, list[str]],
    documents_by_id: dict[str, Any],
    keyword_corpus: KeywordCorpus,
    llm_client: LLMClient,
    vectors_by_doc: dict[str, Any] | None = None,
    probabilities_by_doc: dict[str, float] | None = None,
    min_domain_docs: int = DEFAULT_MIN_DOMAIN_DOCS,
    min_domain_share: float = DEFAULT_MIN_DOMAIN_SHARE,
    max_domains: int = DEFAULT_MAX_DOMAINS,
    representative_docs_per_cluster: int = 8,
    keywords_per_cluster: int = 15,
    uncertain_membership_probability: float = DEFAULT_UNCERTAIN_MEMBERSHIP_PROBABILITY,
) -> DraftTaxonomyResult:
    """Label the eligible clusters and assemble the draft taxonomy + audit.

    One LLM call per eligible cluster, via the existing
    :func:`~src.clustering.label_clusters.generate_domain_draft`. A
    cluster the LLM fails to name is demoted to unmapped rather than
    shipped as a domain called "Cluster 4 (unlabeled)".
    """

    decisions = assess_clusters(
        review,
        min_domain_docs=min_domain_docs,
        min_domain_share=min_domain_share,
        max_domains=max_domains,
    )
    summaries = _summary_by_cluster(review)

    domains: list[DomainDraft] = []
    taken_ids: set[str] = set()
    unmapped_doc_ids: list[str] = []
    sample_size = review.total_documents

    for decision in decisions:
        cluster_doc_ids = doc_ids_by_cluster.get(decision.cluster_id, [])

        if decision.outcome == OUTCOME_UNMAPPED:
            unmapped_doc_ids.extend(cluster_doc_ids)
            continue

        cluster_documents = [
            documents_by_id[d] for d in cluster_doc_ids if d in documents_by_id
        ]
        keywords = extract_cluster_keywords(
            cluster_documents, keyword_corpus, top_k=keywords_per_cluster
        )
        cluster_probabilities = (
            {d: probabilities_by_doc.get(d, 0.0) for d in cluster_doc_ids}
            if probabilities_by_doc
            else None
        )
        representative_ids = select_representative_docs(
            cluster_doc_ids,
            vectors_by_doc or {},
            probabilities=cluster_probabilities,
            top_k=representative_docs_per_cluster,
        )

        draft = generate_domain_draft(
            decision.cluster_id,
            cluster_documents,
            representative_ids,
            keywords,
            sample_size=sample_size,
            llm_client=llm_client,
        )

        if draft.error or _UNLABELED_MARKER in draft.name:
            # No name, no domain. The documents fall through to
            # Other/Uncertain and the cluster is recorded as unmapped, so
            # a failed LLM call can never masquerade as a legal domain.
            decision.outcome = OUTCOME_UNMAPPED
            decision.reasons.append(f"{REASON_UNLABELED}:{draft.error or draft.name}")
            unmapped_doc_ids.extend(cluster_doc_ids)
            logger.warning(
                "Cluster %s could not be labeled; routed to Other/Uncertain",
                decision.cluster_id,
            )
            continue

        summary = summaries.get(decision.cluster_id)
        low_confidence = decision.outcome == OUTCOME_PROMOTED_LOW_CONFIDENCE

        draft.domain_id = assign_domain_id(draft.name, decision.cluster_id, taken_ids)
        draft.confidence = CONFIDENCE_LOW if low_confidence else CONFIDENCE_HIGH
        draft.review_required = low_confidence
        draft.flags = list(decision.flags)
        if low_confidence:
            note = (
                "Cluster review flagged this cluster as possibly mixed: "
                f"{', '.join(decision.flags)}."
            )
            if summary is not None:
                if summary.cohesion is not None:
                    note += f" cohesion={summary.cohesion:.2f}."
                if summary.mean_probability is not None:
                    note += f" mean membership probability={summary.mean_probability:.2f}."
            draft.notes.append(note)
            draft.notes.append(
                "Do not accept this domain without reading the cluster's "
                "membership: its boundary is the least trustworthy in this draft."
            )
        decision.domain_id = draft.domain_id
        domains.append(draft)

    domains.sort(key=lambda d: -d.doc_count)

    noise_docs = doc_ids_by_cluster.get(NOISE_LABEL, [])
    other_bucket = build_other_bucket_draft(unmapped_doc_ids, sample_size=sample_size)

    audit = TaxonomyAudit(
        decisions=decisions,
        unmapped_cluster_ids=[
            d.cluster_id
            for d in decisions
            if d.outcome == OUTCOME_UNMAPPED and d.cluster_id != NOISE_LABEL
        ],
        unmapped_doc_count=len(unmapped_doc_ids),
        noise_doc_count=len(noise_docs),
        low_confidence_domain_ids=[
            d.domain_id for d in domains if d.confidence == CONFIDENCE_LOW
        ],
        uncertain_membership_doc_count=_uncertain_membership_count(
            doc_ids_by_cluster,
            probabilities_by_doc,
            uncertain_membership_probability,
        ),
        notes=[
            "status=draft: nothing here is frozen. config/domains.yaml is "
            "untouched and promoting any of these domains is a human step.",
            f"Eligibility gates: min_domain_docs={min_domain_docs}, "
            f"min_domain_share={min_domain_share}, max_domains={max_domains}.",
        ],
    )

    return DraftTaxonomyResult(
        domains=domains, other_bucket=other_bucket, audit=audit
    )
