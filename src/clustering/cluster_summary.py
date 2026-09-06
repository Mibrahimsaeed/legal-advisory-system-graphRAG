"""Describe and triage what HDBSCAN actually found, without naming it.

Stage 1.2 proper (``domain_discovery_flow``) goes straight from clusters
to LLM-labeled draft domains. Before spending LLM calls -- and before
anyone starts treating cluster 0 as a legal domain -- it is worth looking
at the clustering on its own terms: how big is each cluster, how tightly
does it hold together, which documents sit at its core, and which
clusters look like they are hiding two topics in a trench coat.

That is all this module does. It computes, per cluster:

* size and share of the corpus,
* mean HDBSCAN membership probability (the clusterer's own confidence),
* cohesion -- mean cosine similarity of members to the cluster centroid
  in *embedding* space, not UMAP space, so the number means "how alike
  are these documents" rather than "how alike did the projection make
  them look",
* corpus-relative TF-IDF keywords (no LLM -- see
  :mod:`src.clustering.label_clusters`),
* representative cases with their court/date/citation for eyeballing,
* the court distribution, which is how a "clustered by forum, not by
  subject" failure shows up.

Nothing here names a domain, writes to ``domain_candidates``, or touches
the taxonomy. It is deliberately read-only diagnosis.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field, replace
from typing import Any

import numpy as np

from src.clustering.cluster import NOISE_LABEL
from src.clustering.label_clusters import (
    KeywordCorpus,
    extract_cluster_keywords,
    select_representative_docs,
)
from src.common.logging_utils import get_logger

logger = get_logger(__name__)

# Cluster size classes.
KIND_MAJOR = "major"
KIND_SMALL = "small"
KIND_NOISE = "noise"

# Quality flags -- any flag makes a cluster "suspicious" and worth a look
# before it is allowed to become a domain candidate.
FLAG_LOW_COHESION = "low_cohesion"
FLAG_LOW_MEMBERSHIP_CONFIDENCE = "low_membership_confidence"
FLAG_WEAK_RELATIVE_CONFIDENCE = "weak_relative_confidence"
FLAG_CATCH_ALL_SIZE = "catch_all_size"
FLAG_SINGLE_COURT_DOMINATED = "single_court_dominated"

DEFAULT_MAJOR_MIN_SHARE = 0.05
DEFAULT_MIXED_MAX_MEAN_PROBABILITY = 0.6
DEFAULT_MIXED_MAX_COHESION = 0.35
DEFAULT_MIXED_MAX_SHARE = 0.5
DEFAULT_COURT_DOMINANCE_THRESHOLD = 0.9
DEFAULT_RELATIVE_CONFIDENCE_RATIO = 0.8
DEFAULT_SNIPPET_CHARS = 240
# A median only says something once there are a few clusters to take it over.
_MIN_CLUSTERS_FOR_RELATIVE_FLAG = 3
# Below this many distinct courts corpus-wide, "one court dominates this
# cluster" says nothing about the cluster -- it just describes the corpus.
_MIN_CORPUS_COURTS_FOR_DOMINANCE_FLAG = 3


@dataclass(frozen=True)
class RepresentativeCase:
    """One case picked to stand for its cluster, with enough context to judge it."""

    doc_id: str
    title: str | None = None
    court: str | None = None
    decision_date: str | None = None
    citation: str | None = None
    probability: float | None = None
    source_relpath: str | None = None
    snippet: str = ""


@dataclass(frozen=True)
class ClusterSummary:
    cluster_id: int  # NOISE_LABEL (-1) for the noise bucket
    kind: str  # KIND_MAJOR | KIND_SMALL | KIND_NOISE
    doc_count: int
    share: float
    mean_probability: float | None = None
    cohesion: float | None = None
    keywords: list[str] = field(default_factory=list)
    representative_cases: list[RepresentativeCase] = field(default_factory=list)
    court_distribution: dict[str, int] = field(default_factory=dict)
    earliest_decision_date: str | None = None
    latest_decision_date: str | None = None
    flags: list[str] = field(default_factory=list)
    sample_doc_ids: list[str] = field(default_factory=list)

    @property
    def is_noise(self) -> bool:
        return self.cluster_id == NOISE_LABEL

    @property
    def is_suspicious(self) -> bool:
        return bool(self.flags) and not self.is_noise


@dataclass(frozen=True)
class ClusterReview:
    run_id: str
    total_documents: int
    clustered_documents: int
    noise_documents: int
    n_clusters: int
    summaries: list[ClusterSummary] = field(default_factory=list)
    parameters: dict[str, Any] = field(default_factory=dict)

    @property
    def noise_share(self) -> float:
        return self.noise_documents / self.total_documents if self.total_documents else 0.0

    @property
    def major_clusters(self) -> list[ClusterSummary]:
        return [s for s in self.summaries if s.kind == KIND_MAJOR]

    @property
    def small_clusters(self) -> list[ClusterSummary]:
        return [s for s in self.summaries if s.kind == KIND_SMALL]

    @property
    def noise_summary(self) -> ClusterSummary | None:
        return next((s for s in self.summaries if s.is_noise), None)

    @property
    def suspicious_clusters(self) -> list[ClusterSummary]:
        return [s for s in self.summaries if s.is_suspicious]


def cluster_cohesion(vectors: np.ndarray) -> float | None:
    """Mean cosine similarity of members to their centroid.

    1.0 means every document in the cluster points the same way; values
    near 0 mean the cluster is a bag of unrelated documents that density
    in the reduced space happened to put together. ``None`` when there
    are fewer than two vectors (a cohesion of one point is meaningless).
    """

    if vectors.ndim != 2 or vectors.shape[0] < 2:
        return None

    centroid = vectors.mean(axis=0)
    centroid_norm = np.linalg.norm(centroid)
    if centroid_norm == 0:
        return 0.0
    centroid = centroid / centroid_norm

    norms = np.linalg.norm(vectors, axis=1)
    norms[norms == 0] = 1.0
    similarities = (vectors / norms[:, None]) @ centroid
    return float(similarities.mean())


def _representative_case(
    doc_id: str,
    document: Any,
    probability: float | None,
    snippet_chars: int,
) -> RepresentativeCase:
    body = getattr(document, "body_preview", "") or ""
    return RepresentativeCase(
        doc_id=doc_id,
        title=getattr(document, "title", None),
        court=getattr(document, "court", None),
        decision_date=getattr(document, "decision_date", None),
        citation=getattr(document, "citation", None),
        probability=probability,
        source_relpath=getattr(document, "source_relpath", None),
        snippet=" ".join(body[:snippet_chars].split()),
    )


def _date_range(documents: list[Any]) -> tuple[str | None, str | None]:
    dates = sorted(
        d for d in (getattr(doc, "decision_date", None) for doc in documents) if d
    )
    return (dates[0], dates[-1]) if dates else (None, None)


def _flag_relatively_weak_clusters(
    summaries: list[ClusterSummary], ratio: float
) -> list[ClusterSummary]:
    """Flag clusters whose membership confidence lags well behind their peers.

    The absolute thresholds are corpus-dependent and easy to set too
    loosely: in the first real run of this stage, a cluster that turned
    out to be 20 rent-control cases plus 10 unrelated one-off subjects
    scored a mean membership probability of 0.70 -- comfortably above an
    absolute cut of 0.6, but far below the 0.90-0.99 its sibling clusters
    scored. Relative weakness is what actually exposed it, so it is
    checked as well. If every cluster is equally weak, nothing is flagged
    here (that is a run-wide problem, which the absolute thresholds
    catch).
    """

    scored = [
        s.mean_probability
        for s in summaries
        if not s.is_noise and s.mean_probability is not None
    ]
    if len(scored) < _MIN_CLUSTERS_FOR_RELATIVE_FLAG:
        return summaries

    median = float(np.median(scored))
    if median <= 0:
        return summaries

    cutoff = ratio * median
    return [
        (
            replace(s, flags=[*s.flags, FLAG_WEAK_RELATIVE_CONFIDENCE])
            if (
                not s.is_noise
                and s.mean_probability is not None
                and s.mean_probability < cutoff
                and FLAG_WEAK_RELATIVE_CONFIDENCE not in s.flags
            )
            else s
        )
        for s in summaries
    ]


def summarize_clusters(
    run_id: str,
    doc_ids: list[str],
    labels: np.ndarray,
    probabilities: np.ndarray | None,
    vectors: np.ndarray,
    documents_by_id: dict[str, Any],
    keyword_corpus: KeywordCorpus,
    parameters: dict[str, Any] | None = None,
    representative_docs_per_cluster: int = 8,
    keywords_per_cluster: int = 15,
    max_sample_doc_ids: int = 50,
    major_min_share: float = DEFAULT_MAJOR_MIN_SHARE,
    mixed_max_mean_probability: float = DEFAULT_MIXED_MAX_MEAN_PROBABILITY,
    mixed_max_cohesion: float = DEFAULT_MIXED_MAX_COHESION,
    mixed_max_share: float = DEFAULT_MIXED_MAX_SHARE,
    court_dominance_threshold: float = DEFAULT_COURT_DOMINANCE_THRESHOLD,
    relative_confidence_ratio: float = DEFAULT_RELATIVE_CONFIDENCE_RATIO,
    snippet_chars: int = DEFAULT_SNIPPET_CHARS,
) -> ClusterReview:
    """Summarize one clustering run. Pure function; touches no storage.

    ``doc_ids[i]``, ``labels[i]`` and ``vectors[i]`` must describe the
    same document -- that alignment is the whole basis of the report, so
    a length mismatch raises rather than silently mislabeling cases.
    ``vectors`` may be empty (a resumed run whose embeddings artifact was
    already purged); cohesion is then reported as ``None`` rather than
    guessed at.
    """

    if len(doc_ids) != len(labels):
        raise ValueError(
            f"doc_ids ({len(doc_ids)}) and labels ({len(labels)}) must be the same length"
        )

    total = len(doc_ids)
    have_vectors = vectors.ndim == 2 and vectors.shape[0] == total
    if not have_vectors and total:
        logger.warning(
            "No usable embedding matrix for run %s; cohesion will be omitted", run_id
        )

    index_by_cluster: dict[int, list[int]] = {}
    for i, label in enumerate(labels.tolist()):
        index_by_cluster.setdefault(int(label), []).append(i)

    corpus_courts = {
        getattr(doc, "court", None)
        for doc in documents_by_id.values()
        if getattr(doc, "court", None)
    }

    noise_count = len(index_by_cluster.get(NOISE_LABEL, []))
    n_clusters = len([c for c in index_by_cluster if c != NOISE_LABEL])

    summaries: list[ClusterSummary] = []
    for cluster_id, indices in index_by_cluster.items():
        cluster_doc_ids = [doc_ids[i] for i in indices]
        documents = [documents_by_id[d] for d in cluster_doc_ids if d in documents_by_id]
        share = len(indices) / total if total else 0.0
        is_noise = cluster_id == NOISE_LABEL

        cluster_probabilities = (
            {doc_ids[i]: float(probabilities[i]) for i in indices}
            if probabilities is not None
            else None
        )
        mean_probability = (
            float(np.mean(list(cluster_probabilities.values())))
            if cluster_probabilities
            else None
        )

        cohesion = (
            cluster_cohesion(vectors[indices]) if have_vectors else None
        )

        keywords = extract_cluster_keywords(
            documents, keyword_corpus, top_k=keywords_per_cluster
        )

        representative_ids = select_representative_docs(
            cluster_doc_ids,
            {doc_ids[i]: vectors[i] for i in indices} if have_vectors else {},
            probabilities=cluster_probabilities,
            top_k=representative_docs_per_cluster,
        )
        representative_cases = [
            _representative_case(
                doc_id,
                documents_by_id[doc_id],
                (cluster_probabilities or {}).get(doc_id),
                snippet_chars,
            )
            for doc_id in representative_ids
            if doc_id in documents_by_id
        ]

        courts = Counter(
            getattr(doc, "court", None) or "(unknown)" for doc in documents
        )
        earliest, latest = _date_range(documents)

        # -- triage -----------------------------------------------------
        if is_noise:
            kind = KIND_NOISE
        elif share >= major_min_share:
            kind = KIND_MAJOR
        else:
            kind = KIND_SMALL

        flags: list[str] = []
        if not is_noise:
            if cohesion is not None and cohesion < mixed_max_cohesion:
                flags.append(FLAG_LOW_COHESION)
            if (
                mean_probability is not None
                and mean_probability < mixed_max_mean_probability
            ):
                flags.append(FLAG_LOW_MEMBERSHIP_CONFIDENCE)
            if share > mixed_max_share:
                flags.append(FLAG_CATCH_ALL_SIZE)
            if len(corpus_courts) >= _MIN_CORPUS_COURTS_FOR_DOMINANCE_FLAG and documents:
                dominant = courts.most_common(1)[0]
                if dominant[0] != "(unknown)" and dominant[1] / len(documents) >= court_dominance_threshold:
                    flags.append(FLAG_SINGLE_COURT_DOMINATED)

        summaries.append(
            ClusterSummary(
                cluster_id=cluster_id,
                kind=kind,
                doc_count=len(indices),
                share=share,
                mean_probability=mean_probability,
                cohesion=cohesion,
                keywords=keywords,
                representative_cases=representative_cases,
                court_distribution=dict(courts.most_common()),
                earliest_decision_date=earliest,
                latest_decision_date=latest,
                flags=flags,
                sample_doc_ids=cluster_doc_ids[:max_sample_doc_ids],
            )
        )

    summaries = _flag_relatively_weak_clusters(summaries, relative_confidence_ratio)

    # Largest cluster first; the noise bucket always last, since it is a
    # leftover rather than a finding.
    summaries.sort(key=lambda s: (s.is_noise, -s.doc_count, s.cluster_id))

    return ClusterReview(
        run_id=run_id,
        total_documents=total,
        clustered_documents=total - noise_count,
        noise_documents=noise_count,
        n_clusters=n_clusters,
        summaries=summaries,
        parameters=dict(parameters or {}),
    )
