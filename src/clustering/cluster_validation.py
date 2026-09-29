"""Phase 4: does HDBSCAN actually carry a domain signal?

The clustering is **blind**: :data:`src.clustering.cluster.Clusterer` is
``(vectors) -> ClusterResult`` and structurally cannot receive a label, so
no domain or source-folder information can reach the algorithm even by
accident. Labels enter only here, *after* the fact, purely to score what
the unsupervised run produced.

The question this module answers is narrow and empirical: given the
clusters, would knowing a document's cluster tell you anything useful
about its domain? Three things follow from taking that question
seriously:

* **No arbitrary pass marks.** There is no "90% purity" or "3-8% noise"
  rule here. Purity is reported against the only baseline that makes it
  meaningful -- the majority-label share, i.e. what you would score by
  ignoring the clusters entirely and guessing the commonest domain. The
  *lift* over that baseline is the measurement that matters; raw purity
  on an unbalanced corpus is close to meaningless.
* **Standard agreement metrics too.** Adjusted Rand and normalised mutual
  information are reported because they need no cutoff to interpret: ~0
  means the clustering and the domains are independent, regardless of
  corpus balance.
* **A verdict that can be "no".** :func:`assess_usefulness` may return
  ``not_useful``, and its thresholds are configuration, not law. Every
  raw number is in the report so a human can reach a different
  conclusion. Nothing in this module wires a cluster into a classifier.

Noise (``-1``) is excluded from purity and agreement -- it is not a
cluster and scoring it as one would flatter or punish the run depending
on corpus balance -- and reported separately as ``noise_share`` and
``coverage``.
"""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping

import numpy as np

from src.clustering.cluster import NOISE_LABEL
from src.common.logging_utils import get_logger

logger = get_logger(__name__)

try:  # pragma: no cover - optional, declared in requirements.txt
    from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
except ImportError:  # pragma: no cover
    adjusted_rand_score = None  # type: ignore[assignment]
    normalized_mutual_info_score = None  # type: ignore[assignment]

VERDICT_USEFUL = "useful"
VERDICT_WEAK = "weak"
VERDICT_NOT_USEFUL = "not_useful"
VERDICT_NOT_EVALUABLE = "not_evaluable"

# Tunable defaults, deliberately *not* presented as standards. They decide
# only how the report labels itself; every underlying number is reported
# regardless so the call can be overridden.
DEFAULT_MIN_COVERAGE = 0.50
DEFAULT_MIN_PURITY_LIFT = 0.15
DEFAULT_MIN_ARI = 0.10
DEFAULT_MIN_LABEL_RECALL = 0.50


@dataclass(frozen=True)
class ClusterLabelStats:
    """What one cluster turned out to contain, label-wise."""

    cluster_id: int
    size: int
    share: float
    label_counts: dict[str, int] = field(default_factory=dict)
    dominant_label: str | None = None
    purity: float = 0.0
    margin: float = 0.0  # dominant share minus runner-up share
    mean_probability: float | None = None

    @property
    def is_noise(self) -> bool:
        return self.cluster_id == NOISE_LABEL

    @property
    def is_mixed(self) -> bool:
        """More than one label present in meaningful proportion."""

        return len(self.label_counts) > 1 and self.purity < 1.0


@dataclass(frozen=True)
class ValidationReport:
    """The full, self-describing answer to "is this clustering useful?"."""

    run_id: str
    n_documents: int
    n_clusters: int
    clustered_documents: int
    noise_documents: int
    labels_present: list[str] = field(default_factory=list)
    label_distribution: dict[str, int] = field(default_factory=dict)
    baseline_majority_share: float = 0.0
    weighted_purity: float = 0.0
    purity_lift: float = 0.0
    adjusted_rand_index: float | None = None
    normalized_mutual_info: float | None = None
    per_cluster: list[ClusterLabelStats] = field(default_factory=list)
    per_label_recall: dict[str, float] = field(default_factory=dict)
    contingency: dict[str, dict[str, int]] = field(default_factory=dict)
    verdict: str = VERDICT_NOT_EVALUABLE
    verdict_reasons: list[str] = field(default_factory=list)
    parameters: dict[str, Any] = field(default_factory=dict)
    label_source: str = "source_folder"
    notes: list[str] = field(default_factory=list)

    @property
    def noise_share(self) -> float:
        return self.noise_documents / self.n_documents if self.n_documents else 0.0

    @property
    def coverage(self) -> float:
        """Share of documents that landed in a real cluster."""

        return self.clustered_documents / self.n_documents if self.n_documents else 0.0

    @property
    def mixed_clusters(self) -> list[ClusterLabelStats]:
        return [c for c in self.per_cluster if not c.is_noise and c.is_mixed]

    @property
    def cluster_is_a_useful_signal(self) -> bool:
        """Whether Phase 5+ may treat cluster membership as domain evidence.

        ``weak`` deliberately counts as False: a marginal signal that gets
        wired into a classifier is worse than no signal, because its
        contribution is invisible once blended with the others.
        """

        return self.verdict == VERDICT_USEFUL


def labels_from_source_folder(
    representations: Iterable[Any], depth: int = 1
) -> dict[str, str]:
    """Ground-truth labels taken from each case's source folder path.

    Used **only** for evaluation, never for clustering. ``depth`` is how
    many leading path segments form the label, so ``family/case_001``
    labels as ``family`` at depth 1 and a corpus organised as
    ``criminal/narcotics/case_001`` can label at depth 2 if wanted.

    Documents whose ``source_relpath`` is missing or too shallow are
    omitted: an unlabelled document must not be silently counted as a
    label of its own.
    """

    labels: dict[str, str] = {}
    for representation in representations:
        relpath = getattr(representation, "source_relpath", None) or ""
        segments = [s for s in relpath.split("/") if s]
        if len(segments) <= depth:
            # The last segment is the case folder itself, so a relpath with
            # no parent above `depth` carries no label.
            continue
        labels[representation.doc_id] = "/".join(segments[:depth])
    return labels


def _labels_are_degenerate(labels: Mapping[str, str]) -> str | None:
    """Why this label set cannot support evaluation, or ``None`` if it can."""

    if not labels:
        return "no source-folder labels available"
    distinct = set(labels.values())
    if len(distinct) < 2:
        return f"only one distinct label present ({distinct or 'none'})"
    if len(distinct) == len(labels):
        return "every document has its own label; nothing to measure against"
    return None


def _agreement_metrics(
    cluster_ids: list[int], truth: list[str]
) -> tuple[float | None, float | None]:
    """Adjusted Rand and NMI over the clustered (non-noise) documents."""

    if adjusted_rand_score is None or normalized_mutual_info_score is None:
        logger.warning("scikit-learn unavailable; skipping ARI/NMI")
        return None, None
    if len(set(cluster_ids)) < 2 or len(set(truth)) < 2:
        return None, None
    return (
        float(adjusted_rand_score(truth, cluster_ids)),
        float(normalized_mutual_info_score(truth, cluster_ids)),
    )


def assess_usefulness(
    report_values: Mapping[str, Any],
    min_coverage: float = DEFAULT_MIN_COVERAGE,
    min_purity_lift: float = DEFAULT_MIN_PURITY_LIFT,
    min_ari: float = DEFAULT_MIN_ARI,
    min_label_recall: float = DEFAULT_MIN_LABEL_RECALL,
) -> tuple[str, list[str]]:
    """Judge the clustering, and say exactly why.

    Four questions, each answered from a measurement rather than a rule of
    thumb:

    1. Are there at least two clusters? One cluster cannot discriminate.
    2. Does enough of the corpus land in a cluster (``coverage``)? A
       signal that exists for 5% of documents is not a corpus signal.
    3. Does cluster membership beat guessing the commonest domain
       (``purity_lift``)? This is the question raw purity cannot answer.
    4. Do the clusters and the domains agree beyond chance (``ARI``)?

    All four must hold for ``useful``. Meeting the agreement or lift test
    but failing coverage (or vice versa) is ``weak``: real but partial.
    """

    reasons: list[str] = []
    n_clusters = report_values["n_clusters"]
    coverage = report_values["coverage"]
    purity_lift = report_values["purity_lift"]
    ari = report_values.get("adjusted_rand_index")
    recalls = report_values.get("per_label_recall", {}) or {}

    if n_clusters < 2:
        reasons.append(f"only {n_clusters} cluster(s) found; cannot discriminate domains")
        return VERDICT_NOT_USEFUL, reasons

    passes_coverage = coverage >= min_coverage
    passes_lift = purity_lift >= min_purity_lift
    passes_ari = ari is None or ari >= min_ari
    weakest_recall = min(recalls.values()) if recalls else 0.0
    passes_recall = bool(recalls) and weakest_recall >= min_label_recall

    reasons.append(
        f"coverage {coverage:.2f} vs min {min_coverage:.2f}"
        f" ({'ok' if passes_coverage else 'below'})"
    )
    reasons.append(
        f"purity {report_values['weighted_purity']:.2f} vs majority baseline "
        f"{report_values['baseline_majority_share']:.2f} -> lift {purity_lift:+.2f}"
        f" vs min {min_purity_lift:.2f} ({'ok' if passes_lift else 'below'})"
    )
    if ari is not None:
        reasons.append(
            f"adjusted Rand {ari:.2f} vs min {min_ari:.2f}"
            f" ({'ok' if passes_ari else 'below'})"
        )
    if recalls:
        worst = min(recalls, key=lambda k: recalls[k])
        reasons.append(
            f"weakest per-label recall {weakest_recall:.2f} ({worst}) vs min "
            f"{min_label_recall:.2f} ({'ok' if passes_recall else 'below'})"
        )

    if passes_coverage and passes_lift and passes_ari and passes_recall:
        return VERDICT_USEFUL, reasons
    if passes_lift or passes_ari:
        reasons.append(
            "signal is real but partial -- treat cluster membership as "
            "corroboration only, not as a weighted classifier input"
        )
        return VERDICT_WEAK, reasons

    reasons.append(
        "cluster membership does not predict domain better than the "
        "majority baseline; do NOT feed it to the classifier"
    )
    return VERDICT_NOT_USEFUL, reasons


def evaluate_clustering(
    run_id: str,
    doc_ids: list[str],
    labels: np.ndarray,
    ground_truth: Mapping[str, str],
    probabilities: np.ndarray | None = None,
    parameters: Mapping[str, Any] | None = None,
    label_source: str = "source_folder",
    min_coverage: float = DEFAULT_MIN_COVERAGE,
    min_purity_lift: float = DEFAULT_MIN_PURITY_LIFT,
    min_ari: float = DEFAULT_MIN_ARI,
    min_label_recall: float = DEFAULT_MIN_LABEL_RECALL,
) -> ValidationReport:
    """Score a blind clustering against after-the-fact labels.

    ``doc_ids[i]`` must describe ``labels[i]``; a mismatch raises rather
    than producing a quietly wrong report. Documents with no ground-truth
    label are excluded from every label-based measurement and counted in
    ``notes``.
    """

    if len(doc_ids) != len(labels):
        raise ValueError(
            f"doc_ids ({len(doc_ids)}) and labels ({len(labels)}) must be the same length"
        )

    parameters = dict(parameters or {})
    notes: list[str] = []

    label_by_doc = {d: ground_truth[d] for d in doc_ids if d in ground_truth}
    unlabelled = len(doc_ids) - len(label_by_doc)
    if unlabelled:
        notes.append(f"{unlabelled} document(s) had no source-folder label and were excluded")

    degenerate = _labels_are_degenerate(label_by_doc)

    cluster_of = {d: int(l) for d, l in zip(doc_ids, labels.tolist())}
    probability_of = (
        {d: float(p) for d, p in zip(doc_ids, probabilities.tolist())}
        if probabilities is not None
        else {}
    )

    noise_documents = sum(1 for c in cluster_of.values() if c == NOISE_LABEL)
    real_clusters = sorted({c for c in cluster_of.values() if c != NOISE_LABEL})
    clustered_documents = len(doc_ids) - noise_documents

    if degenerate is not None:
        logger.warning("Cluster validation for run %s is not evaluable: %s", run_id, degenerate)
        return ValidationReport(
            run_id=run_id,
            n_documents=len(doc_ids),
            n_clusters=len(real_clusters),
            clustered_documents=clustered_documents,
            noise_documents=noise_documents,
            verdict=VERDICT_NOT_EVALUABLE,
            verdict_reasons=[degenerate],
            parameters=parameters,
            label_source=label_source,
            notes=notes,
        )

    label_distribution = dict(Counter(label_by_doc.values()).most_common())
    baseline_majority_share = (
        max(label_distribution.values()) / sum(label_distribution.values())
        if label_distribution
        else 0.0
    )

    # -- per cluster ---------------------------------------------------
    per_cluster: list[ClusterLabelStats] = []
    contingency: dict[str, dict[str, int]] = {}
    labelled_in_clusters = 0
    correct_in_clusters = 0

    for cluster_id in real_clusters + ([NOISE_LABEL] if noise_documents else []):
        members = [d for d, c in cluster_of.items() if c == cluster_id]
        member_labels = [label_by_doc[d] for d in members if d in label_by_doc]
        counts = dict(Counter(member_labels).most_common())
        ranked = sorted(counts.values(), reverse=True)
        dominant_label = next(iter(counts), None)
        purity = (ranked[0] / len(member_labels)) if member_labels else 0.0
        runner_up = (ranked[1] / len(member_labels)) if len(ranked) > 1 else 0.0
        probs = [probability_of[d] for d in members if d in probability_of]

        stats = ClusterLabelStats(
            cluster_id=cluster_id,
            size=len(members),
            share=len(members) / len(doc_ids) if doc_ids else 0.0,
            label_counts=counts,
            dominant_label=dominant_label,
            purity=purity,
            margin=purity - runner_up,
            mean_probability=(float(np.mean(probs)) if probs else None),
        )
        per_cluster.append(stats)
        contingency[str(cluster_id)] = counts

        if cluster_id != NOISE_LABEL and member_labels:
            labelled_in_clusters += len(member_labels)
            correct_in_clusters += ranked[0]

    per_cluster.sort(key=lambda c: (c.is_noise, -c.size, c.cluster_id))

    weighted_purity = (
        correct_in_clusters / labelled_in_clusters if labelled_in_clusters else 0.0
    )

    # -- per label: how much of each domain reached a cluster that is
    #    dominated by that domain (i.e. is the domain recoverable?) -----
    dominant_by_cluster = {
        c.cluster_id: c.dominant_label for c in per_cluster if not c.is_noise
    }
    per_label_recall: dict[str, float] = {}
    for label in label_distribution:
        total = label_distribution[label]
        recovered = sum(
            1
            for doc_id, doc_label in label_by_doc.items()
            if doc_label == label
            and dominant_by_cluster.get(cluster_of.get(doc_id, NOISE_LABEL)) == label
        )
        per_label_recall[label] = recovered / total if total else 0.0

    # -- agreement over clustered documents only -----------------------
    clustered_pairs = [
        (cluster_of[d], label_by_doc[d])
        for d in doc_ids
        if d in label_by_doc and cluster_of[d] != NOISE_LABEL
    ]
    ari, nmi = _agreement_metrics(
        [c for c, _ in clustered_pairs], [t for _, t in clustered_pairs]
    )
    if noise_documents:
        notes.append(
            f"{noise_documents} noise document(s) excluded from purity and agreement; "
            "see noise_share and coverage"
        )

    values = {
        "n_clusters": len(real_clusters),
        "coverage": clustered_documents / len(doc_ids) if doc_ids else 0.0,
        "weighted_purity": weighted_purity,
        "baseline_majority_share": baseline_majority_share,
        "purity_lift": weighted_purity - baseline_majority_share,
        "adjusted_rand_index": ari,
        "per_label_recall": per_label_recall,
    }
    verdict, reasons = assess_usefulness(
        values,
        min_coverage=min_coverage,
        min_purity_lift=min_purity_lift,
        min_ari=min_ari,
        min_label_recall=min_label_recall,
    )

    return ValidationReport(
        run_id=run_id,
        n_documents=len(doc_ids),
        n_clusters=len(real_clusters),
        clustered_documents=clustered_documents,
        noise_documents=noise_documents,
        labels_present=sorted(label_distribution),
        label_distribution=label_distribution,
        baseline_majority_share=baseline_majority_share,
        weighted_purity=weighted_purity,
        purity_lift=values["purity_lift"],
        adjusted_rand_index=ari,
        normalized_mutual_info=nmi,
        per_cluster=per_cluster,
        per_label_recall=per_label_recall,
        contingency=contingency,
        verdict=verdict,
        verdict_reasons=reasons,
        parameters=parameters,
        label_source=label_source,
        notes=notes,
    )
