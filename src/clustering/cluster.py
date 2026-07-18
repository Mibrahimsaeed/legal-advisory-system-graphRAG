"""HDBSCAN clustering of (optionally UMAP-reduced) document embeddings.

HDBSCAN is used instead of K-Means because the number of legal domains
present in the corpus is unknown ahead of time, and HDBSCAN's noise label
(``-1``) gives an explicit "doesn't clearly belong anywhere" signal for
free -- which is exactly what later becomes the "Other / Uncertain"
bucket (Stage 1.2 spec, point 8), rather than every document being
forced into some cluster whether it actually fits or not.

Clustering is exposed as a :data:`Clusterer` callable
(``vectors -> ClusterResult``) rather than a hard dependency on the
``hdbscan`` package throughout the rest of the pipeline -- orchestration
code takes a ``Clusterer`` as a parameter, defaulting to
:func:`hdbscan_clusterer`. This is what lets ranking/labeling/orchestration
logic be unit-tested with a deterministic fake clusterer in environments
without ``hdbscan`` installed, while production always uses the real thing.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np

from src.common.exceptions import ClusteringError
from src.common.logging_utils import get_logger

logger = get_logger(__name__)

try:  # pragma: no cover
    import hdbscan as _hdbscan
except ImportError:  # pragma: no cover
    _hdbscan = None  # type: ignore[assignment]

NOISE_LABEL = -1


@dataclass(frozen=True)
class ClusterResult:
    labels: np.ndarray  # shape (n,); NOISE_LABEL (-1) marks noise/unclustered
    probabilities: np.ndarray | None  # per-point membership strength, or None
    n_clusters: int  # count of non-noise clusters found


class Clusterer(Protocol):
    def __call__(self, vectors: np.ndarray) -> ClusterResult: ...


def hdbscan_clusterer(
    min_cluster_size: int = 15,
    min_samples: int | None = None,
    metric: str = "euclidean",
) -> Clusterer:
    """Factory returning a :data:`Clusterer` bound to these HDBSCAN hyperparameters.

    Raises:
        ClusteringError: immediately, if ``hdbscan`` isn't installed --
            unlike the PDF-extraction backends in Stage 1, there is no
            silent lesser fallback here; HDBSCAN is what the spec calls
            for, so a missing dependency should fail loudly rather than
            produce clustering results from something else pretending
            to be HDBSCAN.
    """

    if _hdbscan is None:
        raise ClusteringError(
            "hdbscan is not installed; cannot cluster (see requirements.txt)",
            phase="cluster",
        )

    def _run(vectors: np.ndarray) -> ClusterResult:
        if vectors.shape[0] == 0:
            return ClusterResult(
                labels=np.array([], dtype=int), probabilities=None, n_clusters=0
            )

        try:
            clusterer = _hdbscan.HDBSCAN(
                min_cluster_size=min_cluster_size,
                min_samples=min_samples,
                metric=metric,
            )
            labels = clusterer.fit_predict(vectors)
            probabilities = getattr(clusterer, "probabilities_", None)
        except Exception as exc:
            raise ClusteringError(
                "HDBSCAN clustering failed", phase="cluster", cause=exc
            ) from exc

        labels = np.asarray(labels, dtype=int)
        n_clusters = len(set(labels.tolist()) - {NOISE_LABEL})
        noise_count = int((labels == NOISE_LABEL).sum())

        logger.info(
            "HDBSCAN clustered %d doc(s) into %d cluster(s), %d marked noise",
            len(labels),
            n_clusters,
            noise_count,
        )
        return ClusterResult(
            labels=labels, probabilities=probabilities, n_clusters=n_clusters
        )

    return _run


def available() -> bool:
    """Whether the real HDBSCAN backend can be used in this environment."""

    return _hdbscan is not None