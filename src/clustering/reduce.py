"""Dimensionality reduction (UMAP) ahead of HDBSCAN clustering.

HDBSCAN on very high-dimensional sentence embeddings (768+ dims) tends to
suffer from the curse of dimensionality: density estimates get noisy and
points start looking roughly equidistant. UMAP is the standard fix --
project down to a lower-dimensional space that preserves local
neighborhood structure before clustering.

Reduction is skipped (input returned unchanged) when:

* there are too few sampled documents for a UMAP neighborhood graph to
  be meaningful (``discovery.umap_min_docs`` -- "if appropriate" per the
  Stage 1.2 spec), or
* the embedding is already at or below the target dimensionality.

When ``umap-learn`` isn't installed, a numpy-only PCA (via SVD) fallback
is used instead so the pipeline still runs end-to-end. This is logged
loudly: PCA is a strictly weaker, linear-only substitute for UMAP's
non-linear manifold projection and should not be relied on for a real
discovery run.
"""

from __future__ import annotations

import numpy as np

from src.common.exceptions import ClusteringError
from src.common.logging_utils import get_logger

logger = get_logger(__name__)

try:  # pragma: no cover
    import umap
except ImportError:  # pragma: no cover
    umap = None  # type: ignore[assignment]


def _pca_fallback(vectors: np.ndarray, n_components: int) -> np.ndarray:
    """Numpy-only PCA via SVD -- a linear, dependency-free UMAP substitute."""

    n_components = min(n_components, vectors.shape[0] - 1, vectors.shape[1])
    if n_components <= 0:
        return vectors

    centered = vectors - vectors.mean(axis=0, keepdims=True)
    u, s, _vt = np.linalg.svd(centered, full_matrices=False)
    projected = u[:, :n_components] * s[:n_components]
    return projected.astype(np.float32)


def reduce_dimensions(
    vectors: np.ndarray,
    n_components: int = 50,
    n_neighbors: int = 15,
    min_dist: float = 0.0,
    metric: str = "cosine",
    min_docs: int = 50,
    random_state: int = 42,
) -> np.ndarray:
    """Project ``vectors`` (shape ``[n_docs, dim]``) down to ``n_components``.

    Raises:
        ClusteringError: if UMAP is installed but fitting raises (a
            genuine failure, as opposed to "not installed", which
            triggers the PCA fallback instead).
    """

    n_docs = vectors.shape[0]
    if n_docs == 0:
        return vectors

    if n_docs < min_docs:
        logger.info(
            "Skipping dimensionality reduction: only %d sampled doc(s) (< min_docs=%d)",
            n_docs,
            min_docs,
        )
        return vectors

    if vectors.shape[1] <= n_components:
        logger.info(
            "Skipping dimensionality reduction: embedding dim (%d) <= "
            "target n_components (%d)",
            vectors.shape[1],
            n_components,
        )
        return vectors

    effective_n_neighbors = max(2, min(n_neighbors, n_docs - 1))

    if umap is not None:
        try:
            reducer = umap.UMAP(
                n_components=n_components,
                n_neighbors=effective_n_neighbors,
                min_dist=min_dist,
                metric=metric,
                random_state=random_state,
            )
            reduced = reducer.fit_transform(vectors)
            import matplotlib.pyplot as plt

            plt.figure(figsize=(8, 6))

            plt.scatter(
            reduced[:, 0],
            reduced[:, 1],
            s=4,
            alpha=0.7
            )

            plt.title("UMAP Projection")
            plt.xlabel("UMAP-1")
            plt.ylabel("UMAP-2")
            plt.tight_layout()

            plt.savefig("var/umap_projection.png", dpi=300)

            plt.close()
            
        except Exception as exc:
            raise ClusteringError(
                "UMAP dimensionality reduction failed", phase="reduce", cause=exc
            ) from exc

        logger.info(
            "UMAP reduced %d doc(s) from dim=%d to dim=%d",
            n_docs,
            vectors.shape[1],
            n_components,
        )
        return np.asarray(reduced, dtype=np.float32)

    logger.warning(
        "umap-learn not installed; falling back to a numpy PCA projection "
        "(a weaker, linear-only substitute). Install umap-learn (see "
        "requirements.txt) before running Stage 1.2 for real."
    )
    return _pca_fallback(vectors, n_components)