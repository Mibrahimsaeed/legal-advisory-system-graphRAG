"""Embedding model wrapper for Stage 1.2 clustering-only embeddings.

These embeddings exist purely to feed UMAP/HDBSCAN for a one-time,
sample-based domain-discovery pass. They are explicitly NOT the final
RAG embeddings -- ``src/embedding/vector_store_client.py`` and any
chunk-level embedding pipeline remain untouched by this stage.

``get_embedder`` returns a real ``sentence-transformers`` model when the
package is installed, and a deterministic, dependency-free stand-in
otherwise. The fallback exists so the rest of the pipeline (pooling,
reduction, clustering, ranking, labeling, checkpointing) can be
exercised end-to-end in environments without the ML stack installed
(e.g. CI, this sandbox) -- it is explicitly *not* semantically
meaningful and must never be used for a real discovery run.
"""

from __future__ import annotations

import hashlib
from typing import Protocol, runtime_checkable

import numpy as np

from src.common.exceptions import EmbeddingError
from src.common.logging_utils import get_logger

logger = get_logger(__name__)

try:  # pragma: no cover
    from sentence_transformers import SentenceTransformer
except ImportError:  # pragma: no cover
    SentenceTransformer = None  # type: ignore[assignment]


@runtime_checkable
class EmbeddingModel(Protocol):
    """Minimal interface every embedder (real or fake) must satisfy."""

    dimension: int

    def encode(self, texts: list[str]) -> np.ndarray: ...


class SentenceTransformerEmbedder:
    """Wraps a ``sentence-transformers`` model.

    Defaults to a strong general-purpose model
    (``discovery.embedding_model_name`` in config); a legal-tuned
    sentence-embedding checkpoint can be swapped in via config without
    any code change, since nothing downstream assumes a specific model
    or dimensionality.
    """

    def __init__(
        self,
        model_name: str,
        batch_size: int = 32,
        device: str | None = None,
    ) -> None:
        if SentenceTransformer is None:
            raise EmbeddingError(
                "sentence-transformers is not installed; cannot load embedding model",
                phase="embed",
            )

        self.model_name = model_name
        self.batch_size = batch_size

        try:
            self._model = SentenceTransformer(model_name, device=device)
        except Exception as exc:
            raise EmbeddingError(
                f"Failed to load embedding model {model_name!r}",
                phase="embed",
                cause=exc,
            ) from exc

        self.dimension = self._model.get_sentence_embedding_dimension()

    def encode(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dimension), dtype=np.float32)
        try:
            vectors = self._model.encode(
                texts,
                batch_size=self.batch_size,
                show_progress_bar=False,
                convert_to_numpy=True,
                normalize_embeddings=True,
            )
        except Exception as exc:
            raise EmbeddingError(
                "Embedding encode() failed", phase="embed", cause=exc
            ) from exc
        return np.asarray(vectors, dtype=np.float32)


class DeterministicHashEmbedder:
    """Dependency-free embedding stand-in. NOT semantically meaningful.

    Hashes each text into a fixed, content-seeded pseudo-random unit
    vector: identical text always maps to the identical vector, but
    there is no notion of semantic similarity between different texts.
    Clustering on top of this is meaningless -- it exists only so
    pooling/reduction/clustering/ranking/labeling/orchestration logic can
    be unit-tested and run end-to-end without ``sentence-transformers``
    (and its ``torch`` dependency) installed.
    """

    def __init__(self, dimension: int = 384) -> None:
        self.dimension = dimension

    def _vector_for(self, text: str) -> np.ndarray:
        digest = hashlib.sha256(text.encode("utf-8")).digest()
        seed = int.from_bytes(digest[:8], "big", signed=False)
        rng = np.random.default_rng(seed)
        vec = rng.normal(size=self.dimension).astype(np.float32)
        norm = np.linalg.norm(vec)
        return vec / norm if norm > 0 else vec

    def encode(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dimension), dtype=np.float32)
        return np.stack([self._vector_for(t) for t in texts])


def get_embedder(
    model_name: str,
    batch_size: int = 32,
    device: str | None = None,
) -> EmbeddingModel:
    """Real embedder if ``sentence-transformers`` is installed, else the
    deterministic fallback (with a loud warning -- this should only ever
    happen in a dependency-light dev/test environment, never in production).
    """

    if SentenceTransformer is not None:
        return SentenceTransformerEmbedder(model_name, batch_size=batch_size, device=device)

    logger.warning(
        "sentence-transformers not installed; falling back to "
        "DeterministicHashEmbedder. Clustering results from this fallback "
        "are NOT meaningful -- install sentence-transformers (see "
        "requirements.txt) before running Stage 1.2 for real."
    )
    return DeterministicHashEmbedder()