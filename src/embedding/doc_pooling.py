"""Pool a document signature's embeddable text into a single vector.

A :class:`~src.extraction.signature.DocumentSignature` carries three kinds
of embeddable text -- title, TOC/headings, and a (possibly long) body
preview -- and a long body preview is itself split into chunks before
embedding, since a single ``encode()`` call on a 20,000-character blob
isn't what sentence-embedding models are tuned for. That means every
signature can produce *several* vectors. This module is where "if
multiple vectors exist per signature, pool them into a single document
embedding" (Stage 1.2 spec, point 3) happens: title and TOC are
up-weighted relative to body chunks by default, since they carry a
denser domain signal per character (see ``pdf_extractor.py``'s own
docstring on why front matter matters more than body prose for
document-level classification).

To keep embedding calls cheap, :func:`embed_signatures` flattens every
input across the whole batch into a single ``encode()`` call rather than
one call per document.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.embedding.embed_model import EmbeddingModel
from src.extraction.signature import DocumentSignature

DEFAULT_TITLE_WEIGHT = 2.0
DEFAULT_TOC_WEIGHT = 1.5
DEFAULT_BODY_WEIGHT = 1.0
DEFAULT_BODY_CHUNK_CHARS = 2_000
DEFAULT_MAX_BODY_CHUNKS = 4


@dataclass(frozen=True)
class EmbeddingInput:
    text: str
    weight: float
    kind: str  # "title" | "toc" | "body"


def build_embedding_inputs(
    signature: DocumentSignature,
    title_weight: float = DEFAULT_TITLE_WEIGHT,
    toc_weight: float = DEFAULT_TOC_WEIGHT,
    body_weight: float = DEFAULT_BODY_WEIGHT,
    body_chunk_chars: int = DEFAULT_BODY_CHUNK_CHARS,
    max_body_chunks: int = DEFAULT_MAX_BODY_CHUNKS,
) -> list[EmbeddingInput]:
    """Break one signature into the (text, weight, kind) pieces to embed."""

    inputs: list[EmbeddingInput] = []

    if signature.title:
        inputs.append(EmbeddingInput(signature.title, title_weight, "title"))

    if signature.toc:
        toc_text = "; ".join(signature.toc)
        inputs.append(EmbeddingInput(toc_text, toc_weight, "toc"))

    body = (signature.body_preview or "").strip()
    body_budget = body_chunk_chars * max_body_chunks
    for start in range(0, min(len(body), body_budget), body_chunk_chars):
        chunk = body[start : start + body_chunk_chars].strip()
        if chunk:
            inputs.append(EmbeddingInput(chunk, body_weight, "body"))

    return inputs


def pool_embeddings(vectors: np.ndarray, weights: list[float]) -> np.ndarray:
    """Weighted-mean pool a set of vectors into one, then L2-normalize.

    Falls back to an unweighted mean if every weight is zero (shouldn't
    happen with the defaults, but avoids a divide-by-zero if a caller
    passes all-zero weights).
    """

    if vectors.shape[0] == 0:
        dim = vectors.shape[1] if vectors.ndim == 2 else 0
        return np.zeros((dim,), dtype=np.float32)

    w = np.asarray(weights, dtype=np.float32).reshape(-1, 1)
    if w.sum() <= 0:
        w = np.ones_like(w)

    pooled = (vectors * w).sum(axis=0) / w.sum()
    norm = np.linalg.norm(pooled)
    return (pooled / norm) if norm > 0 else pooled


def embed_signatures(
    signatures: list[DocumentSignature],
    embedder: EmbeddingModel,
    title_weight: float = DEFAULT_TITLE_WEIGHT,
    toc_weight: float = DEFAULT_TOC_WEIGHT,
    body_weight: float = DEFAULT_BODY_WEIGHT,
    body_chunk_chars: int = DEFAULT_BODY_CHUNK_CHARS,
    max_body_chunks: int = DEFAULT_MAX_BODY_CHUNKS,
) -> dict[str, np.ndarray]:
    """Embed and pool a batch of signatures with one flattened ``encode()`` call.

    Signatures with no embeddable text at all (no title, no TOC, no body
    preview -- shouldn't happen for a ``extraction_status != "failed"``
    signature, but guarded anyway) are simply omitted from the result
    rather than producing a meaningless zero vector that would otherwise
    cluster together as a false "domain".

    Returns:
        ``{doc_id: pooled_vector}`` for every signature that had at
        least one embeddable input.
    """

    per_doc_inputs: dict[str, list[EmbeddingInput]] = {}
    flattened_texts: list[str] = []
    flattened_owner: list[str] = []

    for sig in signatures:
        inputs = build_embedding_inputs(
            sig,
            title_weight=title_weight,
            toc_weight=toc_weight,
            body_weight=body_weight,
            body_chunk_chars=body_chunk_chars,
            max_body_chunks=max_body_chunks,
        )
        if not inputs:
            continue
        per_doc_inputs[sig.doc_id] = inputs
        for item in inputs:
            flattened_texts.append(item.text)
            flattened_owner.append(sig.doc_id)

    if not flattened_texts:
        return {}

    all_vectors = embedder.encode(flattened_texts)

    result: dict[str, np.ndarray] = {}
    cursor = 0
    for doc_id, inputs in per_doc_inputs.items():
        n = len(inputs)
        doc_vectors = all_vectors[cursor : cursor + n]
        doc_weights = [item.weight for item in inputs]
        result[doc_id] = pool_embeddings(doc_vectors, doc_weights)
        cursor += n

    return result
