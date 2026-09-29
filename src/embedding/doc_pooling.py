"""Pool a document's embeddable text into a single vector.

An :class:`~src.extraction.doc_representation.EmbeddableDocument` carries
three kinds of embeddable text -- title, headings, and a (possibly long)
body preview -- and a long body preview is itself split into chunks
before embedding, since a single ``encode()`` call on a 20,000-character
blob isn't what sentence-embedding models are tuned for. That means every
document can produce *several* vectors. This module is where "if
multiple vectors exist per document, pool them into a single document
embedding" (Stage 1.2 spec, point 3) happens: title and headings are
up-weighted relative to body chunks by default, since they carry a
denser domain signal per character (see ``pdf_extractor.py``'s own
docstring on why front matter matters more than body prose for
document-level classification).

These embeddings are for domain discovery/classification only -- they are
not the RAG embeddings (see ``src/embedding/embed_model.py``).

Which fields feed the vector (Phase 3 determination)
----------------------------------------------------

Embedded: ``title`` (the cause title -- "X v. The State" or "X v.
Commissioner Inland Revenue" is often the single most domain-predictive
string in a judgment), ``headings``, and ``body_preview`` chunks.

Deliberately **not** embedded, though
:class:`~src.extraction.doc_representation.DocumentRepresentation` now
carries them:

* ``citation``, ``case_number``, ``decision_date`` -- identifiers. "2019
  SCMR 123" carries no subject-matter semantics; embedding it adds noise
  and pulls documents together by reporter/volume.
* ``judges`` -- person names. They would cluster cases by bench
  composition, which is not a legal domain.
* ``court`` -- a confounder. Forum correlates with domain only partly
  (a service tribunal hears service matters, but a High Court hears
  everything), and including it risks the discovered taxonomy coming
  back as "Supreme Court cases" instead of "criminal appeals".

Those fields stay on the record, indexed in SQL, for filtering,
stratification and post-hoc analysis of the clusters -- which is where
forum/date/citation actually belong.

To keep embedding calls cheap, :func:`embed_documents` flattens every
input across a *batch of documents* into one ``encode()`` call rather
than one call per document. A document's inputs never straddle two
batches, so ``doc_batch_size`` bounds peak memory without changing a
single output vector.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.common.exceptions import EmbeddingError
from src.common.logging_utils import get_logger
from src.embedding.embed_model import EmbeddingModel
from src.extraction.doc_representation import EmbeddableDocument

logger = get_logger(__name__)

DEFAULT_TITLE_WEIGHT = 2.0
DEFAULT_TOC_WEIGHT = 1.5
DEFAULT_BODY_WEIGHT = 1.0
DEFAULT_BODY_CHUNK_CHARS = 2_000
DEFAULT_MAX_BODY_CHUNKS = 4
DEFAULT_DOC_BATCH_SIZE = 256


@dataclass(frozen=True)
class EmbeddingInput:
    text: str
    weight: float
    kind: str  # "title" | "toc" | "body"


def build_embedding_inputs(
    document: EmbeddableDocument,
    title_weight: float = DEFAULT_TITLE_WEIGHT,
    toc_weight: float = DEFAULT_TOC_WEIGHT,
    body_weight: float = DEFAULT_BODY_WEIGHT,
    body_chunk_chars: int = DEFAULT_BODY_CHUNK_CHARS,
    max_body_chunks: int = DEFAULT_MAX_BODY_CHUNKS,
) -> list[EmbeddingInput]:
    """Break one document into the (text, weight, kind) pieces to embed."""

    inputs: list[EmbeddingInput] = []

    if document.title:
        inputs.append(EmbeddingInput(document.title, title_weight, "title"))

    if document.headings:
        headings_text = "; ".join(document.headings)
        # kind stays "toc" -- the weight it selects is still
        # ``discovery.toc_weight``; renaming it would change persisted
        # config keys for no behavioral gain.
        inputs.append(EmbeddingInput(headings_text, toc_weight, "toc"))

    body = (document.body_preview or "").strip()
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


def embed_documents(
    documents: list[EmbeddableDocument],
    embedder: EmbeddingModel,
    title_weight: float = DEFAULT_TITLE_WEIGHT,
    toc_weight: float = DEFAULT_TOC_WEIGHT,
    body_weight: float = DEFAULT_BODY_WEIGHT,
    body_chunk_chars: int = DEFAULT_BODY_CHUNK_CHARS,
    max_body_chunks: int = DEFAULT_MAX_BODY_CHUNKS,
    doc_batch_size: int = DEFAULT_DOC_BATCH_SIZE,
) -> dict[str, np.ndarray]:
    """Embed and pool documents, ``doc_batch_size`` documents per ``encode()``.

    One vector per document, always -- no chunk-level vectors escape this
    function; a document's chunks are pooled before it returns.

    Two classes of document produce no vector at all, deliberately:

    * **No embeddable text** (no title, no headings, no body preview --
      e.g. a ``status="failed"`` record that reached here anyway). Omitted
      rather than given a zero vector, which would otherwise make every
      empty document mutually "similar" and surface as a false domain.
    * **A repeated ``doc_id``** within one call. The first occurrence
      wins; later ones are dropped and counted. This is an identity
      guard, not a nicety: silently overwriting would leave the flattened
      text list longer than the per-document index and misalign every
      vector after the duplicate.

    Returns:
        ``{doc_id: pooled_vector}`` for every document that had at
        least one embeddable input, in input order.

    Raises:
        EmbeddingError: if the embedder returns a different number of
            vectors than the texts it was given -- there is no safe way
            to map vectors back to documents after that.
    """

    prepared: list[tuple[str, list[EmbeddingInput]]] = []
    seen_doc_ids: set[str] = set()
    duplicates = 0
    without_content = 0

    for document in documents:
        if document.doc_id in seen_doc_ids:
            duplicates += 1
            continue
        seen_doc_ids.add(document.doc_id)

        inputs = build_embedding_inputs(
            document,
            title_weight=title_weight,
            toc_weight=toc_weight,
            body_weight=body_weight,
            body_chunk_chars=body_chunk_chars,
            max_body_chunks=max_body_chunks,
        )
        if not inputs:
            without_content += 1
            continue
        prepared.append((document.doc_id, inputs))

    if duplicates:
        logger.warning(
            "Skipped %d document(s) with a duplicate doc_id; the first "
            "occurrence of each was embedded",
            duplicates,
        )
    if without_content:
        logger.warning(
            "Skipped %d document(s) with no embeddable text", without_content
        )

    if not prepared:
        return {}

    result: dict[str, np.ndarray] = {}
    for start in range(0, len(prepared), max(doc_batch_size, 1)):
        batch = prepared[start : start + max(doc_batch_size, 1)]
        texts = [item.text for _, inputs in batch for item in inputs]

        vectors = embedder.encode(texts)
        if vectors.shape[0] != len(texts):
            raise EmbeddingError(
                f"Embedder returned {vectors.shape[0]} vector(s) for "
                f"{len(texts)} input text(s); cannot map vectors to documents",
                phase="embed",
            )

        cursor = 0
        for doc_id, inputs in batch:
            n = len(inputs)
            result[doc_id] = pool_embeddings(
                vectors[cursor : cursor + n], [item.weight for item in inputs]
            )
            cursor += n

        logger.info(
            "Embedded %d/%d document(s)", len(result), len(prepared)
        )

    return result
