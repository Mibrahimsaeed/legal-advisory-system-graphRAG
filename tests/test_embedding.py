from __future__ import annotations

import numpy as np
import pytest

from src.embedding.doc_pooling import (
    build_embedding_inputs,
    embed_signatures,
    pool_embeddings,
)
from src.embedding.embed_model import DeterministicHashEmbedder, get_embedder
from src.extraction.signature import DocumentSignature


def _sig(doc_id: str, **kwargs) -> DocumentSignature:
    defaults = dict(
        source_uri=f"s3://x/{doc_id}.pdf",
        signature_hash=f"h_{doc_id}",
        is_scanned=False,
        extraction_status="ok",
        extractor_used="pdfplumber",
    )
    defaults.update(kwargs)
    return DocumentSignature(doc_id=doc_id, **defaults)


# ---------------------------------------------------------------------------
# embed_model.py
# ---------------------------------------------------------------------------


def test_deterministic_hash_embedder_is_deterministic():
    embedder = DeterministicHashEmbedder(dimension=32)
    v1 = embedder.encode(["hello world"])
    v2 = embedder.encode(["hello world"])
    assert np.allclose(v1, v2)


def test_deterministic_hash_embedder_distinguishes_different_text():
    embedder = DeterministicHashEmbedder(dimension=32)
    vectors = embedder.encode(["alpha text", "beta text"])
    assert vectors.shape == (2, 32)
    assert not np.allclose(vectors[0], vectors[1])


def test_deterministic_hash_embedder_produces_unit_vectors():
    embedder = DeterministicHashEmbedder(dimension=16)
    vectors = embedder.encode(["one", "two", "three"])
    norms = np.linalg.norm(vectors, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-5)


def test_deterministic_hash_embedder_handles_empty_input():
    embedder = DeterministicHashEmbedder(dimension=16)
    result = embedder.encode([])
    assert result.shape == (0, 16)


def test_get_embedder_falls_back_without_sentence_transformers():
    # This sandbox/test environment may or may not have sentence-transformers
    # installed; either way get_embedder() must return *something* usable.
    embedder = get_embedder("sentence-transformers/all-mpnet-base-v2")
    vec = embedder.encode(["some text"])
    assert vec.shape[0] == 1
    assert vec.shape[1] == embedder.dimension


# ---------------------------------------------------------------------------
# doc_pooling.py
# ---------------------------------------------------------------------------


def test_build_embedding_inputs_includes_title_toc_and_body_chunks():
    sig = _sig(
        "d1",
        title="Master Services Agreement",
        toc=["1. Definitions", "2. Term"],
        body_preview="x" * 5000,
    )
    inputs = build_embedding_inputs(sig, body_chunk_chars=2000, max_body_chunks=4)
    kinds = [i.kind for i in inputs]
    assert kinds[0] == "title"
    assert kinds[1] == "toc"
    assert kinds.count("body") == 3  # 5000 chars / 2000 per chunk -> 3 chunks


def test_build_embedding_inputs_respects_max_body_chunks():
    sig = _sig("d1", title="T", body_preview="x" * 100_000)
    inputs = build_embedding_inputs(sig, body_chunk_chars=2000, max_body_chunks=4)
    body_chunks = [i for i in inputs if i.kind == "body"]
    assert len(body_chunks) == 4


def test_build_embedding_inputs_empty_signature_yields_nothing():
    sig = _sig("d1")
    assert build_embedding_inputs(sig) == []


def test_pool_embeddings_weighted_mean_and_normalized():
    vectors = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    pooled = pool_embeddings(vectors, weights=[3.0, 1.0])
    # weighted mean before normalization: (3*[1,0] + 1*[0,1]) / 4 = [0.75, 0.25]
    expected_direction = np.array([0.75, 0.25], dtype=np.float32)
    expected_direction /= np.linalg.norm(expected_direction)
    assert np.allclose(pooled, expected_direction, atol=1e-5)
    assert np.isclose(np.linalg.norm(pooled), 1.0)


def test_pool_embeddings_handles_empty_input():
    empty = np.zeros((0, 8), dtype=np.float32)
    pooled = pool_embeddings(empty, weights=[])
    assert pooled.shape == (8,)
    assert np.allclose(pooled, 0)


def test_embed_signatures_skips_docs_with_no_content():
    embedder = DeterministicHashEmbedder(dimension=16)
    sigs = [
        _sig("has_content", title="Something", body_preview="text here"),
        _sig("empty"),  # no title/toc/body -> nothing to embed
    ]
    result = embed_signatures(sigs, embedder)
    assert "has_content" in result
    assert "empty" not in result
    assert result["has_content"].shape == (16,)


def test_embed_signatures_is_deterministic_and_distinguishes_docs():
    embedder = DeterministicHashEmbedder(dimension=16)
    sigs = [
        _sig("d1", title="Contract Agreement", body_preview="services rendered" * 20),
        _sig("d2", title="Notice of Appeal", body_preview="appellant hereby" * 20),
    ]
    result_a = embed_signatures(sigs, embedder)
    result_b = embed_signatures(sigs, embedder)
    assert np.allclose(result_a["d1"], result_b["d1"])
    assert not np.allclose(result_a["d1"], result_a["d2"])


def test_embed_signatures_empty_batch_returns_empty_dict():
    embedder = DeterministicHashEmbedder(dimension=16)
    assert embed_signatures([], embedder) == {}