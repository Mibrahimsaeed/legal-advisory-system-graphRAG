from __future__ import annotations

import numpy as np
import pytest

from src.clustering.cluster import ClusterResult, NOISE_LABEL, hdbscan_clusterer
from src.clustering.label_clusters import (
    build_keyword_corpus,
    extract_cluster_keywords,
    generate_domain_draft,
    select_representative_docs,
)
from src.clustering.reduce import reduce_dimensions
from src.common.exceptions import ClusteringError
from src.extraction.signature import DocumentSignature

try:
    import hdbscan as _hdbscan  # noqa: F401

    HDBSCAN_AVAILABLE = True
except ImportError:
    HDBSCAN_AVAILABLE = False

try:
    import umap as _umap  # noqa: F401

    UMAP_AVAILABLE = True
except ImportError:
    UMAP_AVAILABLE = False


def _sig(doc_id: str, char_count: int, is_scanned: bool = False, extractor: str = "pdfplumber", **kwargs) -> DocumentSignature:
    return DocumentSignature(
        doc_id=doc_id,
        source_uri=f"s3://x/{doc_id}.pdf",
        signature_hash=f"h_{doc_id}",
        is_scanned=is_scanned,
        extraction_status="ok",
        extractor_used=extractor,
        char_count=char_count,
        **kwargs,
    )


# ---------------------------------------------------------------------------
# reduce.py
# ---------------------------------------------------------------------------

# ... (rest of the file — reduce.py / cluster.py / label_clusters.py tests — unchanged)

# ---------------------------------------------------------------------------
# reduce.py
# ---------------------------------------------------------------------------


def test_reduce_dimensions_skips_when_too_few_docs():
    rng = np.random.default_rng(0)
    vectors = rng.normal(size=(10, 100)).astype(np.float32)
    out = reduce_dimensions(vectors, n_components=50, min_docs=50)
    assert out.shape == vectors.shape


def test_reduce_dimensions_skips_when_already_low_dimensional():
    rng = np.random.default_rng(0)
    vectors = rng.normal(size=(200, 10)).astype(np.float32)
    out = reduce_dimensions(vectors, n_components=50, min_docs=50)
    assert out.shape == vectors.shape


def test_reduce_dimensions_produces_target_shape():
    rng = np.random.default_rng(0)
    vectors = rng.normal(size=(200, 300)).astype(np.float32)
    out = reduce_dimensions(vectors, n_components=50, min_docs=50)
    assert out.shape == (200, 50)


def test_reduce_dimensions_empty_input():
    empty = np.zeros((0, 50), dtype=np.float32)
    out = reduce_dimensions(empty, n_components=10)
    assert out.shape == (0, 50)


# ---------------------------------------------------------------------------
# cluster.py
# ---------------------------------------------------------------------------


def test_hdbscan_clusterer_raises_clearly_when_not_installed():
    if HDBSCAN_AVAILABLE:
        pytest.skip("hdbscan is installed in this environment")
    with pytest.raises(ClusteringError):
        hdbscan_clusterer()


@pytest.mark.skipif(not HDBSCAN_AVAILABLE, reason="hdbscan not installed")
def test_hdbscan_clusterer_finds_dense_blobs():
    rng = np.random.default_rng(0)
    blob_a = rng.normal(loc=0.0, scale=0.05, size=(30, 5))
    blob_b = rng.normal(loc=10.0, scale=0.05, size=(30, 5))
    vectors = np.vstack([blob_a, blob_b]).astype(np.float32)

    clusterer = hdbscan_clusterer(min_cluster_size=10)
    result = clusterer(vectors)

    assert isinstance(result, ClusterResult)
    assert result.n_clusters == 2
    assert len(result.labels) == 60


@pytest.mark.skipif(not HDBSCAN_AVAILABLE, reason="hdbscan not installed")
def test_cluster_result_shape_is_stable_for_empty_input():
    clusterer = hdbscan_clusterer(min_cluster_size=5)
    result = clusterer(np.zeros((0, 10)))
    assert result.n_clusters == 0
    assert len(result.labels) == 0


# ---------------------------------------------------------------------------
# label_clusters.py
# ---------------------------------------------------------------------------


def _contract_sig(i: int) -> DocumentSignature:
    return _sig(
        f"c{i}",
        char_count=1000,
        title="Master Services Agreement",
        toc=["1. Definitions", "2. Payment Terms"],
        body_preview="This agreement governs software licensing and payment obligations. " * 10,
    )


def _litigation_sig(i: int) -> DocumentSignature:
    return _sig(
        f"l{i}",
        char_count=1000,
        title="Notice of Appeal",
        toc=["Grounds for Appeal"],
        body_preview="Appellant hereby appeals the judgment of the trial court. " * 10,
    )


def test_extract_cluster_keywords_are_distinctive_per_cluster():
    contracts = [_contract_sig(i) for i in range(10)]
    litigation = [_litigation_sig(i) for i in range(6)]
    corpus = build_keyword_corpus(contracts + litigation)

    contract_kw = set(extract_cluster_keywords(contracts, corpus, top_k=8))
    litigation_kw = set(extract_cluster_keywords(litigation, corpus, top_k=8))

    assert contract_kw != litigation_kw
    assert "software" in contract_kw or "licensing" in contract_kw
    assert "appellant" in litigation_kw or "appeals" in litigation_kw


def test_extract_cluster_keywords_empty_cluster():
    corpus = build_keyword_corpus([_contract_sig(0)])
    assert extract_cluster_keywords([], corpus) == []


def test_select_representative_docs_prefers_probability_when_available():
    doc_ids = [f"d{i}" for i in range(5)]
    vectors = {d: np.zeros(4) for d in doc_ids}
    probabilities = {"d0": 0.1, "d1": 0.9, "d2": 0.95, "d3": 0.2, "d4": 0.5}
    reps = select_representative_docs(doc_ids, vectors, probabilities=probabilities, top_k=2)
    assert reps == ["d2", "d1"]


def test_select_representative_docs_falls_back_to_centroid_distance():
    doc_ids = ["near", "far"]
    vectors = {"near": np.array([0.01, 0.0]), "far": np.array([5.0, 5.0])}
    reps = select_representative_docs(doc_ids, vectors, probabilities=None, top_k=1)
    assert reps == ["near"]


class _FakeLLMClient:
    def __init__(self, response: str):
        self.response = response
        self.last_prompt: str | None = None

    def complete(self, system, prompt, max_tokens=None):
        self.last_prompt = prompt
        return self.response


def test_generate_domain_draft_parses_llm_json():
    sigs = [_contract_sig(i) for i in range(5)]
    llm = _FakeLLMClient(
        '{"name": "Commercial Contracts", "description": "Services and licensing.", '
        '"inclusion_criteria": ["MSAs"], "exclusion_criteria": ["litigation"]}'
    )
    draft = generate_domain_draft(0, sigs, ["c0", "c1"], ["software", "licensing"], sample_size=100, llm_client=llm)
    assert draft.name == "Commercial Contracts"
    assert draft.error is None
    assert draft.inclusion_criteria == ["MSAs"]
    assert draft.doc_count == 5
    assert "Cluster keywords" in llm.last_prompt


def test_generate_domain_draft_degrades_gracefully_on_llm_failure():
    sigs = [_litigation_sig(i) for i in range(3)]

    class BrokenLLM:
        def complete(self, system, prompt, max_tokens=None):
            raise RuntimeError("API unavailable")

    draft = generate_domain_draft(2, sigs, ["l0"], ["appeal"], sample_size=100, llm_client=BrokenLLM())
    assert draft.name == "Cluster 2 (unlabeled)"
    assert draft.error is not None
    assert draft.doc_count == 3
    assert draft.keywords == ["appeal"]


def test_generate_domain_draft_degrades_gracefully_on_malformed_json():
    sigs = [_contract_sig(0)]
    draft = generate_domain_draft(0, sigs, [], [], sample_size=10, llm_client=_FakeLLMClient("not json"))
    assert draft.error is not None
    assert draft.name == "Cluster 0 (unlabeled)"