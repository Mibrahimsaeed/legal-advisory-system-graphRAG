"""Phase 3: classification embeddings over case-law representations.

These vectors exist only to feed domain discovery (UMAP -> HDBSCAN ->
labeling). They are not RAG embeddings, and nothing here produces or
stores a chunk-level vector.

What is pinned down:

* which representation fields contribute to a document's vector,
* that document identity survives the embed step exactly (order,
  batching, skipped documents, duplicate ids),
* that failed / contentless documents get no vector rather than a
  misleading zero vector,
* that the run artifact carries the provenance needed to reuse it safely.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import orchestration.dags.domain_discovery_flow as flow
from orchestration.dags.case_ingest_flow import run_case_ingest
from src.common.checkpoint import CheckpointManager
from src.common.db import init_schema
from src.common.exceptions import ClusteringError, EmbeddingError
from src.common.metrics import MetricsStore
from src.embedding.doc_pooling import build_embedding_inputs, embed_documents
from src.embedding.embed_model import DeterministicHashEmbedder
from src.extraction.doc_representation import DocumentRepresentation
from src.extraction.representation_store import (
    DEFAULT_REPRESENTATION_SCHEMA_FILE,
    list_representations,
    upsert_representations,
)

FIXTURE_ROOT = Path(__file__).resolve().parent / "fixtures" / "caselaw"


def _rep(doc_id: str, **overrides) -> DocumentRepresentation:
    base = dict(
        doc_id=doc_id,
        source_uri=f"/cases/{doc_id}",
        source_type="case_html",
        title=f"{doc_id} v. The State",
        headings=["Facts", "Order"],
        body_preview=f"Judgment text for {doc_id}. " * 20,
        char_count=500,
        court="Supreme Court of Pakistan",
        decision_date="2019-04-11",
        citation="2019 SCMR 123",
        judges=["Mr. Justice A"],
        case_number="Criminal Appeal No. 1 of 2018",
    )
    base.update(overrides)
    return DocumentRepresentation(**base)


class _CountingEmbedder:
    """Deterministic embedder that records how it was called."""

    def __init__(self, dimension: int = 16, tag: str = "") -> None:
        self.dimension = dimension
        self.tag = tag
        self.calls = 0
        self.texts_seen: list[str] = []
        self._inner = DeterministicHashEmbedder(dimension=dimension)

    def encode(self, texts: list[str]) -> np.ndarray:
        self.calls += 1
        self.texts_seen.extend(texts)
        return self._inner.encode([self.tag + t for t in texts])


@pytest.fixture()
def caselaw_db(tmp_path) -> Path:
    db_path = tmp_path / "metadata.db"
    init_schema(db_path=db_path, schema_file=str(DEFAULT_REPRESENTATION_SCHEMA_FILE))
    return db_path


# ---------------------------------------------------------------------------
# Which fields contribute to the classification vector
# ---------------------------------------------------------------------------


def test_only_title_headings_and_body_are_embedded():
    inputs = build_embedding_inputs(_rep("d1"))

    assert [i.kind for i in inputs] == ["title", "toc", "body"]
    joined = " ".join(i.text for i in inputs)
    assert "d1 v. The State" in joined  # title
    assert "Facts; Order" in joined  # headings
    assert "Judgment text for d1" in joined  # body preview


@pytest.mark.parametrize(
    "field",
    ["court", "decision_date", "citation", "judges", "case_number", "metadata"],
)
def test_case_law_facts_do_not_change_the_vector(field):
    """Forum, dates, citations and bench names are stored and indexed, but
    kept out of the vector on purpose -- see src/embedding/doc_pooling.py."""

    changed = {
        "court": "Lahore High Court",
        "decision_date": "2021-01-01",
        "citation": "2021 PLJ 88",
        "judges": ["Mr. Justice Z", "Mr. Justice Y"],
        "case_number": "Writ Petition No. 9 of 2020",
        "metadata": {"result": "dismissed"},
    }[field]

    embedder = DeterministicHashEmbedder(dimension=16)
    baseline = embed_documents([_rep("d1")], embedder)["d1"]
    variant = embed_documents([_rep("d1", **{field: changed})], embedder)["d1"]

    assert np.allclose(baseline, variant)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("title", "Different Cause Title"),
        ("headings", ["Question of Law"]),
        ("body_preview", "An entirely different judgment about income tax."),
    ],
)
def test_classification_text_does_change_the_vector(field, value):
    embedder = DeterministicHashEmbedder(dimension=16)
    baseline = embed_documents([_rep("d1")], embedder)["d1"]
    variant = embed_documents([_rep("d1", **{field: value})], embedder)["d1"]

    assert not np.allclose(baseline, variant)


# ---------------------------------------------------------------------------
# Deterministic document -> embedding mapping
# ---------------------------------------------------------------------------


def test_embedding_is_deterministic_across_runs():
    docs = [_rep(f"d{i}") for i in range(5)]
    embedder = DeterministicHashEmbedder(dimension=16)

    first = embed_documents(docs, embedder)
    second = embed_documents(docs, embedder)

    assert set(first) == set(second) == {d.doc_id for d in docs}
    for doc_id in first:
        assert np.allclose(first[doc_id], second[doc_id])


def test_each_document_gets_its_own_distinct_vector():
    docs = [_rep(f"d{i}") for i in range(5)]

    vectors = embed_documents(docs, DeterministicHashEmbedder(dimension=16))

    stacked = np.stack([vectors[d.doc_id] for d in docs])
    assert len({v.tobytes() for v in stacked}) == len(docs)


def test_input_order_does_not_change_any_document_vector():
    docs = [_rep(f"d{i}") for i in range(6)]
    embedder = DeterministicHashEmbedder(dimension=16)

    in_order = embed_documents(docs, embedder)
    reversed_order = embed_documents(list(reversed(docs)), embedder)

    for doc_id, vector in in_order.items():
        assert np.allclose(vector, reversed_order[doc_id]), doc_id


@pytest.mark.parametrize("doc_batch_size", [1, 2, 3, 7, 1_000])
def test_batching_does_not_change_a_single_vector(doc_batch_size):
    """Batching bounds memory for a 10k-case corpus; it must not be
    observable in the output."""

    docs = [_rep(f"d{i}") for i in range(7)]
    embedder = DeterministicHashEmbedder(dimension=16)

    reference = embed_documents(docs, embedder, doc_batch_size=1_000_000)
    batched = embed_documents(docs, embedder, doc_batch_size=doc_batch_size)

    assert set(batched) == set(reference)
    for doc_id in reference:
        assert np.allclose(batched[doc_id], reference[doc_id]), doc_id


def test_batching_actually_splits_the_encode_calls():
    docs = [_rep(f"d{i}") for i in range(6)]
    embedder = _CountingEmbedder()

    embed_documents(docs, embedder, doc_batch_size=2)

    assert embedder.calls == 3  # 6 documents / 2 per batch


def test_one_vector_per_document_regardless_of_body_length():
    """No chunk-level vectors escape: a long body is pooled, not emitted."""

    long_doc = _rep("long", body_preview="evidence " * 5_000)
    inputs = build_embedding_inputs(long_doc)
    assert len([i for i in inputs if i.kind == "body"]) > 1  # several chunks in

    vectors = embed_documents([long_doc], DeterministicHashEmbedder(dimension=16))

    assert list(vectors) == ["long"]  # ... one vector out
    assert vectors["long"].shape == (16,)


def test_pooled_vectors_are_unit_length():
    vectors = embed_documents(
        [_rep(f"d{i}") for i in range(3)], DeterministicHashEmbedder(dimension=16)
    )

    for vector in vectors.values():
        assert np.isclose(np.linalg.norm(vector), 1.0, atol=1e-5)


# ---------------------------------------------------------------------------
# Failed / contentless / duplicate documents
# ---------------------------------------------------------------------------


def test_document_without_embeddable_text_is_skipped_not_zero_vectored():
    empty = DocumentRepresentation(
        doc_id="empty", source_uri="/cases/empty", source_type="case_html"
    )

    vectors = embed_documents([empty, _rep("d1")], DeterministicHashEmbedder(dimension=16))

    assert "empty" not in vectors
    assert "d1" in vectors


def test_a_skipped_document_does_not_shift_the_others_vectors():
    """The alignment bug this guards against is silent: if a skipped
    document consumed a slot, every later document would get the wrong
    vector."""

    embedder = DeterministicHashEmbedder(dimension=16)
    good = [_rep("a"), _rep("b"), _rep("c")]
    empty = DocumentRepresentation(
        doc_id="hole", source_uri="/cases/hole", source_type="case_html"
    )

    without_hole = embed_documents(good, embedder)
    with_hole = embed_documents([good[0], empty, good[1], good[2]], embedder)

    assert "hole" not in with_hole
    for doc_id in ("a", "b", "c"):
        assert np.allclose(with_hole[doc_id], without_hole[doc_id]), doc_id


def test_duplicate_doc_ids_keep_the_first_and_do_not_misalign():
    embedder = DeterministicHashEmbedder(dimension=16)
    original = _rep("dup")
    impostor = _rep("dup", title="Completely different title", body_preview="other text")
    trailing = _rep("after")

    vectors = embed_documents([original, impostor, trailing], embedder)
    reference = embed_documents([original, trailing], embedder)

    assert set(vectors) == {"dup", "after"}
    assert np.allclose(vectors["dup"], reference["dup"])  # first occurrence won
    assert np.allclose(vectors["after"], reference["after"])  # no drift after it


def test_failed_representations_never_reach_the_embedder(caselaw_db):
    ok = _rep("ok_case")
    failed = DocumentRepresentation(
        doc_id="failed_case",
        source_uri="/cases/failed_case",
        source_type="case_html",
        status="failed",
        error="empty_document: no extractable text in case.html",
    )
    upsert_representations([ok, failed], db_path=caselaw_db)

    corpus = list_representations(db_path=caselaw_db)
    embedder = _CountingEmbedder()
    vectors = embed_documents(corpus, embedder)

    assert [r.doc_id for r in corpus] == ["ok_case"]
    assert set(vectors) == {"ok_case"}
    assert not any("failed_case" in t for t in embedder.texts_seen)


def test_empty_corpus_returns_no_vectors_and_calls_no_model():
    embedder = _CountingEmbedder()

    assert embed_documents([], embedder) == {}
    assert embedder.calls == 0


def test_embedder_returning_the_wrong_number_of_vectors_is_an_error():
    class _TruncatingEmbedder:
        dimension = 8

        def encode(self, texts):
            return np.zeros((len(texts) - 1, self.dimension), dtype=np.float32)

    with pytest.raises(EmbeddingError):
        embed_documents([_rep("d1")], _TruncatingEmbedder())


# ---------------------------------------------------------------------------
# Run artifact: identity + provenance + safe reuse
# ---------------------------------------------------------------------------


def test_embedding_artifact_round_trips_ids_vectors_and_model(tmp_path):
    doc_ids = ["a", "b", "c"]
    vectors = np.arange(12, dtype=np.float32).reshape(3, 4)
    path = tmp_path / "embeddings.npz"

    flow._save_embeddings(path, doc_ids, vectors, "test-model")
    loaded_ids, loaded_vectors, model_name = flow._load_embeddings(path)

    assert loaded_ids == doc_ids
    assert np.array_equal(loaded_vectors, vectors)
    assert model_name == "test-model"
    with np.load(path, allow_pickle=True) as data:
        assert int(data["dimension"]) == 4
        assert str(data["created_at"])  # provenance timestamp recorded


def test_legacy_artifact_without_model_name_is_still_readable(tmp_path):
    path = tmp_path / "embeddings.npz"
    np.savez(
        path,
        doc_ids=np.array(["a"], dtype=object),
        vectors=np.zeros((1, 4), dtype=np.float32),
    )

    doc_ids, vectors, model_name = flow._load_embeddings(path)

    assert doc_ids == ["a"] and vectors.shape == (1, 4)
    assert model_name is None


def _discovery_settings(tmp_path, model_name: str) -> SimpleNamespace:
    return SimpleNamespace(
        corpus_source="representations",
        embedding_model_name=model_name,
        embedding_batch_size=32,
        embedding_doc_batch_size=256,
        title_weight=2.0,
        toc_weight=1.5,
        body_weight=1.0,
        body_chunk_chars=2000,
        max_body_chunks=4,
        umap_n_components=4,
        umap_n_neighbors=3,
        umap_min_dist=0.0,
        umap_metric="cosine",
        umap_min_docs=1_000,  # keep UMAP out of it; this is an embed test
        hdbscan_min_cluster_size=2,
        hdbscan_min_samples=None,
        hdbscan_metric="euclidean",
    )


def _run_embed_phase(tmp_path, db_path, discovery, embedder, clusterer, run_id="r1"):
    checkpoint = CheckpointManager(
        checkpoint_dir=tmp_path / "checkpoints",
        run_key=run_id,
        phases=["sample", "embed", "cluster", "label", "persist"],
    )
    metrics = MetricsStore(tmp_path / "metrics.db")
    metrics.init_schema()
    return flow._run_sample_embed_cluster(
        SimpleNamespace(),
        discovery,
        checkpoint,
        metrics,
        "metrics_run",
        run_id,
        db_path,
        tmp_path / "discovery" / run_id / "embeddings.npz",
        embedder,
        clusterer,
    )


def _exploding_clusterer(vectors):
    raise ClusteringError("clustering backend unavailable")


def test_resume_reuses_cached_embeddings_for_the_same_model(tmp_path, caselaw_db):
    upsert_representations([_rep(f"d{i}") for i in range(4)], db_path=caselaw_db)
    discovery = _discovery_settings(tmp_path, "model-a")
    embedder = _CountingEmbedder(tag="a")

    # First pass: embeds, then the cluster phase fails -- exactly the
    # crash-after-embed case the artifact exists for.
    with pytest.raises(ClusteringError):
        _run_embed_phase(tmp_path, caselaw_db, discovery, embedder, _exploding_clusterer)
    assert embedder.calls == 1

    # Second pass with the same model: the artifact is reused.
    resumed = _CountingEmbedder(tag="a")
    with pytest.raises(ClusteringError):
        _run_embed_phase(tmp_path, caselaw_db, discovery, resumed, _exploding_clusterer)

    assert resumed.calls == 0


def test_resume_re_embeds_when_the_configured_model_changed(tmp_path, caselaw_db):
    """Reusing vectors from another model would make the discovered
    domains an artifact of the model switch."""

    upsert_representations([_rep(f"d{i}") for i in range(4)], db_path=caselaw_db)
    first = _CountingEmbedder(tag="a")
    with pytest.raises(ClusteringError):
        _run_embed_phase(
            tmp_path, caselaw_db, _discovery_settings(tmp_path, "model-a"),
            first, _exploding_clusterer,
        )

    second = _CountingEmbedder(tag="b")
    with pytest.raises(ClusteringError):
        _run_embed_phase(
            tmp_path, caselaw_db, _discovery_settings(tmp_path, "model-b"),
            second, _exploding_clusterer,
        )

    assert second.calls == 1  # re-embedded rather than trusting the artifact


def test_embed_phase_preserves_doc_id_to_vector_alignment(tmp_path, caselaw_db):
    reps = [_rep(f"d{i}") for i in range(4)]
    upsert_representations(reps, db_path=caselaw_db)
    discovery = _discovery_settings(tmp_path, "model-a")
    embedder = _CountingEmbedder(tag="a")

    with pytest.raises(ClusteringError):
        _run_embed_phase(tmp_path, caselaw_db, discovery, embedder, _exploding_clusterer)

    doc_ids, vectors, _ = flow._load_embeddings(
        tmp_path / "discovery" / "r1" / "embeddings.npz"
    )
    standalone = embed_documents(
        list_representations(db_path=caselaw_db), _CountingEmbedder(tag="a")
    )

    assert set(doc_ids) == {r.doc_id for r in reps}
    for i, doc_id in enumerate(doc_ids):
        assert np.allclose(vectors[i], standalone[doc_id]), doc_id


# ---------------------------------------------------------------------------
# End to end over the Phase 2 fixture corpus
# ---------------------------------------------------------------------------


@pytest.fixture()
def fixture_corpus_db(tmp_path, monkeypatch, caselaw_db) -> Path:
    import orchestration.dags.case_ingest_flow as ingest_flow

    monkeypatch.setattr(
        ingest_flow,
        "get_settings",
        lambda: SimpleNamespace(
            document=SimpleNamespace(min_characters=200, min_words=250),
            # The fixture cases are short hand-written excerpts, not full
            # judgments; they are measured against the pre-Phase-1 threshold so
            # these tests stay about metadata/encoding rather than length.
            # The configured production value (1200) is covered by
            # tests/test_phase1_foundation.py.
            caselaw=SimpleNamespace(
                corpus_root=FIXTURE_ROOT,
                case_html_filename="case.html",
                metadata_filename="metadata.json",
                body_preview_char_limit=20_000,
                max_headings=50,
                representation_schema_file=DEFAULT_REPRESENTATION_SCHEMA_FILE,
            )
        ),
    )
    run_case_ingest(root=FIXTURE_ROOT, db_path=caselaw_db)
    return caselaw_db


def test_real_case_corpus_embeds_one_vector_per_stored_case(fixture_corpus_db):
    corpus = list_representations(db_path=fixture_corpus_db)
    embedder = DeterministicHashEmbedder(dimension=32)

    vectors = embed_documents(corpus, embedder)

    assert len(corpus) == 6  # the 7th fixture case failed extraction
    assert set(vectors) == {r.doc_id for r in corpus}
    assert all(v.shape == (32,) for v in vectors.values())
    # Stable across an identical re-run, keyed by doc_id.
    assert all(
        np.allclose(v, embed_documents(corpus, embedder)[k]) for k, v in vectors.items()
    )


def test_provenance_survives_the_embed_step(fixture_corpus_db):
    """A vector can always be traced back to the case folder it came from."""

    corpus = list_representations(db_path=fixture_corpus_db)
    by_id = {r.doc_id: r for r in corpus}

    vectors = embed_documents(corpus, DeterministicHashEmbedder(dimension=16))

    for doc_id in vectors:
        rep = by_id[doc_id]
        assert Path(rep.source_file).exists()
        assert rep.source_relpath and rep.content_hash


def test_representation_fields_are_unchanged_by_embedding(fixture_corpus_db):
    """Embedding is read-only with respect to the representation."""

    corpus = list_representations(db_path=fixture_corpus_db)
    before = [dataclasses.asdict(r) for r in corpus]

    embed_documents(corpus, DeterministicHashEmbedder(dimension=16))

    assert [dataclasses.asdict(r) for r in corpus] == before
