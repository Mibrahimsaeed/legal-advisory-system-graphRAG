"""Phase 4: HDBSCAN over case-law embeddings, and the review of what it found.

Two halves:

* :mod:`src.clustering.cluster_summary` -- does the summary preserve
  cluster ids, document ids, sizes and the noise bucket, and does the
  triage flag the clusters that deserve a second look?
* :mod:`orchestration.dags.cluster_review_flow` -- does the stage run end
  to end over stored case representations, persist assignments, write its
  report, and (critically for Phase 4) leave the taxonomy alone?
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from orchestration.dags.cluster_review_flow import run_cluster_review, write_review_json
from src.clustering.cluster import ClusterResult, NOISE_LABEL, hdbscan_clusterer
from src.clustering.cluster_summary import (
    FLAG_CATCH_ALL_SIZE,
    FLAG_LOW_COHESION,
    FLAG_LOW_MEMBERSHIP_CONFIDENCE,
    FLAG_SINGLE_COURT_DOMINATED,
    FLAG_WEAK_RELATIVE_CONFIDENCE,
    KIND_MAJOR,
    KIND_NOISE,
    KIND_SMALL,
    cluster_cohesion,
    summarize_clusters,
)
from src.clustering.label_clusters import build_keyword_corpus
from src.clustering.reduce import reduce_dimensions
from src.clustering.sampling import default_strata_key, stratified_sample
from src.common.db import connection_scope, init_schema
from src.embedding.doc_pooling import embed_documents
from src.embedding.embed_model import DeterministicHashEmbedder
from src.extraction.doc_representation import DocumentRepresentation
from src.extraction.representation_store import (
    DEFAULT_REPRESENTATION_SCHEMA_FILE,
    upsert_representations,
)

DOMAIN_REGISTRY_SCHEMA_FILE = "schemas/domain_registry_schema.sql"

try:
    import hdbscan as _hdbscan  # noqa: F401

    HDBSCAN_AVAILABLE = True
except ImportError:
    HDBSCAN_AVAILABLE = False


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _rep(doc_id: str, topic: str, court: str = "Supreme Court of Pakistan", **kw):
    base = dict(
        doc_id=doc_id,
        source_uri=f"/cases/{doc_id}",
        source_type="case_html",
        source_relpath=f"{topic}/{doc_id}",
        title=f"{topic.title()} matter {doc_id}",
        headings=["Facts", "Order"],
        body_preview=f"{topic} {topic} judgment text about {topic} proceedings. " * 8,
        char_count=600,
        court=court,
        decision_date="2019-04-11",
        citation=f"2019 SCMR {doc_id[-2:]}",
    )
    base.update(kw)
    return DocumentRepresentation(**base)


def _summarize(reps, labels, probabilities=None, vectors=None, **kwargs):
    doc_ids = [r.doc_id for r in reps]
    labels = np.asarray(labels, dtype=int)
    if vectors is None:
        vectors = np.stack(
            [np.eye(4, dtype=np.float32)[int(l) % 4] for l in labels]
        )
    return summarize_clusters(
        run_id="test_run",
        doc_ids=doc_ids,
        labels=labels,
        probabilities=(
            np.asarray(probabilities, dtype=float) if probabilities is not None else None
        ),
        vectors=vectors,
        documents_by_id={r.doc_id: r for r in reps},
        keyword_corpus=build_keyword_corpus(reps),
        **kwargs,
    )


@pytest.fixture()
def review_db(tmp_path) -> Path:
    db_path = tmp_path / "metadata.db"
    init_schema(db_path=db_path, schema_file=str(DEFAULT_REPRESENTATION_SCHEMA_FILE))
    init_schema(db_path=db_path, schema_file=DOMAIN_REGISTRY_SCHEMA_FILE)
    return db_path


@pytest.fixture()
def review_settings(tmp_path, monkeypatch):
    import orchestration.dags.cluster_review_flow as flow

    settings = SimpleNamespace(
        database=SimpleNamespace(schema_file=Path("schemas/manifest_schema.sql")),
        pipeline=SimpleNamespace(checkpoint_dir=tmp_path / "checkpoints"),
        scratch=SimpleNamespace(root=tmp_path / "scratch"),
        metrics=SimpleNamespace(db_path=tmp_path / "metrics.db"),
        caselaw=SimpleNamespace(
            representation_schema_file=DEFAULT_REPRESENTATION_SCHEMA_FILE
        ),
        discovery=SimpleNamespace(
            corpus_source="representations",
            embedding_model_name="fake-model",
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
            umap_min_docs=10_000,  # keep UMAP out of the unit-level flow tests
            hdbscan_min_cluster_size=3,
            hdbscan_min_samples=None,
            hdbscan_metric="euclidean",
            keywords_per_cluster=10,
            domain_registry_schema_file=Path(DOMAIN_REGISTRY_SCHEMA_FILE),
            review_output_dir=tmp_path / "cluster_review",
            review_representative_docs_per_cluster=3,
            review_max_sample_doc_ids=50,
            review_sample_size=None,
            review_major_min_share=0.05,
            review_mixed_max_mean_probability=0.6,
            review_mixed_max_cohesion=0.35,
            review_mixed_max_share=0.5,
            review_relative_confidence_ratio=0.8,
            review_court_dominance_threshold=0.9,
        ),
    )
    monkeypatch.setattr(flow, "get_settings", lambda: settings)
    return settings


# ---------------------------------------------------------------------------
# cohesion
# ---------------------------------------------------------------------------


def test_cohesion_is_one_for_identical_documents():
    vectors = np.tile(np.array([[0.6, 0.8]], dtype=np.float32), (5, 1))

    assert cluster_cohesion(vectors) == pytest.approx(1.0, abs=1e-6)


def test_cohesion_drops_for_a_scattered_cluster():
    tight = np.array([[1, 0.02], [1, -0.02], [1, 0.01]], dtype=np.float32)
    scattered = np.array([[1, 0], [0, 1], [-1, 0], [0, -1]], dtype=np.float32)

    assert cluster_cohesion(tight) > 0.99
    assert cluster_cohesion(scattered) < 0.2


def test_cohesion_is_undefined_for_fewer_than_two_vectors():
    assert cluster_cohesion(np.zeros((1, 4), dtype=np.float32)) is None
    assert cluster_cohesion(np.zeros((0, 4), dtype=np.float32)) is None


# ---------------------------------------------------------------------------
# summaries preserve ids, sizes and the noise bucket
# ---------------------------------------------------------------------------


def test_summary_preserves_cluster_ids_document_ids_and_sizes():
    reps = [_rep(f"c{i}", "criminal") for i in range(6)] + [
        _rep(f"t{i}", "tax") for i in range(4)
    ]
    labels = [0] * 6 + [1] * 4

    review = _summarize(reps, labels)

    assert review.n_clusters == 2
    assert {s.cluster_id for s in review.summaries} == {0, 1}
    assert [s.doc_count for s in review.summaries] == [6, 4]  # largest first
    assert sum(s.doc_count for s in review.summaries) == review.total_documents == 10
    assert sum(s.share for s in review.summaries) == pytest.approx(1.0)
    by_id = {s.cluster_id: s for s in review.summaries}
    assert set(by_id[0].sample_doc_ids) == {f"c{i}" for i in range(6)}
    assert set(by_id[1].sample_doc_ids) == {f"t{i}" for i in range(4)}


def test_noise_is_kept_as_its_own_bucket_reported_last():
    reps = [_rep(f"c{i}", "criminal") for i in range(5)] + [
        _rep(f"n{i}", "misc") for i in range(3)
    ]
    labels = [0] * 5 + [NOISE_LABEL] * 3

    review = _summarize(reps, labels)

    assert review.n_clusters == 1  # noise is not a cluster
    assert review.noise_documents == 3
    assert review.clustered_documents == 5
    assert review.noise_share == pytest.approx(3 / 8)
    assert review.summaries[-1].cluster_id == NOISE_LABEL
    noise = review.noise_summary
    assert noise.kind == KIND_NOISE
    assert noise.is_suspicious is False  # noise is expected, not suspicious


def test_major_and_small_clusters_are_split_by_share():
    reps = [_rep(f"a{i}", "criminal") for i in range(18)] + [
        _rep(f"b{i}", "tax") for i in range(2)
    ]
    labels = [0] * 18 + [1] * 2

    review = _summarize(reps, labels, major_min_share=0.15)  # 0.90 vs 0.10 share

    assert [s.kind for s in review.summaries] == [KIND_MAJOR, KIND_SMALL]
    assert [s.cluster_id for s in review.major_clusters] == [0]
    assert [s.cluster_id for s in review.small_clusters] == [1]


def test_summary_reports_keywords_representatives_courts_and_dates():
    reps = [
        _rep(f"t{i}", "taxation", court="Lahore High Court", decision_date=f"201{i}-05-01")
        for i in range(4)
    ]

    review = _summarize(reps, [0] * 4, probabilities=[0.9, 0.8, 0.7, 0.6])
    summary = review.summaries[0]

    assert "taxation" in summary.keywords
    assert summary.court_distribution == {"Lahore High Court": 4}
    assert summary.earliest_decision_date == "2010-05-01"
    assert summary.latest_decision_date == "2013-05-01"
    assert summary.mean_probability == pytest.approx(0.75)
    top = summary.representative_cases[0]
    assert top.doc_id == "t0" and top.probability == pytest.approx(0.9)
    assert top.court == "Lahore High Court" and top.citation and top.snippet


def test_summary_tolerates_a_run_without_vectors():
    """A resumed run whose embeddings artifact was purged still summarizes."""

    reps = [_rep(f"c{i}", "criminal") for i in range(4)]

    review = _summarize(reps, [0] * 4, vectors=np.zeros((0, 0)))

    assert review.summaries[0].cohesion is None
    assert review.summaries[0].doc_count == 4
    assert review.summaries[0].representative_cases  # falls back gracefully


def test_mismatched_labels_and_doc_ids_raise():
    reps = [_rep("a", "criminal")]

    with pytest.raises(ValueError):
        _summarize(reps, [0, 1])


# ---------------------------------------------------------------------------
# triage: which clusters look suspicious
# ---------------------------------------------------------------------------


def test_scattered_cluster_is_flagged_as_low_cohesion():
    reps = [_rep(f"d{i}", "mixed") for i in range(4)]
    scattered = np.array([[1, 0], [0, 1], [-1, 0], [0, -1]], dtype=np.float32)

    review = _summarize(reps, [0] * 4, vectors=scattered)

    assert FLAG_LOW_COHESION in review.summaries[0].flags
    assert review.suspicious_clusters


def test_low_membership_probability_is_flagged():
    reps = [_rep(f"d{i}", "mixed") for i in range(4)]

    review = _summarize(reps, [0] * 4, probabilities=[0.3, 0.2, 0.4, 0.1])

    assert FLAG_LOW_MEMBERSHIP_CONFIDENCE in review.summaries[0].flags


def test_a_cluster_swallowing_the_corpus_is_flagged_as_catch_all():
    reps = [_rep(f"d{i}", "everything") for i in range(9)] + [_rep("x", "other")]

    review = _summarize(reps, [0] * 9 + [1])

    by_id = {s.cluster_id: s for s in review.summaries}
    assert FLAG_CATCH_ALL_SIZE in by_id[0].flags
    assert FLAG_CATCH_ALL_SIZE not in by_id[1].flags


def test_cluster_built_from_one_court_is_flagged_when_the_corpus_spans_several():
    single_court = [_rep(f"s{i}", "service", court="Federal Service Tribunal") for i in range(5)]
    others = [
        _rep("a1", "criminal", court="Lahore High Court"),
        _rep("a2", "tax", court="Sindh High Court"),
        _rep("a3", "family", court="Peshawar High Court"),
    ]

    review = _summarize(single_court + others, [0] * 5 + [1, 1, 1])

    by_id = {s.cluster_id: s for s in review.summaries}
    assert FLAG_SINGLE_COURT_DOMINATED in by_id[0].flags
    assert FLAG_SINGLE_COURT_DOMINATED not in by_id[1].flags


def test_contaminated_cluster_is_caught_by_relative_confidence():
    """The failure seen on the first real run: a cluster that mixes in
    unrelated cases still clears the absolute probability threshold, but
    sits far below its sibling clusters."""

    reps = [_rep(f"d{i}", "topic") for i in range(12)]
    labels = [0] * 4 + [1] * 4 + [2] * 4
    probabilities = [0.95] * 4 + [0.92] * 4 + [0.70] * 4  # all above the 0.6 cut

    review = _summarize(reps, labels, probabilities=probabilities)

    by_id = {s.cluster_id: s for s in review.summaries}
    assert FLAG_LOW_MEMBERSHIP_CONFIDENCE not in by_id[2].flags  # absolute cut misses it
    assert FLAG_WEAK_RELATIVE_CONFIDENCE in by_id[2].flags  # relative cut catches it
    assert by_id[0].flags == [] and by_id[1].flags == []


def test_uniformly_confident_clusters_are_not_flagged_relatively():
    reps = [_rep(f"d{i}", "topic") for i in range(12)]

    review = _summarize(
        reps, [0] * 4 + [1] * 4 + [2] * 4, probabilities=[0.9] * 12
    )

    assert all(FLAG_WEAK_RELATIVE_CONFIDENCE not in s.flags for s in review.summaries)


# ---------------------------------------------------------------------------
# real HDBSCAN over real (pooled) embeddings
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not HDBSCAN_AVAILABLE, reason="hdbscan not installed")
def test_hdbscan_over_case_embeddings_recovers_topics_and_marks_outliers():
    class _TopicEmbedder:
        dimension = 12

        def encode(self, texts):
            rng = np.random.default_rng(7)
            out = []
            for text in texts:
                lowered = text.lower()
                centre = np.zeros(self.dimension)
                if "criminal" in lowered:
                    centre[0] = 10
                elif "taxation" in lowered:
                    centre[1] = 10
                elif "oddity" in lowered:
                    # one-off subjects, each pushed somewhere different
                    centre[2:] = rng.normal(scale=8, size=self.dimension - 2)
                # anything else (the shared "Facts; Order" headings) stays
                # at the origin: common to every document, so it carries no
                # topic signal either way.
                out.append((centre + rng.normal(scale=0.05, size=self.dimension)).astype(np.float32))
            return np.stack(out)

    reps = (
        [_rep(f"c{i:02d}", "criminal") for i in range(20)]
        + [_rep(f"t{i:02d}", "taxation") for i in range(20)]
        + [_rep(f"o{i:02d}", f"oddity{i}") for i in range(4)]
    )

    pooled = embed_documents(reps, _TopicEmbedder())
    doc_ids = list(pooled)
    vectors = np.stack([pooled[d] for d in doc_ids])
    reduced = reduce_dimensions(vectors, n_components=5, min_docs=10_000)  # skip UMAP
    result = hdbscan_clusterer(min_cluster_size=5)(reduced)

    review = summarize_clusters(
        run_id="hdbscan_run",
        doc_ids=doc_ids,
        labels=result.labels,
        probabilities=result.probabilities,
        vectors=vectors,
        documents_by_id={r.doc_id: r for r in reps},
        keyword_corpus=build_keyword_corpus(reps),
    )

    assert review.n_clusters == 2
    assert review.total_documents == 44
    assert review.noise_documents == 4  # the one-off subjects
    assert sorted(s.doc_count for s in review.major_clusters) == [20, 20]
    keywords = {kw for s in review.major_clusters for kw in s.keywords}
    assert "criminal" in keywords and "taxation" in keywords


# ---------------------------------------------------------------------------
# the review flow end to end
# ---------------------------------------------------------------------------


def _fake_clusterer(vectors: np.ndarray) -> ClusterResult:
    """Labels by position: first two thirds cluster 0/1, last chunk noise."""

    n = vectors.shape[0]
    labels = np.array([0 if i < 6 else 1 if i < 12 else NOISE_LABEL for i in range(n)])
    probabilities = np.where(labels == NOISE_LABEL, 0.0, 0.9)
    return ClusterResult(
        labels=labels,
        probabilities=probabilities,
        n_clusters=len(set(labels.tolist()) - {NOISE_LABEL}),
    )


def test_review_flow_persists_assignments_and_writes_a_report(review_db, review_settings):
    reps = [_rep(f"c{i:02d}", "criminal") for i in range(6)] + [
        _rep(f"t{i:02d}", "taxation") for i in range(6)
    ] + [_rep(f"o{i:02d}", "oddity") for i in range(3)]
    upsert_representations(reps, db_path=review_db)

    review = run_cluster_review(
        run_id="review_run",
        db_path=review_db,
        embedder=DeterministicHashEmbedder(dimension=16),
        clusterer=_fake_clusterer,
    )

    assert review.total_documents == 15
    assert review.n_clusters == 2
    assert review.noise_documents == 3
    assert review.noise_share == pytest.approx(0.2)

    # cluster ids, doc ids and noise all durable in SQL
    with connection_scope(review_db) as conn:
        rows = conn.execute(
            "SELECT doc_id, cluster_id, confidence FROM cluster_assignments WHERE run_id = ?",
            ("review_run",),
        ).fetchall()
    assert len(rows) == 15
    assert sum(1 for r in rows if r["cluster_id"] == NOISE_LABEL) == 3
    assert {r["doc_id"] for r in rows} == {r.doc_id for r in reps}

    report = review_settings.discovery.review_output_dir / "review_run.json"
    assert report.exists()
    payload = json.loads(report.read_text())
    assert payload["run_id"] == "review_run"
    assert payload["noise_documents"] == 3
    assert payload["parameters"]["hdbscan_min_cluster_size"] == 3
    assert payload["parameters"]["embedding_model"] == "fake-model"
    assert len(payload["summaries"]) == 3  # 2 clusters + noise


def test_review_flow_does_not_touch_the_taxonomy(review_db, review_settings):
    """Phase 4 explicitly stops short of naming domains."""

    upsert_representations(
        [_rep(f"c{i:02d}", "criminal") for i in range(15)], db_path=review_db
    )

    run_cluster_review(
        run_id="review_run",
        db_path=review_db,
        embedder=DeterministicHashEmbedder(dimension=16),
        clusterer=_fake_clusterer,
    )

    with connection_scope(review_db) as conn:
        candidates = conn.execute("SELECT COUNT(*) AS c FROM domain_candidates").fetchone()["c"]
    assert candidates == 0
    assert not (review_settings.discovery.review_output_dir / "taxonomy").exists()


def test_review_flow_clusters_a_controlled_sample_when_asked(review_db, review_settings):
    reps = [
        _rep(f"d{i:03d}", "criminal" if i % 2 else "taxation", char_count=100 * (i % 30))
        for i in range(60)
    ]
    upsert_representations(reps, db_path=review_db)

    review = run_cluster_review(
        run_id="sampled_run",
        db_path=review_db,
        sample_size=20,
        embedder=DeterministicHashEmbedder(dimension=16),
        clusterer=_fake_clusterer,
    )

    assert review.total_documents == 20  # the sample, not the corpus
    assert review.parameters["corpus_size"] == 60
    assert review.parameters["requested_sample_size"] == 20


def test_review_flow_handles_an_empty_corpus(review_db, review_settings):
    review = run_cluster_review(
        run_id="empty_run",
        db_path=review_db,
        embedder=DeterministicHashEmbedder(dimension=16),
        clusterer=_fake_clusterer,
    )

    assert review.total_documents == 0
    assert review.n_clusters == 0
    assert (review_settings.discovery.review_output_dir / "empty_run.json").exists()


def test_write_review_json_includes_derived_fields(tmp_path):
    reps = [_rep(f"c{i}", "criminal") for i in range(4)] + [_rep("n1", "misc")]
    review = _summarize(reps, [0] * 4 + [NOISE_LABEL], probabilities=[0.2] * 4 + [0.0])

    path = write_review_json(review, tmp_path)
    payload = json.loads(path.read_text())

    assert payload["noise_share"] == pytest.approx(0.2)
    assert payload["suspicious_cluster_ids"] == [0]
    assert payload["summaries"][0]["representative_cases"][0]["doc_id"]


# ---------------------------------------------------------------------------
# controlled sampling
# ---------------------------------------------------------------------------


def test_stratified_sample_covers_courts_and_lengths_deterministically():
    reps = [
        _rep(
            f"d{i:03d}",
            "criminal",
            court=["Supreme Court of Pakistan", "Lahore High Court", "Sindh High Court"][i % 3],
            char_count=[500, 5_000, 50_000][i % 3],
        )
        for i in range(90)
    ]

    sample = stratified_sample(reps, sample_min=30, sample_max=30, seed=42)
    again = stratified_sample(reps, sample_min=30, sample_max=30, seed=42)

    assert len(sample) == 30
    assert [d.doc_id for d in sample] == [d.doc_id for d in again]  # deterministic
    assert len({default_strata_key(d) for d in sample}) == 3  # every stratum present


def test_stratified_sample_returns_everything_when_corpus_is_small():
    reps = [_rep(f"d{i}", "criminal") for i in range(5)]

    assert len(stratified_sample(reps, sample_min=50, sample_max=100)) == 5
