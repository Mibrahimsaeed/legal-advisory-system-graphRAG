"""Phase 5: draft domain taxonomy from reviewed clusters -- and what it refuses.

The interesting assertions here are the negative ones: a tiny cluster must
not become a legal domain, a cluster the LLM failed to name must not ship
as one, a flagged/mixed cluster must not pass as clean, and nothing may be
frozen.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from orchestration.dags.taxonomy_draft_flow import run_taxonomy_draft
from src.clustering.cluster import ClusterResult, NOISE_LABEL
from src.clustering.cluster_summary import FLAG_LOW_COHESION, summarize_clusters
from src.clustering.label_clusters import build_keyword_corpus
from src.clustering.taxonomy_card import (
    CONFIDENCE_HIGH,
    CONFIDENCE_LOW,
    get_candidates_for_run,
)
from src.clustering.taxonomy_draft import (
    OUTCOME_PROMOTED,
    OUTCOME_PROMOTED_LOW_CONFIDENCE,
    OUTCOME_UNMAPPED,
    REASON_BELOW_MIN_DOCS,
    REASON_BELOW_MIN_SHARE,
    REASON_NOISE,
    REASON_OUTSIDE_TOP_N,
    REASON_UNLABELED,
    assess_clusters,
    assign_domain_id,
    build_draft_taxonomy,
    slugify_domain_name,
)
from src.common.db import connection_scope, init_schema
from src.embedding.embed_model import DeterministicHashEmbedder
from src.extraction.doc_representation import DocumentRepresentation
from src.extraction.representation_store import (
    DEFAULT_REPRESENTATION_SCHEMA_FILE,
    upsert_representations,
)

DOMAIN_REGISTRY_SCHEMA_FILE = "schemas/domain_registry_schema.sql"


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _rep(doc_id: str, topic: str, **kw) -> DocumentRepresentation:
    base = dict(
        doc_id=doc_id,
        source_uri=f"/cases/{doc_id}",
        source_type="case_html",
        source_relpath=f"{topic}/{doc_id}",
        title=f"{topic} matter {doc_id}",
        headings=["Facts", "Order"],
        body_preview=f"{topic} judgment concerning {topic} proceedings. " * 6,
        char_count=400,
        court="Lahore High Court",
        decision_date="2020-01-01",
    )
    base.update(kw)
    return DocumentRepresentation(**base)


class _NamingLLM:
    """Names a cluster after whichever topic marker shows up in the prompt."""

    def __init__(self, names: dict[str, str], fail_for: set[str] | None = None):
        self.names = names
        self.fail_for = fail_for or set()
        self.calls = 0

    def complete(self, system, prompt, max_tokens=None):
        self.calls += 1
        lowered = prompt.lower()
        for marker in self.fail_for:
            if marker.lower() in lowered:
                raise RuntimeError("LLM unavailable")
        for marker, name in self.names.items():
            if marker.lower() in lowered:
                return json.dumps(
                    {
                        "name": name,
                        "description": f"Cases about {name.lower()}.",
                        "inclusion_criteria": [f"Involves {marker}"],
                        "exclusion_criteria": ["Anything else"],
                    }
                )
        return json.dumps(
            {"name": "Unknown", "description": "", "inclusion_criteria": [], "exclusion_criteria": []}
        )


def _review_for(groups: dict[str, int], noise: int = 0, flags_on: int | None = None):
    """Build (review, doc_ids_by_cluster, documents_by_id) for synthetic clusters."""

    reps, labels, doc_ids = [], [], []
    for cluster_id, (topic, n) in enumerate(groups.items()):
        for i in range(n):
            rep = _rep(f"{topic}_{i}", topic)
            reps.append(rep)
            doc_ids.append(rep.doc_id)
            labels.append(cluster_id)
    for i in range(noise):
        rep = _rep(f"noise_{i}", "oddity")
        reps.append(rep)
        doc_ids.append(rep.doc_id)
        labels.append(NOISE_LABEL)

    labels_arr = np.array(labels, dtype=int)
    # Give the flagged cluster genuinely scattered vectors so the review
    # flags it the same way it would in production.
    vectors = []
    rng = np.random.default_rng(3)
    for label in labels:
        if flags_on is not None and label == flags_on:
            vectors.append(rng.normal(size=6).astype(np.float32))
        else:
            v = np.zeros(6, dtype=np.float32)
            v[(label + 1) % 6] = 1.0
            vectors.append(v)
    vectors = np.stack(vectors)

    review = summarize_clusters(
        run_id="t",
        doc_ids=doc_ids,
        labels=labels_arr,
        probabilities=np.where(labels_arr == NOISE_LABEL, 0.0, 0.95),
        vectors=vectors,
        documents_by_id={r.doc_id: r for r in reps},
        keyword_corpus=build_keyword_corpus(reps),
        # These fixtures deliberately use lopsided cluster sizes to exercise
        # the eligibility gates; the catch-all-size flag has its own tests in
        # tests/test_cluster_review.py and would otherwise fire on every one.
        mixed_max_share=1.0,
    )
    doc_ids_by_cluster: dict[int, list[str]] = {}
    for doc_id, label in zip(doc_ids, labels):
        doc_ids_by_cluster.setdefault(label, []).append(doc_id)
    return review, doc_ids_by_cluster, {r.doc_id: r for r in reps}, reps


# ---------------------------------------------------------------------------
# domain ids
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("Criminal Appeals", "criminal_appeals"),
        ("Landlord-Tenant Ejectment", "landlord_tenant_ejectment"),
        ("Tax & Revenue (Inland)", "tax_revenue_inland"),
        ("  spaced  out  ", "spaced_out"),
        ("", "domain"),
    ],
)
def test_slugify_domain_name(name, expected):
    assert slugify_domain_name(name) == expected


def test_domain_ids_are_unique_within_a_card():
    taken: set[str] = set()

    first = assign_domain_id("Criminal Appeals", 1, taken)
    second = assign_domain_id("Criminal Appeals", 4, taken)

    assert first == "criminal_appeals"
    assert second == "criminal_appeals_c4"  # disambiguated by cluster, not a counter


# ---------------------------------------------------------------------------
# eligibility gates
# ---------------------------------------------------------------------------


def test_small_cluster_is_never_drafted_as_a_domain():
    """Requirement: do not invent a domain from a handful of odd cases."""

    review, *_ = _review_for({"criminal": 40, "oddity": 3})

    decisions = {d.cluster_id: d for d in assess_clusters(review, min_domain_docs=15)}

    assert decisions[0].outcome == OUTCOME_PROMOTED
    assert decisions[1].outcome == OUTCOME_UNMAPPED
    assert any(r.startswith(REASON_BELOW_MIN_DOCS) for r in decisions[1].reasons)


def test_cluster_below_the_share_floor_is_not_drafted():
    review, *_ = _review_for({"criminal": 200, "rare": 16})

    decisions = {
        d.cluster_id: d
        for d in assess_clusters(review, min_domain_docs=15, min_domain_share=0.1)
    }

    assert decisions[1].outcome == OUTCOME_UNMAPPED
    assert any(r.startswith(REASON_BELOW_MIN_SHARE) for r in decisions[1].reasons)


def test_only_the_largest_max_domains_clusters_are_drafted():
    review, *_ = _review_for({"a": 60, "b": 50, "c": 40, "d": 30})

    decisions = {
        d.cluster_id: d
        for d in assess_clusters(review, min_domain_docs=15, min_domain_share=0.0, max_domains=2)
    }

    assert [decisions[i].outcome for i in range(4)] == [
        OUTCOME_PROMOTED, OUTCOME_PROMOTED, OUTCOME_UNMAPPED, OUTCOME_UNMAPPED,
    ]
    assert any(r.startswith(REASON_OUTSIDE_TOP_N) for r in decisions[2].reasons)


def test_noise_is_never_a_domain_candidate():
    review, *_ = _review_for({"criminal": 40}, noise=25)

    decisions = {d.cluster_id: d for d in assess_clusters(review)}

    assert decisions[NOISE_LABEL].outcome == OUTCOME_UNMAPPED
    assert decisions[NOISE_LABEL].reasons == [REASON_NOISE]


def test_flagged_cluster_is_drafted_but_marked_low_confidence():
    review, *_ = _review_for({"criminal": 40, "mixedbag": 30}, flags_on=1)
    assert FLAG_LOW_COHESION in {f for s in review.summaries for f in s.flags}

    decisions = {d.cluster_id: d for d in assess_clusters(review)}

    assert decisions[0].outcome == OUTCOME_PROMOTED
    assert decisions[1].outcome == OUTCOME_PROMOTED_LOW_CONFIDENCE


# ---------------------------------------------------------------------------
# drafting
# ---------------------------------------------------------------------------


def test_draft_taxonomy_produces_full_domain_definitions():
    review, by_cluster, by_id, reps = _review_for({"criminal": 40, "taxation": 30})
    llm = _NamingLLM({"criminal": "Criminal Appeals", "taxation": "Tax References"})

    result = build_draft_taxonomy(
        review=review,
        doc_ids_by_cluster=by_cluster,
        documents_by_id=by_id,
        keyword_corpus=build_keyword_corpus(reps),
        llm_client=llm,
    )

    assert llm.calls == 2  # one call per eligible cluster
    assert [d.domain_id for d in result.domains] == ["criminal_appeals", "tax_references"]
    for domain in result.domains:
        assert domain.domain_id and domain.name and domain.description
        assert domain.inclusion_criteria and domain.exclusion_criteria
        assert domain.doc_count > 0
        assert domain.confidence == CONFIDENCE_HIGH
        assert domain.review_required is False
        assert domain.representative_doc_ids


def test_low_confidence_domain_carries_flags_and_review_notes():
    review, by_cluster, by_id, reps = _review_for(
        {"criminal": 40, "mixedbag": 30}, flags_on=1
    )
    llm = _NamingLLM({"criminal": "Criminal Appeals", "mixedbag": "Rent Matters"})

    result = build_draft_taxonomy(
        review=review,
        doc_ids_by_cluster=by_cluster,
        documents_by_id=by_id,
        keyword_corpus=build_keyword_corpus(reps),
        llm_client=llm,
    )

    flagged = next(d for d in result.domains if d.domain_id == "rent_matters")
    assert flagged.confidence == CONFIDENCE_LOW
    assert flagged.review_required is True
    assert flagged.flags == [FLAG_LOW_COHESION]
    assert any("possibly mixed" in n for n in flagged.notes)
    assert result.audit.low_confidence_domain_ids == ["rent_matters"]


def test_cluster_the_llm_cannot_name_does_not_become_a_domain():
    review, by_cluster, by_id, reps = _review_for({"criminal": 40, "taxation": 30})
    llm = _NamingLLM({"criminal": "Criminal Appeals"}, fail_for={"taxation"})

    result = build_draft_taxonomy(
        review=review,
        doc_ids_by_cluster=by_cluster,
        documents_by_id=by_id,
        keyword_corpus=build_keyword_corpus(reps),
        llm_client=llm,
    )

    assert [d.domain_id for d in result.domains] == ["criminal_appeals"]
    decision = next(d for d in result.audit.decisions if d.cluster_id == 1)
    assert decision.outcome == OUTCOME_UNMAPPED
    assert any(r.startswith(REASON_UNLABELED) for r in decision.reasons)
    # ... and its documents are accounted for rather than dropped.
    assert result.other_bucket.doc_count == 30


def test_every_document_is_accounted_for_across_domains_and_other():
    review, by_cluster, by_id, reps = _review_for(
        {"criminal": 40, "taxation": 30, "tiny": 5}, noise=12
    )
    llm = _NamingLLM({"criminal": "Criminal Appeals", "taxation": "Tax References"})

    result = build_draft_taxonomy(
        review=review,
        doc_ids_by_cluster=by_cluster,
        documents_by_id=by_id,
        keyword_corpus=build_keyword_corpus(reps),
        llm_client=llm,
    )

    total = sum(d.doc_count for d in result.domains) + result.other_bucket.doc_count
    assert total == review.total_documents == 87
    assert result.audit.noise_doc_count == 12
    assert result.audit.unmapped_doc_count == 17  # 5 too-small + 12 noise
    assert result.audit.unmapped_cluster_ids == [2]  # noise is reported separately


def test_audit_records_a_decision_for_every_cluster():
    review, by_cluster, by_id, reps = _review_for({"a": 40, "b": 30, "c": 4}, noise=6)
    llm = _NamingLLM({"a": "Alpha", "b": "Beta"})

    result = build_draft_taxonomy(
        review=review,
        doc_ids_by_cluster=by_cluster,
        documents_by_id=by_id,
        keyword_corpus=build_keyword_corpus(reps),
        llm_client=llm,
    )

    assert {d.cluster_id for d in result.audit.decisions} == {0, 1, 2, NOISE_LABEL}
    assert all(d.reasons or d.outcome == OUTCOME_PROMOTED for d in result.audit.decisions)
    assert "config/domains.yaml is untouched" in " ".join(result.audit.notes)


# ---------------------------------------------------------------------------
# the flow: persistence, and what it must not do
# ---------------------------------------------------------------------------


@pytest.fixture()
def taxonomy_db(tmp_path) -> Path:
    db_path = tmp_path / "metadata.db"
    init_schema(db_path=db_path, schema_file=str(DEFAULT_REPRESENTATION_SCHEMA_FILE))
    init_schema(db_path=db_path, schema_file=DOMAIN_REGISTRY_SCHEMA_FILE)
    return db_path


@pytest.fixture()
def taxonomy_settings(tmp_path, monkeypatch):
    import orchestration.dags.taxonomy_draft_flow as flow

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
            title_weight=2.0, toc_weight=1.5, body_weight=1.0,
            body_chunk_chars=2000, max_body_chunks=4,
            umap_n_components=4, umap_n_neighbors=3, umap_min_dist=0.0,
            umap_metric="cosine", umap_min_docs=10_000,
            hdbscan_min_cluster_size=3, hdbscan_min_samples=None,
            hdbscan_metric="euclidean",
            top_n_domains=15,
            representative_docs_per_cluster=3,
            keywords_per_cluster=10,
            llm_model="fake-model", llm_max_tokens=512,
            taxonomy_output_dir=tmp_path / "taxonomy",
            taxonomy_min_domain_docs=10,
            taxonomy_min_domain_share=0.02,
            taxonomy_uncertain_membership_probability=0.5,
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


def _two_topic_clusterer(vectors: np.ndarray) -> ClusterResult:
    n = vectors.shape[0]
    labels = np.array([0 if i < 20 else 1 if i < 35 else NOISE_LABEL for i in range(n)])
    return ClusterResult(
        labels=labels,
        probabilities=np.where(labels == NOISE_LABEL, 0.0, 0.9),
        n_clusters=2,
    )


def _seed_corpus(db_path: Path) -> None:
    reps = (
        [_rep(f"criminal_{i:02d}", "criminal") for i in range(20)]
        + [_rep(f"taxation_{i:02d}", "taxation") for i in range(15)]
        + [_rep(f"oddity_{i:02d}", "oddity") for i in range(5)]
    )
    upsert_representations(reps, db_path=db_path)


def test_flow_writes_a_draft_card_and_draft_candidates(taxonomy_db, taxonomy_settings):
    _seed_corpus(taxonomy_db)

    result = run_taxonomy_draft(
        run_id="taxonomy_run",
        db_path=taxonomy_db,
        embedder=DeterministicHashEmbedder(dimension=16),
        clusterer=_two_topic_clusterer,
        llm_client=_NamingLLM({"criminal": "Criminal Appeals", "taxation": "Tax References"}),
    )

    assert result.is_frozen is False
    assert result.card.status == "draft"
    assert [d.domain_id for d in result.domains] == ["criminal_appeals", "tax_references"]

    card = json.loads(Path(result.card_path).read_text())
    assert card["status"] == "draft"
    assert card["audit"]["noise_doc_count"] == 5
    assert card["other_bucket"]["doc_count"] == 5
    for domain in card["domains"]:
        assert domain["domain_id"] and domain["inclusion_criteria"] and domain["exclusion_criteria"]

    rows = get_candidates_for_run("taxonomy_run", db_path=taxonomy_db)
    assert len(rows) == 3  # 2 domains + Other/Uncertain
    assert all(r["status"] == "draft" for r in rows)
    assert {r["domain_id"] for r in rows} == {
        "criminal_appeals", "tax_references", "other_uncertain",
    }


def test_flow_does_not_freeze_the_registry(taxonomy_db, taxonomy_settings):
    """config/domains.yaml is the frozen registry; this stage never writes it."""

    registry = Path("config/domains.yaml")
    before = registry.read_bytes() if registry.exists() else None
    _seed_corpus(taxonomy_db)

    run_taxonomy_draft(
        run_id="taxonomy_run",
        db_path=taxonomy_db,
        embedder=DeterministicHashEmbedder(dimension=16),
        clusterer=_two_topic_clusterer,
        llm_client=_NamingLLM({"criminal": "Criminal Appeals", "taxation": "Tax References"}),
    )

    after = registry.read_bytes() if registry.exists() else None
    assert after == before

    with connection_scope(taxonomy_db) as conn:
        statuses = {
            r["status"]
            for r in conn.execute("SELECT DISTINCT status FROM domain_candidates")
        }
    assert statuses == {"draft"}  # nothing accepted, nothing promoted


def test_flow_reruns_idempotently_for_a_run(taxonomy_db, taxonomy_settings):
    _seed_corpus(taxonomy_db)
    kwargs = dict(
        run_id="taxonomy_run",
        db_path=taxonomy_db,
        embedder=DeterministicHashEmbedder(dimension=16),
        clusterer=_two_topic_clusterer,
    )

    first = run_taxonomy_draft(
        llm_client=_NamingLLM({"criminal": "Criminal Appeals", "taxation": "Tax References"}),
        **kwargs,
    )
    second = run_taxonomy_draft(
        llm_client=_NamingLLM({"criminal": "Criminal Appeals", "taxation": "Tax References"}),
        **kwargs,
    )

    assert [d.domain_id for d in first.domains] == [d.domain_id for d in second.domains]
    assert len(get_candidates_for_run("taxonomy_run", db_path=taxonomy_db)) == 3


def test_flow_handles_an_empty_corpus(taxonomy_db, taxonomy_settings):
    result = run_taxonomy_draft(
        run_id="empty_run",
        db_path=taxonomy_db,
        embedder=DeterministicHashEmbedder(dimension=16),
        clusterer=_two_topic_clusterer,
        llm_client=_NamingLLM({}),
    )

    assert result.domains == []
    assert result.card.status == "draft"


def test_persisted_columns_are_added_to_an_older_database(tmp_path):
    """A database created before the review columns existed still upserts."""

    db_path = tmp_path / "old.db"
    legacy_schema = """
    CREATE TABLE domain_candidates (
        candidate_id TEXT PRIMARY KEY, run_id TEXT NOT NULL, cluster_id INTEGER,
        is_other_bucket INTEGER NOT NULL DEFAULT 0, name TEXT, description TEXT,
        inclusion_criteria TEXT, exclusion_criteria TEXT,
        keywords_json TEXT NOT NULL DEFAULT '[]',
        representative_doc_ids_json TEXT NOT NULL DEFAULT '[]',
        doc_count INTEGER NOT NULL DEFAULT 0, sample_size INTEGER NOT NULL DEFAULT 0,
        status TEXT NOT NULL DEFAULT 'draft',
        created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
    );
    """
    with connection_scope(db_path) as conn:
        conn.executescript(legacy_schema)

    from src.clustering.taxonomy_card import (
        build_taxonomy_card,
        build_other_bucket_draft,
        DomainDraft,
        persist_taxonomy_card,
    )

    card = build_taxonomy_card(
        run_id="legacy_run",
        sample_size=10,
        total_signatures=10,
        embedding_model="fake",
        domains=[
            DomainDraft(
                cluster_id=0, name="Criminal Appeals", description="d",
                doc_count=10, sample_size=10, domain_id="criminal_appeals",
            )
        ],
        other_bucket=build_other_bucket_draft([], sample_size=10),
    )

    assert persist_taxonomy_card(card, db_path=db_path) == 2
    rows = get_candidates_for_run("legacy_run", db_path=db_path)
    assert {r["domain_id"] for r in rows} == {"criminal_appeals", "other_uncertain"}
