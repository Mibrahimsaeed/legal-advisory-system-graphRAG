from __future__ import annotations

from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from src.clustering.cluster import ClusterResult, NOISE_LABEL
from src.clustering.taxonomy_card import (
    DomainDraft,
    build_other_bucket_draft,
    build_taxonomy_card,
    get_candidates_for_run,
    new_run_id,
    persist_taxonomy_card,
    write_taxonomy_card_json,
)
from src.common.db import connection_scope, init_schema
from src.common.llm_client import LLMError, extract_json_object
from src.extraction.signature import DocumentSignature
from src.extraction.signature_store import upsert_signatures
from src.ingestion import manifest as manifest_db

MANIFEST_SCHEMA_FILE = "schemas/manifest_schema.sql"
SIGNATURE_SCHEMA_FILE = "schemas/signature_schema.sql"
DOMAIN_REGISTRY_SCHEMA_FILE = "schemas/domain_registry_schema.sql"


# ---------------------------------------------------------------------------
# llm_client.py
# ---------------------------------------------------------------------------


def test_extract_json_object_plain():
    assert extract_json_object('{"a": 1}') == {"a": 1}


def test_extract_json_object_fenced():
    text = '```json\n{"name": "Contracts"}\n```'
    assert extract_json_object(text) == {"name": "Contracts"}


def test_extract_json_object_prose_wrapped():
    text = 'Here you go:\n{"name": "Litigation"}\nLet me know if you need anything else.'
    assert extract_json_object(text) == {"name": "Litigation"}


def test_extract_json_object_raises_on_missing_json():
    with pytest.raises(LLMError):
        extract_json_object("no json in here at all")


def test_extract_json_object_raises_on_malformed_json():
    with pytest.raises(LLMError):
        extract_json_object("{this is not valid json}")


# ---------------------------------------------------------------------------
# taxonomy_card.py
# ---------------------------------------------------------------------------


@pytest.fixture()
def db_path(tmp_path) -> Path:
    p = tmp_path / "metadata.db"
    init_schema(db_path=p, schema_file=MANIFEST_SCHEMA_FILE)
    init_schema(db_path=p, schema_file=SIGNATURE_SCHEMA_FILE)
    init_schema(db_path=p, schema_file=DOMAIN_REGISTRY_SCHEMA_FILE)
    return p


def test_write_taxonomy_card_json(tmp_path):
    domains = [
        DomainDraft(cluster_id=0, name="Contracts", description="d", doc_count=10, sample_size=100)
    ]
    other = build_other_bucket_draft(["o1", "o2"], sample_size=100)
    card = build_taxonomy_card(
        run_id="run_x", sample_size=100, total_signatures=1000,
        embedding_model="fake-model", domains=domains, other_bucket=other,
    )
    path = write_taxonomy_card_json(card, output_dir=tmp_path)
    assert path.exists()
    import json

    loaded = json.loads(path.read_text())
    assert loaded["run_id"] == "run_x"
    assert loaded["domains"][0]["name"] == "Contracts"
    assert loaded["other_bucket"]["doc_count"] == 2


def test_persist_taxonomy_card_round_trip(db_path):
    domains = [
        DomainDraft(cluster_id=0, name="Contracts", description="d", doc_count=10, sample_size=50),
        DomainDraft(cluster_id=1, name="Litigation", description="d2", doc_count=8, sample_size=50),
    ]
    other = build_other_bucket_draft(["o1", "o2", "o3"], sample_size=50)
    run_id = new_run_id()
    card = build_taxonomy_card(
        run_id=run_id, sample_size=50, total_signatures=500,
        embedding_model="fake-model", domains=domains, other_bucket=other,
    )
    n = persist_taxonomy_card(card, db_path=db_path)
    assert n == 3

    rows = get_candidates_for_run(run_id, db_path=db_path)
    assert len(rows) == 3
    names = {r["name"] for r in rows}
    assert names == {"Contracts", "Litigation", "Other / Uncertain"}
    assert all(r["status"] == "draft" for r in rows)


def test_persist_taxonomy_card_is_idempotent_per_run(db_path):
    domains = [DomainDraft(cluster_id=0, name="Contracts", description="d", doc_count=5, sample_size=20)]
    other = build_other_bucket_draft(["o1"], sample_size=20)
    run_id = new_run_id()
    card = build_taxonomy_card(
        run_id=run_id, sample_size=20, total_signatures=200,
        embedding_model="fake-model", domains=domains, other_bucket=other,
    )
    persist_taxonomy_card(card, db_path=db_path)
    persist_taxonomy_card(card, db_path=db_path)  # re-run, e.g. checkpoint resume

    with connection_scope(db_path) as conn:
        count = conn.execute(
            "SELECT COUNT(*) AS c FROM domain_candidates WHERE run_id = ?", (run_id,)
        ).fetchone()["c"]
    assert count == 2  # 1 domain + other bucket, not duplicated


# ---------------------------------------------------------------------------
# orchestration/dags/domain_discovery_flow.py -- full pipeline with fakes
# ---------------------------------------------------------------------------


def _make_signature(doc_id: str, group: str, is_noise: bool = False) -> DocumentSignature:
    marker = group if not is_noise else doc_id
    return DocumentSignature(
        doc_id=doc_id,
        source_uri=f"s3://x/{doc_id}.pdf",
        signature_hash=f"h_{doc_id}",
        is_scanned=False,
        extraction_status="ok",
        extractor_used="pdfplumber",
        title=f"{marker} Document",
        toc=[f"{marker} Section 1", f"{marker} Section 2"],
        body_preview=f"{marker} boilerplate text repeated many times. " * 20,
        pages_used=3,
        char_count=1200,
        quality_score=0.9,
    )


class _GroupedFakeEmbedder:
    """Maps each embedding input's text to a group centroid + small jitter,
    so a *toy* clustering algorithm can recover the synthetic groups below
    without needing sentence-transformers installed."""

    dimension = 12

    def __init__(self, groups: list[str]):
        self.groups = groups
        rng = np.random.default_rng(42)
        self._centroids = {g: rng.normal(size=self.dimension) * 5 for g in groups}

    def encode(self, texts: list[str]) -> np.ndarray:
        rng = np.random.default_rng(123)
        out = []
        for text in texts:
            matched = next((g for g in self.groups if g in text), None)
            base = (
                self._centroids[matched]
                if matched is not None
                else rng.normal(size=self.dimension) * 50  # far from every centroid -> noise
            )
            jitter = rng.normal(size=self.dimension) * 0.05
            out.append((base + jitter).astype(np.float32))
        return np.stack(out)


def _fake_union_find_clusterer(vectors: np.ndarray, eps: float = 1.0, min_cluster_size: int = 3) -> ClusterResult:
    """A deliberately simple density clusterer (union-find on a distance
    threshold) -- not HDBSCAN, but a real, working algorithm, used here
    purely so the rest of the pipeline (rank/label/persist/checkpoint) can
    be exercised end-to-end without the ``hdbscan`` package installed."""

    n = vectors.shape[0]
    parent = list(range(n))

    def find(x: int) -> int:
        while parent[x] != x:
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    for i in range(n):
        for j in range(i + 1, n):
            if np.linalg.norm(vectors[i] - vectors[j]) < eps:
                union(i, j)

    roots = [find(i) for i in range(n)]
    root_counts = Counter(roots)

    root_to_label: dict[int, int] = {}
    next_label = 0
    labels = []
    for root in roots:
        if root_counts[root] < min_cluster_size:
            labels.append(NOISE_LABEL)
            continue
        if root not in root_to_label:
            root_to_label[root] = next_label
            next_label += 1
        labels.append(root_to_label[root])

    labels_arr = np.array(labels, dtype=int)
    n_clusters = len(set(labels_arr.tolist()) - {NOISE_LABEL})
    return ClusterResult(labels=labels_arr, probabilities=None, n_clusters=n_clusters)


class _KeywordAwareFakeLLMClient:
    """Names a cluster from whichever group marker shows up in the prompt --
    good enough to verify labeling wiring without a real API call."""

    def __init__(self, group_names: dict[str, str]):
        self.group_names = group_names

    def complete(self, system, prompt, max_tokens=None):
        for marker, name in self.group_names.items():
            if marker.lower() in prompt.lower():
                return (
                    f'{{"name": "{name}", "description": "test", '
                    f'"inclusion_criteria": ["x"], "exclusion_criteria": ["y"]}}'
                )
        return '{"name": "Unknown", "description": "", "inclusion_criteria": [], "exclusion_criteria": []}'


@pytest.fixture()
def discovery_settings(tmp_path) -> SimpleNamespace:
    """Settings for the LEGACY PDF/book corpus source.

    These tests cover the signature path, which is no longer the default:
    ``corpus_source='signatures'`` is what keeps ``document_signatures``
    readable by Stage 1.2 now that case-law representations are the
    default corpus (see tests/test_caselaw.py for the active path).
    """

    return SimpleNamespace(
        database=SimpleNamespace(schema_file=Path(MANIFEST_SCHEMA_FILE)),
        pipeline=SimpleNamespace(checkpoint_dir=tmp_path / "checkpoints"),
        scratch=SimpleNamespace(root=tmp_path / "scratch"),
        metrics=SimpleNamespace(db_path=tmp_path / "metrics.db"),
        discovery=SimpleNamespace(
            corpus_source="signatures",
            embedding_model_name="fake-model",
            embedding_batch_size=32,
            embedding_doc_batch_size=256,
            title_weight=2.0,
            toc_weight=1.5,
            body_weight=1.0,
            body_chunk_chars=2000,
            max_body_chunks=4,
            umap_n_components=8,
            umap_n_neighbors=5,
            umap_min_dist=0.0,
            umap_metric="cosine",
            umap_min_docs=5,
            hdbscan_min_cluster_size=3,
            hdbscan_min_samples=None,
            hdbscan_metric="euclidean",
            top_n_domains=3,
            representative_docs_per_cluster=3,
            keywords_per_cluster=10,
            llm_model="claude-sonnet-5",
            llm_max_tokens=512,
            taxonomy_output_dir=tmp_path / "taxonomy",
            domain_registry_schema_file=Path(DOMAIN_REGISTRY_SCHEMA_FILE),
        ),
    )

@pytest.fixture()
def populated_db(tmp_path) -> Path:
    db_path = tmp_path / "metadata.db"
    init_schema(db_path=db_path, schema_file=MANIFEST_SCHEMA_FILE)
    init_schema(db_path=db_path, schema_file=SIGNATURE_SCHEMA_FILE)
    init_schema(db_path=db_path, schema_file=DOMAIN_REGISTRY_SCHEMA_FILE)

    groups = {"ALPHACONTRACT": 20, "BETALITIGATION": 12, "GAMMAIP": 8}
    signatures = []
    entries = []
    for group, count in groups.items():
        for i in range(count):
            sig = _make_signature(f"{group.lower()}_{i}", group)
            signatures.append(sig)
            entries.append({"doc_id": sig.doc_id, "source_uri": sig.source_uri, "checksum": "x"})

    for i in range(6):
        sig = _make_signature(f"noise_{i}", group=f"UNIQUE{i}", is_noise=True)
        signatures.append(sig)
        entries.append({"doc_id": sig.doc_id, "source_uri": sig.source_uri, "checksum": "x"})

    manifest_db.register_documents(entries, db_path=db_path)
    upsert_signatures(signatures, db_path=db_path)
    return db_path


def test_run_stage1_2_end_to_end(monkeypatch, populated_db, discovery_settings):
    import orchestration.dags.domain_discovery_flow as flow

    monkeypatch.setattr(flow, "get_settings", lambda: discovery_settings)

    embedder = _GroupedFakeEmbedder(groups=["ALPHACONTRACT", "BETALITIGATION", "GAMMAIP"])
    llm_client = _KeywordAwareFakeLLMClient(
        {
            "alphacontract": "Commercial Contracts",
            "betalitigation": "Litigation",
            "gammaip": "Intellectual Property",
        }
    )

    result = flow.run_stage1_2(
        run_id="test_run_pytest",
        db_path=populated_db,
        embedder=embedder,
        clusterer=_fake_union_find_clusterer,
        llm_client=llm_client,
    )
    # No sampling: the full corpus registered in populated_db
# (20 + 12 + 8 grouped docs + 6 noise docs = 46)
# is loaded and clustered in one pass.
    assert result.sample_size == result.total_signatures == 46
    assert len(result.domains) == 3
    names = {d.name for d in result.domains}
    assert names == {"Commercial Contracts", "Litigation", "Intellectual Property"}

    counts = [d.doc_count for d in result.domains]
    assert counts == sorted(counts, reverse=True)
    assert all(c > 0 for c in counts)
    assert result.other_bucket is not None
    assert result.other_bucket.doc_count > 0

    total = sum(counts) + result.other_bucket.doc_count
    assert total == result.sample_size

    assert result.taxonomy_card_path is not None
    assert result.taxonomy_card_path.exists()

    # scratch artifacts are cleaned up once the run completes
    artifact_dir = discovery_settings.scratch.root / "discovery" / "test_run_pytest"
    assert not artifact_dir.exists()

    rows = get_candidates_for_run("test_run_pytest", db_path=populated_db)
    assert len(rows) == 4  # 3 domains + other bucket
    assert all(r["status"] == "draft" for r in rows)


def test_run_stage1_2_end_to_end(monkeypatch, populated_db, discovery_settings):
    import orchestration.dags.domain_discovery_flow as flow

    monkeypatch.setattr(flow, "get_settings", lambda: discovery_settings)

    embedder = _GroupedFakeEmbedder(
        groups=["ALPHACONTRACT", "BETALITIGATION", "GAMMAIP"]
    )
    llm_client = _KeywordAwareFakeLLMClient(
        {
            "alphacontract": "Commercial Contracts",
            "betalitigation": "Litigation",
            "gammaip": "Intellectual Property",
        }
    )

    result = flow.run_stage1_2(
        run_id="test_run_pytest",
        db_path=populated_db,
        embedder=embedder,
        clusterer=_fake_union_find_clusterer,
        llm_client=llm_client,
    )

    assert result.sample_size == result.total_signatures == 46
    assert len(result.domains) == 3

    names = {d.name for d in result.domains}
    assert names == {
        "Commercial Contracts",
        "Litigation",
        "Intellectual Property",
    }

    counts = [d.doc_count for d in result.domains]
    assert counts == sorted(counts, reverse=True)
    assert all(c > 0 for c in counts)

    # The Other bucket always exists, but it may legitimately be empty
    # if every sampled document belongs to one of the top clusters.
    assert result.other_bucket is not None

    # Every sampled document must be accounted for.
    assert (
        sum(d.doc_count for d in result.domains)
        + result.other_bucket.doc_count
        == result.sample_size
    )

    assert result.taxonomy_card_path is not None
    assert result.taxonomy_card_path.exists()

    # Scratch artifacts are cleaned up once the run completes.
    artifact_dir = (
        discovery_settings.scratch.root / "discovery" / "test_run_pytest"
    )
    assert not artifact_dir.exists()

    rows = get_candidates_for_run("test_run_pytest", db_path=populated_db)

    # Three discovered domains + the Other bucket.
    assert len(rows) == 4
    assert all(r["status"] == "draft" for r in rows)

     
def test_run_stage1_2_handles_empty_signature_store(monkeypatch, tmp_path, discovery_settings):
    import orchestration.dags.domain_discovery_flow as flow

    monkeypatch.setattr(flow, "get_settings", lambda: discovery_settings)

    db_path = tmp_path / "empty.db"
    init_schema(db_path=db_path, schema_file=MANIFEST_SCHEMA_FILE)
    init_schema(db_path=db_path, schema_file=SIGNATURE_SCHEMA_FILE)
    init_schema(db_path=db_path, schema_file=DOMAIN_REGISTRY_SCHEMA_FILE)

    result = flow.run_stage1_2(run_id="empty_run", db_path=db_path)
    assert result.sample_size == 0
    assert result.domains == []
    