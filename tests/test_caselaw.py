"""Phase 1: the case-law path runs end to end without any signature.

Covers, in order:

1. ``case.html`` + ``metadata.json`` -> DocumentRepresentation (no
   signature built, stored or required).
2. The representation store round-trips those records through SQLite.
3. A case folder enters the pipeline via ``run_case_ingest``.
4. The existing classification embeddings still work on representations.
5. The existing UMAP -> HDBSCAN input still works on those embeddings.
6. Full Stage 1.2 domain discovery over a case-law corpus.
7. No production module on the case-law path imports signature code.
"""

from __future__ import annotations

import json
import subprocess
import sys
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from orchestration.dags.case_ingest_flow import run_case_ingest
from src.clustering.cluster import ClusterResult, NOISE_LABEL, hdbscan_clusterer
from src.clustering.label_clusters import build_keyword_corpus, extract_cluster_keywords
from src.clustering.reduce import reduce_dimensions
from src.clustering.taxonomy_card import get_candidates_for_run
from src.common.db import init_schema
from src.embedding.doc_pooling import embed_documents
from src.embedding.embed_model import DeterministicHashEmbedder
from src.extraction.case_loader import (
    doc_id_for_case,
    iter_case_folders,
    load_case_folder,
    parse_case_html,
)
from src.extraction.doc_representation import (
    DocumentRepresentation,
    EmbeddableDocument,
)
from src.extraction.representation_store import (
    DEFAULT_REPRESENTATION_SCHEMA_FILE,
    get_representation,
    list_representations,
    upsert_representations,
)

MANIFEST_SCHEMA_FILE = "schemas/manifest_schema.sql"
DOMAIN_REGISTRY_SCHEMA_FILE = "schemas/domain_registry_schema.sql"
REPRESENTATION_SCHEMA_FILE = str(DEFAULT_REPRESENTATION_SCHEMA_FILE)

try:
    import hdbscan as _hdbscan  # noqa: F401

    HDBSCAN_AVAILABLE = True
except ImportError:
    HDBSCAN_AVAILABLE = False


# ---------------------------------------------------------------------------
# fixtures / helpers
# ---------------------------------------------------------------------------


def _case_html(title: str, headings: list[str], body: str) -> str:
    heading_markup = "\n".join(f"<h2>{h}</h2>" for h in headings)
    return f"""<!DOCTYPE html>
<html>
<head><title>{title} (portal)</title>
<style>.hidden {{ display: none; }}</style>
<script>var tracking = 1;</script>
</head>
<body>
<h1>{title}</h1>
{heading_markup}
<p>{body}</p>
<p>&amp; the appeal is disposed of accordingly.</p>
</body>
</html>
"""


def _write_case(
    root: Path,
    folder_name: str,
    title: str,
    headings: list[str],
    body: str,
    metadata: dict | None = None,
) -> Path:
    folder = root / folder_name
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "case.html").write_text(_case_html(title, headings, body), encoding="utf-8")
    if metadata is not None:
        (folder / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
    return folder


@pytest.fixture()
def case_corpus(tmp_path) -> Path:
    """A small on-disk corpus: two cases, one with metadata, one without."""

    root = tmp_path / "cases"
    _write_case(
        root,
        "2019/SC/123",
        "Muhammad Aslam v. The State",
        ["Facts", "Arguments of Counsel", "Findings"],
        "The appellant was convicted under section 302 of the Pakistan Penal Code.",
        metadata={
            "case_title": "Muhammad Aslam v. The State",
            "court": "Supreme Court of Pakistan",
            "date": "2019-04-11",
            "citation": "2019 SCMR 123",
        },
    )
    _write_case(
        root,
        "2020/LHC/77",
        "Zainab Bibi v. Commissioner Inland Revenue",
        ["Question of Law"],
        "The reference concerns assessment of income tax for the year 2016.",
        metadata=None,
    )
    return root


@pytest.fixture()
def caselaw_db(tmp_path) -> Path:
    db_path = tmp_path / "metadata.db"
    init_schema(db_path=db_path, schema_file=MANIFEST_SCHEMA_FILE)
    init_schema(db_path=db_path, schema_file=REPRESENTATION_SCHEMA_FILE)
    init_schema(db_path=db_path, schema_file=DOMAIN_REGISTRY_SCHEMA_FILE)
    return db_path


def _rep(doc_id: str, title: str, headings: list[str], body: str) -> DocumentRepresentation:
    return DocumentRepresentation(
        doc_id=doc_id,
        source_uri=f"/cases/{doc_id}",
        source_type="case_html",
        title=title,
        headings=headings,
        body_preview=body,
        char_count=len(body),
    )


# ---------------------------------------------------------------------------
# 1. case.html + metadata.json -> lightweight representation (no signature)
# ---------------------------------------------------------------------------


def test_load_case_folder_builds_representation_without_a_signature(case_corpus):
    folder = case_corpus / "2019/SC/123"

    rep = load_case_folder(folder, root=case_corpus)

    assert rep.status == "ok"
    assert rep.source_type == "case_html"
    assert rep.title == "Muhammad Aslam v. The State"  # from metadata.json
    assert rep.headings == ["Facts", "Arguments of Counsel", "Findings"]  # <h1> == title, dropped
    assert "section 302 of the Pakistan Penal Code" in rep.body_preview
    assert rep.char_count == len(rep.body_preview)
    assert rep.metadata["court"] == "Supreme Court of Pakistan"
    assert rep.error is None

    # The whole point of Phase 1: no signature is produced or required.
    assert not hasattr(rep, "signature_hash")
    assert not hasattr(rep, "is_scanned")
    assert not hasattr(rep, "pages_used")


def test_parse_case_html_drops_script_and_style_and_decodes_entities():
    text, headings, html_title = parse_case_html(
        _case_html("Case A", ["Order"], "Body text here.")
    )

    assert "tracking" not in text  # <script> content dropped
    assert "display: none" not in text  # <style> content dropped
    assert "& the appeal is disposed of" in text  # &amp; decoded
    assert headings == ["Case A", "Order"]  # <h1> counts as a heading
    assert html_title == "Case A (portal)"


def test_load_case_folder_without_metadata_json_still_works(case_corpus):
    folder = case_corpus / "2020/LHC/77"

    rep = load_case_folder(folder, root=case_corpus)

    assert rep.status == "ok"
    assert rep.metadata == {}
    # No metadata title -> falls back to the first heading in the markup.
    assert rep.title == "Zainab Bibi v. Commissioner Inland Revenue"
    assert "income tax" in rep.body_preview


def test_load_case_folder_with_malformed_metadata_is_not_fatal(tmp_path):
    folder = _write_case(tmp_path, "bad_meta", "Case B", ["Order"], "Body text.")
    (folder / "metadata.json").write_text("{not valid json", encoding="utf-8")

    rep = load_case_folder(folder, root=tmp_path)

    assert rep.status == "ok"
    assert rep.metadata == {}
    # Phase 2 moved non-fatal problems out of `error` (reserved for
    # failures) and into `warnings`; see tests/test_case_representation.py.
    assert rep.error is None
    assert any(w.startswith("invalid_metadata_json") for w in rep.warnings)


def test_load_case_folder_missing_html_is_marked_failed(tmp_path):
    folder = tmp_path / "empty_case"
    folder.mkdir()

    rep = load_case_folder(folder, root=tmp_path)

    assert rep.status == "failed"
    assert "case.html not found" in rep.error


def test_body_preview_is_capped_but_char_count_is_not(tmp_path):
    folder = _write_case(tmp_path, "long_case", "Long Case", ["Order"], "word " * 5_000)

    rep = load_case_folder(folder, root=tmp_path, body_preview_char_limit=500)

    assert len(rep.body_preview) == 500
    assert rep.char_count > 500


def test_headings_are_deduplicated_and_capped(tmp_path):
    folder = _write_case(
        tmp_path, "many_headings", "Case C", ["A", "B", "A", "C", "D"], "Body."
    )

    rep = load_case_folder(folder, root=tmp_path, max_headings=3)

    # <h1>Case C</h1>, A, B survive the cap of 3; the repeated "A" is
    # dropped as a duplicate, and "Case C" then drops out of headings
    # because it is already the title.
    assert rep.title == "Case C"
    assert rep.headings == ["A", "B"]


def test_doc_id_is_stable_across_scans_and_unique_per_case(case_corpus):
    first = case_corpus / "2019/SC/123"
    second = case_corpus / "2020/LHC/77"

    assert doc_id_for_case(case_corpus, first) == doc_id_for_case(case_corpus, first)
    assert doc_id_for_case(case_corpus, first) != doc_id_for_case(case_corpus, second)


def test_iter_case_folders_finds_nested_cases(case_corpus):
    folders = list(iter_case_folders(case_corpus))

    assert {f.name for f in folders} == {"123", "77"}


# ---------------------------------------------------------------------------
# 2. SQL storage for representations
# ---------------------------------------------------------------------------


def test_representation_store_round_trip(caselaw_db):
    rep = _rep("case_1", "Case One", ["Facts"], "body text")

    assert upsert_representations([rep], db_path=caselaw_db) == 1

    fetched = get_representation("case_1", db_path=caselaw_db)
    assert fetched is not None
    assert fetched.title == "Case One"
    assert fetched.headings == ["Facts"]
    assert fetched.body_preview == "body text"


def test_representation_store_upsert_is_idempotent(caselaw_db):
    rep = _rep("case_1", "Case One", ["Facts"], "body text")
    upsert_representations([rep], db_path=caselaw_db)
    upsert_representations(
        [_rep("case_1", "Case One (revised)", ["Facts"], "body text")],
        db_path=caselaw_db,
    )

    all_reps = list_representations(db_path=caselaw_db)
    assert len(all_reps) == 1
    assert all_reps[0].title == "Case One (revised)"


def test_list_representations_excludes_failed_records(caselaw_db):
    ok = _rep("case_ok", "Case OK", ["Facts"], "body")
    failed = DocumentRepresentation(
        doc_id="case_bad",
        source_uri="/cases/case_bad",
        source_type="case_html",
        status="failed",
        error="case.html not found",
    )
    upsert_representations([ok, failed], db_path=caselaw_db)

    assert [r.doc_id for r in list_representations(db_path=caselaw_db)] == ["case_ok"]


# ---------------------------------------------------------------------------
# 3. a case-law folder enters the pipeline (no signature stage involved)
# ---------------------------------------------------------------------------


def test_run_case_ingest_stores_representations(case_corpus, caselaw_db, monkeypatch):
    import orchestration.dags.case_ingest_flow as flow

    monkeypatch.setattr(
        flow,
        "get_settings",
        lambda: SimpleNamespace(
            document=SimpleNamespace(min_characters=200, min_words=250),
            # The fixture cases are short hand-written excerpts, not full
            # judgments; they are measured against the pre-Phase-1 threshold so
            # these tests stay about metadata/encoding rather than length.
            # The configured production value (1200) is covered by
            # tests/test_phase1_foundation.py.
            caselaw=SimpleNamespace(
                corpus_root=case_corpus,
                case_html_filename="case.html",
                metadata_filename="metadata.json",
                body_preview_char_limit=20_000,
                max_headings=50,
                representation_schema_file=Path(REPRESENTATION_SCHEMA_FILE),
            )
        ),
    )

    result = run_case_ingest(root=case_corpus, db_path=caselaw_db, batch_id="batch_cases")

    assert len(result.succeeded_doc_ids) == 2
    assert result.failed_doc_ids == []

    stored = list_representations(db_path=caselaw_db)
    assert len(stored) == 2
    assert {r.source_type for r in stored} == {"case_html"}
    assert all(r.batch_id == "batch_cases" for r in stored)
    # Nothing was written to the signature table -- it was never touched.
    assert all(not hasattr(r, "signature_hash") for r in stored)


# ---------------------------------------------------------------------------
# 4. classification embeddings still work, straight off representations
# ---------------------------------------------------------------------------


def test_classification_embeddings_work_on_case_representations(case_corpus):
    reps = [load_case_folder(f, root=case_corpus) for f in iter_case_folders(case_corpus)]
    embedder = DeterministicHashEmbedder(dimension=16)

    vectors = embed_documents(reps, embedder)

    assert len(vectors) == len(reps)
    assert all(v.shape == (16,) for v in vectors.values())
    # Deterministic, and different cases get different vectors.
    assert np.allclose(
        vectors[reps[0].doc_id], embed_documents(reps, embedder)[reps[0].doc_id]
    )
    assert not np.allclose(vectors[reps[0].doc_id], vectors[reps[1].doc_id])


def test_representation_with_no_content_is_skipped_by_embedding():
    empty = DocumentRepresentation(
        doc_id="empty", source_uri="/cases/empty", source_type="case_html"
    )
    ok = _rep("case_ok", "Case OK", ["Facts"], "body text")

    vectors = embed_documents([empty, ok], DeterministicHashEmbedder(dimension=8))

    assert "case_ok" in vectors
    assert "empty" not in vectors


def test_cluster_keyword_extraction_works_on_representations():
    criminal = [_rep(f"c{i}", "Criminal Appeal", ["Facts"], "murder conviction sentence " * 10) for i in range(6)]
    tax = [_rep(f"t{i}", "Tax Reference", ["Question of Law"], "income assessment revenue " * 10) for i in range(6)]
    corpus = build_keyword_corpus(criminal + tax)

    criminal_kw = set(extract_cluster_keywords(criminal, corpus, top_k=5))
    tax_kw = set(extract_cluster_keywords(tax, corpus, top_k=5))

    assert "murder" in criminal_kw
    assert "income" in tax_kw
    assert not (criminal_kw & tax_kw)


# ---------------------------------------------------------------------------
# 5. HDBSCAN input still works
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not HDBSCAN_AVAILABLE, reason="hdbscan not installed")
def test_hdbscan_accepts_representation_embeddings():
    """Representation -> embed -> reduce -> HDBSCAN, unchanged from the
    signature path: two synthetic domains must come back as two clusters."""

    class _TwoDomainEmbedder:
        """Two well-separated *directions* -- pooling L2-normalizes, so
        two blobs that differ only in magnitude would collapse onto the
        same unit vector."""

        dimension = 8

        def encode(self, texts: list[str]) -> np.ndarray:
            rng = np.random.default_rng(0)
            out = []
            for text in texts:
                centre = np.zeros(self.dimension)
                centre[0 if "criminal" in text.lower() else 1] = 10.0
                out.append((centre + rng.normal(scale=0.05, size=self.dimension)).astype(np.float32))
            return np.stack(out)

    reps = [
        _rep(f"crim_{i}", "Criminal Appeal", ["Facts"], "criminal appeal against conviction")
        for i in range(20)
    ] + [
        _rep(f"tax_{i}", "Tax Reference", ["Question"], "tax reference on assessment")
        for i in range(20)
    ]

    pooled = embed_documents(reps, _TwoDomainEmbedder())
    doc_ids = list(pooled.keys())
    vectors = np.stack(list(pooled.values()))

    # min_docs=50 -> UMAP is skipped for this small set, exactly as in
    # production for a sub-threshold corpus; the array still flows through.
    reduced = reduce_dimensions(vectors, n_components=5, min_docs=50)
    assert reduced.shape[0] == len(doc_ids)

    result = hdbscan_clusterer(min_cluster_size=5)(reduced)

    assert isinstance(result, ClusterResult)
    assert result.n_clusters == 2
    assert len(result.labels) == len(doc_ids)


# ---------------------------------------------------------------------------
# 6. full Stage 1.2 over a case-law corpus
# ---------------------------------------------------------------------------


class _GroupedFakeEmbedder:
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
                else rng.normal(size=self.dimension) * 50
            )
            out.append((base + rng.normal(size=self.dimension) * 0.05).astype(np.float32))
        return np.stack(out)


def _fake_union_find_clusterer(
    vectors: np.ndarray, eps: float = 1.0, min_cluster_size: int = 3
) -> ClusterResult:
    """Same toy density clusterer tests/test_domain_discovery.py uses, so
    the flow can be exercised without depending on HDBSCAN's tuning."""

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
    labels = []
    for root in roots:
        if root_counts[root] < min_cluster_size:
            labels.append(NOISE_LABEL)
            continue
        if root not in root_to_label:
            root_to_label[root] = len(root_to_label)
        labels.append(root_to_label[root])

    labels_arr = np.array(labels, dtype=int)
    return ClusterResult(
        labels=labels_arr,
        probabilities=None,
        n_clusters=len(set(labels_arr.tolist()) - {NOISE_LABEL}),
    )


class _KeywordAwareFakeLLMClient:
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
def caselaw_discovery_settings(tmp_path) -> SimpleNamespace:
    return SimpleNamespace(
        database=SimpleNamespace(schema_file=Path(MANIFEST_SCHEMA_FILE)),
        pipeline=SimpleNamespace(checkpoint_dir=tmp_path / "checkpoints"),
        scratch=SimpleNamespace(root=tmp_path / "scratch"),
        metrics=SimpleNamespace(db_path=tmp_path / "metrics.db"),
        caselaw=SimpleNamespace(
            representation_schema_file=Path(REPRESENTATION_SCHEMA_FILE)
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


def test_domain_discovery_runs_on_case_representations(
    monkeypatch, caselaw_db, caselaw_discovery_settings
):
    """The full active flow: case representations -> embed -> reduce ->
    cluster -> rank -> label -> taxonomy card, with no signature anywhere."""

    import orchestration.dags.domain_discovery_flow as flow

    monkeypatch.setattr(flow, "get_settings", lambda: caselaw_discovery_settings)

    reps = []
    for group, count in {"CRIMINALAPPEAL": 20, "TAXREFERENCE": 12, "SERVICEMATTER": 8}.items():
        for i in range(count):
            reps.append(
                _rep(
                    f"{group.lower()}_{i}",
                    f"{group} Case",
                    [f"{group} Facts", f"{group} Findings"],
                    f"{group} recurring case text. " * 20,
                )
            )
    for i in range(6):
        reps.append(_rep(f"noise_{i}", f"UNIQUE{i} Case", [f"UNIQUE{i}"], f"UNIQUE{i} text. " * 20))
    upsert_representations(reps, db_path=caselaw_db)

    result = flow.run_stage1_2(
        run_id="caselaw_run",
        db_path=caselaw_db,
        embedder=_GroupedFakeEmbedder(
            groups=["CRIMINALAPPEAL", "TAXREFERENCE", "SERVICEMATTER"]
        ),
        clusterer=_fake_union_find_clusterer,
        llm_client=_KeywordAwareFakeLLMClient(
            {
                "criminalappeal": "Criminal Appeals",
                "taxreference": "Tax References",
                "servicematter": "Service Matters",
            }
        ),
    )

    assert result.sample_size == result.total_signatures == 46
    assert {d.name for d in result.domains} == {
        "Criminal Appeals",
        "Tax References",
        "Service Matters",
    }
    assert result.other_bucket is not None
    assert (
        sum(d.doc_count for d in result.domains) + result.other_bucket.doc_count
        == result.sample_size
    )
    assert result.taxonomy_card_path is not None and result.taxonomy_card_path.exists()

    rows = get_candidates_for_run("caselaw_run", db_path=caselaw_db)
    assert len(rows) == 4  # 3 domains + Other/Uncertain
    assert all(r["status"] == "draft" for r in rows)


def test_domain_discovery_handles_empty_representation_store(
    monkeypatch, caselaw_db, caselaw_discovery_settings
):
    import orchestration.dags.domain_discovery_flow as flow

    monkeypatch.setattr(flow, "get_settings", lambda: caselaw_discovery_settings)

    result = flow.run_stage1_2(run_id="empty_caselaw_run", db_path=caselaw_db)

    assert result.sample_size == 0
    assert result.domains == []


# ---------------------------------------------------------------------------
# 7. no production code on this path requires a signature
# ---------------------------------------------------------------------------


def test_both_record_types_satisfy_the_shared_document_contract():
    """The classification path is typed against EmbeddableDocument only --
    which is why the legacy signature record still flows through it."""

    from src.extraction.signature import DocumentSignature

    representation = _rep("case_1", "Case One", ["Facts"], "body")
    signature = DocumentSignature(
        doc_id="doc_1",
        source_uri="s3://x/doc_1.pdf",
        signature_hash="h",
        is_scanned=False,
        extraction_status="ok",
        extractor_used="pdfplumber",
        title="Book",
        toc=["Chapter 1"],
        body_preview="body",
    )

    assert isinstance(representation, EmbeddableDocument)
    assert isinstance(signature, EmbeddableDocument)
    assert signature.headings == ["Chapter 1"]  # toc alias keeps the legacy path working


def test_case_law_pipeline_imports_no_signature_module():
    """Importing and running the active path must not pull in signature code.

    Run in a subprocess so this is a real import-graph assertion, not a
    reflection of whatever other tests already imported.
    """

    script = """
import sys

import orchestration.dags.case_ingest_flow  # noqa: F401
import orchestration.dags.domain_discovery_flow  # noqa: F401
import src.clustering.label_clusters  # noqa: F401
import src.embedding.doc_pooling  # noqa: F401
import src.extraction.case_loader  # noqa: F401
import src.extraction.representation_store  # noqa: F401

first_party = [m for m in sys.modules if m.startswith(("src.", "orchestration.", "scripts."))]
leaked = sorted(
    m for m in first_party
    if "signature" in m or "pdf_extractor" in m or "ocr_fallback" in m
)
print(",".join(leaked))
"""
    completed = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        cwd=str(Path(__file__).resolve().parent.parent),
    )

    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.strip() == "", (
        f"case-law pipeline imported signature/PDF modules: {completed.stdout.strip()}"
    )
