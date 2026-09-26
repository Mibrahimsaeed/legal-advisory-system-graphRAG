"""Durable Phase 3 case signatures.

The signature is the compact case representation, now stored rather than
held only in memory. What is under test:

* it is created from a real document and round-trips losslessly,
* it is retrievable by ``doc_id``,
* Phase 3 reuses a stored signature when the source text and the builder
  version are both unchanged, and regenerates it when either moves,
* a rerun neither corrupts existing rows nor re-stamps unchanged ones,
* the full judgment is not duplicated into this table, and the legacy
  ``document_signatures`` table is left completely alone.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from orchestration.dags.domain_signal_flow import run_domain_signals
from src.classification.case_representation import (
    SIGNATURE_VERSION,
    build_case_representation,
)
from src.classification.signature_store import (
    DEFAULT_SIGNATURE_SCHEMA_FILE,
    WRITE_INSERTED,
    WRITE_UNCHANGED,
    WRITE_UPDATED,
    CaseSignature,
    get_case_signature,
    get_case_signatures,
    signature_stats,
    upsert_case_signatures,
)
from src.classification.signal_store import DEFAULT_SIGNALS_SCHEMA_FILE
from src.clustering.cluster import ClusterResult, NOISE_LABEL
from src.embedding.embed_model import DeterministicHashEmbedder
from src.common.config import get_settings
from src.common.db import connection_scope, init_schema
from src.extraction.doc_representation import (
    CLASSIFICATION_STATUS_DROPPED_PROCEDURAL,
    CLASSIFICATION_STATUS_PENDING,
    DocumentRepresentation,
)
from src.extraction.representation_store import (
    DEFAULT_REPRESENTATION_SCHEMA_FILE,
    upsert_representations,
)

DOMAIN_REGISTRY_SCHEMA_FILE = "schemas/domain_registry_schema.sql"

JUDGMENT = (
    "The respondent instituted a suit for recovery of dower, dowry articles and "
    "maintenance allowance before the learned Judge Family Court. The wife seeks "
    "dissolution of marriage on the basis of khula. Custody of the minor is to be "
    "decided with reference to the welfare of the minor under the Guardians and "
    "Wards Act. The nikahnama was exhibited without objection. "
)


def _doc(doc_id: str, text: str = JUDGMENT, **overrides) -> DocumentRepresentation:
    base = dict(
        doc_id=doc_id,
        source_uri=f"/corpus/{doc_id}",
        source_relpath=doc_id,
        source_type="case_html",
        title=f"{doc_id} cause title",
        headings=["Facts", "Arguments", "Order"],
        body_preview=text[:500],
        cleaned_text=text * 3,
        char_count=len(text) * 3,
        content_hash=f"hash_{doc_id}",
        court="Lahore High Court",
        decision_date="2021-04-11",
        classification_status=CLASSIFICATION_STATUS_PENDING,
    )
    base.update(overrides)
    return DocumentRepresentation(**base)


@pytest.fixture()
def db(tmp_path) -> Path:
    db_path = tmp_path / "signatures.db"
    init_schema(db_path=db_path, schema_file=str(DEFAULT_REPRESENTATION_SCHEMA_FILE))
    init_schema(db_path=db_path, schema_file=str(DEFAULT_SIGNATURE_SCHEMA_FILE))
    return db_path


@pytest.fixture()
def flow_db(db) -> Path:
    """The signature DB plus everything the Phase 3 flow needs."""

    init_schema(db_path=db, schema_file=str(DEFAULT_SIGNALS_SCHEMA_FILE))
    init_schema(db_path=db, schema_file=DOMAIN_REGISTRY_SCHEMA_FILE)
    return db


@pytest.fixture()
def flow_settings(monkeypatch, tmp_path):
    """Real Pydantic models, so these fakes cannot drift from the schema."""

    import orchestration.dags.domain_signal_flow as flow

    real = get_settings()
    monkeypatch.setattr(
        flow, "get_settings",
        lambda: SimpleNamespace(
            domain_signals=real.domain_signals.model_copy(
                update={"batch_size": 5, "llm_enabled": False}
            ),
            classification=real.classification,
            caselaw=real.caselaw,
            pipeline=SimpleNamespace(checkpoint_dir=tmp_path / "checkpoints"),
            metrics=SimpleNamespace(db_path=tmp_path / "metrics.db"),
            discovery=real.discovery.model_copy(
                update={"umap_min_docs": 10_000, "hdbscan_min_cluster_size": 2}
            ),
        ),
    )
    return real


def _fake_clusterer(vectors: np.ndarray) -> ClusterResult:
    n = vectors.shape[0]
    labels = np.array([0 if i % 2 == 0 else 1 for i in range(n)])
    if n:
        labels[-1] = NOISE_LABEL
    return ClusterResult(
        labels=labels,
        probabilities=np.where(labels == NOISE_LABEL, 0.0, 0.9),
        n_clusters=len(set(labels.tolist()) - {NOISE_LABEL}),
    )


def _seed_doc(db, doc_id: str = "d1", text: str = JUDGMENT, **overrides):
    """Store the document, then return the signature built from it.

    The parent row is required: case_signatures.doc_id has a FOREIGN KEY to
    document_representations, and foreign keys are enforced here -- a
    signature for a document that does not exist would be meaningless.
    """

    document = _doc(doc_id, text, **overrides)
    upsert_representations([document], db_path=db)
    return build_case_representation(document)


# ---------------------------------------------------------------------------
# 1. Creation
# ---------------------------------------------------------------------------


def test_a_signature_is_built_from_the_document(db):
    representation = _seed_doc(db)
    upsert_case_signatures([representation], {"d1": "hash_d1"}, db_path=db)

    stored = get_case_signature("d1", db_path=db)
    assert stored is not None
    assert stored.signature_version == SIGNATURE_VERSION
    assert stored.signature_hash == representation.representation_hash
    assert stored.source_content_hash == "hash_d1"
    assert stored.text_source == "cleaned_text"
    assert stored.char_count == representation.char_count
    assert stored.word_count == representation.word_count


def test_the_signature_is_a_bounded_derivative_not_the_judgment(db):
    """The full text stays in document_representations; this is the derivative.

    "Compact" means *bounded*, not merely shorter: the signature prepends
    the title and headings, so for a short judgment it can exceed the
    judgment's own length. What holds for every document is that its body
    is a truncated prefix of cleaned_text, capped at max_text_chars.
    """

    cap = get_settings().domain_signals.max_text_chars
    long_judgment = JUDGMENT * 60  # comfortably past the cap
    assert len(long_judgment) > cap, "the fixture must actually exercise the bound"

    document = _doc("d1", long_judgment)
    upsert_representations([document], db_path=db)
    representation = build_case_representation(document, max_text_chars=cap)
    upsert_case_signatures([representation], {"d1": "hash_d1"}, db_path=db)

    stored = get_case_signature("d1", db_path=db)
    assert stored.signature_text == representation.signal_text

    body = stored.to_representation().body_preview
    assert len(body) <= cap                          # bounded
    assert document.cleaned_text.startswith(body)    # a prefix of the full text
    assert len(body) < len(document.cleaned_text)    # genuinely truncated

    with connection_scope(db) as conn:
        row = conn.execute(
            "SELECT cleaned_text FROM document_representations WHERE doc_id = 'd1'"
        ).fetchone()
    assert row["cleaned_text"] == document.cleaned_text  # untouched


def test_title_and_headings_survive_so_the_weighting_can_be_rebuilt(db):
    """A flattened string cannot be split back into title/headings/body."""

    representation = _seed_doc(db)
    upsert_case_signatures([representation], {"d1": "hash_d1"}, db_path=db)

    stored = get_case_signature("d1", db_path=db)
    assert stored.title == representation.title
    assert stored.headings == representation.headings


def test_a_signature_round_trips_to_an_identical_representation(db):
    representation = _seed_doc(db)
    upsert_case_signatures([representation], {"d1": "hash_d1"}, db_path=db)

    rebuilt = get_case_signature("d1", db_path=db).to_representation()

    assert rebuilt.doc_id == representation.doc_id
    assert rebuilt.title == representation.title
    assert rebuilt.headings == representation.headings
    assert rebuilt.body_preview == representation.body_preview
    assert rebuilt.signal_text == representation.signal_text
    assert rebuilt.representation_hash == representation.representation_hash


def test_a_document_with_no_title_or_headings_round_trips(db):
    """The body is recovered by stripping a prefix -- so the empty case matters."""

    representation = _seed_doc(db, "bare", title=None, headings=[])
    upsert_case_signatures([representation], {"bare": "hash_bare"}, db_path=db)

    rebuilt = get_case_signature("bare", db_path=db).to_representation()
    assert rebuilt.body_preview == representation.body_preview
    assert rebuilt.signal_text == representation.signal_text


# ---------------------------------------------------------------------------
# 2. Retrieval
# ---------------------------------------------------------------------------


def test_retrieval_by_doc_id(db):
    upsert_case_signatures(
        [_seed_doc(db, "a"), _seed_doc(db, "b")],
        {"a": "hash_a", "b": "hash_b"}, db_path=db,
    )

    assert get_case_signature("a", db_path=db).doc_id == "a"
    assert get_case_signature("b", db_path=db).doc_id == "b"
    assert get_case_signature("missing", db_path=db) is None


def test_bulk_retrieval_returns_only_what_was_asked_for(db):
    upsert_case_signatures(
        [_seed_doc(db, d) for d in ("a", "b", "c")],
        {"a": "h", "b": "h", "c": "h"}, db_path=db,
    )

    assert set(get_case_signatures(["a", "c"], db_path=db)) == {"a", "c"}
    assert set(get_case_signatures(db_path=db)) == {"a", "b", "c"}
    assert get_case_signatures([], db_path=db) == {}


def test_bulk_retrieval_handles_more_ids_than_sqlite_allows_variables(db):
    """Over 999 ids must not blow up on the variable limit."""

    docs = [_seed_doc(db, f"d{i:04d}") for i in range(1200)]
    upsert_case_signatures(docs, {d.doc_id: "h" for d in docs}, db_path=db)

    found = get_case_signatures([d.doc_id for d in docs], db_path=db)
    assert len(found) == 1200


def test_stats_summarise_the_store(db):
    upsert_case_signatures(
        [_seed_doc(db, "a"), _seed_doc(db, "b")],
        {"a": "h", "b": "h"}, db_path=db,
    )
    stats = signature_stats(db_path=db)

    assert stats["signatures"] == 2
    assert stats["by_version"] == {SIGNATURE_VERSION: 2}
    assert stats["by_text_source"] == {"cleaned_text": 2}
    assert stats["mean_char_count"] > 0


# ---------------------------------------------------------------------------
# 3. Reuse: when is a stored signature still correct?
# ---------------------------------------------------------------------------


def test_a_signature_is_current_when_source_and_version_match():
    signature = CaseSignature(
        doc_id="d1", signature_text="t", signature_version=SIGNATURE_VERSION,
        signature_hash="h", source_content_hash="src1",
    )
    assert signature.is_current("src1")


def test_changed_source_text_invalidates_a_signature():
    signature = CaseSignature(
        doc_id="d1", signature_text="t", signature_version=SIGNATURE_VERSION,
        signature_hash="h", source_content_hash="src1",
    )
    assert not signature.is_current("src2")


def test_a_new_builder_version_invalidates_a_signature():
    signature = CaseSignature(
        doc_id="d1", signature_text="t", signature_version="case_signature/0.9",
        signature_hash="h", source_content_hash="src1",
    )
    assert not signature.is_current("src1")


def test_a_signature_with_no_recorded_source_hash_is_never_reused():
    """Nothing to compare against, so trusting it would defeat the check."""

    signature = CaseSignature(
        doc_id="d1", signature_text="t", signature_version=SIGNATURE_VERSION,
        signature_hash="h", source_content_hash=None,
    )
    assert not signature.is_current("src1")
    assert not signature.is_current(None)


# ---------------------------------------------------------------------------
# 4. Writes: idempotency and no corruption
# ---------------------------------------------------------------------------


def test_first_write_inserts_second_identical_write_is_a_no_op(db):
    representation = _seed_doc(db)
    first = upsert_case_signatures([representation], {"d1": "hash_d1"}, db_path=db)
    second = upsert_case_signatures([representation], {"d1": "hash_d1"}, db_path=db)

    assert first[WRITE_INSERTED] == 1
    assert second[WRITE_UNCHANGED] == 1
    assert second[WRITE_INSERTED] == 0 and second[WRITE_UPDATED] == 0


def test_an_unchanged_write_does_not_touch_updated_at(db):
    representation = _seed_doc(db)
    upsert_case_signatures([representation], {"d1": "hash_d1"}, db_path=db)
    before = get_case_signature("d1", db_path=db).updated_at

    upsert_case_signatures([representation], {"d1": "hash_d1"}, db_path=db)
    assert get_case_signature("d1", db_path=db).updated_at == before


def test_changed_text_updates_the_row_in_place(db):
    upsert_case_signatures([_seed_doc(db, "d1")], {"d1": "hash_v1"}, db_path=db)
    revised = build_case_representation(_doc("d1", "A different judgment entirely. " * 40))

    outcomes = upsert_case_signatures([revised], {"d1": "hash_v2"}, db_path=db)

    assert outcomes[WRITE_UPDATED] == 1
    stored = get_case_signature("d1", db_path=db)
    assert stored.signature_hash == revised.representation_hash
    assert stored.source_content_hash == "hash_v2"
    with connection_scope(db) as conn:
        assert conn.execute(
            "SELECT COUNT(*) AS n FROM case_signatures"
        ).fetchone()["n"] == 1  # one row per doc_id, never a second


def test_one_row_per_doc_id(db):
    for _ in range(3):
        upsert_case_signatures([_seed_doc(db, "d1")], {"d1": "h"}, db_path=db)

    with connection_scope(db) as conn:
        assert conn.execute(
            "SELECT COUNT(*) AS n FROM case_signatures"
        ).fetchone()["n"] == 1


# ---------------------------------------------------------------------------
# 5. Phase 3 end to end
# ---------------------------------------------------------------------------


_FAKES = {"clusterer": _fake_clusterer, "embedder": DeterministicHashEmbedder()}


def _seed(db, n=4, dropped=1):
    docs = [_doc(f"case_{i}") for i in range(n)]
    docs += [
        _doc(
            f"dropped_{i}", "Case called. None present. Adjourned.",
            classification_status=CLASSIFICATION_STATUS_DROPPED_PROCEDURAL,
            drop_reason="procedural_adjournment", cleaned_text=None,
        )
        for i in range(dropped)
    ]
    upsert_representations(docs, db_path=db)
    return docs


def test_phase_3_persists_a_signature_for_every_case(flow_db, flow_settings):
    _seed(flow_db)
    run_domain_signals("sig1", db_path=flow_db, **_FAKES)

    stored = get_case_signatures(db_path=flow_db)
    assert set(stored) == {f"case_{i}" for i in range(4)}
    for signature in stored.values():
        assert signature.signature_text
        assert signature.signature_version == SIGNATURE_VERSION
        assert signature.source_content_hash


def test_phase_3_does_not_sign_a_phase_2_drop(flow_db, flow_settings):
    """The short-circuit holds for signatures too."""

    _seed(flow_db)
    run_domain_signals("sig1", db_path=flow_db, **_FAKES)

    assert get_case_signature("dropped_0", db_path=flow_db) is None


def test_rerunning_phase_3_reuses_signatures_and_rewrites_nothing(flow_db, flow_settings):
    _seed(flow_db)
    run_domain_signals("sig1", db_path=flow_db, **_FAKES)
    before = {
        doc_id: (s.signature_hash, s.updated_at)
        for doc_id, s in get_case_signatures(db_path=flow_db).items()
    }

    run_domain_signals("sig2", db_path=flow_db, **_FAKES)
    after = {
        doc_id: (s.signature_hash, s.updated_at)
        for doc_id, s in get_case_signatures(db_path=flow_db).items()
    }

    assert before == after, "a rerun must neither change nor re-stamp signatures"


def test_a_rerun_regenerates_a_signature_whose_source_changed(flow_db, flow_settings):
    _seed(flow_db)
    run_domain_signals("sig1", db_path=flow_db, **_FAKES)
    original = get_case_signature("case_0", db_path=flow_db)

    # The judgment is re-scraped with different text and a new content hash.
    upsert_representations(
        [_doc("case_0", "A wholly different criminal appeal judgment. " * 40,
              content_hash="hash_case_0_v2")],
        db_path=flow_db,
    )
    run_domain_signals("sig2", db_path=flow_db, **_FAKES)

    revised = get_case_signature("case_0", db_path=flow_db)
    assert revised.signature_hash != original.signature_hash
    assert revised.source_content_hash == "hash_case_0_v2"
    # Every other signature is untouched.
    assert get_case_signature("case_1", db_path=flow_db).updated_at == (
        get_case_signatures(["case_1"], db_path=flow_db)["case_1"].updated_at
    )


def test_a_rerun_does_not_corrupt_the_stored_documents(flow_db, flow_settings):
    docs = _seed(flow_db)
    run_domain_signals("sig1", db_path=flow_db, **_FAKES)
    run_domain_signals("sig2", db_path=flow_db, **_FAKES)

    with connection_scope(flow_db) as conn:
        rows = {
            r["doc_id"]: (r["cleaned_text"], r["classification_status"], r["content_hash"])
            for r in conn.execute(
                "SELECT doc_id, cleaned_text, classification_status, content_hash "
                "FROM document_representations"
            )
        }

    for document in docs:
        assert rows[document.doc_id] == (
            document.cleaned_text,
            document.classification_status,
            document.content_hash,
        )


def test_the_legacy_signature_table_is_untouched(flow_db, flow_settings):
    """`case_signatures` exists precisely so the legacy table is not reused."""

    _seed(flow_db)
    run_domain_signals("sig1", db_path=flow_db, **_FAKES)

    with connection_scope(flow_db) as conn:
        tables = {
            r["name"] for r in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            )
        }
    assert "case_signatures" in tables
    assert "document_signatures" not in tables


def test_no_embeddings_are_persisted_by_this_change(flow_db, flow_settings):
    """Embedding persistence is a separate concern, deliberately not done here."""

    _seed(flow_db)
    run_domain_signals("sig1", db_path=flow_db, **_FAKES)

    with connection_scope(flow_db) as conn:
        columns = [r["name"] for r in conn.execute("PRAGMA table_info(case_signatures)")]
        tables = {
            r["name"] for r in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            )
        }
    assert not [c for c in columns if "embed" in c or "vector" in c]
    assert not [t for t in tables if "embedding" in t]
