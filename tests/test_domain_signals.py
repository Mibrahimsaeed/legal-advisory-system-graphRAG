"""Phase 3: case representation and domain signal generation.

Phase 3 gathers evidence and decides nothing. The tests are organised
around that contract:

* the representation is compact, deterministic, and free of court/date,
* embedding and clustering run on it and never touch a Phase 2 drop,
* keyword profiles and the LLM produce *independent* signals,
* a cluster id is never treated as a domain,
* malformed LLM output degrades to a recorded failure, never a guess.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from orchestration.dags.domain_signal_flow import run_domain_signals
from src.classification.case_representation import (
    build_case_representation,
    build_case_representations,
)
from src.classification.domain_assessment import (
    ASSESSMENT_VERSION,
    STATUS_FAILED,
    STATUS_OK,
    assess_domain,
    build_assessment_prompt,
    render_domain_definitions,
    validate_assessment,
)
from src.classification.keyword_signals import (
    compile_profiles,
    detect_keyword_signals,
)
from src.classification.signal_store import (
    DEFAULT_SIGNALS_SCHEMA_FILE,
    get_signal_history,
    get_signalled_doc_ids,
    get_signals_for_run,
    persist_domain_signals,
    signal_stats,
)
from src.classification.taxonomy_registry import OTHER_DOMAIN_ID, load_frozen_taxonomy
from src.clustering.cluster import ClusterResult, NOISE_LABEL
from src.common.config import get_settings
from src.common.db import connection_scope, init_schema
from src.common.exceptions import ConfigurationError
from src.embedding.doc_pooling import embed_documents
from src.embedding.embed_model import DeterministicHashEmbedder
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

FAMILY_TEXT = (
    "The respondent instituted a suit for recovery of dower, dowry articles and "
    "maintenance allowance before the learned Judge Family Court. The wife seeks "
    "dissolution of marriage on the basis of khula. Custody of the minor is to be "
    "decided with reference to the welfare of the minor under the Guardians and "
    "Wards Act. The nikahnama was exhibited without objection and maintenance was "
    "fixed accordingly. "
)
CRIMINAL_TEXT = (
    "The accused was convicted under the Penal Code and sentenced by the trial "
    "court. Learned counsel challenges the conviction, arguing the ocular account "
    "is doubtful and the complainant deposed with delay. The investigating officer "
    "did not join any private witness. The prosecution failed to prove the charge "
    "and the appeal against conviction is allowed; bail is confirmed. "
)
NEUTRAL_TEXT = (
    "This reference concerns the assessment of income tax for the year in question "
    "and the limitation period prescribed by the Ordinance. The taxpayer contends "
    "the amendment was time barred. "
)


def _doc(doc_id: str, text: str, **overrides) -> DocumentRepresentation:
    base = dict(
        doc_id=doc_id,
        source_uri=f"/corpus/{doc_id}",
        source_relpath=doc_id,
        source_type="case_html",
        title=f"{doc_id} cause title",
        headings=["Facts", "Order"],
        body_preview=text[:500],
        cleaned_text=text * 3,
        char_count=len(text) * 3,
        court="Lahore High Court",
        decision_date="2021-04-11",
        citation="2021 PLJ 88",
        classification_status=CLASSIFICATION_STATUS_PENDING,
    )
    base.update(overrides)
    return DocumentRepresentation(**base)


@pytest.fixture()
def taxonomy():
    return load_frozen_taxonomy()


@pytest.fixture()
def profiles(taxonomy):
    return compile_profiles(get_settings().domain_signals.profiles, taxonomy)


@pytest.fixture()
def db(tmp_path) -> Path:
    db_path = tmp_path / "signals.db"
    init_schema(db_path=db_path, schema_file=str(DEFAULT_REPRESENTATION_SCHEMA_FILE))
    init_schema(db_path=db_path, schema_file=str(DEFAULT_SIGNALS_SCHEMA_FILE))
    init_schema(db_path=db_path, schema_file=DOMAIN_REGISTRY_SCHEMA_FILE)
    return db_path


class _ScriptedLLM:
    def __init__(self, responses=None, default=None):
        self.responses = list(responses or [])
        self.default = default
        self.calls = 0
        self.prompts: list[str] = []

    def complete(self, system, prompt, max_tokens=None):
        self.calls += 1
        self.prompts.append(prompt)
        response = self.responses.pop(0) if self.responses else self.default
        if response is None:
            raise AssertionError("LLM called more times than scripted")
        if isinstance(response, Exception):
            raise response
        return response if isinstance(response, str) else json.dumps(response)


def _ok_llm(domain="criminal_law", confidence=0.9, reason="The text discusses a conviction."):
    return json.dumps({"domain": domain, "confidence": confidence, "reason": reason})


# ---------------------------------------------------------------------------
# 1. Case representation
# ---------------------------------------------------------------------------


def test_representation_prefers_cleaned_text_from_phase_2():
    rep = build_case_representation(_doc("d1", CRIMINAL_TEXT))

    assert rep.text_source == "cleaned_text"
    assert rep.body_preview.startswith("The accused was convicted")


def test_representation_falls_back_to_body_preview():
    rep = build_case_representation(_doc("d1", CRIMINAL_TEXT, cleaned_text=None))

    assert rep.text_source == "body_preview"
    assert rep.body_preview


def test_representation_excludes_court_and_date():
    """Metadata must not dominate -- or even enter -- the embedded text."""

    rep = build_case_representation(_doc("d1", CRIMINAL_TEXT))

    assert "Lahore High Court" not in rep.signal_text
    assert "2021-04-11" not in rep.signal_text
    assert "2021 PLJ 88" not in rep.signal_text


def test_representation_is_deterministic():
    doc = _doc("d1", CRIMINAL_TEXT)

    first, second = build_case_representation(doc), build_case_representation(doc)

    assert first == second
    assert first.representation_hash == second.representation_hash


def test_representation_hash_tracks_the_text_that_is_analysed():
    base = build_case_representation(_doc("d1", CRIMINAL_TEXT))
    changed_text = build_case_representation(_doc("d1", FAMILY_TEXT))
    changed_court = build_case_representation(_doc("d1", CRIMINAL_TEXT, court="Sindh High Court"))

    assert base.representation_hash != changed_text.representation_hash
    assert base.representation_hash == changed_court.representation_hash


def test_representation_is_bounded():
    rep = build_case_representation(_doc("d1", "word " * 50_000), max_text_chars=5_000)

    assert rep.char_count == 5_000


def test_documents_without_text_are_skipped():
    """Nothing to embed means no representation -- a near-zero vector would
    cluster with every other empty document."""

    docs = [
        _doc("d1", CRIMINAL_TEXT),
        _doc("empty", "", cleaned_text=None, body_preview="", title=None, headings=[]),
    ]

    reps = build_case_representations(docs)

    assert [r.doc_id for r in reps] == ["d1"]


def test_a_title_alone_is_still_representable():
    """Thin, but not empty: a cause title carries real domain signal."""

    doc = _doc("title_only", "", cleaned_text=None, body_preview="", headings=[],
               title="Zainab Bibi v. The State")

    reps = build_case_representations([doc])

    assert [r.doc_id for r in reps] == ["title_only"]


def test_representation_satisfies_the_embeddable_contract():
    from src.extraction.doc_representation import EmbeddableDocument

    assert isinstance(build_case_representation(_doc("d1", CRIMINAL_TEXT)), EmbeddableDocument)


# ---------------------------------------------------------------------------
# 2. Embedding
# ---------------------------------------------------------------------------


def test_one_embedding_per_case():
    reps = build_case_representations([_doc(f"d{i}", CRIMINAL_TEXT) for i in range(4)])

    vectors = embed_documents(reps, DeterministicHashEmbedder(dimension=16))

    assert set(vectors) == {r.doc_id for r in reps}
    assert all(v.shape == (16,) for v in vectors.values())


def test_embedding_is_deterministic_and_ignores_court():
    embedder = DeterministicHashEmbedder(dimension=16)
    base = build_case_representation(_doc("d1", CRIMINAL_TEXT))
    other_court = build_case_representation(_doc("d1", CRIMINAL_TEXT, court="Peshawar High Court"))

    assert np.allclose(
        embed_documents([base], embedder)["d1"],
        embed_documents([other_court], embedder)["d1"],
    )


# ---------------------------------------------------------------------------
# 3. Keyword signals
# ---------------------------------------------------------------------------


def test_family_text_scores_family_law(profiles):
    signals = detect_keyword_signals("d1", FAMILY_TEXT, profiles)

    assert signals.top_domain == "family_law"
    assert signals.score_for("family_law") > signals.score_for("criminal_law")


def test_criminal_text_scores_criminal_law(profiles):
    signals = detect_keyword_signals("d1", CRIMINAL_TEXT, profiles)

    assert signals.top_domain == "criminal_law"
    assert signals.score_for("criminal_law") > signals.score_for("family_law")


def test_unrelated_text_matches_nothing(profiles):
    signals = detect_keyword_signals("d1", NEUTRAL_TEXT, profiles)

    assert signals.top_domain is None
    assert signals.total_matches == 0


def test_single_passing_mention_does_not_register(profiles):
    """One stray "bail" in a family judgment is not criminal evidence."""

    signals = detect_keyword_signals(
        "d1", "The wife filed for khula and dower. Bail was mentioned once.", profiles,
        min_matches=2,
    )

    assert signals.score_for("criminal_law") == 0.0


def test_keyword_detection_is_deterministic(profiles):
    assert detect_keyword_signals("d1", FAMILY_TEXT, profiles) == detect_keyword_signals(
        "d1", FAMILY_TEXT, profiles
    )


def test_keyword_matching_is_whole_word(profiles):
    """"Charge" must not match inside "surcharged"."""

    signals = detect_keyword_signals("d1", "The surcharged amounts were recomputed. " * 5, profiles)

    assert signals.score_for("criminal_law") == 0.0


def test_evidence_is_serialisable_and_bounded(profiles):
    evidence = detect_keyword_signals("d1", FAMILY_TEXT, profiles).as_evidence()

    assert json.loads(json.dumps(evidence))
    assert evidence["top_domain"] == "family_law"
    assert len(evidence["domains"]["family_law"]["matched_terms"]) <= 10


def test_profile_for_an_unknown_domain_is_rejected(taxonomy):
    with pytest.raises(ConfigurationError, match="unknown domain"):
        compile_profiles({"tax_law": ["income"]}, taxonomy)


def test_profiles_are_configuration_driven():
    assert set(get_settings().domain_signals.profiles) == {"family_law", "criminal_law"}


# ---------------------------------------------------------------------------
# 4. LLM broad assessment
# ---------------------------------------------------------------------------


def test_valid_llm_output_is_accepted(taxonomy):
    assessment = validate_assessment(
        {"domain": "family_law", "confidence": 0.82, "reason": "Dower and custody are in issue."},
        taxonomy, "d1",
    )

    assert assessment.status == STATUS_OK
    assert assessment.domain == "family_law"
    assert assessment.confidence == 0.82
    assert assessment.reason


def test_other_uncertain_is_a_valid_answer(taxonomy):
    assessment = validate_assessment(
        {"domain": OTHER_DOMAIN_ID, "confidence": 0.4, "reason": "Tax matter, not family or criminal."},
        taxonomy, "d1",
    )

    assert assessment.status == STATUS_OK
    assert assessment.is_uncertain


@pytest.mark.parametrize(
    ("payload", "error_prefix"),
    [
        ({"confidence": 0.9, "reason": "x"}, "missing_domain"),
        ({"domain": "tax_law", "confidence": 0.9, "reason": "x"}, "unknown_domain_id"),
        ({"domain": "family_law", "confidence": "very high", "reason": "x"}, "invalid_confidence"),
        ({"domain": "family_law", "confidence": 1.7, "reason": "x"}, "invalid_confidence"),
        ({"domain": "family_law", "confidence": 0.9}, "missing_reason"),
        ({"domain": "family_law", "confidence": 0.9, "reason": "  "}, "missing_reason"),
    ],
)
def test_malformed_llm_output_fails_without_guessing(taxonomy, payload, error_prefix):
    assessment = validate_assessment(payload, taxonomy, "d1")

    assert assessment.status == STATUS_FAILED
    assert assessment.domain is None  # never a guessed domain
    assert assessment.error.startswith(error_prefix)


def test_percentage_confidence_is_normalised(taxonomy):
    assessment = validate_assessment(
        {"domain": "family_law", "confidence": 85, "reason": "x"}, taxonomy, "d1"
    )

    assert assessment.confidence == pytest.approx(0.85)


def test_non_json_response_is_a_recorded_failure(taxonomy):
    rep = build_case_representation(_doc("d1", CRIMINAL_TEXT))

    assessment = assess_domain(rep, taxonomy, _ScriptedLLM(["I think it is criminal."]))

    assert assessment.status == STATUS_FAILED
    assert assessment.domain is None


def test_llm_exception_is_a_recorded_failure(taxonomy):
    rep = build_case_representation(_doc("d1", CRIMINAL_TEXT))

    assessment = assess_domain(rep, taxonomy, _ScriptedLLM([RuntimeError("ollama down")]))

    assert assessment.status == STATUS_FAILED
    assert "ollama down" in assessment.error


def test_prompt_asks_only_for_a_broad_reading(taxonomy):
    rep = build_case_representation(_doc("d1", CRIMINAL_TEXT))

    prompt = build_assessment_prompt(rep, render_domain_definitions(taxonomy))

    assert "family_law" in prompt and "criminal_law" in prompt
    assert OTHER_DOMAIN_ID in prompt
    assert "Lahore High Court" not in prompt  # court stays out of the representation


def test_the_prompt_carries_no_keyword_evidence(taxonomy, profiles):
    """Superseded: the prompt used to append a labelled keyword "hint".

    It was removed because Phase 5 weights keyword and LLM at 0.40 each as
    two *independent* readings, and a model shown the keyword verdict
    agrees with it more often than it would blind -- which quietly inflates
    corroboration and suppresses the conflict rule. The signals now reach
    the decision separately. See tests/test_pipeline_safety_fixes.py for
    the end-to-end regression.
    """

    rep = build_case_representation(_doc("d1", CRIMINAL_TEXT))
    prompt = build_assessment_prompt(rep, render_domain_definitions(taxonomy))

    lowered = prompt.lower()
    assert "keyword" not in lowered
    assert "lexical signal" not in lowered
    # The document itself is still there.
    assert "accused" in lowered


# ---------------------------------------------------------------------------
# 5. The flow: storage, short-circuit, batching, idempotency
# ---------------------------------------------------------------------------


@pytest.fixture()
def flow_settings(monkeypatch, tmp_path):
    import orchestration.dags.domain_signal_flow as flow

    real = get_settings()
    monkeypatch.setattr(
        flow,
        "get_settings",
        lambda: SimpleNamespace(
            domain_signals=real.domain_signals.model_copy(update={"batch_size": 2}),
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


def _seed(db, n_family=3, n_criminal=3, dropped=2):
    docs = [_doc(f"family_{i}", FAMILY_TEXT) for i in range(n_family)]
    docs += [_doc(f"criminal_{i}", CRIMINAL_TEXT) for i in range(n_criminal)]
    docs += [
        _doc(
            f"dropped_{i}", "Case called. None present. Adjourned.",
            classification_status=CLASSIFICATION_STATUS_DROPPED_PROCEDURAL,
            drop_reason="procedural_adjournment",
            cleaned_text=None,
        )
        for i in range(dropped)
    ]
    upsert_representations(docs, db_path=db)
    return docs


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


def test_dropped_documents_never_reach_embedding_or_signals(db, flow_settings):
    _seed(db)
    llm = _ScriptedLLM(default=_ok_llm())

    result = run_domain_signals(
        run_id="r1", db_path=db, llm_client=llm,
        embedder=DeterministicHashEmbedder(dimension=16), clusterer=_fake_clusterer,
    )

    signalled = {r["doc_id"] for r in get_signals_for_run("r1", db_path=db)}
    assert not any(d.startswith("dropped_") for d in signalled)
    assert len(signalled) == 6
    assert result.corpus_size == 6  # the Phase 2 drops are not even loaded
    assert all("None present" not in p for p in llm.prompts)


def test_every_signal_is_stored_with_provenance(db, flow_settings):
    _seed(db, n_family=2, n_criminal=2, dropped=0)

    run_domain_signals(
        run_id="r1", db_path=db, llm_client=_ScriptedLLM(default=_ok_llm("family_law", 0.77)),
        embedder=DeterministicHashEmbedder(dimension=16), clusterer=_fake_clusterer,
    )

    rows = get_signals_for_run("r1", db_path=db)
    assert len(rows) == 4
    row = rows[0]
    assert row["representation_hash"] and row["text_source"] == "cleaned_text"
    assert row["cluster_id"] is not None            # clustering signal
    assert row["keyword_signals"]["domains"]        # keyword signal
    assert row["llm_domain"] == "family_law"        # LLM signal
    assert row["llm_confidence"] == pytest.approx(0.77)
    assert row["llm_reason"] and row["llm_status"] == "ok"
    assert row["signal_version"] == ASSESSMENT_VERSION
    assert row["embedding_model"] and row["created_at"]


def test_cluster_assignments_are_preserved_for_phase_4(db, flow_settings):
    _seed(db, dropped=0)

    run_domain_signals(
        run_id="r1", db_path=db, llm_client=_ScriptedLLM(default=_ok_llm()),
        embedder=DeterministicHashEmbedder(dimension=16), clusterer=_fake_clusterer,
    )

    with connection_scope(db) as conn:
        rows = conn.execute(
            "SELECT doc_id, cluster_id FROM cluster_assignments WHERE run_id = 'r1'"
        ).fetchall()

    assert len(rows) == 6
    assert NOISE_LABEL in {r["cluster_id"] for r in rows}  # noise preserved, not dropped


def test_cluster_id_is_never_treated_as_a_domain(db, flow_settings):
    """A cluster is a discovery signal: no row may equate one to a domain."""

    _seed(db, dropped=0)

    run_domain_signals(
        run_id="r1", db_path=db, llm_client=_ScriptedLLM(default=_ok_llm("family_law")),
        embedder=DeterministicHashEmbedder(dimension=16), clusterer=_fake_clusterer,
    )

    rows = get_signals_for_run("r1", db_path=db)
    # Documents in the same cluster are free to carry different domain
    # evidence -- nothing derives one from the other.
    by_cluster: dict[int, set] = {}
    for row in rows:
        by_cluster.setdefault(row["cluster_id"], set()).add(row["keyword_top_domain"])
    assert any(len(v) >= 1 for v in by_cluster.values())
    # And no classification was written anywhere.
    with connection_scope(db) as conn:
        tables = {r["name"] for r in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        )}
    assert "document_classifications" not in tables


def test_uncertain_assessment_is_stored_as_such(db, flow_settings):
    _seed(db, n_family=1, n_criminal=1, dropped=0)

    run_domain_signals(
        run_id="r1", db_path=db,
        llm_client=_ScriptedLLM(default=_ok_llm(OTHER_DOMAIN_ID, 0.35, "Neither domain fits.")),
        embedder=DeterministicHashEmbedder(dimension=16), clusterer=_fake_clusterer,
    )

    rows = get_signals_for_run("r1", db_path=db)
    assert {r["llm_domain"] for r in rows} == {OTHER_DOMAIN_ID}
    assert all(r["llm_status"] == "ok" for r in rows)  # uncertainty is not failure


def test_llm_failure_is_recorded_without_a_domain(db, flow_settings):
    _seed(db, n_family=1, n_criminal=0, dropped=0)

    run_domain_signals(
        run_id="r1", db_path=db, llm_client=_ScriptedLLM(default="not json at all"),
        embedder=DeterministicHashEmbedder(dimension=16), clusterer=_fake_clusterer,
    )

    row = get_signals_for_run("r1", db_path=db)[0]
    assert row["llm_status"] == "failed"
    assert row["llm_domain"] is None
    assert row["llm_error"]
    # The deterministic signals still landed.
    assert row["keyword_top_domain"] == "family_law"
    assert row["cluster_id"] is not None


def test_llm_can_be_disabled_for_a_deterministic_only_run(db, flow_settings):
    _seed(db, n_family=1, n_criminal=1, dropped=0)
    llm = _ScriptedLLM()  # any call raises

    run_domain_signals(
        run_id="r1", db_path=db, llm_client=llm, llm_enabled=False,
        embedder=DeterministicHashEmbedder(dimension=16), clusterer=_fake_clusterer,
    )

    assert llm.calls == 0
    rows = get_signals_for_run("r1", db_path=db)
    assert {r["llm_status"] for r in rows} == {"skipped"}
    assert all(r["keyword_signals"] for r in rows)


def test_batching_persists_progressively(db, flow_settings):
    _seed(db, n_family=3, n_criminal=2, dropped=0)

    run_domain_signals(
        run_id="r1", db_path=db, llm_client=_ScriptedLLM(default=_ok_llm()),
        embedder=DeterministicHashEmbedder(dimension=16), clusterer=_fake_clusterer,
    )

    batches = {r["batch_id"] for r in get_signals_for_run("r1", db_path=db)}
    assert batches == {"r1_b0000", "r1_b0001", "r1_b0002"}  # batch_size=2 over 5 cases


def test_resume_does_not_re_spend_llm_calls(db, flow_settings):
    _seed(db, n_family=2, n_criminal=2, dropped=0)

    first = run_domain_signals(
        run_id="r1", db_path=db, limit=2, llm_client=_ScriptedLLM(default=_ok_llm()),
        embedder=DeterministicHashEmbedder(dimension=16), clusterer=_fake_clusterer,
    )
    assert first.processed == 2

    resumed = _ScriptedLLM(default=_ok_llm())
    second = run_domain_signals(
        run_id="r1", db_path=db, llm_client=resumed,
        embedder=DeterministicHashEmbedder(dimension=16), clusterer=_fake_clusterer,
    )

    assert second.processed == 2
    assert second.skipped_already_done == 2
    assert resumed.calls == 2
    assert len(get_signals_for_run("r1", db_path=db)) == 4


def test_rerunning_is_idempotent(db, flow_settings):
    _seed(db, dropped=0)
    kwargs = dict(
        db_path=db, embedder=DeterministicHashEmbedder(dimension=16), clusterer=_fake_clusterer,
    )

    run_domain_signals(run_id="r1", llm_client=_ScriptedLLM(default=_ok_llm()), **kwargs)
    before = {
        r["doc_id"]: (r["representation_hash"], r["keyword_top_domain"], r["cluster_id"])
        for r in get_signals_for_run("r1", db_path=db)
    }

    run_domain_signals(run_id="r1", llm_client=_ScriptedLLM(default=_ok_llm()), **kwargs)
    after = {
        r["doc_id"]: (r["representation_hash"], r["keyword_top_domain"], r["cluster_id"])
        for r in get_signals_for_run("r1", db_path=db)
    }

    assert before == after
    with connection_scope(db) as conn:
        assert conn.execute(
            "SELECT COUNT(*) c FROM document_domain_signals"
        ).fetchone()["c"] == 6  # no duplicates


def test_a_new_run_adds_evidence_without_destroying_the_old(db, flow_settings):
    _seed(db, n_family=1, n_criminal=0, dropped=0)
    kwargs = dict(
        db_path=db, embedder=DeterministicHashEmbedder(dimension=16), clusterer=_fake_clusterer,
    )

    run_domain_signals(run_id="r1", llm_client=_ScriptedLLM(default=_ok_llm("family_law")), **kwargs)
    run_domain_signals(run_id="r2", llm_client=_ScriptedLLM(default=_ok_llm(OTHER_DOMAIN_ID)), **kwargs)

    history = get_signal_history("family_0", db_path=db)
    assert len(history) == 2
    assert {h["run_id"] for h in history} == {"r1", "r2"}
    assert {h["llm_domain"] for h in history} == {"family_law", OTHER_DOMAIN_ID}


def test_court_does_not_partition_the_run(db, flow_settings):
    """Same text under four courts: same keyword signal, one row each."""

    docs = [
        _doc(f"c{i}", CRIMINAL_TEXT, court=court)
        for i, court in enumerate(
            ["Supreme Court of Pakistan", "Lahore High Court", "Sindh High Court", None]
        )
    ]
    upsert_representations(docs, db_path=db)

    run_domain_signals(
        run_id="r1", db_path=db, llm_client=_ScriptedLLM(default=_ok_llm()),
        embedder=DeterministicHashEmbedder(dimension=16), clusterer=_fake_clusterer,
    )

    rows = get_signals_for_run("r1", db_path=db)
    assert len(rows) == 4
    assert {r["keyword_top_domain"] for r in rows} == {"criminal_law"}
    assert len({r["representation_hash"] for r in rows}) == 4  # differ only by doc_id


def test_stats_summarise_the_run(db, flow_settings):
    _seed(db, n_family=2, n_criminal=2, dropped=1)

    result = run_domain_signals(
        run_id="r1", db_path=db, llm_client=_ScriptedLLM(default=_ok_llm("criminal_law")),
        embedder=DeterministicHashEmbedder(dimension=16), clusterer=_fake_clusterer,
    )

    stats = signal_stats("r1", db_path=db)
    assert stats["total"] == 4
    assert stats["by_llm_domain"]["criminal_law"] == 4
    assert set(stats["by_keyword_domain"]) == {"family_law", "criminal_law"}
    assert stats["by_llm_status"] == {"ok": 4}
    assert 0.0 <= stats["keyword_llm_agreement_rate"] <= 1.0
    assert result.n_clusters >= 1


def test_empty_corpus_is_handled(db, flow_settings):
    result = run_domain_signals(
        run_id="r1", db_path=db, llm_client=_ScriptedLLM(),
        embedder=DeterministicHashEmbedder(dimension=16), clusterer=_fake_clusterer,
    )

    assert result.processed == 0
    assert get_signals_for_run("r1", db_path=db) == []


# ---------------------------------------------------------------------------
# 6. Retry of a failed LLM assessment
#
# A row whose llm_status is 'failed' used to count as "already done", so a
# transient outage (Ollama unreachable) was permanent: every rerun skipped
# the very documents that still needed an assessment.
# ---------------------------------------------------------------------------


def _signal_row(doc_id: str, llm_status: str, **overrides) -> dict:
    """One evidence record, as build_signal_record would produce it."""

    base = dict(
        doc_id=doc_id,
        representation_hash=f"hash_{doc_id}",
        text_source="cleaned_text",
        char_count=1200,
        word_count=200,
        cluster_id=3,
        cluster_confidence=0.87,
        embedding_model="sentence-transformers/all-mpnet-base-v2",
        keyword_signals={"top_domain": "family_law", "total_matches": 9, "domains": {}},
        keyword_top_domain="family_law",
        keyword_margin=0.42,
        llm_status=llm_status,
    )
    if llm_status == "ok":
        base.update(
            llm_domain="family_law", llm_confidence=0.94,
            llm_reason="dower and maintenance", llm_model="qwen3:14b",
        )
    elif llm_status == "failed":
        base.update(llm_error="Failed to connect to Ollama", llm_domain=None)
    base.update(overrides)
    return base


def test_a_successful_assessment_counts_as_already_done(db):
    """Test A."""

    persist_domain_signals(
        "r1", [_signal_row("ok_doc", "ok")],
        signal_version=ASSESSMENT_VERSION, db_path=db,
    )
    assert get_signalled_doc_ids("r1", db_path=db, require_llm_ok=True) == {"ok_doc"}


def test_a_failed_assessment_does_not_count_as_already_done(db):
    """Test B -- the bug: 187 failed rows blocked their own retry."""

    persist_domain_signals(
        "r1", [_signal_row("failed_doc", "failed")],
        signal_version=ASSESSMENT_VERSION, db_path=db,
    )
    assert get_signalled_doc_ids("r1", db_path=db, require_llm_ok=True) == set()
    # The row is still there -- nothing was deleted.
    assert len(get_signals_for_run("r1", db_path=db)) == 1


def test_a_skipped_assessment_is_retried_once_the_llm_is_enabled(db):
    """'skipped' is a row without a reading, exactly like 'failed'."""

    persist_domain_signals(
        "r1", [_signal_row("skipped_doc", "skipped")],
        signal_version=ASSESSMENT_VERSION, db_path=db,
    )
    assert get_signalled_doc_ids("r1", db_path=db, require_llm_ok=True) == set()


def test_a_deterministic_only_run_still_treats_any_row_as_done(db):
    """With the LLM disabled, re-deriving the same signals is pure waste."""

    persist_domain_signals(
        "r1",
        [_signal_row("a", "skipped"), _signal_row("b", "failed"), _signal_row("c", "ok")],
        signal_version=ASSESSMENT_VERSION, db_path=db,
    )
    assert get_signalled_doc_ids("r1", db_path=db, require_llm_ok=False) == {"a", "b", "c"}


def test_only_failed_rows_are_offered_for_retry(db):
    """A mixed run retries the failures and leaves the successes alone."""

    persist_domain_signals(
        "r1",
        [_signal_row("ok_1", "ok"), _signal_row("ok_2", "ok"),
         _signal_row("bad_1", "failed"), _signal_row("bad_2", "failed")],
        signal_version=ASSESSMENT_VERSION, db_path=db,
    )
    done = get_signalled_doc_ids("r1", db_path=db, require_llm_ok=True)
    assert done == {"ok_1", "ok_2"}


def test_the_retry_is_scoped_to_its_own_run(db):
    """A failure in one run does not offer another run's document for retry."""

    persist_domain_signals(
        "r1", [_signal_row("d1", "failed")],
        signal_version=ASSESSMENT_VERSION, db_path=db,
    )
    persist_domain_signals(
        "r2", [_signal_row("d1", "ok")],
        signal_version=ASSESSMENT_VERSION, db_path=db,
    )

    assert get_signalled_doc_ids("r1", db_path=db, require_llm_ok=True) == set()
    assert get_signalled_doc_ids("r2", db_path=db, require_llm_ok=True) == {"d1"}


def test_retrying_a_failed_row_keeps_every_other_signal(db, flow_settings):
    """Test C -- the retry must not cost the deterministic evidence."""

    _seed(db, n_family=1, n_criminal=0, dropped=0)
    kwargs = dict(
        db_path=db, embedder=DeterministicHashEmbedder(dimension=16),
        clusterer=_fake_clusterer,
    )

    # First pass: the model is unreachable, exactly as during the pilot.
    run_domain_signals(
        run_id="r1", llm_client=_ScriptedLLM(default=ConnectionError("no server")),
        **kwargs,
    )
    failed = get_signals_for_run("r1", db_path=db)[0]
    assert failed["llm_status"] == "failed"
    assert failed["keyword_top_domain"] == "family_law"

    # Second pass: the model is back.
    retried = _ScriptedLLM(default=_ok_llm(domain="family_law", confidence=0.93))
    result = run_domain_signals(run_id="r1", llm_client=retried, **kwargs)

    assert retried.calls == 1, "the failed document should have been retried"
    assert result.processed == 1
    assert result.skipped_already_done == 0

    row = get_signals_for_run("r1", db_path=db)[0]
    assert row["llm_status"] == "ok"
    assert row["llm_domain"] == "family_law"
    assert row["llm_error"] is None
    # ...and none of the other evidence was lost.
    assert row["keyword_signals"] == failed["keyword_signals"]
    assert row["keyword_top_domain"] == failed["keyword_top_domain"]
    assert row["keyword_margin"] == failed["keyword_margin"]
    assert row["cluster_id"] == failed["cluster_id"]
    assert row["cluster_confidence"] == failed["cluster_confidence"]
    assert row["representation_hash"] == failed["representation_hash"]
    assert row["text_source"] == failed["text_source"]
    assert row["char_count"] == failed["char_count"]
    assert row["word_count"] == failed["word_count"]


def test_a_retry_updates_in_place_and_adds_no_row(db, flow_settings):
    """The (run_id, doc_id) upsert is preserved: one row per document."""

    _seed(db, n_family=1, n_criminal=0, dropped=0)
    kwargs = dict(
        db_path=db, embedder=DeterministicHashEmbedder(dimension=16),
        clusterer=_fake_clusterer,
    )

    run_domain_signals(
        run_id="r1", llm_client=_ScriptedLLM(default="not json"), **kwargs
    )
    run_domain_signals(
        run_id="r1", llm_client=_ScriptedLLM(default=_ok_llm()), **kwargs
    )

    with connection_scope(db) as conn:
        assert conn.execute(
            "SELECT COUNT(*) c FROM document_domain_signals"
        ).fetchone()["c"] == 1


def test_successful_rows_are_not_reprocessed_on_a_later_run(db, flow_settings):
    """Test D -- success stays resumable; only the failure is retried."""

    _seed(db, n_family=2, n_criminal=0, dropped=0)
    kwargs = dict(
        db_path=db, embedder=DeterministicHashEmbedder(dimension=16),
        clusterer=_fake_clusterer,
    )

    # One document succeeds, the next fails.
    run_domain_signals(
        run_id="r1",
        llm_client=_ScriptedLLM(responses=[_ok_llm(), "not json at all"]),
        **kwargs,
    )
    statuses = {r["doc_id"]: r["llm_status"] for r in get_signals_for_run("r1", db_path=db)}
    assert sorted(statuses.values()) == ["failed", "ok"]

    retry = _ScriptedLLM(default=_ok_llm())
    result = run_domain_signals(run_id="r1", llm_client=retry, **kwargs)

    assert retry.calls == 1, "only the failed document should be re-asked"
    assert result.processed == 1
    assert result.skipped_already_done == 1
    assert {
        r["llm_status"] for r in get_signals_for_run("r1", db_path=db)
    } == {"ok"}
