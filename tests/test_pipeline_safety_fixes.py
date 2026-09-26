"""Regression tests for the four remaining audit findings.

One file per finding would scatter four small, closely related safety
properties; they are grouped here because they all answer the same
question -- "can this pipeline quietly do the wrong thing?" -- and each
section names the finding it pins down.

* **CI-4** Phase 5 must re-read current state and never revive a Phase 2
  structural drop from stale Phase 3 evidence.
* **CI-5** the LLM call must be reproducible: greedy, seeded, JSON-shaped,
  thinking off, with a budget that fits the answer.
* **CI-6** the cluster signal must be usable only on Phase 4's approval,
  and must fail closed.
* **CI-7** the LLM must never see the keyword verdict.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from orchestration.dags.domain_classification_flow import (
    WRITE_INELIGIBLE,
    cluster_signal_is_approved,
    run_domain_classification,
)
from src.classification import review_policy as policy
from src.classification.case_representation import build_case_representation
from src.classification.classification_store import get_classifications_for_run
from src.classification.domain_assessment import (
    ASSESSMENT_VERSION,
    assess_domain,
    build_assessment_prompt,
    render_domain_definitions,
)
from src.classification.keyword_signals import compile_profiles, detect_keyword_signals
from src.classification.review_store import DEFAULT_REVIEW_SCHEMA_FILE
from src.classification.signal_store import (
    DEFAULT_SIGNALS_SCHEMA_FILE,
    persist_domain_signals,
)
from src.classification.taxonomy_registry import load_frozen_taxonomy
from src.common.config import get_settings
from src.common.db import init_schema
from src.common.llm_client import OllamaLLMClient
from src.extraction.doc_representation import (
    CLASSIFICATION_STATUS_AUTO_ACCEPTED,
    CLASSIFICATION_STATUS_DROPPED_PROCEDURAL,
    CLASSIFICATION_STATUS_PENDING,
    DocumentRepresentation,
)
from src.extraction.representation_store import (
    DEFAULT_REPRESENTATION_SCHEMA_FILE,
    get_representation,
    update_classification_state,
    upsert_representations,
)

CLASSIFICATION_SCHEMA_FILE = "schemas/classification_schema.sql"

FAMILY_TEXT = (
    "Suit for recovery of dower, dowry articles and maintenance before the Judge "
    "Family Court. The wife seeks dissolution of marriage by khula; custody of the "
    "minor falls under the Guardians and Wards Act and the nikahnama was exhibited. "
)
CRIMINAL_TEXT = (
    "The accused was convicted under the Penal Code and sentenced by the trial "
    "court. The ocular account of the complainant is doubtful, the investigating "
    "officer joined no private witness, and the prosecution failed to prove the "
    "charge; bail is confirmed and the conviction set aside. "
)


@pytest.fixture()
def taxonomy():
    return load_frozen_taxonomy()


@pytest.fixture()
def profiles(taxonomy):
    return compile_profiles(get_settings().domain_signals.profiles, taxonomy)


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
        content_hash=f"hash_{doc_id}",
        court="Lahore High Court",
        decision_date="2021-04-11",
        classification_status=CLASSIFICATION_STATUS_PENDING,
    )
    base.update(overrides)
    return DocumentRepresentation(**base)


@pytest.fixture()
def db(tmp_path) -> Path:
    db_path = tmp_path / "safety.db"
    init_schema(db_path=db_path, schema_file=str(DEFAULT_REPRESENTATION_SCHEMA_FILE))
    init_schema(db_path=db_path, schema_file=str(DEFAULT_SIGNALS_SCHEMA_FILE))
    init_schema(db_path=db_path, schema_file=CLASSIFICATION_SCHEMA_FILE)
    init_schema(db_path=db_path, schema_file=str(DEFAULT_REVIEW_SCHEMA_FILE))
    return db_path


@pytest.fixture()
def validation_dir(tmp_path) -> Path:
    return tmp_path / "cluster_validation"


@pytest.fixture()
def flow_settings(monkeypatch, tmp_path, validation_dir):
    """Real Pydantic models, so these fakes cannot drift from the schema."""

    import orchestration.dags.domain_classification_flow as flow

    real = get_settings()
    monkeypatch.setattr(
        flow, "get_settings",
        lambda: SimpleNamespace(
            domain_decision=real.domain_decision.model_copy(update={"batch_size": 25}),
            domain_signals=real.domain_signals,
            classification=real.classification,
            caselaw=real.caselaw,
            review=real.review,
            discovery=real.discovery,
            cluster_validation=real.cluster_validation.model_copy(
                update={"output_dir": validation_dir}
            ),
            pipeline=SimpleNamespace(checkpoint_dir=tmp_path / "checkpoints"),
            metrics=SimpleNamespace(db_path=tmp_path / "metrics.db"),
        ),
    )
    return real


def _keyword_evidence(text: str, profiles) -> dict:
    signals = detect_keyword_signals("x", text, profiles, min_matches=2)
    return {
        "keyword_signals": signals.as_evidence(),
        "keyword_top_domain": signals.top_domain,
        "keyword_margin": signals.margin,
    }


def _seed_signals(db, profiles, docs, signal_run="sig1"):
    """Store documents plus Phase 3 evidence for each."""

    upsert_representations(docs, db_path=db)
    records = []
    for document in docs:
        text = document.cleaned_text or ""
        family = "dower" in text
        records.append({
            "doc_id": document.doc_id,
            "cluster_id": 0 if family else 1,
            "cluster_confidence": 0.9,
            "llm_domain": "family_law" if family else "criminal_law",
            "llm_confidence": 0.95,
            "llm_status": "ok",
            "llm_reason": "seed",
            "llm_model": "qwen3:14b",
            **_keyword_evidence(text, profiles),
        })
    persist_domain_signals(
        signal_run, records, signal_version=ASSESSMENT_VERSION,
        batch_id="b0", db_path=db,
    )


def _write_phase_4_report(
    directory: Path, run_id: str, useful: bool, verdict: str, n_documents: int = 100
):
    """A Phase 4 report as cluster_validation_flow writes one.

    ``n_documents`` matters: the gate checks it against the run being
    decided, because validating 200 documents does not approve a
    14,000-document clustering.
    """

    directory.mkdir(parents=True, exist_ok=True)
    (directory / f"{run_id}.json").write_text(
        json.dumps({
            "run_id": run_id,
            "n_documents": n_documents,
            "verdict": verdict,
            "cluster_is_a_useful_signal": useful,
        }),
        encoding="utf-8",
    )


# ===========================================================================
# CI-4 -- stale-state protection
# ===========================================================================


def test_eligibility_refuses_only_a_phase_2_structural_drop():
    assert policy.is_eligible_for_decision(CLASSIFICATION_STATUS_PENDING)
    assert policy.is_eligible_for_decision(CLASSIFICATION_STATUS_AUTO_ACCEPTED)
    assert policy.is_eligible_for_decision("needs_review")
    # Phase 5 may re-decide its OWN prior drop...
    assert policy.is_eligible_for_decision("dropped_off_domain")
    # ...but never Phase 2's.
    assert not policy.is_eligible_for_decision(CLASSIFICATION_STATUS_DROPPED_PROCEDURAL)


def test_a_document_that_no_longer_exists_is_not_eligible():
    assert not policy.is_eligible_for_decision(None)


def test_phase_5_does_not_revive_a_document_phase_2_dropped_after_signalling(
    db, flow_settings, profiles, validation_dir
):
    """The CI-4 regression: valid at Phase 3, dropped later, must stay dropped.

    Phase 3 gathered evidence while the document looked substantive. Phase 2
    was then re-run with stricter thresholds and dropped it. The old
    evidence is stale, and using it would put a cause list or an
    adjournment slip into the accepted corpus.
    """

    _seed_signals(db, profiles, [_doc("case_0", FAMILY_TEXT)])
    _write_phase_4_report(
        validation_dir, "sig1", useful=True, verdict="useful", n_documents=1
    )

    # Phase 2 re-runs and drops it as structural noise.
    upsert_representations(
        [_doc("case_0", FAMILY_TEXT,
              classification_status=CLASSIFICATION_STATUS_DROPPED_PROCEDURAL,
              drop_reason="cause_list", cleaned_text=None)],
        db_path=db,
    )

    result = run_domain_classification("dec1", "sig1", db_path=db)

    after = get_representation("case_0", db_path=db)
    assert after.classification_status == CLASSIFICATION_STATUS_DROPPED_PROCEDURAL
    assert after.drop_reason == "cause_list"
    assert after.primary_domain is None
    assert result.ineligible_for_decision == 1
    assert result.write_outcomes.get(WRITE_INELIGIBLE) == 1


def test_the_refusal_is_still_recorded_in_the_audit_trail(
    db, flow_settings, profiles, validation_dir
):
    """What the pipeline WOULD have said stays on record."""

    _seed_signals(db, profiles, [_doc("case_0", FAMILY_TEXT)])
    upsert_representations(
        [_doc("case_0", FAMILY_TEXT,
              classification_status=CLASSIFICATION_STATUS_DROPPED_PROCEDURAL,
              drop_reason="cause_list", cleaned_text=None)],
        db_path=db,
    )
    run_domain_classification("dec1", "sig1", db_path=db)

    rows = {r["doc_id"]: r for r in get_classifications_for_run("dec1", db_path=db)}
    assert "case_0" in rows, "the audit row must be written even when the write is refused"
    assert rows["case_0"]["primary_domain"] == "family_law"


def test_eligible_documents_are_still_decided_normally(
    db, flow_settings, profiles, validation_dir
):
    """The guard must not block the ordinary path."""

    _seed_signals(
        db, profiles,
        [_doc("fam_0", FAMILY_TEXT), _doc("crim_0", CRIMINAL_TEXT)],
    )
    result = run_domain_classification("dec1", "sig1", db_path=db)

    assert result.ineligible_for_decision == 0
    assert result.decided == 2
    assert get_representation("fam_0", db_path=db).primary_domain == "family_law"
    assert get_representation("crim_0", db_path=db).primary_domain == "criminal_law"


def test_the_invariant_cannot_be_configured_away(db, flow_settings, profiles):
    """An operator may freeze extra statuses, never un-freeze a structural drop."""

    _seed_signals(db, profiles, [_doc("case_0", FAMILY_TEXT)])
    upsert_representations(
        [_doc("case_0", FAMILY_TEXT,
              classification_status=CLASSIFICATION_STATUS_DROPPED_PROCEDURAL,
              drop_reason="cause_list", cleaned_text=None)],
        db_path=db,
    )

    # Explicitly empty protection, and reviewed-protection off.
    run_domain_classification(
        "dec1", "sig1", db_path=db, protect_statuses=(), protect_reviewed=False
    )

    assert get_representation("case_0", db_path=db).classification_status == (
        CLASSIFICATION_STATUS_DROPPED_PROCEDURAL
    )


# ===========================================================================
# CI-5 -- deterministic LLM
# ===========================================================================


class _RecordingOllamaClient(OllamaLLMClient):
    """Captures the kwargs that would reach ollama.Client.chat()."""

    def __post_init__(self) -> None:  # skip the real connection
        self.calls: list[dict] = []

        class _Fake:
            def chat(_self, **kwargs):
                self.calls.append(kwargs)
                return {"message": {"content": '{"domain": "family_law", '
                                               '"confidence": 0.9, "reason": "dower"}'}}

        self._client = _Fake()


def test_the_llm_request_is_greedy_seeded_json_and_not_thinking():
    client = _RecordingOllamaClient(model="qwen3:14b")
    client.complete(system="s", prompt="p")

    (call,) = client.calls
    assert call["options"]["temperature"] == 0
    assert call["options"]["seed"] == 42
    assert call["options"]["num_predict"] == 1024
    assert call["format"] == "json"
    assert call["think"] is False


def test_the_token_budget_is_1024_not_the_512_that_truncated_json():
    """512 let qwen3's reasoning block consume the whole budget."""

    assert get_settings().domain_signals.llm_max_tokens == 1024
    assert OllamaLLMClient(model="m").max_tokens == 1024


def test_determinism_settings_are_overridable_for_a_deliberate_experiment():
    client = _RecordingOllamaClient(model="m", temperature=0.7, seed=7)
    client.complete(system="s", prompt="p")

    (call,) = client.calls
    assert call["options"]["temperature"] == 0.7
    assert call["options"]["seed"] == 7


def test_the_same_prompt_yields_the_same_recorded_request():
    """Two identical calls must produce byte-identical requests."""

    client = _RecordingOllamaClient(model="qwen3:14b")
    client.complete(system="s", prompt="p")
    client.complete(system="s", prompt="p")

    first, second = client.calls
    assert first == second


# ===========================================================================
# CI-7 -- the LLM never sees the keyword verdict
# ===========================================================================


def test_the_prompt_carries_no_keyword_evidence(taxonomy, profiles):
    """The CI-7 regression: the old prompt appended a labelled keyword hint."""

    representation = build_case_representation(_doc("d1", CRIMINAL_TEXT))
    prompt = build_assessment_prompt(
        representation, render_domain_definitions(taxonomy)
    )

    lowered = prompt.lower()
    assert "keyword" not in lowered
    assert "lexical signal" not in lowered
    assert "hint" not in lowered
    # No score-shaped leak either, e.g. "criminal_law=0.53".
    assert "=0." not in prompt


def test_the_prompt_builder_cannot_be_handed_keyword_signals(taxonomy, profiles):
    """Not merely unused -- the parameter is gone, so a caller cannot pass it."""

    representation = build_case_representation(_doc("d1", CRIMINAL_TEXT))
    signals = detect_keyword_signals("d1", representation.signal_text, profiles)

    with pytest.raises(TypeError):
        build_assessment_prompt(
            representation, render_domain_definitions(taxonomy), signals
        )


def test_assess_domain_cannot_be_handed_keyword_signals(taxonomy, profiles):
    representation = build_case_representation(_doc("d1", CRIMINAL_TEXT))
    signals = detect_keyword_signals("d1", representation.signal_text, profiles)

    class _Unused:
        def complete(self, system, prompt, max_tokens=None):  # pragma: no cover
            raise AssertionError("should not be reached")

    with pytest.raises(TypeError):
        assess_domain(
            representation, taxonomy, _Unused(), keyword_signals=signals
        )


def test_the_prompt_still_carries_the_document_itself(taxonomy):
    """Removing the hint must not remove the intended inputs."""

    representation = build_case_representation(_doc("d1", FAMILY_TEXT))
    prompt = build_assessment_prompt(
        representation, render_domain_definitions(taxonomy)
    )

    assert "DOMAINS:" in prompt
    assert representation.title in prompt
    assert "Facts" in prompt                       # headings
    assert "dower" in prompt                       # the text
    assert "family_law" in prompt                  # the domain menu


def test_the_assessment_version_records_the_prompt_change():
    """A 1.0 assessment was made by a model shown the keyword verdict."""

    assert ASSESSMENT_VERSION == "domain_assessment/2.0"


def test_phase_3_never_sends_keyword_evidence_to_the_model(
    tmp_path, monkeypatch, profiles
):
    """The actual CI-7 regression: the FLOW used to pass the keyword result.

    Checking `build_assessment_prompt` alone is not enough -- the old
    builder only added the hint when it was handed one, and it was the
    Phase 3 flow doing the handing. This captures every prompt the flow
    really sends.
    """

    import numpy as np

    import orchestration.dags.domain_signal_flow as flow
    from src.classification.signal_store import DEFAULT_SIGNALS_SCHEMA_FILE
    from src.clustering.cluster import ClusterResult, NOISE_LABEL
    from src.embedding.embed_model import DeterministicHashEmbedder

    db_path = tmp_path / "ci7.db"
    init_schema(db_path=db_path, schema_file=str(DEFAULT_REPRESENTATION_SCHEMA_FILE))
    init_schema(db_path=db_path, schema_file=str(DEFAULT_SIGNALS_SCHEMA_FILE))
    init_schema(db_path=db_path, schema_file="schemas/domain_registry_schema.sql")
    init_schema(
        db_path=db_path,
        schema_file=str(get_settings().domain_signals.signature_schema_file),
    )
    upsert_representations(
        [_doc("fam_0", FAMILY_TEXT), _doc("crim_0", CRIMINAL_TEXT)], db_path=db_path
    )

    real = get_settings()
    monkeypatch.setattr(
        flow, "get_settings",
        lambda: SimpleNamespace(
            domain_signals=real.domain_signals,
            classification=real.classification,
            caselaw=real.caselaw,
            pipeline=SimpleNamespace(checkpoint_dir=tmp_path / "checkpoints"),
            metrics=SimpleNamespace(db_path=tmp_path / "metrics.db"),
            discovery=real.discovery.model_copy(
                update={"umap_min_docs": 10_000, "hdbscan_min_cluster_size": 2}
            ),
        ),
    )

    prompts: list[str] = []

    class _Recording:
        def complete(self, system, prompt, max_tokens=None):
            prompts.append(prompt)
            return json.dumps(
                {"domain": "family_law", "confidence": 0.9, "reason": "dower"}
            )

    def _clusterer(vectors: np.ndarray) -> ClusterResult:
        labels = np.array([0 if i % 2 == 0 else 1 for i in range(vectors.shape[0])])
        return ClusterResult(
            labels=labels,
            probabilities=np.full(labels.shape, 0.9),
            n_clusters=len(set(labels.tolist()) - {NOISE_LABEL}),
        )

    flow.run_domain_signals(
        "sig1", db_path=db_path,
        embedder=DeterministicHashEmbedder(),
        clusterer=_clusterer,
        llm_client=_Recording(),
    )

    assert prompts, "the flow should have prompted the model"
    for prompt in prompts:
        lowered = prompt.lower()
        assert "keyword" not in lowered
        assert "lexical signal" not in lowered
        assert "=0." not in prompt          # no "criminal_law=0.53" style leak
        assert "dower" in lowered or "accused" in lowered  # the document IS there


# ===========================================================================
# CI-6 -- the cluster signal is used only on Phase 4's approval
# ===========================================================================


def test_no_phase_4_report_means_no_cluster_signal(validation_dir):
    approved, reason = cluster_signal_is_approved(
        "sig_missing", output_dir=validation_dir
    )
    assert approved is False
    assert "no Phase 4 validation report" in reason


def test_an_unreadable_phase_4_report_fails_closed(validation_dir):
    validation_dir.mkdir(parents=True, exist_ok=True)
    (validation_dir / "sig1.json").write_text("{ truncated", encoding="utf-8")

    approved, reason = cluster_signal_is_approved("sig1", output_dir=validation_dir)
    assert approved is False
    assert "unreadable" in reason


def test_a_negative_phase_4_verdict_disables_the_signal(validation_dir):
    _write_phase_4_report(validation_dir, "sig1", useful=False, verdict="not_useful")

    approved, reason = cluster_signal_is_approved("sig1", output_dir=validation_dir)
    assert approved is False
    assert "not_useful" in reason


def test_a_weak_phase_4_verdict_disables_the_signal(validation_dir):
    """`weak` is not approval: Phase 4 sets the flag False for it."""

    _write_phase_4_report(validation_dir, "sig1", useful=False, verdict="weak")

    approved, _ = cluster_signal_is_approved("sig1", output_dir=validation_dir)
    assert approved is False


def test_an_approving_phase_4_verdict_enables_the_signal(validation_dir):
    _write_phase_4_report(validation_dir, "sig1", useful=True, verdict="useful")

    approved, reason = cluster_signal_is_approved("sig1", output_dir=validation_dir)
    assert approved is True
    assert "useful" in reason


def test_phase_5_disables_the_cluster_weight_without_phase_4_approval(
    db, flow_settings, profiles
):
    """No report written: the cluster contribution must carry no weight."""

    _seed_signals(db, profiles, [_doc("fam_0", FAMILY_TEXT)])
    result = run_domain_classification("dec1", "sig1", db_path=db)

    assert result.cluster_signal_enabled is False
    assert result.clusters_profiled == 0
    evidence = json.loads(
        get_classifications_for_run("dec1", db_path=db)[0]["justification"]
        .split("evidence=", 1)[1]
    )
    cluster = next(s for s in evidence["signals"] if s["signal"] == "cluster")
    assert cluster["weight"] == 0.0
    assert cluster["available"] is False


def test_phase_5_uses_the_cluster_weight_once_phase_4_approves(
    db, flow_settings, profiles, validation_dir
):
    _seed_signals(
        db, profiles,
        [_doc(f"fam_{i}", FAMILY_TEXT) for i in range(3)]
        + [_doc(f"crim_{i}", CRIMINAL_TEXT) for i in range(3)],
    )
    _write_phase_4_report(
        validation_dir, "sig1", useful=True, verdict="useful", n_documents=6
    )

    result = run_domain_classification("dec1", "sig1", db_path=db)

    assert result.cluster_signal_enabled is True
    assert result.clusters_profiled > 0
    evidence = json.loads(
        get_classifications_for_run("dec1", db_path=db)[0]["justification"]
        .split("evidence=", 1)[1]
    )
    cluster = next(s for s in evidence["signals"] if s["signal"] == "cluster")
    assert cluster["weight"] > 0


def test_an_explicit_caller_override_still_wins(db, flow_settings, profiles):
    """A deliberate experiment can force the signal on despite no report."""

    _seed_signals(db, profiles, [_doc("fam_0", FAMILY_TEXT)])
    result = run_domain_classification(
        "dec1", "sig1", db_path=db, cluster_signal_enabled=True
    )
    assert result.cluster_signal_enabled is True


def test_disabling_the_cluster_signal_neutralises_it_rather_than_renormalising(
    db, flow_settings, profiles
):
    """Coverage drops; the other signals keep their own weights."""

    _seed_signals(db, profiles, [_doc("fam_0", FAMILY_TEXT)])
    run_domain_classification("dec1", "sig1", db_path=db)

    evidence = json.loads(
        get_classifications_for_run("dec1", db_path=db)[0]["justification"]
        .split("evidence=", 1)[1]
    )
    weights = {s["signal"]: s["weight"] for s in evidence["signals"]}
    assert weights["cluster"] == 0.0
    # Untouched, not rescaled to absorb the missing 0.15.
    assert weights["keyword"] == pytest.approx(0.40)
    assert weights["llm"] == pytest.approx(0.40)
    assert evidence["coverage"] < 1.0


# ===========================================================================
# CI-4 (widened) -- stale evidence must not overwrite ANY newer state
# ===========================================================================


def _age_the_state(db, doc_id: str, status: str, **columns):
    """Move a document's state to `status`, stamped AFTER its signals.

    Simulates the gap CI-4 is about: Phase 3 gathered evidence, then
    something else changed the document.
    """

    from src.common.db import connection_scope

    later = "2099-01-01T00:00:00+00:00"
    assignments = ", ".join(f"{name} = ?" for name in columns)
    prefix = f"{assignments}, " if columns else ""
    with connection_scope(db) as conn:
        conn.execute(
            f"UPDATE document_representations SET {prefix}"
            "classification_status = ?, updated_at = ? WHERE doc_id = ?",
            (*columns.values(), status, later, doc_id),
        )


def test_evidence_older_than_the_state_is_stale():
    early, late = "2026-01-01T10:00:00+00:00", "2026-01-01T11:00:00+00:00"
    assert policy.evidence_is_stale(early, late, "sig_other", "sig1") is True


def test_evidence_newer_than_the_state_is_not_stale():
    early, late = "2026-01-01T10:00:00+00:00", "2026-01-01T11:00:00+00:00"
    assert policy.evidence_is_stale(late, early, None, "sig1") is False


def test_a_states_own_evidence_is_never_stale_against_it():
    """Phase 5's own write makes the state newer -- that must stay decidable."""

    early, late = "2026-01-01T10:00:00+00:00", "2026-01-01T11:00:00+00:00"
    assert policy.evidence_is_stale(early, late, "sig1", "sig1") is False


def test_missing_timestamps_do_not_block_the_first_pass():
    assert policy.evidence_is_stale(None, "2026-01-01T11:00:00+00:00", None, "s") is False
    assert policy.evidence_is_stale("2026-01-01T11:00:00+00:00", None, None, "s") is False


def test_sqlite_default_timestamp_format_is_compared_correctly():
    """A lexicographic compare across the two formats would be wrong."""

    assert policy.evidence_is_stale(
        "2026-01-01T10:00:00+00:00", "2026-01-01 11:00:00", None, "sig1"
    ) is True


def test_stale_evidence_does_not_overwrite_a_later_off_domain_drop(
    db, flow_settings, profiles
):
    _seed_signals(db, profiles, [_doc("case_0", FAMILY_TEXT)])
    _age_the_state(db, "case_0", "dropped_off_domain",
                   drop_reason="off_domain", domain_confidence=0.91)

    result = run_domain_classification("dec1", "sig1", db_path=db)

    after = get_representation("case_0", db_path=db)
    assert after.classification_status == "dropped_off_domain"
    assert after.drop_reason == "off_domain"
    assert result.skipped_stale_evidence == 1


def test_stale_evidence_does_not_overwrite_a_later_needs_review_state(
    db, flow_settings, profiles
):
    _seed_signals(db, profiles, [_doc("case_0", FAMILY_TEXT)])
    _age_the_state(db, "case_0", "needs_review",
                   primary_domain="criminal_law", domain_confidence=0.55)

    result = run_domain_classification("dec1", "sig1", db_path=db)

    after = get_representation("case_0", db_path=db)
    assert after.classification_status == "needs_review"
    assert after.primary_domain == "criminal_law"
    assert result.skipped_stale_evidence == 1


def test_stale_evidence_does_not_overwrite_a_later_acceptance(
    db, flow_settings, profiles
):
    _seed_signals(db, profiles, [_doc("case_0", CRIMINAL_TEXT)])
    _age_the_state(db, "case_0", CLASSIFICATION_STATUS_AUTO_ACCEPTED,
                   primary_domain="family_law", domain_confidence=0.99)

    run_domain_classification("dec1", "sig1", db_path=db)

    after = get_representation("case_0", db_path=db)
    assert after.primary_domain == "family_law"
    assert after.domain_confidence == pytest.approx(0.99)


def test_a_stale_skip_is_still_fully_audited(db, flow_settings, profiles):
    _seed_signals(db, profiles, [_doc("case_0", FAMILY_TEXT)])
    _age_the_state(db, "case_0", "needs_review", primary_domain="criminal_law")

    run_domain_classification("dec1", "sig1", db_path=db)

    rows = {r["doc_id"]: r for r in get_classifications_for_run("dec1", db_path=db)}
    assert rows["case_0"]["primary_domain"] == "family_law"
    assert rows["case_0"]["signal_run_id"] == "sig1"


def test_a_current_document_is_still_written_normally(db, flow_settings, profiles):
    """The guard must not block the ordinary path: evidence newer than state."""

    _seed_signals(db, profiles, [_doc("fam_0", FAMILY_TEXT)])

    result = run_domain_classification("dec1", "sig1", db_path=db)

    assert result.skipped_stale_evidence == 0
    assert get_representation("fam_0", db_path=db).primary_domain == "family_law"


def test_resuming_the_same_run_is_not_blocked_by_its_own_writes(
    db, flow_settings, profiles
):
    """CI-3 rerun protection intact: the same evidence may re-decide."""

    _seed_signals(db, profiles, [_doc("fam_0", FAMILY_TEXT)])
    run_domain_classification("dec1", "sig1", db_path=db)

    # A second decision run over the SAME evidence: its own prior write made
    # the state newer, which must not make the evidence stale.
    second = run_domain_classification("dec2", "sig1", db_path=db)

    assert second.skipped_stale_evidence == 0
    assert second.decided == 1
    assert get_representation("fam_0", db_path=db).primary_domain == "family_law"


def test_newer_evidence_may_still_supersede_an_older_decision(
    db, flow_settings, profiles
):
    """Staleness is directional -- fresh evidence is allowed to win."""

    docs = [_doc("fam_0", FAMILY_TEXT)]
    _seed_signals(db, profiles, docs, signal_run="sig1")
    run_domain_classification("dec1", "sig1", db_path=db)

    # A NEW signal run, gathered after that decision.
    _seed_signals(db, profiles, docs, signal_run="sig2")
    second = run_domain_classification("dec2", "sig2", db_path=db)

    assert second.skipped_stale_evidence == 0
    assert second.decided == 1


# ===========================================================================
# Contract tests named by the hardening spec
# ===========================================================================


def test_the_llm_output_contract_is_unchanged(taxonomy):
    """CI-5: {domain, confidence, reason} still validates as before."""

    from src.classification.domain_assessment import STATUS_OK, validate_assessment

    assessment = validate_assessment(
        {"domain": "family_law", "confidence": 0.9, "reason": "dower claimed"},
        taxonomy,
        "d1",
    )
    assert assessment.status == STATUS_OK
    assert assessment.domain == "family_law"
    assert assessment.confidence == pytest.approx(0.9)
    assert assessment.reason == "dower claimed"


def test_keyword_classification_still_works_on_its_own(profiles):
    """CI-7: removing the coupling must not disturb the keyword signal."""

    signals = detect_keyword_signals("d1", FAMILY_TEXT, profiles, min_matches=2)
    assert signals.top_domain == "family_law"
    assert signals.scores["family_law"].score > 0


def test_phase_5_receives_keyword_and_llm_as_separate_signals(
    db, flow_settings, profiles
):
    """CI-7: both signals reach the decision, independently weighted."""

    _seed_signals(db, profiles, [_doc("fam_0", FAMILY_TEXT)])
    run_domain_classification("dec1", "sig1", db_path=db)

    evidence = json.loads(
        get_classifications_for_run("dec1", db_path=db)[0]["justification"]
        .split("evidence=", 1)[1]
    )
    by_signal = {s["signal"]: s for s in evidence["signals"]}
    assert by_signal["keyword"]["available"] is True
    assert by_signal["llm"]["available"] is True
    assert by_signal["keyword"]["weight"] == pytest.approx(0.40)
    assert by_signal["llm"]["weight"] == pytest.approx(0.40)


# ===========================================================================
# CI-4 (hardening) -- the refusal is atomic, and signal runs never collide
# ===========================================================================


def test_both_refusals_are_decided_inside_the_write_transaction(db, profiles):
    """The row is read at write time, not from a snapshot taken earlier.

    A caller that checked a snapshot would leave a window in which the row
    could move between the check and the UPDATE. These call the store
    directly, so what is under test is the store's own guard.
    """

    from src.extraction.representation_store import (
        WRITE_INELIGIBLE as STORE_INELIGIBLE,
        WRITE_STALE as STORE_STALE,
        WRITE_UPDATED,
    )

    upsert_representations([_doc("case_0", FAMILY_TEXT)], db_path=db)

    # Structurally dropped between the caller's read and its write.
    _age_the_state(db, "case_0", CLASSIFICATION_STATUS_DROPPED_PROCEDURAL,
                   drop_reason="cause_list")
    assert update_classification_state(
        "case_0", classification_status=CLASSIFICATION_STATUS_AUTO_ACCEPTED,
        primary_domain="family_law",
        ineligible_statuses=("dropped_procedural",),
        db_path=db,
    ) == STORE_INELIGIBLE

    # Moved by someone else after this caller's evidence was gathered.
    _age_the_state(db, "case_0", "needs_review", primary_domain="criminal_law")
    assert update_classification_state(
        "case_0", classification_status=CLASSIFICATION_STATUS_AUTO_ACCEPTED,
        primary_domain="family_law",
        evidence_created_at="2026-01-01T00:00:00+00:00",
        evidence_run_id="sig1",
        state_evidence_run_id="sig_other",
        db_path=db,
    ) == STORE_STALE

    # Its own evidence: allowed.
    assert update_classification_state(
        "case_0", classification_status=CLASSIFICATION_STATUS_AUTO_ACCEPTED,
        primary_domain="family_law",
        evidence_created_at="2026-01-01T00:00:00+00:00",
        evidence_run_id="sig1",
        state_evidence_run_id="sig1",
        db_path=db,
    ) == WRITE_UPDATED


def test_a_write_with_no_evidence_metadata_still_works(db):
    """Other callers (Phase 6 review application) pass no evidence at all."""

    from src.extraction.representation_store import WRITE_UPDATED

    upsert_representations([_doc("case_0", FAMILY_TEXT)], db_path=db)
    assert update_classification_state(
        "case_0", classification_status=CLASSIFICATION_STATUS_AUTO_ACCEPTED,
        primary_domain="family_law", db_path=db,
    ) == WRITE_UPDATED


def test_an_older_signal_run_cannot_overwrite_a_newer_ones_evidence(
    db, flow_settings, profiles
):
    """Signals are keyed by (run_id, doc_id): runs accumulate, never collide."""

    from src.classification.signal_store import get_signals_for_run

    docs = [_doc("case_0", FAMILY_TEXT)]
    _seed_signals(db, profiles, docs, signal_run="sig_old")
    _seed_signals(db, profiles, docs, signal_run="sig_new")

    old_rows = get_signals_for_run("sig_old", db_path=db)
    new_rows = get_signals_for_run("sig_new", db_path=db)

    assert len(old_rows) == 1 and len(new_rows) == 1
    # Both survive independently -- neither run overwrote the other.
    assert old_rows[0]["doc_id"] == new_rows[0]["doc_id"]


def test_a_stale_decision_run_cannot_undo_a_newer_decision_run(
    db, flow_settings, profiles, validation_dir
):
    """dec2 (newer evidence) decides; dec1 (older evidence) must not undo it."""

    docs = [_doc("case_0", CRIMINAL_TEXT)]
    _seed_signals(db, profiles, docs, signal_run="sig_old")
    _seed_signals(db, profiles, docs, signal_run="sig_new")

    run_domain_classification("dec_new", "sig_new", db_path=db)
    after_new = get_representation("case_0", db_path=db)

    stale = run_domain_classification("dec_old", "sig_old", db_path=db)

    assert stale.skipped_stale_evidence == 1
    assert get_representation("case_0", db_path=db).updated_at == after_new.updated_at \
        if hasattr(after_new, "updated_at") else True
    assert get_representation("case_0", db_path=db).primary_domain == "criminal_law"


# ===========================================================================
# CI-6 (hardening) -- an approval must still apply to what is being decided
# ===========================================================================


def test_validation_of_a_different_document_count_is_incompatible(
    db, flow_settings, profiles, validation_dir
):
    """Validating 200 documents does not approve a 14,000-document clustering."""

    _seed_signals(db, profiles, [_doc(f"fam_{i}", FAMILY_TEXT) for i in range(3)])
    _write_phase_4_report(
        validation_dir, "sig1", useful=True, verdict="useful", n_documents=200
    )

    result = run_domain_classification("dec1", "sig1", db_path=db)
    assert result.cluster_signal_enabled is False


def test_validation_computed_on_another_embedding_model_is_incompatible(
    db, flow_settings, profiles, validation_dir
):
    """Different vectors mean different clusters."""

    from src.classification.signal_store import persist_domain_signals

    docs = [_doc("fam_0", FAMILY_TEXT)]
    upsert_representations(docs, db_path=db)
    persist_domain_signals(
        "sig1",
        [{
            "doc_id": "fam_0", "cluster_id": 0, "cluster_confidence": 0.9,
            "embedding_model": "some-other/encoder",      # not the configured one
            "llm_domain": "family_law", "llm_confidence": 0.95, "llm_status": "ok",
            "llm_reason": "seed", "llm_model": "qwen3:14b",
            **_keyword_evidence(FAMILY_TEXT, profiles),
        }],
        signal_version=ASSESSMENT_VERSION, batch_id="b0", db_path=db,
    )
    _write_phase_4_report(
        validation_dir, "sig1", useful=True, verdict="useful", n_documents=1
    )

    result = run_domain_classification("dec1", "sig1", db_path=db)
    assert result.cluster_signal_enabled is False


def test_a_compatible_approval_enables_the_signal(
    db, flow_settings, profiles, validation_dir
):
    _seed_signals(db, profiles, [_doc(f"fam_{i}", FAMILY_TEXT) for i in range(3)])
    _write_phase_4_report(
        validation_dir, "sig1", useful=True, verdict="useful", n_documents=3
    )

    result = run_domain_classification("dec1", "sig1", db_path=db)
    assert result.cluster_signal_enabled is True


def test_compatibility_is_not_checked_when_no_evidence_is_supplied(validation_dir):
    """The bare approval check stays usable for callers without signal rows."""

    _write_phase_4_report(validation_dir, "sig1", useful=True, verdict="useful")
    approved, _ = cluster_signal_is_approved("sig1", output_dir=validation_dir)
    assert approved is True


def test_a_raw_cluster_id_never_becomes_a_domain_label(
    db, flow_settings, profiles, validation_dir
):
    """Cluster 0 is not family_law -- it only votes via its other members."""

    _seed_signals(
        db, profiles,
        [_doc(f"fam_{i}", FAMILY_TEXT) for i in range(3)]
        + [_doc(f"crim_{i}", CRIMINAL_TEXT) for i in range(3)],
    )
    _write_phase_4_report(
        validation_dir, "sig1", useful=True, verdict="useful", n_documents=6
    )
    run_domain_classification("dec1", "sig1", db_path=db)

    for row in get_classifications_for_run("dec1", db_path=db):
        evidence = json.loads(row["justification"].split("evidence=", 1)[1])
        cluster = next(s for s in evidence["signals"] if s["signal"] == "cluster")
        # Scores are shares over the taxonomy's domains, never a cluster id.
        assert set(cluster["domain_scores"]) <= {"family_law", "criminal_law"}
        assert row["primary_domain"] in {"family_law", "criminal_law", "other_uncertain"}


def test_phase_5_still_decides_when_the_cluster_signal_is_neutral(
    db, flow_settings, profiles
):
    """keyword + llm + title + coverage must carry the decision alone."""

    _seed_signals(db, profiles, [_doc("fam_0", FAMILY_TEXT)])
    result = run_domain_classification("dec1", "sig1", db_path=db)

    assert result.cluster_signal_enabled is False
    assert result.decided == 1
    assert get_representation("fam_0", db_path=db).primary_domain == "family_law"


# ===========================================================================
# CI-5 (hardening) -- determinism end to end, and no invented parameters
# ===========================================================================


def test_identical_input_gives_identical_parsed_output(taxonomy):
    """The whole path -- request, response, parse -- must be reproducible."""

    from src.classification.case_representation import build_case_representation
    from src.classification.domain_assessment import assess_domain

    representation = build_case_representation(_doc("d1", FAMILY_TEXT))
    client = _RecordingOllamaClient(model="qwen3:14b")

    first = assess_domain(representation, taxonomy, client)
    second = assess_domain(representation, taxonomy, client)

    assert (first.domain, first.confidence, first.reason, first.status) == (
        second.domain, second.confidence, second.reason, second.status
    )
    # ...and both requests were byte-identical.
    assert client.calls[0] == client.calls[1]


def test_no_unsupported_generation_parameters_are_passed():
    """Only parameters this provider's chat() actually accepts."""

    import inspect

    import ollama

    client = _RecordingOllamaClient(model="qwen3:14b")
    client.complete(system="s", prompt="p")
    (call,) = client.calls

    accepted = set(inspect.signature(ollama.Client.chat).parameters)
    assert set(call) <= accepted, f"unsupported kwargs: {set(call) - accepted}"
    # And no invented options keys.
    assert set(call["options"]) == {"num_predict", "temperature", "seed"}


def test_malformed_model_output_is_handled_safely(taxonomy):
    """A bad response becomes a recorded failure, never a guessed domain."""

    from src.classification.case_representation import build_case_representation
    from src.classification.domain_assessment import STATUS_FAILED, assess_domain

    class _Garbage:
        def complete(self, system, prompt, max_tokens=None):
            return "I think this is probably a family matter, honestly."

    assessment = assess_domain(
        build_case_representation(_doc("d1", FAMILY_TEXT)), taxonomy, _Garbage()
    )
    assert assessment.status == STATUS_FAILED
    assert assessment.domain is None


def test_schema_validation_rejects_a_domain_outside_the_taxonomy(taxonomy):
    from src.classification.domain_assessment import STATUS_FAILED, validate_assessment

    assessment = validate_assessment(
        {"domain": "tax_law", "confidence": 0.9, "reason": "tax"}, taxonomy, "d1"
    )
    assert assessment.status == STATUS_FAILED
    assert assessment.domain is None


# ===========================================================================
# CI-7 (hardening) -- keyword matches cannot reach the model at all
# ===========================================================================


def test_changing_keyword_matches_does_not_change_the_prompt(taxonomy, profiles):
    """The decisive test: vary the keyword result, the prompt must not move."""

    from src.classification.case_representation import build_case_representation
    from src.classification.domain_assessment import (
        build_assessment_prompt,
        render_domain_definitions,
    )

    representation = build_case_representation(_doc("d1", FAMILY_TEXT))
    block = render_domain_definitions(taxonomy)
    baseline = build_assessment_prompt(representation, block)

    # A profile that matches nothing, and one that matches heavily -- the
    # prompt builder has no parameter through which either could enter.
    for alternative in ({"family_law": ["zzzz"]}, {"family_law": ["dower"] * 3}):
        compiled = compile_profiles(alternative, taxonomy)
        signals = detect_keyword_signals(
            "d1", representation.signal_text, compiled, min_matches=1
        )
        assert signals is not None  # the keyword signal still exists...
        assert build_assessment_prompt(representation, block) == baseline  # ...unseen


def test_phase_5_combines_all_three_independent_signals(
    db, flow_settings, profiles, validation_dir
):
    """Qwen + keywords + HDBSCAN reach the decision separately."""

    _seed_signals(
        db, profiles,
        [_doc(f"fam_{i}", FAMILY_TEXT) for i in range(3)]
        + [_doc(f"crim_{i}", CRIMINAL_TEXT) for i in range(3)],
    )
    _write_phase_4_report(
        validation_dir, "sig1", useful=True, verdict="useful", n_documents=6
    )
    run_domain_classification("dec1", "sig1", db_path=db)

    evidence = json.loads(
        get_classifications_for_run("dec1", db_path=db)[0]["justification"]
        .split("evidence=", 1)[1]
    )
    by_signal = {s["signal"]: s for s in evidence["signals"]}
    assert {"keyword", "llm", "cluster", "title"} == set(by_signal)
    for name in ("keyword", "llm", "cluster"):
        assert by_signal[name]["available"] is True, f"{name} should contribute"
