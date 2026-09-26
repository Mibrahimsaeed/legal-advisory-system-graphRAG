"""Phase 5: multi-signal broad domain decision.

The contract under test:

* the decision is **deterministic** -- same evidence, same verdict, no
  model call,
* only four signals feed it, and source-folder labels are not among them,
* ambiguity becomes ``other_uncertain``, and conflict becomes review --
  never a forced label,
* a cluster never votes for its own member, and carries no weight at all
  when Phase 4 says it is not a useful signal,
* the verdict lands on Phase 1's existing fields, with the evidence
  recoverable from the audit row.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from orchestration.dags.domain_classification_flow import run_domain_classification
from src.classification.classification_store import (
    get_classification_history,
    get_classifications_for_run,
)
from src.classification.domain_decision import (
    DECISION_VERSION,
    DROP_REASON_OFF_DOMAIN,
    REVIEW_LOW_COVERAGE,
    REVIEW_NARROW_MARGIN,
    REVIEW_NO_DOMAIN_SUPPORT,
    REVIEW_NO_EVIDENCE,
    REVIEW_SIGNAL_CONFLICT,
    REVIEW_UNCERTAIN,
    STATUS_AUTO_ACCEPTED,
    STATUS_DROPPED_OFF_DOMAIN,
    STATUS_NEEDS_REVIEW,
    build_cluster_profiles,
    decide_domain,
)
from src.classification.keyword_signals import compile_profiles, detect_keyword_signals
from src.classification.signal_store import (
    DEFAULT_SIGNALS_SCHEMA_FILE,
    persist_domain_signals,
)
from src.classification.taxonomy_registry import OTHER_DOMAIN_ID, load_frozen_taxonomy
from src.common.config import get_settings
from src.common.db import connection_scope, init_schema
from src.extraction.doc_representation import (
    CLASSIFICATION_STATUS_DROPPED_PROCEDURAL,
    CLASSIFICATION_STATUS_PENDING,
    DocumentRepresentation,
)
from src.extraction.representation_store import (
    DEFAULT_REPRESENTATION_SCHEMA_FILE,
    get_representation,
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
NEUTRAL_TEXT = (
    "This reference concerns the assessment of income tax and the limitation "
    "period prescribed by the Ordinance; the taxpayer contends it was time barred. "
)

WEIGHTS = {"keyword": 0.40, "llm": 0.40, "cluster": 0.15, "title": 0.05}


@pytest.fixture()
def taxonomy():
    return load_frozen_taxonomy()


@pytest.fixture()
def profiles(taxonomy):
    return compile_profiles(get_settings().domain_signals.profiles, taxonomy)


def _keyword_evidence(text: str, profiles) -> dict:
    """Real keyword evidence, so the tests exercise the stored shape."""

    signals = detect_keyword_signals("x", text, profiles, min_matches=2)
    return {
        "keyword_signals": signals.as_evidence(),
        "keyword_top_domain": signals.top_domain,
        "keyword_margin": signals.margin,
    }


def _row(
    doc_id="d1",
    text=None,
    profiles=None,
    llm_domain=None,
    llm_confidence=None,
    llm_status="ok",
    cluster_id=None,
    **overrides,
):
    row = {
        "doc_id": doc_id,
        "cluster_id": cluster_id,
        "keyword_signals": {},
        "keyword_top_domain": None,
        "keyword_margin": None,
        "llm_domain": llm_domain,
        "llm_confidence": llm_confidence,
        "llm_status": llm_status if llm_domain or llm_status != "ok" else "skipped",
        "llm_reason": "",
    }
    if text is not None:
        row.update(_keyword_evidence(text, profiles))
    row.update(overrides)
    return row


def _decide(row, taxonomy, **kwargs):
    kwargs.setdefault("weights", WEIGHTS)
    return decide_domain(row, taxonomy, **kwargs)


# ---------------------------------------------------------------------------
# 1. The decision is deterministic and makes no model call
# ---------------------------------------------------------------------------


def test_same_evidence_always_yields_the_same_verdict(taxonomy, profiles):
    row = _row(text=FAMILY_TEXT, profiles=profiles, llm_domain="family_law",
               llm_confidence=0.9)

    verdicts = [_decide(row, taxonomy) for _ in range(5)]
    assert len({(v.primary_domain, v.confidence, v.status) for v in verdicts}) == 1


def test_decision_makes_no_llm_call(taxonomy, profiles):
    """The stochastic step stayed in Phase 3; this one only reads it."""

    import src.common.llm_client as llm_client

    def _explode(*args, **kwargs):  # pragma: no cover - must never run
        raise AssertionError("Phase 5 must not call an LLM")

    original = llm_client.get_llm_client
    llm_client.get_llm_client = _explode
    try:
        decision = _decide(
            _row(text=CRIMINAL_TEXT, profiles=profiles, llm_domain="criminal_law",
                 llm_confidence=0.9),
            taxonomy,
        )
    finally:
        llm_client.get_llm_client = original

    assert decision.primary_domain == "criminal_law"


def test_evidence_is_json_serialisable_for_the_audit_trail(taxonomy, profiles):
    decision = _decide(
        _row(text=FAMILY_TEXT, profiles=profiles, llm_domain="family_law",
             llm_confidence=0.85),
        taxonomy,
    )
    payload = json.loads(json.dumps(decision.evidence()))

    assert payload["decision_version"] == DECISION_VERSION
    assert {s["signal"] for s in payload["signals"]} == {
        "keyword", "llm", "cluster", "title"
    }
    # Every signal records its weight, so a verdict can be re-derived.
    assert all("weight" in s for s in payload["signals"])
    assert decision.justification()


# ---------------------------------------------------------------------------
# 2. Agreement, disagreement and the absence of evidence
# ---------------------------------------------------------------------------


def test_agreeing_signals_auto_accept(taxonomy, profiles):
    decision = _decide(
        _row(text=FAMILY_TEXT, profiles=profiles, llm_domain="family_law",
             llm_confidence=1.0),
        taxonomy,
    )
    assert decision.primary_domain == "family_law"
    assert decision.status == STATUS_AUTO_ACCEPTED
    assert decision.review_reasons == []


def test_criminal_evidence_decides_criminal(taxonomy, profiles):
    decision = _decide(
        _row(text=CRIMINAL_TEXT, profiles=profiles, llm_domain="criminal_law",
             llm_confidence=1.0),
        taxonomy,
    )
    assert decision.primary_domain == "criminal_law"
    assert decision.status == STATUS_AUTO_ACCEPTED


def test_conflicting_keyword_and_llm_go_to_review_not_arithmetic(taxonomy, profiles):
    """A genuine disagreement is a human's call, however the sums land."""

    decision = _decide(
        _row(text=FAMILY_TEXT, profiles=profiles, llm_domain="criminal_law",
             llm_confidence=1.0),
        taxonomy,
    )
    assert decision.status == STATUS_NEEDS_REVIEW
    assert REVIEW_SIGNAL_CONFLICT in decision.review_reasons


def test_no_evidence_at_all_is_uncertain_never_a_coin_flip(taxonomy):
    decision = _decide(_row(llm_status="skipped"), taxonomy)

    assert decision.primary_domain == OTHER_DOMAIN_ID
    assert decision.status == STATUS_NEEDS_REVIEW
    assert REVIEW_NO_EVIDENCE in decision.review_reasons
    assert decision.confidence == 0.0
    assert decision.coverage == 0.0


def test_a_gap_in_the_evidence_reads_differently_from_a_neither_answer(taxonomy, profiles):
    """A reviewer must be able to tell "nothing was visible" from "nothing applies"."""

    nothing_observed = _decide(_row(llm_status="skipped"), taxonomy)
    neither_answered = _decide(
        _row(text=NEUTRAL_TEXT, profiles=profiles, llm_domain=OTHER_DOMAIN_ID,
             llm_confidence=0.9),
        taxonomy,
    )

    assert REVIEW_NO_EVIDENCE in nothing_observed.review_reasons
    assert REVIEW_NO_DOMAIN_SUPPORT in neither_answered.review_reasons
    assert REVIEW_NO_EVIDENCE not in neither_answered.review_reasons


def test_weak_support_for_everything_is_not_support_for_something(taxonomy, profiles):
    decision = _decide(
        _row(text=NEUTRAL_TEXT, profiles=profiles, llm_domain="family_law",
             llm_confidence=0.3),
        taxonomy,
        min_domain_score=0.35,
    )
    assert decision.primary_domain == OTHER_DOMAIN_ID
    assert REVIEW_UNCERTAIN in decision.review_reasons


def test_a_narrow_margin_resolves_to_uncertain(taxonomy):
    """Both domains scoring alike means the evidence did not separate them."""

    row = _row(
        keyword_signals={
            "total_matches": 8,
            "domains": {
                "family_law": {"score": 0.62},
                "criminal_law": {"score": 0.60},
            },
        },
        keyword_top_domain="family_law",
        llm_status="skipped",
    )
    decision = _decide(row, taxonomy, min_margin=0.10)

    assert decision.primary_domain == OTHER_DOMAIN_ID
    assert REVIEW_NARROW_MARGIN in decision.review_reasons
    assert decision.margin < 0.10


def test_low_confidence_verdict_is_reviewed_not_accepted(taxonomy, profiles):
    decision = _decide(
        _row(text=FAMILY_TEXT, profiles=profiles, llm_status="skipped"),
        taxonomy,
        auto_accept_threshold=0.95,
    )
    assert decision.primary_domain == "family_law"
    assert decision.status == STATUS_NEEDS_REVIEW


# ---------------------------------------------------------------------------
# 3. other_uncertain and the off-domain drop
# ---------------------------------------------------------------------------


def test_confident_neither_with_no_keyword_support_is_dropped_off_domain(taxonomy, profiles):
    decision = _decide(
        _row(text=NEUTRAL_TEXT, profiles=profiles, llm_domain=OTHER_DOMAIN_ID,
             llm_confidence=0.95),
        taxonomy,
        off_domain_drop_confidence=0.70,
    )
    assert decision.primary_domain == OTHER_DOMAIN_ID
    assert decision.status == STATUS_DROPPED_OFF_DOMAIN
    assert decision.drop_reason == DROP_REASON_OFF_DOMAIN


def test_unconfident_neither_is_reviewed_not_discarded(taxonomy, profiles):
    decision = _decide(
        _row(text=NEUTRAL_TEXT, profiles=profiles, llm_domain=OTHER_DOMAIN_ID,
             llm_confidence=0.4),
        taxonomy,
        off_domain_drop_confidence=0.70,
    )
    assert decision.status == STATUS_NEEDS_REVIEW
    assert decision.drop_reason is None


def test_keyword_evidence_blocks_an_off_domain_drop(taxonomy, profiles):
    """The LLM saying "neither" does not discard a document the profiles matched."""

    decision = _decide(
        _row(text=FAMILY_TEXT, profiles=profiles, llm_domain=OTHER_DOMAIN_ID,
             llm_confidence=0.99),
        taxonomy,
        off_domain_drop_confidence=0.70,
    )
    assert decision.status != STATUS_DROPPED_OFF_DOMAIN


def test_a_failed_assessment_is_not_treated_as_neither(taxonomy, profiles):
    decision = _decide(
        _row(text=FAMILY_TEXT, profiles=profiles, llm_domain=None,
             llm_status="failed"),
        taxonomy,
    )
    llm = next(c for c in decision.contributions if c.signal == "llm")
    assert not llm.available
    assert decision.primary_domain == "family_law"  # decided on the rest


def test_only_the_three_broad_outcomes_are_possible(taxonomy, profiles):
    rows = [
        _row(text=FAMILY_TEXT, profiles=profiles, llm_domain="family_law", llm_confidence=0.9),
        _row(text=CRIMINAL_TEXT, profiles=profiles, llm_domain="criminal_law", llm_confidence=0.9),
        _row(text=NEUTRAL_TEXT, profiles=profiles, llm_domain=OTHER_DOMAIN_ID, llm_confidence=0.9),
        _row(llm_status="skipped"),
    ]
    outcomes = {_decide(r, taxonomy).primary_domain for r in rows}
    assert outcomes <= {"family_law", "criminal_law", OTHER_DOMAIN_ID}


# ---------------------------------------------------------------------------
# 4. The cluster signal: corroboration, never a label
# ---------------------------------------------------------------------------


def test_cluster_profile_ignores_noise_and_tiny_clusters():
    rows = [
        {"cluster_id": 0, "keyword_top_domain": "family_law", "llm_domain": "family_law"},
        {"cluster_id": 0, "keyword_top_domain": "family_law", "llm_domain": "family_law"},
        {"cluster_id": 0, "keyword_top_domain": "family_law", "llm_domain": None},
        {"cluster_id": 1, "keyword_top_domain": "criminal_law", "llm_domain": "criminal_law"},
        {"cluster_id": -1, "keyword_top_domain": "family_law", "llm_domain": "family_law"},
    ]
    profiles = build_cluster_profiles(rows, ["family_law", "criminal_law"], min_cluster_members=3)

    assert set(profiles) == {0}  # cluster 1 too small, -1 is not a cluster


def test_a_document_never_votes_for_itself(taxonomy):
    """Leave-one-out: otherwise a cluster merely echoes its members back."""

    domains = ["family_law", "criminal_law"]
    rows = [
        _row(doc_id=f"f{i}", cluster_id=0, keyword_top_domain="family_law",
             llm_domain="family_law", llm_confidence=0.9)
        for i in range(3)
    ]
    # A lone criminal document inside an otherwise family cluster.
    odd = _row(doc_id="c0", cluster_id=0, keyword_top_domain="criminal_law",
               llm_domain="criminal_law", llm_confidence=0.9)
    profiles = build_cluster_profiles(rows + [odd], domains, min_cluster_members=3)

    contribution = _decide(
        odd, taxonomy, cluster_profiles=profiles
    ).contributions
    cluster = next(c for c in contribution if c.signal == "cluster")

    # Its own criminal votes are removed, so the cluster reports family.
    assert cluster.domain_scores["family_law"] == 1.0
    assert cluster.domain_scores["criminal_law"] == 0.0


def test_cluster_noise_is_not_a_signal(taxonomy, profiles):
    decision = _decide(
        _row(text=FAMILY_TEXT, profiles=profiles, cluster_id=-1, llm_status="skipped"),
        taxonomy,
        cluster_profiles={},
    )
    cluster = next(c for c in decision.contributions if c.signal == "cluster")
    assert not cluster.available
    assert "noise" in cluster.detail


def test_zero_cluster_weight_removes_the_signal_entirely(taxonomy, profiles):
    """Phase 4's "not useful" verdict must be honourable in configuration."""

    row = _row(doc_id="f0", text=FAMILY_TEXT, profiles=profiles, cluster_id=0,
               keyword_top_domain="family_law", llm_status="skipped")
    cluster_profiles = build_cluster_profiles(
        [_row(doc_id=f"x{i}", cluster_id=0, keyword_top_domain="criminal_law",
              llm_domain="criminal_law", llm_confidence=0.9) for i in range(4)],
        ["family_law", "criminal_law"],
    )

    with_cluster = _decide(row, taxonomy, cluster_profiles=cluster_profiles)
    without = _decide(
        row, taxonomy,
        weights={**WEIGHTS, "cluster": 0.0},
        cluster_profiles=cluster_profiles,
    )

    assert with_cluster.scores["criminal_law"] > 0
    assert without.scores["criminal_law"] == 0
    cluster = next(c for c in without.contributions if c.signal == "cluster")
    assert not cluster.available


def test_disabling_cluster_lowers_coverage_not_the_verdict(taxonomy, profiles):
    """Losing a signal is recorded as thinner evidence, not as a different answer.

    Confidence is a share of what fired, so dropping a corroborating
    signal does not change where the remaining evidence points -- the
    honest record of the loss is coverage, which the review rules read.
    """

    row = _row(doc_id="f0", text=FAMILY_TEXT, profiles=profiles, cluster_id=0,
               keyword_top_domain="family_law", llm_domain="family_law",
               llm_confidence=1.0)
    cluster_profiles = build_cluster_profiles(
        [_row(doc_id=f"f{i}", cluster_id=0, keyword_top_domain="family_law",
              llm_domain="family_law", llm_confidence=0.9) for i in range(1, 5)]
        + [row],
        ["family_law", "criminal_law"],
    )

    with_cluster = _decide(row, taxonomy, cluster_profiles=cluster_profiles)
    without = _decide(row, taxonomy, weights={**WEIGHTS, "cluster": 0.0},
                      cluster_profiles=cluster_profiles)

    assert without.coverage < with_cluster.coverage
    assert without.primary_domain == with_cluster.primary_domain


def test_thin_evidence_cannot_auto_accept(taxonomy, profiles):
    """One signal alone is never enough, however unanimous it looks."""

    keyword_only = _decide(
        _row(text=FAMILY_TEXT, profiles=profiles, llm_status="skipped"), taxonomy
    )
    assert keyword_only.primary_domain == "family_law"
    assert keyword_only.coverage == pytest.approx(0.40)
    assert keyword_only.status == STATUS_NEEDS_REVIEW
    assert REVIEW_LOW_COVERAGE in keyword_only.review_reasons


def test_coverage_shortfall_scales_confidence_down(taxonomy, profiles):
    """A lone weak signal must not report near-certainty.

    Without this, share-normalisation would hand a single title match a
    confidence of 1.00 -- the winner's share of a very small pie.
    """

    title_only = _decide(
        _row(llm_status="skipped"), taxonomy,
        compiled_profiles=profiles,
        title="Mst. Zainab v. Muhammad Ali (suit for dower and maintenance)",
    )
    assert title_only.coverage == pytest.approx(0.05)
    assert title_only.confidence < 0.2

    # At or above the coverage floor the tempering does nothing at all.
    full = _decide(
        _row(text=FAMILY_TEXT, profiles=profiles, llm_domain="family_law",
             llm_confidence=1.0),
        taxonomy,
    )
    assert full.confidence == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# 5. What must NOT be a signal
# ---------------------------------------------------------------------------


def test_source_folder_is_never_read_as_a_signal(taxonomy, profiles):
    """Folder labels are Phase 4 validation ground truth, not evidence."""

    base = _row(text=NEUTRAL_TEXT, profiles=profiles, llm_status="skipped")
    misleading = {
        **base,
        "source_relpath": "family_law/case_0001/case.html",
        "source_folder": "family_law",
        "label": "family_law",
    }
    assert _decide(base, taxonomy).scores == _decide(misleading, taxonomy).scores


def test_phase_5_does_not_import_the_validation_module():
    """Importing it would make the Phase 4 evaluation circular."""

    source = Path("src/classification/domain_decision.py").read_text()
    flow = Path("orchestration/dags/domain_classification_flow.py").read_text()
    for text in (source, flow):
        assert "cluster_validation" not in text.replace(
            # the docstrings name it only to say it is absent
            "cluster_validation`", ""
        ).split("\"\"\"")[-1]


def test_the_superseded_weighting_scheme_is_not_implemented():
    """No statute/court/source_folder weights anywhere in the Phase 5 engine."""

    source = Path("src/classification/domain_decision.py").read_text()
    for absent in ("0.45", "statute", "court"):
        # allowed only inside the docstring that explains their absence
        body = source.split('"""', 2)[-1]
        assert absent not in body, f"{absent!r} leaked into the decision engine"


def test_court_and_date_are_not_signals(taxonomy, profiles):
    base = _row(text=NEUTRAL_TEXT, profiles=profiles, llm_status="skipped")
    with_metadata = {**base, "court": "Federal Shariat Court", "decision_date": "2021-01-01"}
    assert _decide(base, taxonomy).scores == _decide(with_metadata, taxonomy).scores


def test_weights_are_configuration_not_hardcoded():
    settings = get_settings().domain_decision
    assert set(settings.weights.as_mapping()) == {"keyword", "llm", "cluster", "title"}
    assert abs(settings.weights.total - 1.0) < 1e-9


def test_domain_ids_are_not_hardcoded_in_the_engine():
    source = Path("src/classification/domain_decision.py").read_text()
    body = source.split('"""', 2)[-1]
    assert "family_law" not in body
    assert "criminal_law" not in body


# ---------------------------------------------------------------------------
# 6. The title signal
# ---------------------------------------------------------------------------


def test_title_contributes_but_only_lightly(taxonomy, profiles):
    row = _row(text=NEUTRAL_TEXT, profiles=profiles, llm_status="skipped")
    decided = _decide(
        row, taxonomy, compiled_profiles=profiles,
        title="Mst. Zainab v. Muhammad Ali (suit for dower and maintenance)",
    )
    title = next(c for c in decided.contributions if c.signal == "title")

    assert title.available
    assert title.weight == 0.05
    # A title alone cannot carry a document over the assignment floor:
    # its coverage shortfall scales the score below min_domain_score.
    assert decided.primary_domain == OTHER_DOMAIN_ID
    assert decided.status == STATUS_NEEDS_REVIEW


def test_a_missing_title_is_simply_unavailable(taxonomy, profiles):
    decided = _decide(
        _row(text=FAMILY_TEXT, profiles=profiles, llm_status="skipped"),
        taxonomy, compiled_profiles=profiles, title=None,
    )
    title = next(c for c in decided.contributions if c.signal == "title")
    assert not title.available


# ---------------------------------------------------------------------------
# 7. The flow: storage, resumability, versioning
# ---------------------------------------------------------------------------


def _doc(doc_id: str, text: str, **overrides) -> DocumentRepresentation:
    base = dict(
        doc_id=doc_id,
        source_uri=f"/corpus/{doc_id}",
        source_relpath=f"family_law/{doc_id}",  # a label that must be ignored
        source_type="case_html",
        title=f"{doc_id} cause title",
        headings=["Facts", "Order"],
        body_preview=text[:500],
        cleaned_text=text * 3,
        char_count=len(text) * 3,
        court="Lahore High Court",
        decision_date="2021-04-11",
        classification_status=CLASSIFICATION_STATUS_PENDING,
    )
    base.update(overrides)
    return DocumentRepresentation(**base)


@pytest.fixture()
def db(tmp_path) -> Path:
    db_path = tmp_path / "decide.db"
    init_schema(db_path=db_path, schema_file=str(DEFAULT_REPRESENTATION_SCHEMA_FILE))
    init_schema(db_path=db_path, schema_file=str(DEFAULT_SIGNALS_SCHEMA_FILE))
    init_schema(db_path=db_path, schema_file=CLASSIFICATION_SCHEMA_FILE)
    return db_path


@pytest.fixture()
def flow_settings(monkeypatch, tmp_path):
    """Real Pydantic models, so these fakes cannot drift from the schema."""

    import orchestration.dags.domain_classification_flow as flow

    real = get_settings()
    monkeypatch.setattr(
        flow,
        "get_settings",
        lambda: SimpleNamespace(
            domain_decision=real.domain_decision.model_copy(update={"batch_size": 2}),
            domain_signals=real.domain_signals,
            classification=real.classification,
            caselaw=real.caselaw,
            review=real.review,
            discovery=real.discovery,
            cluster_validation=real.cluster_validation,
            pipeline=SimpleNamespace(checkpoint_dir=tmp_path / "checkpoints"),
            metrics=SimpleNamespace(db_path=tmp_path / "metrics.db"),
        ),
    )
    return real


def _seed(db, profiles, signal_run="sig1", n_family=3, n_criminal=3):
    docs = [_doc(f"family_{i}", FAMILY_TEXT) for i in range(n_family)]
    docs += [_doc(f"criminal_{i}", CRIMINAL_TEXT) for i in range(n_criminal)]
    upsert_representations(docs, db_path=db)

    records = []
    for i in range(n_family):
        records.append({
            "doc_id": f"family_{i}", "cluster_id": 0, "cluster_confidence": 0.9,
            "llm_domain": "family_law", "llm_confidence": 0.95, "llm_status": "ok",
            "llm_reason": "dower and maintenance", "llm_model": "qwen3:14b",
            **_keyword_evidence(FAMILY_TEXT, profiles),
        })
    for i in range(n_criminal):
        records.append({
            "doc_id": f"criminal_{i}", "cluster_id": 1, "cluster_confidence": 0.9,
            "llm_domain": "criminal_law", "llm_confidence": 0.95, "llm_status": "ok",
            "llm_reason": "conviction", "llm_model": "qwen3:14b",
            **_keyword_evidence(CRIMINAL_TEXT, profiles),
        })
    persist_domain_signals(
        signal_run, records, signal_version="domain_assessment/1.0",
        batch_id="b0", db_path=db,
    )
    return docs


def test_verdict_lands_on_phase_1s_existing_fields(db, flow_settings, profiles):
    _seed(db, profiles)
    result = run_domain_classification("run1", "sig1", db_path=db)

    assert result.decided == 6
    rep = get_representation("family_0", db_path=db)
    assert rep.primary_domain == "family_law"
    assert rep.classification_status == STATUS_AUTO_ACCEPTED
    assert 0.0 < rep.domain_confidence <= 1.0
    assert rep.secondary_domain is None  # broad decision: no subdomains


def test_the_verdict_needs_no_table_of_its_own(db, flow_settings, profiles):
    """Phase 5 writes into Phase 1's fields, not into an invented table.

    The run does create ``document_review_decisions`` -- that is Phase 6's
    human-review ledger, which records who decided what and is the reason
    a rerun cannot reset a reviewed document. Nothing else new appears,
    and in particular no table holds the classification result.
    """

    _seed(db, profiles)
    with connection_scope(db) as conn:
        before = {
            r["name"] for r in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            )
        }
    run_domain_classification("run1", "sig1", db_path=db)
    with connection_scope(db) as conn:
        after = {
            r["name"] for r in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            )
        }

    assert after - before <= {"document_review_decisions"}
    # The verdict itself lives on the Phase 1 row.
    assert get_representation("family_0", db_path=db).primary_domain == "family_law"


def test_audit_row_carries_the_evidence_and_its_provenance(db, flow_settings, profiles, taxonomy):
    _seed(db, profiles)
    run_domain_classification("run1", "sig1", db_path=db)

    rows = {r["doc_id"]: r for r in get_classifications_for_run("run1", db_path=db)}
    row = rows["criminal_0"]

    assert row["primary_domain"] == "criminal_law"
    assert row["classifier_version"] == DECISION_VERSION
    assert row["taxonomy_version"] == taxonomy.version
    assert row["signal_run_id"] == "sig1"  # the audit chain: verdict -> evidence
    assert row["model_name"] is None  # no model call in Phase 5
    assert row["cluster_id"] == 1

    evidence = json.loads(row["justification"].split("evidence=", 1)[1])
    assert {s["signal"] for s in evidence["signals"]} == {
        "keyword", "llm", "cluster", "title"
    }
    assert evidence["signal_run_id"] == "sig1"


def test_extraction_fields_are_left_untouched(db, flow_settings, profiles):
    _seed(db, profiles)
    before = get_representation("family_0", db_path=db)
    run_domain_classification("run1", "sig1", db_path=db)
    after = get_representation("family_0", db_path=db)

    assert after.cleaned_text == before.cleaned_text
    assert after.content_hash == before.content_hash
    assert after.court == before.court
    assert after.status == before.status


def test_rerunning_the_same_run_is_idempotent(db, flow_settings, profiles):
    _seed(db, profiles)
    first = run_domain_classification("run1", "sig1", db_path=db)
    second = run_domain_classification("run1", "sig1", db_path=db)

    assert first.decided == 6
    assert second.decided == 0  # everything already decided
    assert second.skipped_already_done == 6
    assert len(get_classifications_for_run("run1", db_path=db)) == 6


def test_a_new_run_versions_rather_than_overwrites(db, flow_settings, profiles):
    _seed(db, profiles)
    run_domain_classification("run1", "sig1", db_path=db)
    run_domain_classification("run2", "sig1", db_path=db)

    history = get_classification_history("family_0", db_path=db)
    assert {h["run_id"] for h in history} == {"run1", "run2"}
    assert len(get_classifications_for_run("run1", db_path=db)) == 6


def test_batches_persist_progressively(db, flow_settings, profiles):
    _seed(db, profiles)
    result = run_domain_classification("run1", "sig1", db_path=db)
    batches = {
        r["batch_id"] for r in get_classifications_for_run("run1", db_path=db)
    }
    assert len(batches) == 3  # batch_size 2 over 6 documents
    assert result.decided == 6


def test_dry_run_records_the_audit_without_changing_current_state(db, flow_settings, profiles):
    _seed(db, profiles)
    run_domain_classification("run1", "sig1", db_path=db, write_current_state=False)

    assert len(get_classifications_for_run("run1", db_path=db)) == 6
    rep = get_representation("family_0", db_path=db)
    assert rep.classification_status == CLASSIFICATION_STATUS_PENDING
    assert rep.primary_domain is None


def test_a_document_without_evidence_is_not_decided(db, flow_settings, profiles):
    _seed(db, profiles)
    upsert_representations([_doc("unsignalled", FAMILY_TEXT)], db_path=db)
    run_domain_classification("run1", "sig1", db_path=db)

    rep = get_representation("unsignalled", db_path=db)
    assert rep.classification_status == CLASSIFICATION_STATUS_PENDING


def test_phase_2_drops_are_never_revived(db, flow_settings, profiles):
    """Phase 5 decides only what Phase 3 gathered evidence for."""

    _seed(db, profiles)
    upsert_representations(
        [_doc("dropped_0", "Case called. None present. Adjourned.",
              classification_status=CLASSIFICATION_STATUS_DROPPED_PROCEDURAL,
              drop_reason="procedural_adjournment", cleaned_text=None)],
        db_path=db,
    )
    run_domain_classification("run1", "sig1", db_path=db)

    rep = get_representation("dropped_0", db_path=db)
    assert rep.classification_status == CLASSIFICATION_STATUS_DROPPED_PROCEDURAL
    assert rep.drop_reason == "procedural_adjournment"


def test_missing_evidence_run_is_reported_not_guessed(db, flow_settings, profiles):
    _seed(db, profiles)
    result = run_domain_classification("run1", "nonexistent_run", db_path=db)

    assert result.decided == 0
    assert result.signals_available == 0
    assert get_representation("family_0", db_path=db).classification_status == (
        CLASSIFICATION_STATUS_PENDING
    )


def test_cluster_signal_can_be_disabled_for_the_whole_run(db, flow_settings, profiles):
    _seed(db, profiles)
    result = run_domain_classification(
        "run1", "sig1", db_path=db, cluster_signal_enabled=False
    )

    assert not result.cluster_signal_enabled
    assert result.clusters_profiled == 0
    evidence = json.loads(
        get_classifications_for_run("run1", db_path=db)[0]["justification"]
        .split("evidence=", 1)[1]
    )
    cluster = next(s for s in evidence["signals"] if s["signal"] == "cluster")
    assert cluster["weight"] == 0.0
    assert not cluster["available"]


def test_run_summary_reports_the_distribution(db, flow_settings, profiles):
    _seed(db, profiles)
    result = run_domain_classification("run1", "sig1", db_path=db)

    assert result.by_domain == {"family_law": 3, "criminal_law": 3}
    assert result.auto_accepted + result.needs_review + result.dropped_off_domain == 6
    assert 0.0 <= result.auto_accept_rate <= 1.0
    assert result.signal_run_id == "sig1"
