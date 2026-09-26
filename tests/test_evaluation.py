"""Phase 7: evaluation against human labels, and the dataset freeze.

The contract under test:

* precision, recall, F1 and the confusion matrix are computed correctly,
  including the cases where they are *undefined* rather than zero,
* the validation set comes from people, never from folder names, and an
  undecided review contributes no label,
* clustering metrics are reported but **cannot** move the verdict,
* the audit plan is a 100% census of needs_review plus reproducible
  samples of what a reviewer would otherwise never see,
* the verdict can say "not evaluable" and "not ready", and a freeze is
  impossible until it does not,
* a freeze is a snapshot: nothing is deleted, and an existing freeze is
  immutable.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from orchestration.dags.domain_classification_flow import run_domain_classification
from orchestration.dags.evaluation_flow import (
    render_evaluation_summary,
    run_evaluation,
)
from src.classification import review_policy as policy
from src.classification.keyword_signals import compile_profiles, detect_keyword_signals
from src.classification.review_store import (
    DEFAULT_REVIEW_SCHEMA_FILE,
    apply_review_decisions,
    persist_review_decisions,
)
from src.classification.signal_store import (
    DEFAULT_SIGNALS_SCHEMA_FILE,
    persist_domain_signals,
)
from src.classification.taxonomy_registry import OTHER_DOMAIN_ID, load_frozen_taxonomy
from src.common.config import get_settings
from src.common.db import connection_scope, init_schema
from src.common.exceptions import ConfigurationError
from src.evaluation.audit_sampling import (
    STRATUM_NEEDS_REVIEW,
    STRATUM_OTHER_UNCERTAIN,
    build_audit_plan,
)
from src.evaluation.classification_metrics import (
    evaluate_classification,
    render_confusion_matrix,
)
from src.evaluation.dataset_freeze import (
    DEFAULT_FREEZE_SCHEMA_FILE,
    accepted_documents,
    freeze_corpus,
    get_freeze,
    get_frozen_doc_ids,
    list_freezes,
)
from src.evaluation.readiness import (
    VERDICT_NOT_EVALUABLE,
    VERDICT_NOT_READY,
    VERDICT_READY,
    VERDICT_READY_WITH_RESERVATIONS,
    ReadinessThresholds,
    assess_readiness,
)
from src.evaluation.validation_set import (
    SOURCE_LABELS_FILE,
    ValidationSet,
    load_validation_set,
)
from src.extraction.doc_representation import (
    CLASSIFICATION_STATUS_PENDING,
    DocumentRepresentation,
)
from src.extraction.representation_store import (
    DEFAULT_REPRESENTATION_SCHEMA_FILE,
    get_representation,
    upsert_representations,
)

CLASSIFICATION_SCHEMA_FILE = "schemas/classification_schema.sql"
DOMAINS = ["family_law", "criminal_law", OTHER_DOMAIN_ID]
TARGETS = ["family_law", "criminal_law"]

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


# ---------------------------------------------------------------------------
# 1. Classification metrics
# ---------------------------------------------------------------------------


def test_perfect_agreement_scores_one():
    gold = {"a": "family_law", "b": "criminal_law", "c": OTHER_DOMAIN_ID}
    report = evaluate_classification(gold, dict(gold), DOMAINS)

    assert report.accuracy == pytest.approx(1.0)
    assert report.macro_f1 == pytest.approx(1.0)
    for domain in DOMAINS:
        assert report.per_domain[domain].f1 == pytest.approx(1.0)


def test_precision_and_recall_are_not_transposed():
    """The classic silent error: one over-prediction must hit precision only."""

    gold = {"a": "family_law", "b": "criminal_law", "c": "criminal_law"}
    predicted = {"a": "family_law", "b": "family_law", "c": "criminal_law"}
    report = evaluate_classification(gold, predicted, DOMAINS)

    family = report.per_domain["family_law"]
    # Predicted family twice, only one was right.
    assert family.precision == pytest.approx(0.5)
    # The one true family document was found.
    assert family.recall == pytest.approx(1.0)

    criminal = report.per_domain["criminal_law"]
    assert criminal.precision == pytest.approx(1.0)
    assert criminal.recall == pytest.approx(0.5)


def test_support_travels_with_every_score():
    gold = {"a": "family_law", "b": "family_law", "c": "criminal_law"}
    report = evaluate_classification(gold, dict(gold), DOMAINS)

    assert report.per_domain["family_law"].support == 2
    assert report.per_domain["criminal_law"].support == 1
    assert report.per_domain[OTHER_DOMAIN_ID].support == 0


def test_a_domain_with_no_gold_examples_is_undefined_not_zero():
    """Reporting 0.00 would read as failure; the truth is 'not measured'."""

    gold = {"a": "family_law"}
    report = evaluate_classification(gold, dict(gold), DOMAINS)

    criminal = report.per_domain["criminal_law"]
    assert criminal.support == 0
    assert criminal.recall is None
    assert criminal.precision is None
    assert not criminal.is_measurable
    # ...and it does not drag the macro average down.
    assert report.macro_f1 == pytest.approx(1.0)
    assert report.measurable_domains == ["family_law"]


def test_a_never_classified_document_counts_against_recall():
    """Refusing to answer is a failure mode and has to show up somewhere."""

    gold = {"a": "family_law", "b": "family_law"}
    report = evaluate_classification(gold, {"a": "family_law"}, DOMAINS)

    family = report.per_domain["family_law"]
    assert family.recall == pytest.approx(0.5)
    assert family.precision == pytest.approx(1.0)
    assert report.unlabelled == ["b"]


def test_the_catch_all_is_scored_as_a_class():
    """A classifier that dumps hard cases into 'other' must be visible."""

    gold = {"a": "family_law", "b": "family_law", "c": OTHER_DOMAIN_ID}
    predicted = {"a": "family_law", "b": OTHER_DOMAIN_ID, "c": OTHER_DOMAIN_ID}
    report = evaluate_classification(gold, predicted, DOMAINS)

    assert report.per_domain["family_law"].recall == pytest.approx(0.5)
    assert report.per_domain[OTHER_DOMAIN_ID].precision == pytest.approx(0.5)


def test_confusion_matrix_rows_are_gold_and_columns_are_predictions():
    gold = {"a": "family_law"}
    predicted = {"a": "criminal_law"}
    report = evaluate_classification(gold, predicted, DOMAINS)

    assert report.confusion["family_law"]["criminal_law"] == 1
    assert report.confusion["criminal_law"]["family_law"] == 0
    rendered = render_confusion_matrix(report)
    assert "gold \\ predicted" in rendered


def test_only_reviewed_documents_are_scored():
    """A validation set is what a human looked at, not the whole corpus."""

    gold = {"a": "family_law"}
    predicted = {"a": "family_law", "b": "criminal_law", "c": "family_law"}
    report = evaluate_classification(gold, predicted, DOMAINS)

    assert report.n_documents == 1


def test_weakest_domain_is_identified_for_follow_up():
    gold = {f"f{i}": "family_law" for i in range(4)}
    gold.update({f"c{i}": "criminal_law" for i in range(4)})
    predicted = {**gold, "c0": "family_law", "c1": "family_law"}
    report = evaluate_classification(gold, predicted, DOMAINS)

    assert report.weakest_domain().domain == "criminal_law"


# ---------------------------------------------------------------------------
# 2. The validation set
# ---------------------------------------------------------------------------


def _doc(doc_id: str, text: str, **overrides) -> DocumentRepresentation:
    base = dict(
        doc_id=doc_id,
        source_uri=f"/corpus/{doc_id}",
        # A folder name that must never be mistaken for a gold label.
        source_relpath=f"family_law/{doc_id}",
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
def taxonomy():
    return load_frozen_taxonomy()


@pytest.fixture()
def profiles(taxonomy):
    return compile_profiles(get_settings().domain_signals.profiles, taxonomy)


@pytest.fixture()
def db(tmp_path) -> Path:
    db_path = tmp_path / "evaluation.db"
    init_schema(db_path=db_path, schema_file=str(DEFAULT_REPRESENTATION_SCHEMA_FILE))
    init_schema(db_path=db_path, schema_file=str(DEFAULT_SIGNALS_SCHEMA_FILE))
    init_schema(db_path=db_path, schema_file=CLASSIFICATION_SCHEMA_FILE)
    init_schema(db_path=db_path, schema_file=str(DEFAULT_REVIEW_SCHEMA_FILE))
    init_schema(db_path=db_path, schema_file=str(DEFAULT_FREEZE_SCHEMA_FILE))
    return db_path


@pytest.fixture()
def flow_settings(monkeypatch, tmp_path):
    """Real Pydantic models, so these fakes cannot drift from the schema."""

    import orchestration.dags.domain_classification_flow as decision_flow
    import orchestration.dags.evaluation_flow as eval_flow

    real = get_settings()
    fake = SimpleNamespace(
        domain_decision=real.domain_decision.model_copy(update={"batch_size": 25}),
        domain_signals=real.domain_signals,
        classification=real.classification,
        caselaw=real.caselaw,
        review=real.review,
        cluster_validation=real.cluster_validation.model_copy(
            update={"output_dir": tmp_path / "cluster_validation"}
        ),
        evaluation=real.evaluation.model_copy(
            update={"output_dir": tmp_path / "evaluation"}
        ),
        pipeline=SimpleNamespace(checkpoint_dir=tmp_path / "checkpoints"),
        metrics=SimpleNamespace(db_path=tmp_path / "metrics.db"),
    )
    monkeypatch.setattr(decision_flow, "get_settings", lambda: fake)
    monkeypatch.setattr(eval_flow, "get_settings", lambda: fake)
    return real


def _keyword_evidence(text: str, profiles) -> dict:
    signals = detect_keyword_signals("x", text, profiles, min_matches=2)
    return {
        "keyword_signals": signals.as_evidence(),
        "keyword_top_domain": signals.top_domain,
        "keyword_margin": signals.margin,
    }


def _seed(db, profiles, n_family=120, n_criminal=120, n_other=20, signal_run="sig1"):
    """A corpus large enough to clear the readiness thresholds."""

    docs, records = [], []

    def add(doc_id, text, cluster, llm_domain, llm_confidence):
        docs.append(_doc(doc_id, text))
        records.append({
            "doc_id": doc_id, "cluster_id": cluster, "cluster_confidence": 0.9,
            "llm_domain": llm_domain, "llm_confidence": llm_confidence,
            "llm_status": "ok" if llm_domain else "failed",
            "llm_reason": "seed", "llm_model": "qwen3:14b",
            **_keyword_evidence(text, profiles),
        })

    for i in range(n_family):
        add(f"fam_{i:03d}", FAMILY_TEXT, 0, "family_law", 0.95)
    for i in range(n_criminal):
        add(f"crim_{i:03d}", CRIMINAL_TEXT, 1, "criminal_law", 0.95)
    for i in range(n_other):
        add(f"tax_{i:03d}", NEUTRAL_TEXT, -1, OTHER_DOMAIN_ID, 0.95)

    upsert_representations(docs, db_path=db)
    persist_domain_signals(
        signal_run, records, signal_version="domain_assessment/1.0",
        batch_id="b0", db_path=db,
    )
    return docs


def _review_everything(db, run_id="run1", reviewer="ibrahim", limit=None):
    """Accept every machine verdict, creating a gold set that agrees."""

    with connection_scope(db) as conn:
        rows = conn.execute(
            "SELECT doc_id, primary_domain, status FROM document_classifications "
            "WHERE run_id = ? ORDER BY doc_id", (run_id,),
        ).fetchall()

    decisions = []
    for row in rows[:limit] if limit else rows:
        if row["primary_domain"] == OTHER_DOMAIN_ID:
            decisions.append({
                "doc_id": row["doc_id"], "decision": policy.REVIEW_REJECTED,
                "reviewer": reviewer,
            })
        else:
            decisions.append({
                "doc_id": row["doc_id"], "decision": policy.REVIEW_ACCEPTED,
                "reviewer": reviewer,
            })
    apply_review_decisions(decisions, db_path=db, decision_run_id=run_id)
    return decisions


def test_gold_labels_come_from_the_review_ledger(db, flow_settings, profiles):
    _seed(db, profiles, n_family=5, n_criminal=5, n_other=2)
    run_domain_classification("run1", "sig1", db_path=db)
    _review_everything(db)

    validation = load_validation_set(db_path=db, domains=DOMAINS)
    assert len(validation) == 12
    assert validation.reviewers == {"ibrahim": 12}
    assert validation.by_domain["family_law"] == 5


def test_an_undecided_review_contributes_no_label(db, flow_settings, profiles):
    _seed(db, profiles, n_family=3, n_criminal=3, n_other=0)
    run_domain_classification("run1", "sig1", db_path=db)

    persist_review_decisions(
        [{"doc_id": "fam_000", "decision": policy.REVIEW_STILL_UNCERTAIN,
          "reviewer": "ibrahim"}],
        db_path=db, decision_run_id="run1",
    )
    validation = load_validation_set(db_path=db, domains=DOMAINS)

    assert "fam_000" not in validation.labels
    assert validation.unresolved == ["fam_000"]


def test_folder_names_are_never_gold_labels(db, flow_settings, profiles):
    """Every seeded document sits under a 'family_law/' path; none is labelled."""

    _seed(db, profiles, n_family=2, n_criminal=2, n_other=0)
    validation = load_validation_set(db_path=db, domains=DOMAINS)

    assert len(validation) == 0, "source folders must not produce gold labels"


def test_an_external_labels_file_can_supply_the_validation_set(db, tmp_path):
    path = tmp_path / "gold.csv"
    path.write_text(
        "doc_id,domain,reviewer\n"
        "a,family_law,supervisor\n"
        "b,criminal_law,supervisor\n",
        encoding="utf-8",
    )
    validation = load_validation_set(db_path=db, labels_file=path, domains=DOMAINS)

    assert validation.source == SOURCE_LABELS_FILE
    assert validation.labels == {"a": "family_law", "b": "criminal_law"}


def test_a_label_outside_the_taxonomy_is_rejected(db, tmp_path):
    path = tmp_path / "bad.csv"
    path.write_text("doc_id,domain\na,tax_law\n", encoding="utf-8")

    with pytest.raises(ConfigurationError, match="not a domain in the frozen taxonomy"):
        load_validation_set(db_path=db, labels_file=path, domains=DOMAINS)


def test_a_single_reviewer_is_recorded_as_a_limitation():
    validation = ValidationSet(labels={"a": "family_law"}, reviewers={"ibrahim": 1})
    assert validation.is_single_reviewer

    two = ValidationSet(labels={"a": "family_law"}, reviewers={"a": 1, "b": 1})
    assert not two.is_single_reviewer


# ---------------------------------------------------------------------------
# 3. The audit plan
# ---------------------------------------------------------------------------


def test_needs_review_is_a_census_not_a_sample(db, flow_settings, profiles):
    _seed(db, profiles, n_family=5, n_criminal=5, n_other=2)
    run_domain_classification("run1", "sig1", db_path=db)

    plan = build_audit_plan("run1", TARGETS, db_path=db, sample_per_domain=2)
    census = plan.strata[STRATUM_NEEDS_REVIEW]

    assert census.is_census
    assert census.required == census.population


def test_auto_accepted_documents_are_sampled_per_domain(db, flow_settings, profiles):
    _seed(db, profiles, n_family=50, n_criminal=50, n_other=0)
    run_domain_classification("run1", "sig1", db_path=db)

    plan = build_audit_plan("run1", TARGETS, db_path=db, sample_per_domain=10)

    for domain in TARGETS:
        stratum = plan.strata[f"auto_accepted:{domain}"]
        assert stratum.population == 50
        assert len(stratum.doc_ids) == 10


def test_the_audit_sample_is_reproducible(db, flow_settings, profiles):
    _seed(db, profiles, n_family=50, n_criminal=50, n_other=0)
    run_domain_classification("run1", "sig1", db_path=db)

    first = build_audit_plan("run1", TARGETS, db_path=db, sample_per_domain=10, seed=7)
    second = build_audit_plan("run1", TARGETS, db_path=db, sample_per_domain=10, seed=7)
    different = build_audit_plan("run1", TARGETS, db_path=db, sample_per_domain=10, seed=8)

    assert first.doc_ids() == second.doc_ids()
    assert first.doc_ids() != different.doc_ids()


def test_reviewed_documents_leave_the_audit_plan(db, flow_settings, profiles):
    _seed(db, profiles, n_family=20, n_criminal=20, n_other=0)
    run_domain_classification("run1", "sig1", db_path=db)

    before = build_audit_plan("run1", TARGETS, db_path=db, sample_per_domain=5)
    _review_everything(db, limit=10)
    after = build_audit_plan("run1", TARGETS, db_path=db, sample_per_domain=5)

    reviewed_after = sum(
        s.already_reviewed for name, s in after.strata.items()
        if name.startswith("auto_accepted:")
    )
    assert reviewed_after == 10
    assert before.total_outstanding >= after.total_outstanding


def test_the_backlog_is_cleared_only_when_every_review_is_done(db, flow_settings, profiles):
    _seed(db, profiles, n_family=5, n_criminal=5, n_other=2)
    run_domain_classification("run1", "sig1", db_path=db)

    plan = build_audit_plan("run1", TARGETS, db_path=db)
    if plan.strata[STRATUM_NEEDS_REVIEW].population:
        assert not plan.review_backlog_cleared
        _review_everything(db)
        assert build_audit_plan("run1", TARGETS, db_path=db).review_backlog_cleared


def test_other_uncertain_is_inspected_for_leakage(db, flow_settings, profiles):
    _seed(db, profiles, n_family=5, n_criminal=5, n_other=10)
    run_domain_classification("run1", "sig1", db_path=db)

    plan = build_audit_plan("run1", TARGETS, db_path=db)
    assert STRATUM_OTHER_UNCERTAIN in plan.strata


# ---------------------------------------------------------------------------
# 4. The readiness verdict
# ---------------------------------------------------------------------------


def _report(gold_per_domain=110, accuracy=1.0, accepted=500, reviewers=None):
    gold, predicted = {}, {}
    for domain in TARGETS:
        wrong = int(gold_per_domain * (1 - accuracy))
        for i in range(gold_per_domain):
            doc_id = f"{domain}_{i}"
            gold[doc_id] = domain
            predicted[doc_id] = (
                OTHER_DOMAIN_ID if i < wrong else domain
            )
    classification = evaluate_classification(gold, predicted, DOMAINS)
    validation = ValidationSet(
        labels=gold,
        reviewers=reviewers or {"a": len(gold) // 2, "b": len(gold) - len(gold) // 2},
        by_domain={d: gold_per_domain for d in TARGETS},
    )
    return validation, classification, {d: accepted for d in TARGETS}


def test_no_validation_set_is_not_evaluable_not_ready():
    validation = ValidationSet()
    classification = evaluate_classification({}, {}, DOMAINS)
    report = assess_readiness(validation, classification, {"family_law": 900}, TARGETS)

    assert report.verdict == VERDICT_NOT_EVALUABLE
    assert not report.can_freeze
    assert "not been measured" in report.answer


def test_a_thin_validation_set_is_not_evaluable():
    validation, classification, accepted = _report(gold_per_domain=5)
    report = assess_readiness(validation, classification, accepted, TARGETS)

    assert report.verdict == VERDICT_NOT_EVALUABLE
    assert any("below the" in b for b in report.blockers)


def test_a_poorly_scoring_domain_blocks_readiness():
    validation, classification, accepted = _report(accuracy=0.4)
    report = assess_readiness(validation, classification, accepted, TARGETS)

    assert report.verdict == VERDICT_NOT_READY
    assert not report.can_freeze
    assert any("F1" in b for b in report.blockers)


def test_too_small_an_accepted_corpus_blocks_readiness():
    validation, classification, accepted = _report(accepted=5)
    report = assess_readiness(validation, classification, accepted, TARGETS)

    assert report.verdict == VERDICT_NOT_READY
    assert any("worth indexing" in b for b in report.blockers)


def test_a_good_corpus_is_ready():
    validation, classification, accepted = _report()
    report = assess_readiness(
        validation, classification, accepted, TARGETS,
        audit_plan=SimpleNamespace(
            review_backlog_cleared=True, total_outstanding=0, strata={},
            as_dict=lambda: {},
        ),
    )
    assert report.verdict == VERDICT_READY
    assert report.can_freeze
    assert report.answer.startswith("Yes.")


def test_a_single_reviewer_becomes_a_stated_reservation():
    validation, classification, accepted = _report(reviewers={"ibrahim": 120})
    report = assess_readiness(validation, classification, accepted, TARGETS)

    assert report.verdict == VERDICT_READY_WITH_RESERVATIONS
    assert report.can_freeze  # usable, with the caveat stated
    assert any("single reviewer" in r for r in report.reservations)


def test_an_uncleared_backlog_is_a_reservation():
    validation, classification, accepted = _report()
    report = assess_readiness(
        validation, classification, accepted, TARGETS,
        audit_plan=SimpleNamespace(
            review_backlog_cleared=False, total_outstanding=12,
            strata={"needs_review": SimpleNamespace(outstanding=12)},
            as_dict=lambda: {},
        ),
    )
    assert report.verdict == VERDICT_READY_WITH_RESERVATIONS
    assert any("needs_review" in r for r in report.reservations)


def test_clustering_cannot_change_the_verdict():
    """The explicit instruction: no cluster threshold is an acceptance gate."""

    validation, classification, accepted = _report()
    audit = SimpleNamespace(
        review_backlog_cleared=True, total_outstanding=0, strata={}, as_dict=lambda: {}
    )

    def clustering(noise_share, purity, ari, verdict):
        return SimpleNamespace(
            n_clusters=2, noise_documents=int(noise_share * 100),
            noise_share=noise_share, coverage=1 - noise_share, per_cluster=[],
            weighted_purity=purity, purity_lift=purity - 0.5,
            adjusted_rand_index=ari, normalized_mutual_info=ari,
            per_label_recall={}, contingency={}, verdict=verdict,
        )

    excellent = assess_readiness(
        validation, classification, accepted, TARGETS, audit_plan=audit,
        clustering_report=clustering(0.02, 0.99, 0.95, "useful"),
    )
    terrible = assess_readiness(
        validation, classification, accepted, TARGETS, audit_plan=audit,
        clustering_report=clustering(0.85, 0.30, 0.01, "not_useful"),
    )

    assert excellent.verdict == terrible.verdict == VERDICT_READY
    assert excellent.blockers == terrible.blockers == []
    assert excellent.reservations == terrible.reservations
    # ...but both are reported.
    assert terrible.clustering_diagnostics["noise_share"] == 0.85
    assert terrible.clustering_diagnostics["adjusted_rand_index"] == 0.01
    assert "does not gate" in terrible.clustering_diagnostics["note"]


def test_low_llm_agreement_is_a_reservation_never_a_blocker():
    validation, classification, accepted = _report()
    report = assess_readiness(
        validation, classification, accepted, TARGETS, llm_agreement=0.2,
    )
    assert report.can_freeze
    assert any("LLM agreement" in r for r in report.reservations)


def test_thresholds_are_configurable():
    validation, classification, accepted = _report(accuracy=0.8)
    strict = assess_readiness(
        validation, classification, accepted, TARGETS,
        thresholds=ReadinessThresholds(min_domain_f1=0.99),
    )
    lenient = assess_readiness(
        validation, classification, accepted, TARGETS,
        thresholds=ReadinessThresholds(min_domain_f1=0.50),
    )
    assert strict.verdict == VERDICT_NOT_READY
    assert lenient.can_freeze


def test_the_verdict_carries_the_measurements_it_rests_on():
    validation, classification, accepted = _report()
    report = assess_readiness(validation, classification, accepted, TARGETS)

    assert report.classification["per_domain"]["family_law"]["f1"] is not None
    assert report.validation["size"] == len(validation)
    assert report.thresholds["min_domain_f1"] == 0.75
    json.dumps(report.as_dict(), default=str)  # must be serialisable


# ---------------------------------------------------------------------------
# 5. The end-to-end flow
# ---------------------------------------------------------------------------


def test_evaluation_reports_not_evaluable_without_review(db, flow_settings, profiles):
    _seed(db, profiles, n_family=20, n_criminal=20, n_other=5)
    run_domain_classification("run1", "sig1", db_path=db)

    result = run_evaluation("eval1", "run1", signal_run_id="sig1", db_path=db)

    assert result.verdict == VERDICT_NOT_EVALUABLE
    assert not result.can_freeze
    assert Path(result.report_paths["text"]).exists()


def test_evaluation_measures_a_reviewed_corpus(db, flow_settings, profiles):
    _seed(db, profiles)
    run_domain_classification("run1", "sig1", db_path=db)
    _review_everything(db)

    result = run_evaluation("eval1", "run1", signal_run_id="sig1", db_path=db)

    assert len(result.validation) >= 100
    assert result.classification.accuracy is not None
    assert result.llm_agreement is not None
    assert result.clustering is not None, "clustering diagnostics should be reported"
    assert result.accepted_by_domain["family_law"] > 0


def test_a_corrected_document_is_scored_against_the_machine_not_the_human(
    db, flow_settings, profiles
):
    """Otherwise every correction would count as a success."""

    _seed(db, profiles, n_family=60, n_criminal=60, n_other=0)
    run_domain_classification("run1", "sig1", db_path=db)

    # A reviewer overturns 10 family verdicts to criminal.
    corrections = [
        {"doc_id": f"fam_{i:03d}", "decision": policy.REVIEW_CORRECTED,
         "decision_domain": "criminal_law", "reviewer": "ibrahim"}
        for i in range(10)
    ]
    accepted = [
        {"doc_id": f"fam_{i:03d}", "decision": policy.REVIEW_ACCEPTED,
         "reviewer": "ibrahim"}
        for i in range(10, 60)
    ] + [
        {"doc_id": f"crim_{i:03d}", "decision": policy.REVIEW_ACCEPTED,
         "reviewer": "ibrahim"}
        for i in range(60)
    ]
    apply_review_decisions(corrections + accepted, db_path=db, decision_run_id="run1")

    result = run_evaluation("eval1", "run1", signal_run_id="sig1", db_path=db)

    # The 10 corrections must appear as family-predicted, criminal-gold.
    assert result.classification.confusion["criminal_law"]["family_law"] == 10
    assert result.classification.accuracy < 1.0


def test_the_summary_answers_the_question_it_was_asked(db, flow_settings, profiles):
    _seed(db, profiles, n_family=20, n_criminal=20, n_other=5)
    run_domain_classification("run1", "sig1", db_path=db)
    result = run_evaluation("eval1", "run1", signal_run_id="sig1", db_path=db)

    summary = render_evaluation_summary(result)
    assert "Can we confidently produce a usable Family Law and Criminal Law" in summary
    assert "VERDICT:" in summary
    assert "diagnostic only" in summary.lower()
    assert "MANUAL AUDIT" in summary


# ---------------------------------------------------------------------------
# 6. The dataset freeze
# ---------------------------------------------------------------------------


def _ready_report():
    validation, classification, accepted = _report()
    return assess_readiness(
        validation, classification, accepted, TARGETS,
        audit_plan=SimpleNamespace(
            review_backlog_cleared=True, total_outstanding=0, strata={},
            as_dict=lambda: {},
        ),
    )


def test_a_freeze_is_refused_without_a_passing_evaluation(db, flow_settings, profiles):
    _seed(db, profiles, n_family=20, n_criminal=20, n_other=0)
    run_domain_classification("run1", "sig1", db_path=db)

    validation = ValidationSet()
    classification = evaluate_classification({}, {}, DOMAINS)
    not_evaluable = assess_readiness(validation, classification, {}, TARGETS)

    with pytest.raises(ConfigurationError, match="cannot freeze corpus"):
        freeze_corpus(
            "freeze1", "run1", TARGETS, "taxonomy@v1", not_evaluable, db_path=db
        )


def test_freezing_snapshots_the_accepted_corpus(db, flow_settings, profiles):
    _seed(db, profiles, n_family=30, n_criminal=30, n_other=5)
    run_domain_classification("run1", "sig1", db_path=db)

    frozen = freeze_corpus(
        "freeze1", "run1", TARGETS, "taxonomy@v1", _ready_report(),
        db_path=db, signal_run_id="sig1",
    )

    assert frozen.document_count == 60
    assert frozen.domain_counts == {"criminal_law": 30, "family_law": 30}
    assert len(get_frozen_doc_ids("freeze1", db_path=db)) == 60
    assert len(get_frozen_doc_ids("freeze1", db_path=db, domain="family_law")) == 30


def test_a_freeze_records_which_documents_a_human_confirmed(db, flow_settings, profiles):
    _seed(db, profiles, n_family=30, n_criminal=30, n_other=0)
    run_domain_classification("run1", "sig1", db_path=db)
    _review_everything(db, limit=20)

    frozen = freeze_corpus(
        "freeze1", "run1", TARGETS, "taxonomy@v1", _ready_report(), db_path=db
    )
    assert frozen.human_reviewed_count == 20
    assert 0 < frozen.human_reviewed_share < 1


def test_a_freeze_deletes_nothing(db, flow_settings, profiles):
    _seed(db, profiles, n_family=30, n_criminal=30, n_other=5)
    run_domain_classification("run1", "sig1", db_path=db)

    with connection_scope(db) as conn:
        before = conn.execute(
            "SELECT COUNT(*) AS n FROM document_representations"
        ).fetchone()["n"]

    freeze_corpus("freeze1", "run1", TARGETS, "taxonomy@v1", _ready_report(), db_path=db)

    with connection_scope(db) as conn:
        after = conn.execute(
            "SELECT COUNT(*) AS n FROM document_representations"
        ).fetchone()["n"]

    assert after == before
    # The off-domain documents are still there, withheld but intact.
    assert get_representation("tax_000", db_path=db) is not None


def test_a_freeze_is_immutable(db, flow_settings, profiles):
    _seed(db, profiles, n_family=30, n_criminal=30, n_other=0)
    run_domain_classification("run1", "sig1", db_path=db)
    freeze_corpus("freeze1", "run1", TARGETS, "taxonomy@v1", _ready_report(), db_path=db)

    with pytest.raises(ConfigurationError, match="already exists"):
        freeze_corpus(
            "freeze1", "run1", TARGETS, "taxonomy@v1", _ready_report(), db_path=db
        )


def test_a_later_freeze_leaves_an_earlier_one_intact(db, flow_settings, profiles):
    _seed(db, profiles, n_family=30, n_criminal=30, n_other=0)
    run_domain_classification("run1", "sig1", db_path=db)
    freeze_corpus("freeze1", "run1", TARGETS, "taxonomy@v1", _ready_report(), db_path=db)

    # A reviewer withdraws a document, then a second freeze is taken.
    apply_review_decisions(
        [{"doc_id": "fam_000", "decision": policy.REVIEW_REJECTED, "reviewer": "r"}],
        db_path=db, decision_run_id="run1",
    )
    freeze_corpus("freeze2", "run1", TARGETS, "taxonomy@v1", _ready_report(), db_path=db)

    assert get_freeze("freeze1", db_path=db).document_count == 60
    assert get_freeze("freeze2", db_path=db).document_count == 59
    assert "fam_000" in get_frozen_doc_ids("freeze1", db_path=db)
    assert "fam_000" not in get_frozen_doc_ids("freeze2", db_path=db)
    assert len(list_freezes(db_path=db)) == 2


def test_an_empty_freeze_is_refused(db, flow_settings, profiles):
    _seed(db, profiles, n_family=2, n_criminal=2, n_other=0)
    # No classification run: nothing is accepted.
    with pytest.raises(ConfigurationError, match="no accepted documents"):
        freeze_corpus(
            "freeze1", "run1", TARGETS, "taxonomy@v1", _ready_report(), db_path=db
        )


def test_the_freeze_carries_the_evaluation_that_permitted_it(db, flow_settings, profiles):
    _seed(db, profiles, n_family=30, n_criminal=30, n_other=0)
    run_domain_classification("run1", "sig1", db_path=db)
    freeze_corpus(
        "freeze1", "run1", TARGETS, "taxonomy@v1", _ready_report(),
        db_path=db, signal_run_id="sig1", notes="first frozen corpus",
    )

    with connection_scope(db) as conn:
        row = conn.execute(
            "SELECT * FROM corpus_freezes WHERE freeze_id = 'freeze1'"
        ).fetchone()

    assert row["readiness_verdict"] == VERDICT_READY
    assert row["signal_run_id"] == "sig1"
    assert row["macro_f1"] is not None
    stored = json.loads(row["readiness_json"])
    assert stored["classification"]["per_domain"]["family_law"]["f1"] is not None


def test_accepted_documents_reflect_human_corrections(db, flow_settings, profiles):
    """The freeze reads current state, so a review has already taken effect."""

    _seed(db, profiles, n_family=10, n_criminal=10, n_other=0)
    run_domain_classification("run1", "sig1", db_path=db)
    apply_review_decisions(
        [{"doc_id": "fam_000", "decision": policy.REVIEW_CORRECTED,
          "decision_domain": "criminal_law", "reviewer": "r"}],
        db_path=db, decision_run_id="run1",
    )
    accepted = {d["doc_id"]: d["primary_domain"] for d in accepted_documents(TARGETS, db_path=db)}
    assert accepted["fam_000"] == "criminal_law"
