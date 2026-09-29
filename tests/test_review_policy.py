"""Phase 6: review routing and decision policy.

The contract under test:

* the three bands route as specified -- high auto-accepts, medium and
  uncertain go to review, low falls to the catch-all,
* **nothing is deleted**: a dropped document keeps its row, its text, its
  metadata and its provenance, and the drop is reversible,
* every decision stays auditable by doc_id / result / confidence and
  evidence / status / reason,
* a rerun does not reset a document a human has decided, and does not
  churn rows it would not change.
"""

from __future__ import annotations

import csv
from pathlib import Path
from types import SimpleNamespace

import pytest

from orchestration.dags.domain_classification_flow import run_domain_classification
from src.classification import review_policy as policy
from src.classification.classification_store import get_classifications_for_run
from src.classification.keyword_signals import compile_profiles, detect_keyword_signals
from src.classification.review_store import (
    DEFAULT_REVIEW_SCHEMA_FILE,
    apply_review_decisions,
    build_review_queue,
    export_review_queue,
    get_current_reviews,
    get_review_history,
    get_reviewed_doc_ids,
    human_domain,
    load_review_decisions_csv,
    persist_review_decisions,
    review_stats,
)
from src.classification.signal_store import (
    DEFAULT_SIGNALS_SCHEMA_FILE,
    persist_domain_signals,
)
from src.classification.taxonomy_registry import OTHER_DOMAIN_ID, load_frozen_taxonomy
from src.common.config import get_settings
from src.common.db import connection_scope, init_schema
from src.extraction.doc_representation import (
    CLASSIFICATION_STATUS_AUTO_ACCEPTED,
    CLASSIFICATION_STATUS_NEEDS_REVIEW,
    CLASSIFICATION_STATUS_PENDING,
    DocumentRepresentation,
)
from src.extraction.representation_store import (
    DEFAULT_REPRESENTATION_SCHEMA_FILE,
    WRITE_MISSING,
    WRITE_PROTECTED,
    WRITE_UNCHANGED,
    WRITE_UPDATED,
    get_representation,
    list_representations,
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
NEUTRAL_TEXT = (
    "This reference concerns the assessment of income tax and the limitation "
    "period prescribed by the Ordinance; the taxpayer contends it was time barred. "
)

THRESHOLDS = policy.ReviewThresholds()


def _summary(**overrides) -> policy.ScoreSummary:
    base = dict(
        top_domain="family_law",
        top_score=0.95,
        margin=0.90,
        coverage=0.95,
        any_signal_available=True,
        any_domain_support=True,
        keyword_supports_a_domain=True,
        llm_other_confidence=0.0,
        signals_conflict=False,
    )
    base.update(overrides)
    return policy.ScoreSummary(**base)


# ---------------------------------------------------------------------------
# 1. The bands
# ---------------------------------------------------------------------------


def test_high_confidence_is_auto_accepted():
    disposition = policy.route(_summary(), THRESHOLDS)

    assert disposition.band == policy.BAND_HIGH
    assert disposition.status == policy.STATUS_AUTO_ACCEPTED
    assert disposition.primary_domain == "family_law"
    assert disposition.review_reasons == ()


def test_medium_confidence_needs_review_with_a_label_only_proposed():
    disposition = policy.route(_summary(top_score=0.65, margin=0.50), THRESHOLDS)

    assert disposition.band == policy.BAND_MEDIUM
    assert disposition.status == policy.STATUS_NEEDS_REVIEW
    # The domain is still reported -- proposed, not final.
    assert disposition.primary_domain == "family_law"
    assert policy.REVIEW_LOW_CONFIDENCE in disposition.review_reasons


def test_low_confidence_falls_to_the_catch_all():
    disposition = policy.route(_summary(top_score=0.20, margin=0.20), THRESHOLDS)

    assert disposition.band == policy.BAND_LOW
    assert disposition.primary_domain == OTHER_DOMAIN_ID
    assert disposition.status == policy.STATUS_NEEDS_REVIEW
    assert policy.REVIEW_UNCERTAIN in disposition.review_reasons


def test_clearly_outside_the_target_domains_is_dropped_off_domain():
    disposition = policy.route(
        _summary(
            any_domain_support=False,
            keyword_supports_a_domain=False,
            llm_other_confidence=0.9,
            top_score=0.0,
            margin=0.0,
        ),
        THRESHOLDS,
    )
    assert disposition.band == policy.BAND_OFF_DOMAIN
    assert disposition.status == policy.STATUS_DROPPED_OFF_DOMAIN
    assert disposition.drop_reason == policy.DROP_REASON_OFF_DOMAIN
    assert disposition.withheld_from_corpus


def test_an_uncertain_neither_is_reviewed_rather_than_dropped():
    disposition = policy.route(
        _summary(
            any_domain_support=False, keyword_supports_a_domain=False,
            llm_other_confidence=0.4, top_score=0.0, margin=0.0,
        ),
        THRESHOLDS,
    )
    assert disposition.status == policy.STATUS_NEEDS_REVIEW
    assert disposition.drop_reason is None


def test_conflict_always_routes_to_review_whatever_the_score():
    disposition = policy.route(_summary(top_score=0.99, signals_conflict=True), THRESHOLDS)

    assert disposition.status == policy.STATUS_NEEDS_REVIEW
    assert disposition.band == policy.BAND_MEDIUM
    assert policy.REVIEW_SIGNAL_CONFLICT in disposition.review_reasons


def test_a_conflicted_document_is_never_dropped():
    """Disagreement is a reason to look, not a reason to discard."""

    disposition = policy.route(
        _summary(
            any_domain_support=False, keyword_supports_a_domain=False,
            llm_other_confidence=0.99, signals_conflict=True,
            top_score=0.0, margin=0.0,
        ),
        THRESHOLDS,
    )
    assert disposition.status == policy.STATUS_NEEDS_REVIEW


def test_thin_coverage_never_auto_accepts():
    disposition = policy.route(_summary(coverage=0.40), THRESHOLDS)

    assert disposition.status == policy.STATUS_NEEDS_REVIEW
    assert policy.REVIEW_LOW_COVERAGE in disposition.review_reasons


def test_routing_is_total_and_deterministic():
    """Every combination lands somewhere, and lands there every time."""

    seen = set()
    for score in (0.0, 0.2, 0.34, 0.36, 0.6, 0.79, 0.8, 1.0):
        for coverage in (0.0, 0.4, 0.55, 1.0):
            for conflict in (False, True):
                for support in (False, True):
                    summary = _summary(
                        top_score=score, margin=score, coverage=coverage,
                        signals_conflict=conflict, any_domain_support=support,
                        keyword_supports_a_domain=support,
                    )
                    first = policy.route(summary, THRESHOLDS)
                    assert first == policy.route(summary, THRESHOLDS)
                    seen.add(first.status)
    assert seen <= {
        policy.STATUS_AUTO_ACCEPTED,
        policy.STATUS_NEEDS_REVIEW,
        policy.STATUS_DROPPED_OFF_DOMAIN,
    }


def test_bands_come_from_configuration():
    config = get_settings().domain_decision
    thresholds = policy.ReviewThresholds.from_settings(config)

    assert thresholds.auto_accept_threshold == config.auto_accept_threshold
    assert thresholds.min_coverage == config.min_coverage


# ---------------------------------------------------------------------------
# 2. Human decisions map onto Phase 1's fields
# ---------------------------------------------------------------------------


def test_acceptance_endorses_the_machine_domain():
    for target in ("family_law", "criminal_law"):
        status, domain, drop = policy.resolve_human_decision(
            policy.REVIEW_ACCEPTED, machine_domain=target
        )
        assert (status, domain, drop) == (policy.STATUS_AUTO_ACCEPTED, target, None)


def test_accepting_the_catch_all_withholds_rather_than_accepts():
    """Endorsing ``other_uncertain`` is a finding of "neither", not an accept.

    Regression: this returned ``auto_accepted`` with the domain nulled
    downstream, so a document the reviewer had said belongs to neither
    target domain sat in the accepted corpus carrying no domain at all.
    """

    status, domain, drop = policy.resolve_human_decision(
        policy.REVIEW_ACCEPTED, machine_domain=OTHER_DOMAIN_ID
    )
    assert status == policy.STATUS_DROPPED_OFF_DOMAIN
    assert domain == OTHER_DOMAIN_ID
    assert drop == policy.DROP_REASON_OFF_DOMAIN


def test_correction_replaces_the_domain():
    status, domain, drop = policy.resolve_human_decision(
        policy.REVIEW_CORRECTED, machine_domain="criminal_law",
        corrected_domain="family_law",
    )
    assert (status, domain, drop) == (policy.STATUS_AUTO_ACCEPTED, "family_law", None)


def test_a_correction_must_say_what_the_right_answer_is():
    with pytest.raises(ValueError, match="requires corrected_domain"):
        policy.resolve_human_decision(
            policy.REVIEW_CORRECTED, machine_domain="criminal_law"
        )


def test_rejection_withholds_rather_than_deletes():
    status, domain, drop = policy.resolve_human_decision(
        policy.REVIEW_REJECTED, machine_domain="criminal_law"
    )
    assert status == policy.STATUS_DROPPED_OFF_DOMAIN
    assert domain == OTHER_DOMAIN_ID
    assert drop == policy.DROP_REASON_OFF_DOMAIN


def test_an_undecided_review_forces_no_label():
    status, domain, _ = policy.resolve_human_decision(
        policy.REVIEW_STILL_UNCERTAIN, machine_domain="family_law"
    )
    assert status == policy.STATUS_NEEDS_REVIEW
    assert domain == OTHER_DOMAIN_ID


def test_an_unknown_decision_is_rejected_loudly():
    with pytest.raises(ValueError, match="unknown review decision"):
        policy.resolve_human_decision("looks_fine_to_me", machine_domain="family_law")


# ---------------------------------------------------------------------------
# 3. Rerun protection
# ---------------------------------------------------------------------------


def test_a_reviewed_document_is_protected():
    assert policy.is_protected("d1", None, reviewed_doc_ids={"d1"})
    assert not policy.is_protected("d2", None, reviewed_doc_ids={"d1"})


def test_protection_can_be_turned_off_deliberately():
    assert not policy.is_protected(
        "d1", None, reviewed_doc_ids={"d1"}, protect_reviewed=False
    )


def test_machine_statuses_can_be_frozen_by_configuration():
    assert policy.is_protected(
        "d1", CLASSIFICATION_STATUS_AUTO_ACCEPTED,
        protect_statuses=(CLASSIFICATION_STATUS_AUTO_ACCEPTED,),
    )
    assert not policy.is_protected(
        "d1", CLASSIFICATION_STATUS_NEEDS_REVIEW,
        protect_statuses=(CLASSIFICATION_STATUS_AUTO_ACCEPTED,),
    )


# ---------------------------------------------------------------------------
# 4. The store: queue, ledger, application
# ---------------------------------------------------------------------------


def _doc(doc_id: str, text: str, **overrides) -> DocumentRepresentation:
    base = dict(
        doc_id=doc_id,
        source_uri=f"/corpus/{doc_id}",
        source_relpath=f"corpus/{doc_id}",
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
    db_path = tmp_path / "review.db"
    init_schema(db_path=db_path, schema_file=str(DEFAULT_REPRESENTATION_SCHEMA_FILE))
    init_schema(db_path=db_path, schema_file=str(DEFAULT_SIGNALS_SCHEMA_FILE))
    init_schema(db_path=db_path, schema_file=CLASSIFICATION_SCHEMA_FILE)
    init_schema(db_path=db_path, schema_file=str(DEFAULT_REVIEW_SCHEMA_FILE))
    return db_path


@pytest.fixture()
def flow_settings(monkeypatch, tmp_path):
    """Real Pydantic models, so these fakes cannot drift from the schema."""

    import orchestration.dags.domain_classification_flow as flow

    real = get_settings()
    monkeypatch.setattr(
        flow, "get_settings",
        lambda: SimpleNamespace(
            domain_decision=real.domain_decision.model_copy(update={"batch_size": 3}),
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


def _keyword_evidence(text: str, profiles) -> dict:
    signals = detect_keyword_signals("x", text, profiles, min_matches=2)
    return {
        "keyword_signals": signals.as_evidence(),
        "keyword_top_domain": signals.top_domain,
        "keyword_margin": signals.margin,
    }


def _seed(db, profiles, signal_run="sig1"):
    """Three clean documents, two borderline, one clearly off-domain."""

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

    for i in range(3):
        add(f"fam_{i}", FAMILY_TEXT, 0, "family_law", 0.95)
    # borderline: the LLM disagrees with the keyword profiles
    add("mixed_0", FAMILY_TEXT, 0, "criminal_law", 0.6)
    # thin: no LLM reading at all
    add("thin_0", FAMILY_TEXT, -1, None, None)
    # clearly outside both domains
    add("tax_0", NEUTRAL_TEXT, -1, OTHER_DOMAIN_ID, 0.95)

    upsert_representations(docs, db_path=db)
    persist_domain_signals(
        signal_run, records, signal_version="domain_assessment/1.0",
        batch_id="b0", db_path=db,
    )
    return docs


def test_the_queue_carries_everything_an_audit_needs(db, flow_settings, profiles):
    _seed(db, profiles)
    run_domain_classification("run1", "sig1", db_path=db)

    queue = build_review_queue("run1", db_path=db)
    assert queue, "borderline documents should be queued"

    row = queue[0]
    for required in ("doc_id", "primary_domain", "confidence", "status", "review_reason"):
        assert required in row
    assert row["evidence"]["signals"], "the evidence must survive into the queue"
    assert row["evidence"]["band"]
    # And enough to find the judgment itself.
    assert row["title"] and row["source_relpath"]


def test_only_review_documents_are_queued(db, flow_settings, profiles):
    _seed(db, profiles)
    run_domain_classification("run1", "sig1", db_path=db)

    queued = {r["doc_id"] for r in build_review_queue("run1", db_path=db)}
    rows = {r["doc_id"]: r for r in get_classifications_for_run("run1", db_path=db)}
    for doc_id in queued:
        assert rows[doc_id]["status"] == policy.STATUS_NEEDS_REVIEW
    # A clean, unanimous document does not occupy a reviewer.
    assert "fam_0" not in queued


def test_the_queue_is_ordered_least_confident_first(db, flow_settings, profiles):
    _seed(db, profiles)
    run_domain_classification("run1", "sig1", db_path=db)

    confidences = [r["confidence"] for r in build_review_queue("run1", db_path=db)]
    assert confidences == sorted(confidences)


def test_queue_export_round_trips_through_a_reviewer(db, flow_settings, profiles, tmp_path):
    _seed(db, profiles)
    run_domain_classification("run1", "sig1", db_path=db)

    exported = export_review_queue("run1", tmp_path / "queue", db_path=db)
    assert exported["queued"] > 0

    csv_path = Path(exported["csv"])
    with csv_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert rows and all(r["decision"] == "" for r in rows)

    # The reviewer fills in the same file and hands it back.
    rows[0]["decision"] = policy.REVIEW_ACCEPTED
    rows[0]["reviewer"] = "ibrahim"
    rows[0]["notes"] = "clearly a dower and maintenance suit"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    loaded = load_review_decisions_csv(csv_path)
    assert len(loaded) == 1
    assert loaded[0]["decision"] == policy.REVIEW_ACCEPTED
    assert loaded[0]["reviewer"] == "ibrahim"


def test_an_unreviewed_row_is_skipped_not_an_error(db, tmp_path):
    path = tmp_path / "partial.csv"
    path.write_text(
        "doc_id,decision,decision_domain,reviewer,notes\n"
        "a,human_accepted,,ibrahim,ok\n"
        "b,,,,\n",
        encoding="utf-8",
    )
    loaded = load_review_decisions_csv(path)
    assert [d["doc_id"] for d in loaded] == ["a"]


def test_a_typo_in_a_decision_is_never_silently_discarded(db, tmp_path):
    path = tmp_path / "typo.csv"
    path.write_text(
        "doc_id,decision,decision_domain,reviewer,notes\n"
        "a,accpeted,,ibrahim,ok\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="unknown decision"):
        load_review_decisions_csv(path)


def test_a_review_without_a_reviewer_is_rejected(db, flow_settings, profiles):
    _seed(db, profiles)
    with pytest.raises(ValueError, match="no reviewer"):
        persist_review_decisions(
            [{"doc_id": "fam_0", "decision": policy.REVIEW_ACCEPTED, "reviewer": ""}],
            db_path=db,
        )


def test_the_ledger_records_the_machine_context(db, flow_settings, profiles):
    _seed(db, profiles)
    run_domain_classification("run1", "sig1", db_path=db)

    persist_review_decisions(
        [{"doc_id": "fam_0", "decision": policy.REVIEW_ACCEPTED, "reviewer": "ibrahim"}],
        db_path=db, decision_run_id="run1",
    )
    review = get_current_reviews(db_path=db)["fam_0"]

    assert review["machine_domain"] == "family_law"
    assert review["machine_status"] == policy.STATUS_AUTO_ACCEPTED
    assert review["machine_confidence"] is not None
    assert review["signal_run_id"] == "sig1"
    assert review["decision_run_id"] == "run1"


def test_re_reviewing_appends_rather_than_overwrites(db, flow_settings, profiles):
    _seed(db, profiles)
    run_domain_classification("run1", "sig1", db_path=db)

    persist_review_decisions(
        [{"doc_id": "fam_0", "decision": policy.REVIEW_ACCEPTED, "reviewer": "a"}],
        db_path=db, decision_run_id="run1",
    )
    persist_review_decisions(
        [{
            "doc_id": "fam_0", "decision": policy.REVIEW_CORRECTED,
            "decision_domain": "criminal_law", "reviewer": "b",
            "notes": "second look: this is a criminal matter",
        }],
        db_path=db, decision_run_id="run1",
    )

    history = get_review_history("fam_0", db_path=db)
    assert len(history) == 2
    assert [h["reviewer"] for h in history] == ["a", "b"]
    # The newest decision is the current one; the first is still on record.
    assert get_current_reviews(db_path=db)["fam_0"]["decision"] == policy.REVIEW_CORRECTED


def test_applying_a_correction_moves_the_document(db, flow_settings, profiles):
    _seed(db, profiles)
    run_domain_classification("run1", "sig1", db_path=db)

    apply_review_decisions(
        [{
            "doc_id": "mixed_0", "decision": policy.REVIEW_CORRECTED,
            "decision_domain": "criminal_law", "reviewer": "ibrahim",
        }],
        db_path=db, decision_run_id="run1",
    )
    rep = get_representation("mixed_0", db_path=db)
    assert rep.primary_domain == "criminal_law"
    assert rep.classification_status == CLASSIFICATION_STATUS_AUTO_ACCEPTED


def test_accepting_the_catch_all_stores_no_domain_and_withholds(db, flow_settings, profiles):
    """The F1 regression, end to end: status withheld, primary_domain NULL."""

    _seed(db, profiles)
    run_domain_classification("run1", "sig1", db_path=db)

    apply_review_decisions(
        [{
            "doc_id": "thin_0", "decision": policy.REVIEW_ACCEPTED,
            "machine_domain": OTHER_DOMAIN_ID, "reviewer": "ibrahim",
        }],
        db_path=db, decision_run_id="run1",
    )
    rep = get_representation("thin_0", db_path=db)
    assert rep.classification_status == policy.STATUS_DROPPED_OFF_DOMAIN
    assert rep.primary_domain is None
    assert rep.drop_reason == policy.DROP_REASON_OFF_DOMAIN


def test_review_stats_measure_agreement_honestly(db, flow_settings, profiles):
    _seed(db, profiles)
    run_domain_classification("run1", "sig1", db_path=db)

    apply_review_decisions(
        [
            {"doc_id": "fam_0", "decision": policy.REVIEW_ACCEPTED, "reviewer": "r"},
            {"doc_id": "fam_1", "decision": policy.REVIEW_CORRECTED,
             "decision_domain": "criminal_law", "reviewer": "r"},
            {"doc_id": "thin_0", "decision": policy.REVIEW_STILL_UNCERTAIN,
             "reviewer": "r"},
        ],
        db_path=db, decision_run_id="run1",
    )
    stats = review_stats(db_path=db)

    assert stats["reviewed_documents"] == 3
    # The undecided review is not counted as agreement either way.
    assert stats["comparable"] == 2
    assert stats["agreed"] == 1
    assert stats["agreement_rate"] == pytest.approx(0.5)


def test_an_undecided_review_names_no_domain():
    assert human_domain({"decision": policy.REVIEW_STILL_UNCERTAIN,
                         "machine_domain": "family_law"}) is None


# ---------------------------------------------------------------------------
# 5. Nothing is deleted, and reruns leave settled work alone
# ---------------------------------------------------------------------------


def test_a_dropped_document_keeps_its_row_and_its_provenance(db, flow_settings, profiles):
    _seed(db, profiles)
    before = get_representation("tax_0", db_path=db)
    run_domain_classification("run1", "sig1", db_path=db)
    after = get_representation("tax_0", db_path=db)

    assert after is not None, "a dropped document must never be deleted"
    assert after.classification_status == "dropped_off_domain"
    # Everything that made it a document is untouched.
    assert after.cleaned_text == before.cleaned_text
    assert after.content_hash == before.content_hash
    assert after.source_uri == before.source_uri
    assert after.court == before.court
    assert after.citation == before.citation
    assert after.title == before.title


def test_a_drop_is_reversible(db, flow_settings, profiles):
    """Withholding is a status, so a reviewer can put a document back."""

    _seed(db, profiles)
    run_domain_classification("run1", "sig1", db_path=db)
    assert "tax_0" not in {r.doc_id for r in list_representations(db_path=db)}

    apply_review_decisions(
        [{
            "doc_id": "tax_0", "decision": policy.REVIEW_CORRECTED,
            "decision_domain": "family_law", "reviewer": "ibrahim",
            "notes": "misread; this is a maintenance matter",
        }],
        db_path=db, decision_run_id="run1",
    )
    assert "tax_0" in {r.doc_id for r in list_representations(db_path=db)}


def test_the_corpus_row_count_never_shrinks(db, flow_settings, profiles):
    _seed(db, profiles)
    with connection_scope(db) as conn:
        before = conn.execute("SELECT COUNT(*) AS n FROM document_representations").fetchone()["n"]

    run_domain_classification("run1", "sig1", db_path=db)
    run_domain_classification("run2", "sig1", db_path=db)

    with connection_scope(db) as conn:
        after = conn.execute("SELECT COUNT(*) AS n FROM document_representations").fetchone()["n"]
    assert after == before


def test_a_rerun_does_not_reset_a_reviewed_document(db, flow_settings, profiles):
    _seed(db, profiles)
    run_domain_classification("run1", "sig1", db_path=db)

    apply_review_decisions(
        [{
            "doc_id": "mixed_0", "decision": policy.REVIEW_CORRECTED,
            "decision_domain": "criminal_law", "reviewer": "ibrahim",
        }],
        db_path=db, decision_run_id="run1",
    )

    result = run_domain_classification("run2", "sig1", db_path=db)

    rep = get_representation("mixed_0", db_path=db)
    assert rep.primary_domain == "criminal_law"
    assert rep.classification_status == CLASSIFICATION_STATUS_AUTO_ACCEPTED
    assert result.protected_by_review == 1


def test_a_protected_document_still_gets_an_audit_row(db, flow_settings, profiles):
    """What the pipeline would have said is always recorded."""

    _seed(db, profiles)
    run_domain_classification("run1", "sig1", db_path=db)
    apply_review_decisions(
        [{
            "doc_id": "mixed_0", "decision": policy.REVIEW_CORRECTED,
            "decision_domain": "criminal_law", "reviewer": "ibrahim",
        }],
        db_path=db, decision_run_id="run1",
    )
    run_domain_classification("run2", "sig1", db_path=db)

    rows = {r["doc_id"]: r for r in get_classifications_for_run("run2", db_path=db)}
    assert "mixed_0" in rows
    assert rows["mixed_0"]["status"] == policy.STATUS_NEEDS_REVIEW


def test_protection_can_be_overridden_deliberately(db, flow_settings, profiles):
    _seed(db, profiles)
    run_domain_classification("run1", "sig1", db_path=db)
    apply_review_decisions(
        [{
            "doc_id": "mixed_0", "decision": policy.REVIEW_CORRECTED,
            "decision_domain": "criminal_law", "reviewer": "ibrahim",
        }],
        db_path=db, decision_run_id="run1",
    )
    run_domain_classification("run2", "sig1", db_path=db, protect_reviewed=False)

    rep = get_representation("mixed_0", db_path=db)
    assert rep.classification_status == CLASSIFICATION_STATUS_NEEDS_REVIEW


def test_a_rerun_over_settled_work_changes_nothing(db, flow_settings, profiles):
    """The second pass should be almost entirely no-ops."""

    _seed(db, profiles)
    run_domain_classification("run1", "sig1", db_path=db)
    second = run_domain_classification("run2", "sig1", db_path=db)

    assert second.write_outcomes.get(WRITE_UPDATED, 0) == 0
    assert second.write_outcomes.get(WRITE_UNCHANGED, 0) == second.decided


def test_an_unchanged_write_does_not_touch_updated_at(db, flow_settings, profiles):
    _seed(db, profiles)
    run_domain_classification("run1", "sig1", db_path=db)

    with connection_scope(db) as conn:
        before = conn.execute(
            "SELECT updated_at FROM document_representations WHERE doc_id = 'fam_0'"
        ).fetchone()["updated_at"]

    run_domain_classification("run2", "sig1", db_path=db)

    with connection_scope(db) as conn:
        after = conn.execute(
            "SELECT updated_at FROM document_representations WHERE doc_id = 'fam_0'"
        ).fetchone()["updated_at"]
    assert after == before


def test_write_outcomes_are_reported_not_swallowed(db, flow_settings, profiles):
    _seed(db, profiles)
    assert update_classification_state(
        "fam_0", classification_status=CLASSIFICATION_STATUS_AUTO_ACCEPTED,
        primary_domain="family_law", domain_confidence=0.9, db_path=db,
    ) == WRITE_UPDATED
    assert update_classification_state(
        "fam_0", classification_status=CLASSIFICATION_STATUS_AUTO_ACCEPTED,
        primary_domain="family_law", domain_confidence=0.9, db_path=db,
    ) == WRITE_UNCHANGED
    assert update_classification_state(
        "fam_0", classification_status=CLASSIFICATION_STATUS_NEEDS_REVIEW,
        protect_statuses=(CLASSIFICATION_STATUS_AUTO_ACCEPTED,), db_path=db,
    ) == WRITE_PROTECTED
    assert update_classification_state(
        "no_such_doc", classification_status=CLASSIFICATION_STATUS_AUTO_ACCEPTED,
        db_path=db,
    ) == WRITE_MISSING


def test_reviewed_doc_ids_drive_protection(db, flow_settings, profiles):
    _seed(db, profiles)
    assert get_reviewed_doc_ids(db_path=db) == set()

    persist_review_decisions(
        [{"doc_id": "fam_0", "decision": policy.REVIEW_ACCEPTED, "reviewer": "r"}],
        db_path=db,
    )
    assert get_reviewed_doc_ids(db_path=db) == {"fam_0"}


def test_a_reviewed_document_leaves_the_queue(db, flow_settings, profiles):
    _seed(db, profiles)
    run_domain_classification("run1", "sig1", db_path=db)

    first = build_review_queue("run1", db_path=db)
    assert first

    apply_review_decisions(
        [{"doc_id": first[0]["doc_id"], "decision": policy.REVIEW_STILL_UNCERTAIN,
          "reviewer": "ibrahim"}],
        db_path=db, decision_run_id="run1",
    )
    remaining = {r["doc_id"] for r in build_review_queue("run1", db_path=db)}
    assert first[0]["doc_id"] not in remaining
