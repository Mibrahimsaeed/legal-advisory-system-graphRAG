"""Phase 6: full-corpus domain classification against the frozen taxonomy.

The assertions that matter here are the refusals and the bookkeeping:

* a taxonomy that was never frozen stops the stage dead,
* a hallucinated domain id never becomes a label,
* an unsure verdict goes to review instead of being rounded up,
* a resumed run does not re-spend LLM calls,
* and no run ever overwrites another run's verdicts.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from orchestration.dags.classification_flow import run_classification
from src.classification.classification_store import (
    DEFAULT_CLASSIFICATION_SCHEMA_FILE,
    classification_stats,
    get_classification_history,
    get_classifications_for_run,
    get_current_classifications,
    persist_classifications,
)
from src.classification.domain_classifier import (
    CLASSIFIER_VERSION,
    REVIEW_LOW_CONFIDENCE,
    REVIEW_MULTI_DOMAIN,
    REVIEW_OTHER_BUCKET,
    STATUS_CLASSIFIED,
    STATUS_FAILED,
    STATUS_NEEDS_REVIEW,
    ClassificationResult,
    apply_review_policy,
    build_classification_prompt,
    classify_document,
    render_taxonomy_prompt,
    validate_classification,
)
from src.classification.taxonomy_registry import (
    OTHER_DOMAIN_ID,
    load_frozen_taxonomy,
)
from src.clustering.taxonomy_freeze import freeze_taxonomy
from src.common.db import connection_scope, init_schema
from src.common.exceptions import ConfigurationError
from src.extraction.doc_representation import DocumentRepresentation
from src.extraction.representation_store import (
    DEFAULT_REPRESENTATION_SCHEMA_FILE,
    upsert_representations,
)

DOMAIN_REGISTRY_SCHEMA_FILE = "schemas/domain_registry_schema.sql"

TAXONOMY_YAML = {
    "version": "test_taxonomy@v1",
    "domains": [
        {
            "id": "criminal_appeals",
            "name": "Criminal Appeals",
            "description": "Appeals against criminal convictions.",
            "inclusion_criteria": ["Conviction under the Pakistan Penal Code"],
            "exclusion_criteria": ["Tax matters"],
        },
        {
            "id": "tax_references",
            "name": "Tax References",
            "description": "Income and sales tax references.",
            "inclusion_criteria": ["Income Tax Ordinance references"],
            "exclusion_criteria": ["Customs"],
        },
    ],
}


@pytest.fixture()
def taxonomy_file(tmp_path) -> Path:
    path = tmp_path / "domains.yaml"
    path.write_text(yaml.safe_dump(TAXONOMY_YAML), encoding="utf-8")
    return path


@pytest.fixture()
def taxonomy(taxonomy_file):
    return load_frozen_taxonomy(taxonomy_file)


@pytest.fixture()
def classification_db(tmp_path) -> Path:
    db_path = tmp_path / "metadata.db"
    init_schema(db_path=db_path, schema_file=str(DEFAULT_REPRESENTATION_SCHEMA_FILE))
    init_schema(db_path=db_path, schema_file=str(DEFAULT_CLASSIFICATION_SCHEMA_FILE))
    init_schema(db_path=db_path, schema_file=DOMAIN_REGISTRY_SCHEMA_FILE)
    return db_path


def _rep(doc_id: str, topic: str = "criminal") -> DocumentRepresentation:
    return DocumentRepresentation(
        doc_id=doc_id,
        source_uri=f"/cases/{doc_id}",
        source_type="case_html",
        source_relpath=f"{topic}/{doc_id}",
        title=f"{topic} matter {doc_id}",
        headings=["Facts", "Order"],
        body_preview=f"A judgment about {topic} proceedings. " * 5,
        char_count=300,
        court="Lahore High Court",
        decision_date="2020-01-01",
    )


class _ScriptedLLM:
    """Returns a canned response per call; records how many calls were made."""

    def __init__(self, responses, default=None):
        self.responses = list(responses)
        self.default = default
        self.calls = 0

    def complete(self, system, prompt, max_tokens=None):
        self.calls += 1
        if self.responses:
            response = self.responses.pop(0)
        elif self.default is not None:
            response = self.default
        else:
            raise AssertionError("LLM called more times than scripted")
        if isinstance(response, Exception):
            raise response
        return response if isinstance(response, str) else json.dumps(response)


def _ok_response(domain="criminal_appeals", confidence=0.9, secondary=None):
    return json.dumps(
        {
            "primary_domain": domain,
            "secondary_domains": secondary or [],
            "confidence": confidence,
            "justification": "The judgment reviews a conviction under section 302 PPC.",
        }
    )


# ---------------------------------------------------------------------------
# frozen taxonomy: it must exist, and be traceable
# ---------------------------------------------------------------------------


def test_classification_refuses_to_run_without_a_frozen_taxonomy(tmp_path):
    empty = tmp_path / "domains.yaml"
    empty.write_text("", encoding="utf-8")

    with pytest.raises(ConfigurationError) as excinfo:
        load_frozen_taxonomy(empty)

    assert "empty" in str(excinfo.value).lower()


def test_missing_taxonomy_file_is_an_error(tmp_path):
    with pytest.raises(ConfigurationError):
        load_frozen_taxonomy(tmp_path / "nope.yaml")


@pytest.mark.parametrize(
    ("payload", "fragment"),
    [
        ({"domains": []}, "no domains"),
        ({"domains": [{"name": "No Id"}]}, "no id"),
        ({"domains": [{"id": "x"}]}, "no name"),
        (
            {"domains": [{"id": "x", "name": "X"}, {"id": "x", "name": "Y"}]},
            "duplicate",
        ),
        ({"domains": [{"id": OTHER_DOMAIN_ID, "name": "Other"}]}, "at least one real domain"),
    ],
)
def test_malformed_taxonomy_is_rejected(tmp_path, payload, fragment):
    path = tmp_path / "domains.yaml"
    path.write_text(yaml.safe_dump(payload), encoding="utf-8")

    with pytest.raises(ConfigurationError) as excinfo:
        load_frozen_taxonomy(path)

    assert fragment in str(excinfo.value).lower()


def test_taxonomy_version_defaults_to_a_content_hash(tmp_path):
    payload = copy.deepcopy({k: v for k, v in TAXONOMY_YAML.items() if k != "version"})
    path = tmp_path / "domains.yaml"
    path.write_text(yaml.safe_dump(payload), encoding="utf-8")

    first = load_frozen_taxonomy(path)
    assert first.version.startswith("sha256:")

    # Editing a criterion changes the version; nothing else needs to.
    payload["domains"][0]["inclusion_criteria"] = ["Something else"]
    path.write_text(yaml.safe_dump(payload), encoding="utf-8")
    assert load_frozen_taxonomy(path).version != first.version


def test_frozen_taxonomy_exposes_assignable_ids(taxonomy):
    assert taxonomy.assignable_ids == {
        "criminal_appeals", "tax_references", OTHER_DOMAIN_ID,
    }
    assert taxonomy.get("tax_references").name == "Tax References"


# ---------------------------------------------------------------------------
# freezing (the human step this stage depends on)
# ---------------------------------------------------------------------------


def _seed_candidates(db_path: Path) -> None:
    from src.clustering.taxonomy_card import (
        DomainDraft,
        build_other_bucket_draft,
        build_taxonomy_card,
        persist_taxonomy_card,
    )

    card = build_taxonomy_card(
        run_id="draft_run",
        sample_size=100,
        total_signatures=100,
        embedding_model="fake",
        domains=[
            DomainDraft(
                cluster_id=0, name="Criminal Appeals", description="d",
                inclusion_criteria=["conviction"], exclusion_criteria=["tax"],
                doc_count=40, sample_size=100, domain_id="criminal_appeals",
            ),
            DomainDraft(
                cluster_id=1, name="Rent Matters", description="d",
                doc_count=30, sample_size=100, domain_id="rent_matters",
                confidence="low", review_required=True, flags=["weak_relative_confidence"],
            ),
        ],
        other_bucket=build_other_bucket_draft(["x"], sample_size=100),
    )
    persist_taxonomy_card(card, db_path=db_path)


def test_freeze_skips_domains_flagged_for_review(classification_db, tmp_path):
    _seed_candidates(classification_db)
    out = tmp_path / "domains.yaml"

    freeze_taxonomy("draft_run", db_path=classification_db, output_path=out)

    frozen = load_frozen_taxonomy(out)
    assert frozen.domain_ids == {"criminal_appeals"}  # rent_matters was flagged


def test_freeze_accepts_a_flagged_domain_when_named_explicitly(classification_db, tmp_path):
    _seed_candidates(classification_db)
    out = tmp_path / "domains.yaml"

    freeze_taxonomy(
        "draft_run",
        accepted_domain_ids=["criminal_appeals", "rent_matters"],
        db_path=classification_db,
        output_path=out,
    )

    assert load_frozen_taxonomy(out).domain_ids == {"criminal_appeals", "rent_matters"}


def test_freeze_marks_candidates_accepted(classification_db, tmp_path):
    _seed_candidates(classification_db)

    freeze_taxonomy("draft_run", db_path=classification_db, output_path=tmp_path / "d.yaml")

    with connection_scope(classification_db) as conn:
        rows = {
            r["domain_id"]: r["status"]
            for r in conn.execute("SELECT domain_id, status FROM domain_candidates")
        }
    assert rows["criminal_appeals"] == "accepted"
    assert rows["rent_matters"] == "draft"  # untouched, still needs a human


def test_freeze_refuses_to_clobber_an_existing_taxonomy(classification_db, tmp_path):
    _seed_candidates(classification_db)
    out = tmp_path / "domains.yaml"
    freeze_taxonomy("draft_run", db_path=classification_db, output_path=out)

    with pytest.raises(ConfigurationError):
        freeze_taxonomy("draft_run", db_path=classification_db, output_path=out)

    freeze_taxonomy("draft_run", db_path=classification_db, output_path=out, overwrite=True)


# ---------------------------------------------------------------------------
# structured output validation
# ---------------------------------------------------------------------------


def test_valid_response_is_accepted(taxonomy):
    parsed = json.loads(_ok_response(secondary=["tax_references"]))

    result = validate_classification(parsed, taxonomy, "d1")

    assert result.status == STATUS_CLASSIFIED
    assert result.primary_domain == "criminal_appeals"
    assert result.secondary_domains == ["tax_references"]
    assert result.confidence == 0.9
    assert result.justification


def test_hallucinated_domain_id_never_becomes_a_label(taxonomy):
    parsed = json.loads(_ok_response(domain="constitutional_law"))

    result = validate_classification(parsed, taxonomy, "d1")

    assert result.status == STATUS_FAILED
    assert result.primary_domain is None
    assert result.error.startswith("unknown_domain_id")


@pytest.mark.parametrize(
    ("payload", "error_prefix"),
    [
        ({"secondary_domains": [], "confidence": 0.9, "justification": "x"}, "missing_primary_domain"),
        ({"primary_domain": "criminal_appeals", "confidence": "high", "justification": "x"}, "invalid_confidence"),
        ({"primary_domain": "criminal_appeals", "confidence": 1.7, "justification": "x"}, "invalid_confidence"),
        ({"primary_domain": "criminal_appeals", "confidence": 0.9}, "missing_justification"),
        ({"primary_domain": "criminal_appeals", "confidence": 0.9, "justification": "   "}, "missing_justification"),
    ],
)
def test_malformed_responses_fail_loudly(taxonomy, payload, error_prefix):
    result = validate_classification(payload, taxonomy, "d1")

    assert result.status == STATUS_FAILED
    assert result.error.startswith(error_prefix)
    assert result.primary_domain is None


def test_percentage_confidence_is_normalized(taxonomy):
    result = validate_classification(
        {"primary_domain": "criminal_appeals", "confidence": 85, "justification": "x"},
        taxonomy, "d1",
    )

    assert result.confidence == pytest.approx(0.85)


def test_unknown_secondary_domains_are_dropped_not_fatal(taxonomy):
    parsed = json.loads(_ok_response(secondary=["tax_references", "made_up", "criminal_appeals"]))

    result = validate_classification(parsed, taxonomy, "d1")

    assert result.status == STATUS_CLASSIFIED
    assert result.secondary_domains == ["tax_references"]  # unknown + self-duplicate dropped


def test_non_json_response_is_a_failure_not_an_exception(taxonomy):
    result = classify_document(
        _rep("d1"), taxonomy, _ScriptedLLM(["I think this is a criminal case."])
    )

    assert result.status == STATUS_FAILED
    assert "LLMError" in result.error or "Could not find" in result.error


def test_llm_exception_is_a_failure_not_an_exception(taxonomy):
    result = classify_document(
        _rep("d1"), taxonomy, _ScriptedLLM([RuntimeError("ollama down")])
    )

    assert result.status == STATUS_FAILED
    assert "ollama down" in result.error


# ---------------------------------------------------------------------------
# review routing
# ---------------------------------------------------------------------------


def test_low_confidence_is_routed_to_review_with_its_label_kept():
    result = apply_review_policy(
        ClassificationResult(
            doc_id="d1", primary_domain="criminal_appeals", confidence=0.4,
            justification="weak", status=STATUS_CLASSIFIED,
        ),
        min_confidence=0.6,
    )

    assert result.status == STATUS_NEEDS_REVIEW
    assert result.review_reason == REVIEW_LOW_CONFIDENCE
    assert result.primary_domain == "criminal_appeals"  # proposal preserved for the reviewer


def test_multi_domain_case_is_routed_to_review():
    result = apply_review_policy(
        ClassificationResult(
            doc_id="d1", primary_domain="criminal_appeals",
            secondary_domains=["tax_references"], confidence=0.95,
            justification="both", status=STATUS_CLASSIFIED,
        ),
    )

    assert result.status == STATUS_NEEDS_REVIEW
    assert REVIEW_MULTI_DOMAIN in result.review_reason


def test_other_uncertain_is_routed_to_review():
    result = apply_review_policy(
        ClassificationResult(
            doc_id="d1", primary_domain=OTHER_DOMAIN_ID, confidence=0.99,
            justification="fits nothing", status=STATUS_CLASSIFIED,
        ),
    )

    assert result.status == STATUS_NEEDS_REVIEW
    assert REVIEW_OTHER_BUCKET in result.review_reason


def test_failed_results_are_never_re_routed_as_review():
    result = apply_review_policy(
        ClassificationResult(doc_id="d1", status=STATUS_FAILED, error="unknown_domain_id:x")
    )

    assert result.status == STATUS_FAILED
    assert result.review_reason is None


def test_confident_single_domain_is_accepted(taxonomy):
    result = classify_document(_rep("d1"), taxonomy, _ScriptedLLM([_ok_response()]))

    assert result.status == STATUS_CLASSIFIED
    assert result.review_reason is None


# ---------------------------------------------------------------------------
# prompt construction
# ---------------------------------------------------------------------------


def test_prompt_lists_every_domain_plus_the_catch_all(taxonomy):
    block = render_taxonomy_prompt(taxonomy)

    assert "criminal_appeals" in block and "tax_references" in block
    assert OTHER_DOMAIN_ID in block
    assert "include: Conviction under the Pakistan Penal Code" in block
    assert "exclude: Tax matters" in block


def test_prompt_carries_document_identity_and_text(taxonomy):
    prompt = build_classification_prompt(_rep("d1"), render_taxonomy_prompt(taxonomy))

    assert "criminal matter d1" in prompt
    assert "Court: Lahore High Court" in prompt
    assert "Facts; Order" in prompt


# ---------------------------------------------------------------------------
# storage: versioning and history
# ---------------------------------------------------------------------------


def _result(doc_id, domain="criminal_appeals", status=STATUS_CLASSIFIED, confidence=0.9):
    return ClassificationResult(
        doc_id=doc_id, primary_domain=domain, confidence=confidence,
        justification="because", status=status, cluster_id=3,
    )


def test_persist_records_full_provenance(classification_db):
    persist_classifications(
        "run_a", [_result("d1")], taxonomy_version="tax@v1",
        classifier_version=CLASSIFIER_VERSION, model_name="qwen3:14b",
        batch_id="run_a_b0000", db_path=classification_db,
    )

    row = get_classifications_for_run("run_a", db_path=classification_db)[0]
    assert row["doc_id"] == "d1"
    assert row["cluster_id"] == 3
    assert row["primary_domain"] == "criminal_appeals"
    assert row["confidence"] == 0.9
    assert row["justification"] == "because"
    assert row["taxonomy_version"] == "tax@v1"
    assert row["classifier_version"] == CLASSIFIER_VERSION
    assert row["model_name"] == "qwen3:14b"
    assert row["batch_id"] == "run_a_b0000"
    assert row["created_at"]
    assert row["secondary_domains"] == []


def test_a_new_run_never_overwrites_an_earlier_verdict(classification_db):
    persist_classifications(
        "run_a", [_result("d1", domain="criminal_appeals")],
        taxonomy_version="tax@v1", classifier_version="c/1.0", db_path=classification_db,
    )
    persist_classifications(
        "run_b", [_result("d1", domain="tax_references")],
        taxonomy_version="tax@v2", classifier_version="c/2.0", db_path=classification_db,
    )

    history = get_classification_history("d1", db_path=classification_db)
    assert len(history) == 2
    assert {h["taxonomy_version"] for h in history} == {"tax@v1", "tax@v2"}
    # The older verdict is still readable, unchanged.
    old = next(h for h in history if h["run_id"] == "run_a")
    assert old["primary_domain"] == "criminal_appeals"


def test_current_classification_is_the_newest_row(classification_db):
    persist_classifications(
        "run_a", [_result("d1", domain="criminal_appeals")],
        taxonomy_version="tax@v1", classifier_version="c/1.0", db_path=classification_db,
    )
    persist_classifications(
        "run_b", [_result("d1", domain="tax_references")],
        taxonomy_version="tax@v2", classifier_version="c/2.0", db_path=classification_db,
    )

    current = get_current_classifications(db_path=classification_db)
    assert len(current) == 1
    assert current[0]["primary_domain"] == "tax_references"

    scoped = get_current_classifications(db_path=classification_db, taxonomy_version="tax@v1")
    assert scoped[0]["primary_domain"] == "criminal_appeals"


def test_re_persisting_within_a_run_is_idempotent(classification_db):
    persist_classifications(
        "run_a", [_result("d1", confidence=0.5)], taxonomy_version="tax@v1",
        classifier_version="c/1.0", db_path=classification_db,
    )
    persist_classifications(
        "run_a", [_result("d1", confidence=0.8)], taxonomy_version="tax@v1",
        classifier_version="c/1.0", db_path=classification_db,
    )

    rows = get_classifications_for_run("run_a", db_path=classification_db)
    assert len(rows) == 1  # the retried batch refreshed its own row
    assert rows[0]["confidence"] == 0.8


def test_stats_summarize_a_run(classification_db):
    persist_classifications(
        "run_a",
        [
            _result("d1", confidence=0.9),
            _result("d2", domain="tax_references", confidence=0.8),
            _result("d3", status=STATUS_NEEDS_REVIEW, confidence=0.4),
            ClassificationResult(doc_id="d4", status=STATUS_FAILED, error="unknown_domain_id:zzz"),
        ],
        taxonomy_version="tax@v1", classifier_version="c/1.0", db_path=classification_db,
    )

    stats = classification_stats("run_a", db_path=classification_db)

    assert stats["total"] == 4
    assert stats["by_status"] == {"classified": 2, "needs_review": 1, "failed": 1}
    assert stats["by_primary_domain"]["criminal_appeals"] == 2
    assert stats["failure_rate"] == pytest.approx(0.25)
    assert stats["needs_review_rate"] == pytest.approx(0.25)
    assert stats["mean_confidence"] == pytest.approx((0.9 + 0.8 + 0.4) / 3)


# ---------------------------------------------------------------------------
# the flow: batching, resumability, pilot
# ---------------------------------------------------------------------------


@pytest.fixture()
def classification_settings(tmp_path, monkeypatch, taxonomy_file):
    import orchestration.dags.classification_flow as flow

    settings = SimpleNamespace(
        pipeline=SimpleNamespace(checkpoint_dir=tmp_path / "checkpoints"),
        metrics=SimpleNamespace(db_path=tmp_path / "metrics.db"),
        caselaw=SimpleNamespace(
            representation_schema_file=DEFAULT_REPRESENTATION_SCHEMA_FILE
        ),
        # classification_flow now also initialises the domain-registry
        # schema, so cluster_assignments exists on a fresh database.
        discovery=SimpleNamespace(
            domain_registry_schema_file=Path(DOMAIN_REGISTRY_SCHEMA_FILE)
        ),
        classification=SimpleNamespace(
            taxonomy_file=taxonomy_file,
            schema_file=DEFAULT_CLASSIFICATION_SCHEMA_FILE,
            batch_size=2,
            pilot_size=2,
            llm_model="test-model",
            llm_max_tokens=512,
            body_chars=3000,
            min_confidence=0.6,
            review_multi_domain=True,
            review_other_bucket=True,
        ),
    )
    monkeypatch.setattr(flow, "get_settings", lambda: settings)
    return settings


def test_flow_classifies_in_batches_and_reports_stats(classification_db, classification_settings):
    upsert_representations([_rep(f"d{i}") for i in range(5)], db_path=classification_db)
    llm = _ScriptedLLM([], default=_ok_response())

    result = run_classification(
        run_id="run_1", db_path=classification_db, llm_client=llm, batch_size=2,
    )

    assert result.processed == 5
    assert llm.calls == 5
    assert result.stats["by_status"] == {"classified": 5}
    assert result.taxonomy_version == "test_taxonomy@v1"
    assert result.classifier_version == CLASSIFIER_VERSION
    # Batches are persisted as they go, and tagged.
    rows = get_classifications_for_run("run_1", db_path=classification_db)
    assert {r["batch_id"] for r in rows} == {"run_1_b0000", "run_1_b0001", "run_1_b0002"}


def test_flow_resumes_without_re_spending_llm_calls(classification_db, classification_settings):
    upsert_representations([_rep(f"d{i}") for i in range(4)], db_path=classification_db)

    first = run_classification(
        run_id="run_1", db_path=classification_db,
        llm_client=_ScriptedLLM([_ok_response(), _ok_response()]), pilot_size=2,
    )
    assert first.processed == 2

    resumed_llm = _ScriptedLLM([], default=_ok_response())
    second = run_classification(
        run_id="run_1", db_path=classification_db, llm_client=resumed_llm,
    )

    assert second.processed == 2  # only the two that were left
    assert second.skipped_already_done == 2
    assert resumed_llm.calls == 2
    assert len(get_classifications_for_run("run_1", db_path=classification_db)) == 4


def test_flow_retries_failed_documents_on_resume(classification_db, classification_settings):
    upsert_representations([_rep("d0")], db_path=classification_db)

    run_classification(
        run_id="run_1", db_path=classification_db,
        llm_client=_ScriptedLLM([RuntimeError("transient")]),
    )
    assert get_classifications_for_run("run_1", db_path=classification_db)[0]["status"] == STATUS_FAILED

    run_classification(
        run_id="run_1", db_path=classification_db,
        llm_client=_ScriptedLLM([_ok_response()]),
    )

    rows = get_classifications_for_run("run_1", db_path=classification_db)
    assert len(rows) == 1
    assert rows[0]["status"] == STATUS_CLASSIFIED


def test_pilot_limits_the_run_and_flags_itself(classification_db, classification_settings):
    upsert_representations([_rep(f"d{i}") for i in range(10)], db_path=classification_db)

    result = run_classification(
        run_id="pilot_1", db_path=classification_db,
        llm_client=_ScriptedLLM([], default=_ok_response()), pilot_size=3,
    )

    assert result.is_pilot is True
    assert result.processed == 3
    assert len(get_classifications_for_run("pilot_1", db_path=classification_db)) == 3


def test_flow_attaches_cluster_ids_from_a_discovery_run(classification_db, classification_settings):
    upsert_representations([_rep("d0"), _rep("d1")], db_path=classification_db)
    with connection_scope(classification_db) as conn:
        conn.executemany(
            "INSERT INTO cluster_assignments (run_id, doc_id, cluster_id, confidence, created_at) "
            "VALUES (?, ?, ?, ?, datetime('now'))",
            [("disc_run", "d0", 2, 0.9), ("disc_run", "d1", -1, 0.0)],
        )

    run_classification(
        run_id="run_1", db_path=classification_db, cluster_run_id="disc_run",
        llm_client=_ScriptedLLM([], default=_ok_response()),
    )

    rows = {r["doc_id"]: r["cluster_id"] for r in get_classifications_for_run("run_1", db_path=classification_db)}
    assert rows == {"d0": 2, "d1": -1}


def test_flow_routes_weak_verdicts_to_review(classification_db, classification_settings):
    upsert_representations([_rep("d0"), _rep("d1"), _rep("d2")], db_path=classification_db)
    llm = _ScriptedLLM(
        [
            _ok_response(confidence=0.95),
            _ok_response(confidence=0.3),
            _ok_response(domain=OTHER_DOMAIN_ID, confidence=0.9),
        ]
    )

    result = run_classification(run_id="run_1", db_path=classification_db, llm_client=llm)

    assert result.stats["by_status"] == {"classified": 1, "needs_review": 2}
    assert set(result.stats["by_review_reason"]) == {REVIEW_LOW_CONFIDENCE, REVIEW_OTHER_BUCKET}
    assert len(result.needs_review) == 2


def test_flow_records_failures_without_labels(classification_db, classification_settings):
    upsert_representations([_rep("d0")], db_path=classification_db)
    llm = _ScriptedLLM([json.dumps({"primary_domain": "invented", "confidence": 0.9, "justification": "x"})])

    result = run_classification(run_id="run_1", db_path=classification_db, llm_client=llm)

    assert result.stats["by_status"] == {"failed": 1}
    row = get_classifications_for_run("run_1", db_path=classification_db)[0]
    assert row["primary_domain"] is None
    assert row["error"].startswith("unknown_domain_id")


def test_flow_with_nothing_pending_is_a_no_op(classification_db, classification_settings):
    upsert_representations([_rep("d0")], db_path=classification_db)
    run_classification(
        run_id="run_1", db_path=classification_db,
        llm_client=_ScriptedLLM([_ok_response()]),
    )

    llm = _ScriptedLLM([])  # any call would raise
    again = run_classification(run_id="run_1", db_path=classification_db, llm_client=llm)

    assert again.processed == 0
    assert llm.calls == 0
    assert again.stats["total"] == 1
