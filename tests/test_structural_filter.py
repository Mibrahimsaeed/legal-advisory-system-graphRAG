"""Phase 2: structural noise pre-filtering.

Two halves: the pure rule engine (:mod:`src.extraction.structural_filter`),
and its integration into case ingest -- where the decision is persisted and,
critically, where a dropped document is short-circuited so it never reaches
embedding or HDBSCAN.

The filter is meant to be high-precision, so the most important tests here
are the ones asserting that genuine judgments are *kept*: a false drop
silently deletes real law from the corpus.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from orchestration.dags.case_ingest_flow import run_case_ingest
from src.common.config import get_settings
from src.common.db import connection_scope, init_schema
from src.extraction.doc_representation import (
    CLASSIFICATION_STATUS_AUTO_ACCEPTED,
    CLASSIFICATION_STATUS_DROPPED_PROCEDURAL,
    CLASSIFICATION_STATUS_PENDING,
)
from src.extraction.representation_store import (
    DEFAULT_REPRESENTATION_SCHEMA_FILE,
    get_representation,
    list_representations,
)
from src.extraction.structural_filter import (
    REASON_CAUSE_LIST,
    REASON_INCOMPLETE_SCRAPE,
    REASON_OFFICE_REPORT,
    REASON_PROCEDURAL_ADJOURNMENT,
    REASON_SHORT_DOCUMENT,
    validate_structure,
)

# ---------------------------------------------------------------------------
# document builders
# ---------------------------------------------------------------------------

JUDGMENT_BODY = (
    "The appellant was convicted under section 302 of the Pakistan Penal Code, 1860 "
    "and sentenced to imprisonment for life by the learned Additional Sessions Judge. "
    "Learned counsel for the appellant contends that the ocular account was furnished "
    "by chance witnesses whose presence at the place of occurrence was doubtful, and "
    "that the recovery of the crime weapon was planted. The learned Deputy Prosecutor "
    "General defended the concurrent findings and submitted that no misreading of "
    "evidence had been pointed out. We have perused the record with the assistance of "
    "the learned counsel for the parties. The medical evidence is at variance with the "
    "ocular account regarding the seat of the injuries. It is held that benefit of the "
    "doubt is not a matter of grace but of right. The appeal is allowed and the "
    "impugned judgment is set aside. "
)


def _judgment(times: int = 4) -> str:
    return "ORDER\n\n" + JUDGMENT_BODY * times


def _family_judgment() -> str:
    """A genuine judgment using entirely different terminology."""

    return (
        "JUDGMENT\n\n"
        + (
            "The respondent instituted a suit for recovery of dower, dowry articles and "
            "maintenance allowance before the learned Judge Family Court. The parties were "
            "married on 05.03.2012 against a prompt dower and the nikahnama was exhibited "
            "without objection. The wife seeks dissolution of marriage on the basis of khula, "
            "asserting a fixed aversion. The welfare of the minor is the paramount "
            "consideration under the Guardians and Wards Act, 1890. Learned counsel submitted "
            "that maintenance was fixed at an excessive rate. In view of the evidence led by "
            "the parties, the appeal is partly allowed and the decree is modified. "
        )
        * 4
    )


def _cause_list(entries: int = 14) -> str:
    rows = "\n".join(
        f"{i}. C.P. No. {100 + i} of 2021    Ahmed Ali v. Bashir Khan    Mr. A. Counsel"
        for i in range(1, entries + 1)
    )
    return f"DAILY CAUSE LIST\nCourt No. 3\n\n{rows}\n"


def _html(body: str, title: str = "Case") -> str:
    return f"<html><head><title>{title}</title></head><body>{body}</body></html>"


def _write_case(root: Path, name: str, body_html: str, metadata: dict | None = None) -> Path:
    folder = root / name
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "case.html").write_text(_html(body_html), encoding="utf-8")
    if metadata is not None:
        (folder / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
    return folder


# ---------------------------------------------------------------------------
# Rule engine: genuine judgments must survive
# ---------------------------------------------------------------------------


def test_substantive_judgment_passes():
    decision = validate_structure(_judgment())

    assert decision.passed
    assert decision.drop_reason is None
    assert decision.char_count > 1200
    assert decision.judgment_markers > 0 and decision.reasoning_markers > 0


def test_judgment_with_different_terminology_passes():
    """No single keyword may be mandatory -- family law reads nothing like
    a criminal appeal."""

    decision = validate_structure(_family_judgment())

    assert decision.passed, decision.drop_reason


def test_long_judgment_mentioning_an_adjournment_is_not_dropped():
    """The substantive guard: a procedural phrase inside a real judgment
    must not trigger the procedural rule."""

    text = _judgment() + " The matter was earlier adjourned to 12.05.2021 at the request of counsel."

    decision = validate_structure(text)

    assert decision.passed, decision.drop_reason


def test_judgment_citing_many_cases_is_not_mistaken_for_a_cause_list():
    citations = " ".join(
        f"Reliance is placed on C.P. No. {100 + i} of 2019, wherein it was held otherwise."
        for i in range(12)
    )
    decision = validate_structure(_judgment() + citations)

    assert decision.passed, decision.drop_reason


def test_court_metadata_does_not_affect_the_structural_decision():
    """Court is metadata; it must not change structural validity."""

    text = _judgment()
    verdicts = {
        court: validate_structure(text, title=f"Case heard at {court}").passed
        for court in (
            "Supreme Court of Pakistan",
            "Lahore High Court",
            "Federal Service Tribunal",
            None,
        )
    }

    assert set(verdicts.values()) == {True}


# ---------------------------------------------------------------------------
# Rule engine: structural noise must be caught
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "text",
    [
        "Order sheet. Case called. None present. Adjourned to 12.05.2021.",
        "Relisted before the appropriate Bench. Put up on 03.09.2020.",
        "At the request of learned counsel, the case is adjourned till 21.01.2022.",
        "Case called, none present. List on 14.07.2019.",
    ],
)
def test_procedural_orders_are_dropped(text):
    decision = validate_structure(text)

    assert decision.dropped
    assert decision.drop_reason == REASON_PROCEDURAL_ADJOURNMENT


def test_cause_list_is_dropped():
    decision = validate_structure(_cause_list())

    assert decision.dropped
    assert decision.drop_reason == REASON_CAUSE_LIST
    assert decision.case_numbers >= 8


@pytest.mark.parametrize(
    "text",
    [
        "Office report submitted. Office objection regarding deficiency of court fee. "
        "The registry is directed to place the matter before the Honourable Judge in chambers.",
        "Registry report: the paper book is incomplete and the office note is awaited.",
    ],
)
def test_office_reports_are_dropped(text):
    decision = validate_structure(text)

    assert decision.dropped
    assert decision.drop_reason == REASON_OFFICE_REPORT


@pytest.mark.parametrize("text", ["", "   \n  ", "Judgment."])
def test_empty_or_near_empty_documents_are_dropped(text):
    decision = validate_structure(text)

    assert decision.dropped
    assert decision.drop_reason == REASON_INCOMPLETE_SCRAPE


def test_portal_error_page_is_dropped():
    text = (
        "404 Not Found. The page you requested could not be located on this server. "
        "Please check the citation and try again. Return to the search page. "
    ) * 5

    decision = validate_structure(text)

    assert decision.dropped
    assert decision.drop_reason == REASON_INCOMPLETE_SCRAPE


def test_document_below_min_characters_is_dropped():
    text = (
        "ORDER. The petitioner seeks relief. The learned counsel submitted that the "
        "impugned order is unsustainable. The petition is disposed of accordingly. "
    ) * 2

    decision = validate_structure(text, min_characters=1200, min_words=250)

    assert decision.dropped
    assert decision.drop_reason == REASON_SHORT_DOCUMENT
    assert decision.char_count < 1200


def test_document_below_min_words_is_dropped():
    """Long enough in characters, too few words -- a padded/garbled scrape."""

    text = "ORDER " + ("Supercalifragilisticexpialidocious " * 120)

    decision = validate_structure(text, min_characters=1200, min_words=250)

    assert decision.dropped
    assert decision.drop_reason == REASON_SHORT_DOCUMENT
    assert decision.char_count >= 1200 and decision.word_count < 250


def test_long_text_with_no_legal_markers_is_dropped():
    decision = validate_structure("lorem ipsum dolor sit amet " * 300)

    assert decision.dropped
    assert decision.drop_reason == "insufficient_substantive_text"


def test_decision_is_deterministic():
    text = _judgment()

    assert validate_structure(text) == validate_structure(text)


# ---------------------------------------------------------------------------
# Integration: ingest applies, persists and short-circuits
# ---------------------------------------------------------------------------


@pytest.fixture()
def db(tmp_path) -> Path:
    db_path = tmp_path / "phase2.db"
    init_schema(db_path=db_path, schema_file=str(DEFAULT_REPRESENTATION_SCHEMA_FILE))
    return db_path


@pytest.fixture()
def settings(monkeypatch):
    """Real document thresholds, so the flow tests exercise production config."""

    import orchestration.dags.case_ingest_flow as flow

    real = get_settings()
    monkeypatch.setattr(
        flow,
        "get_settings",
        lambda: SimpleNamespace(
            document=real.document,
            caselaw=SimpleNamespace(
                corpus_root=None,
                case_html_filename="case.html",
                metadata_filename="metadata.json",
                body_preview_char_limit=20_000,
                max_headings=50,
                representation_schema_file=DEFAULT_REPRESENTATION_SCHEMA_FILE,
            ),
        ),
    )
    return real


@pytest.fixture()
def mixed_corpus(tmp_path) -> Path:
    """Four folders: pass, drop, pass, drop -- processed independently."""

    root = tmp_path / "corpus"
    _write_case(root, "case_001", f"<h1>Criminal Appeal</h1><p>{_judgment()}</p>",
                {"court": "Lahore High Court", "case_title": "Criminal Appeal"})
    _write_case(root, "case_002", "<p>Case called. None present. Adjourned to 12.05.2021.</p>",
                {"court": "Supreme Court of Pakistan"})
    _write_case(root, "case_003", f"<h1>Family Appeal</h1><p>{_family_judgment()}</p>", None)
    _write_case(root, "case_004", f"<pre>{_cause_list()}</pre>",
                {"court": "Sindh High Court"})
    return root


def test_passing_document_is_pending_with_cleaned_text(mixed_corpus, db, settings):
    run_case_ingest(root=mixed_corpus, db_path=db)

    stored = [r for r in list_representations(db_path=db) if r.source_relpath == "case_001"]
    assert len(stored) == 1
    rep = stored[0]
    assert rep.classification_status == CLASSIFICATION_STATUS_PENDING
    assert rep.drop_reason is None
    assert rep.cleaned_text and "section 302" in rep.cleaned_text


def test_dropped_document_records_status_and_reason(mixed_corpus, db, settings):
    run_case_ingest(root=mixed_corpus, db_path=db)

    with connection_scope(db) as conn:
        rows = {
            r["source_relpath"]: (r["classification_status"], r["drop_reason"], r["cleaned_text"])
            for r in conn.execute(
                "SELECT source_relpath, classification_status, drop_reason, cleaned_text "
                "FROM document_representations"
            )
        }

    assert rows["case_002"][0] == CLASSIFICATION_STATUS_DROPPED_PROCEDURAL
    assert rows["case_002"][1] == REASON_PROCEDURAL_ADJOURNMENT
    assert rows["case_002"][2] is None  # no cleaned_text for dropped documents
    assert rows["case_004"][1] == REASON_CAUSE_LIST


def test_dropped_documents_are_still_stored_and_auditable(mixed_corpus, db, settings):
    """The record must survive: a drop is a decision, not a deletion."""

    run_case_ingest(root=mixed_corpus, db_path=db)

    with connection_scope(db) as conn:
        total = conn.execute("SELECT COUNT(*) c FROM document_representations").fetchone()["c"]
    assert total == 4

    audit = list_representations(db_path=db, exclude_dropped=False)
    dropped = [r for r in audit if r.classification_status == CLASSIFICATION_STATUS_DROPPED_PROCEDURAL]
    assert len(dropped) == 2
    assert all(r.source_file and r.body_preview for r in dropped)  # provenance + evidence kept


# ---------------------------------------------------------------------------
# THE SHORT-CIRCUIT
# ---------------------------------------------------------------------------


def test_dropped_documents_never_reach_the_embedding_stage(mixed_corpus, db, settings):
    """The Phase 2 contract: the corpus the embedder sees excludes drops."""

    result = run_case_ingest(root=mixed_corpus, db_path=db)
    assert sorted(result.pending_doc_ids) and len(result.dropped_doc_ids) == 2

    corpus = list_representations(db_path=db)  # what domain_discovery/classification load

    assert {r.source_relpath for r in corpus} == {"case_001", "case_003"}
    assert all(r.classification_status == CLASSIFICATION_STATUS_PENDING for r in corpus)


def test_embedding_function_is_never_called_for_dropped_documents(mixed_corpus, db, settings):
    """Instrument the real embedder: dropped text must not be embedded."""

    from src.embedding.doc_pooling import embed_documents
    from src.embedding.embed_model import DeterministicHashEmbedder

    run_case_ingest(root=mixed_corpus, db_path=db)

    seen_texts: list[str] = []

    class _RecordingEmbedder(DeterministicHashEmbedder):
        def encode(self, texts):
            seen_texts.extend(texts)
            return super().encode(texts)

    corpus = list_representations(db_path=db)
    vectors = embed_documents(corpus, _RecordingEmbedder(dimension=16))

    assert len(vectors) == 2  # only the two substantive judgments
    blob = " ".join(seen_texts)
    assert "None present" not in blob
    assert "DAILY CAUSE LIST" not in blob


def test_audit_query_can_still_reach_dropped_documents(mixed_corpus, db, settings):
    run_case_ingest(root=mixed_corpus, db_path=db)

    assert len(list_representations(db_path=db)) == 2
    assert len(list_representations(db_path=db, exclude_dropped=False)) == 4


# ---------------------------------------------------------------------------
# Batch behaviour, failures, idempotency
# ---------------------------------------------------------------------------


def test_one_bad_document_does_not_stop_the_corpus(tmp_path, db, settings):
    root = tmp_path / "corpus"
    _write_case(root, "good_1", f"<p>{_judgment()}</p>")
    (root / "broken").mkdir(parents=True)          # no case.html at all
    _write_case(root, "malformed", "<p><h1>Unclosed <b>tags " + _judgment())
    bad_meta = _write_case(root, "bad_meta", f"<p>{_judgment()}</p>")
    (bad_meta / "metadata.json").write_text("{not json", encoding="utf-8")
    _write_case(root, "good_2", f"<p>{_family_judgment()}</p>")

    result = run_case_ingest(root=root, db_path=db)

    # The scanner never sees "broken" (no case.html); the rest all processed.
    assert len(result.representations) == 4
    assert set(result.pending_doc_ids) == {
        r.doc_id for r in result.representations
    } - set(result.dropped_doc_ids)
    assert "good_1" in {r.source_relpath for r in list_representations(db_path=db)}


def test_missing_case_html_is_dropped_as_incomplete_scrape(tmp_path, db, settings):
    from src.extraction.case_loader import load_case_folder
    from orchestration.dags.case_ingest_flow import _apply_structural_filter

    folder = tmp_path / "no_html"
    folder.mkdir()
    (folder / "metadata.json").write_text(json.dumps({"court": "LHC"}), encoding="utf-8")

    rep = load_case_folder(folder, root=tmp_path)
    _apply_structural_filter(rep, None, get_settings().document)

    assert rep.status == "failed"
    assert rep.classification_status == CLASSIFICATION_STATUS_DROPPED_PROCEDURAL
    assert rep.drop_reason == REASON_INCOMPLETE_SCRAPE


def test_html_noise_does_not_inflate_the_length_check(tmp_path, db, settings):
    """Navigation/script boilerplate must not rescue a short order."""

    noise = (
        "<nav>" + ("Home | Search | Judgments | Advanced Search | Contact | " * 60) + "</nav>"
        "<script>var x = " + ("'padding',") * 200 + "1;</script>"
        "<style>" + (".c{color:#fff;}" * 100) + "</style>"
    )
    _write_case(tmp_path / "corpus", "stub",
                noise + "<p>Case called. None present. Adjourned to 12.05.2021.</p>")

    run_case_ingest(root=tmp_path / "corpus", db_path=db)

    rep = list_representations(db_path=db, exclude_dropped=False)[0]
    assert rep.classification_status == CLASSIFICATION_STATUS_DROPPED_PROCEDURAL
    assert rep.drop_reason == REASON_PROCEDURAL_ADJOURNMENT


def test_rerunning_is_idempotent(mixed_corpus, db, settings):
    first = run_case_ingest(root=mixed_corpus, db_path=db)
    before = {
        r.doc_id: (r.classification_status, r.drop_reason, r.cleaned_text)
        for r in list_representations(db_path=db, exclude_dropped=False)
    }

    second = run_case_ingest(root=mixed_corpus, db_path=db)
    after = {
        r.doc_id: (r.classification_status, r.drop_reason, r.cleaned_text)
        for r in list_representations(db_path=db, exclude_dropped=False)
    }

    assert before == after
    assert sorted(first.dropped_doc_ids) == sorted(second.dropped_doc_ids)
    with connection_scope(db) as conn:
        assert conn.execute(
            "SELECT COUNT(*) c FROM document_representations"
        ).fetchone()["c"] == 4  # no duplicates


def test_rerun_does_not_reset_a_document_a_later_phase_decided(mixed_corpus, db, settings):
    """A re-scan must not wipe an accepted or reviewed classification."""

    run_case_ingest(root=mixed_corpus, db_path=db)
    target = next(r for r in list_representations(db_path=db) if r.source_relpath == "case_001")

    with connection_scope(db) as conn:
        conn.execute(
            "UPDATE document_representations SET classification_status = ?, "
            "primary_domain = ?, domain_confidence = ? WHERE doc_id = ?",
            (CLASSIFICATION_STATUS_AUTO_ACCEPTED, "criminal_law", 0.91, target.doc_id),
        )

    run_case_ingest(root=mixed_corpus, db_path=db)

    after = get_representation(target.doc_id, db_path=db)
    assert after.classification_status == CLASSIFICATION_STATUS_AUTO_ACCEPTED
    assert after.primary_domain == "criminal_law"
    assert after.domain_confidence == pytest.approx(0.91)


def test_drops_are_reported_by_reason(mixed_corpus, db, settings):
    result = run_case_ingest(root=mixed_corpus, db_path=db)

    assert result.drops_by_reason == {
        REASON_PROCEDURAL_ADJOURNMENT: 1,
        REASON_CAUSE_LIST: 1,
    }


def test_court_does_not_change_whether_a_document_is_kept(tmp_path, db, settings):
    """Same text, four courts -- identical structural outcome."""

    root = tmp_path / "corpus"
    for i, court in enumerate(
        ["Supreme Court of Pakistan", "Lahore High Court", "Sindh High Court", "Peshawar High Court"]
    ):
        _write_case(root, f"case_{i}", f"<p>{_judgment()}</p>", {"court": court})

    run_case_ingest(root=root, db_path=db)

    stored = list_representations(db_path=db, exclude_dropped=False)
    assert len(stored) == 4
    assert {r.classification_status for r in stored} == {CLASSIFICATION_STATUS_PENDING}
    assert len({r.court for r in stored}) == 4  # courts differ, verdicts do not
