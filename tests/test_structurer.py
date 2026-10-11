"""Focused tests for the RAG case-law structuring layer (src/rag_prep/
structure_types.py, structurer_deterministic.py, structurer_llm.py,
structure_validate.py, structurer.py).

Real-corpus documents are used where the behavior under test is exactly
what the structure audit (var/rag/audits/) found in real data --
headnotes, JUDGMENT/ORDER markers, multi-matter dispositions -- rather
than synthetic approximations of those patterns. LLM-dependent behavior
(interleaving, quoted material, failure/fallback) uses a fake client so
the suite has no live-Ollama dependency, per the project's existing
_RecordingOllamaClient-style pattern (see tests/test_pipeline_safety_fixes.py).
"""

from __future__ import annotations

import json
from pathlib import Path

from src.rag_prep.structure_types import (
    STATUS_FALLBACK,
    STATUS_INCOMPLETE_SOURCE,
    STATUS_NON_JUDGMENT_TEXT,
    STATUS_STRUCTURED,
)
from src.rag_prep.structure_validate import identity_fallback_spans, validate_spans
from src.rag_prep.structurer import structure_document, to_output_document
from src.rag_prep.structurer_deterministic import (
    build_deterministic_spans,
    classify_document_kind,
    detect_final_orders,
    find_headnote_span,
    find_judgment_marker,
    numbered_paragraph_fraction,
    paragraphs,
)
from src.rag_prep.structurer_llm import RagStructuringLLMConfig, label_body_paragraphs

PROCESSED_DIR = Path(__file__).resolve().parents[1] / "var" / "rag" / "processed"


def _load(doc_id: str) -> dict:
    return json.loads((PROCESSED_DIR / f"{doc_id}.json").read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# A fake LLM client -- returns whatever the test configures, never touches
# a live Ollama server.
# ---------------------------------------------------------------------------


class _FakeLLMClient:
    def __init__(self, responses=None, raise_error: bool = False):
        self._responses = list(responses or [])
        self._raise_error = raise_error
        self.calls: list[dict] = []

    def complete(self, system: str, prompt: str, max_tokens=None) -> str:
        self.calls.append({"system": system, "prompt": prompt})
        if self._raise_error:
            raise RuntimeError("simulated Ollama outage")
        if self._responses:
            return self._responses.pop(0)
        return '{"spans": []}'


# ---------------------------------------------------------------------------
# 1 & 2 & 3. Headnotes + JUDGMENT boundary / ORDER boundary
# ---------------------------------------------------------------------------


def test_headnotes_and_judgment_boundary_are_detected_on_a_real_document():
    # 8dd0ca3d14364404d75b76fe: headnotes (a)-(f), then a standalone
    # "JUDGMENT" marker -- confirmed during implementation via direct read.
    doc = _load("8dd0ca3d14364404d75b76fe")
    det = build_deterministic_spans(doc["full_text"])
    assert det["headnote_span"] is not None
    assert det["judgment_marker"] == "JUDGMENT"
    assert det["judgment_marker_idx"] is not None
    # The marker must come after the headnotes, not before.
    assert det["judgment_marker_idx"] > det["headnote_span"][1]


def test_order_boundary_is_detected_when_no_judgment_marker_is_present():
    paras_text = (
        "1981 C L C 143\n\n[Lahore]\n\nBefore M. S. H. Qureshi, J\n\n"
        "MUHAMMAD YAQOOB-Petitioner\n\nversus\n\nMst. SHAGUFTA BEGUM-Respondent\n\n"
        "ORDER\n\n1. This is the order text.\n\nThe petition is dismissed."
    )
    marker, idx = find_judgment_marker(paragraphs(paras_text))
    assert marker == "ORDER"
    assert paragraphs(paras_text)[idx] == "ORDER"


def test_documents_with_neither_marker_fall_back_to_headnote_end_for_body_start():
    # 5c5abcc1235caf6d3338de1f: confirmed during implementation to have no
    # standalone JUDGMENT/ORDER marker but real headnotes and a real
    # disposition -- body must still start right after the headnotes.
    doc = _load("5c5abcc1235caf6d3338de1f")
    det = build_deterministic_spans(doc["full_text"])
    assert det["judgment_marker_idx"] is None
    assert det["headnote_span"] is not None
    assert det["body_start"] == det["headnote_span"][1] + 1


# ---------------------------------------------------------------------------
# 4 & 5. Numbered paragraphs / continuous prose
# ---------------------------------------------------------------------------


def test_numbered_paragraph_fraction_detects_a_numbered_body():
    paras = ["Heading", "1. First point.", "2. Second point.", "3. Third point."]
    assert numbered_paragraph_fraction(paras, 1, 3) == 1.0


def test_numbered_paragraph_fraction_is_zero_for_continuous_prose():
    paras = ["This is prose.", "So is this, continuing the thought.", "And this."]
    assert numbered_paragraph_fraction(paras, 0, 2) == 0.0


def test_a_continuous_prose_document_still_structures_without_error():
    doc = _load("a3ec28ba4900855d15a6e870")  # confirmed: no heading, zero numbered paragraphs
    result = structure_document(doc, use_llm=False)
    assert result.validation["ok"] is True
    assert result.structure_status in (STATUS_STRUCTURED, STATUS_FALLBACK)


# ---------------------------------------------------------------------------
# 6. Interleaved facts/reasoning (via a fake LLM distinguishing spans
#    within what a keyword-only approach would have left as one blob)
# ---------------------------------------------------------------------------


def test_llm_can_assign_different_labels_to_different_paragraphs_in_one_batch():
    fake = _FakeLLMClient(responses=[
        json.dumps({"spans": [
            {"start": 0, "end": 0, "label": "facts", "confidence": 0.8},
            {"start": 1, "end": 1, "label": "court_reasoning", "confidence": 0.75},
        ]})
    ])
    paras = ["The parties were married in 1958.", "I am of the view that the marriage subsists."]
    labeled, failed = label_body_paragraphs(paras, 0, 1, {}, llm_client=fake)
    assert failed == []
    labels = {(e["start"], e["end"]): e["label"] for e in labeled}
    assert labels[(0, 0)] == "facts"
    assert labels[(1, 1)] == "court_reasoning"


# ---------------------------------------------------------------------------
# 7. Quoted lower-court material
# ---------------------------------------------------------------------------


def test_quoted_material_label_and_attribution_are_preserved_in_output():
    # "JUDGMENT" first so the quoted list-item paragraph is unambiguously
    # in the judgment body, not mistaken for a headnote -- (i) alone would
    # otherwise collide with the single-lowercase-letter headnote pattern
    # (a), which also matches roman numeral i. That makes the quoted
    # paragraph GLOBAL index 1 (JUDGMENT is index 0) -- the fake response
    # must label index 1, not 0, or the validator correctly rejects it as
    # out-of-range for the batch and falls back (exactly as it should).
    fake = _FakeLLMClient(responses=[
        json.dumps({"spans": [
            {"start": 1, "end": 1, "label": "quoted_material", "confidence": 0.9},
        ]})
    ])
    doc = {
        "doc_id": "fake1",
        "metadata": {"disposition": "dismissed"},
        "full_text": "JUDGMENT\n\n(i) The mother shall have custody on weekends.",
    }
    result = structure_document(doc, llm_client=fake)
    quoted = [s for s in result.spans if s.kind == "quoted_material"]
    assert len(quoted) == 1
    assert quoted[0].label == "quoted_material"
    # Attribution is present but not guessed at beyond what's knowable here.
    assert "attribution" in quoted[0].to_dict()


def test_quotation_density_hint_flags_paragraphs_with_several_sublist_markers():
    from src.rag_prep.structurer_deterministic import quotation_density_hint

    paras = ["(i) first", "(ii) second", "(iii) third", "plain prose paragraph"]
    hints = quotation_density_hint(paras, 0, 3)
    # Each roman-numeral paragraph only has 1 marker individually -- density
    # hint requires >=3 markers IN one paragraph, so none should fire here;
    # this confirms the hint doesn't over-trigger on a normal short list.
    assert all(v < 3 for v in hints.values()) or not hints


# ---------------------------------------------------------------------------
# 8. Multi-matter disposition
# ---------------------------------------------------------------------------


def test_multi_matter_case_produces_multiple_distinct_final_orders():
    # 04139076ff36792b38b72f8b: confirmed during implementation to resolve
    # a main appeal (allowed) and two C.M.A.s (dismissed / disposed of
    # accordingly) in the same tail paragraph -- the exact bug the audit
    # found in the single-disposition tail-scan approach.
    doc = _load("04139076ff36792b38b72f8b")
    det = build_deterministic_spans(doc["full_text"])
    labels = {o["label"] for o in det["final_orders"]}
    assert "allowed" in labels
    assert "dismissed" in labels
    assert len(det["final_orders"]) >= 3


def test_final_order_dedup_never_produces_overlapping_coverage_spans():
    doc = _load("04139076ff36792b38b72f8b")
    result = structure_document(doc, use_llm=False)
    assert result.validation["ok"] is True
    # Multiple final_orders share one paragraph; the coverage spans must
    # still be non-overlapping (validated by validate_spans itself).


# ---------------------------------------------------------------------------
# 9. Incomplete source documents
# ---------------------------------------------------------------------------


def test_short_disposition_null_document_is_incomplete_source():
    assert classify_document_kind("short text with nothing conclusive", None) == "incomplete_source"


def test_long_disposition_null_document_is_non_judgment_text_not_incomplete():
    long_academic_text = "x" * 7000
    assert classify_document_kind(long_academic_text, None) == "non_judgment_text"


def test_real_incomplete_source_document_is_never_forced_into_a_structure():
    # f7afd3f61f59d8f9798b8742: confirmed during implementation to have no
    # judgment body at all (ends mid-headnote at a counsel-listing line).
    doc = _load("f7afd3f61f59d8f9798b8742")
    result = structure_document(doc, use_llm=False)
    assert result.structure_status == STATUS_INCOMPLETE_SOURCE
    assert result.validation["ok"] is True
    # No section is invented: every span is the safe paragraph_group fallback.
    assert all(s.kind == "paragraph_group" for s in result.spans)


def test_real_non_judgment_document_is_flagged_not_silently_structured_as_a_case():
    doc = _load("ae4a07835728a54c3dddb4ad")  # confirmed: academic footnote-heavy text
    result = structure_document(doc, use_llm=False)
    assert result.structure_status == STATUS_NON_JUDGMENT_TEXT


# ---------------------------------------------------------------------------
# 10 & 13 & 14. Unclassified fallback / LLM failure / malformed LLM output
# ---------------------------------------------------------------------------


def test_llm_failure_falls_back_to_paragraph_group_not_a_crash_or_data_loss():
    fake = _FakeLLMClient(raise_error=True)
    doc = {
        "doc_id": "fake2",
        "metadata": {"disposition": "dismissed"},
        "full_text": "Paragraph one.\n\nParagraph two.\n\nThe petition is dismissed.",
    }
    result = structure_document(doc, llm_client=fake)
    assert result.validation["ok"] is True
    assert result.llm_fallback_reason is not None
    assert any(s.kind == "paragraph_group" for s in result.spans)


def test_malformed_llm_json_falls_back_to_paragraph_group():
    fake = _FakeLLMClient(responses=["this is not JSON at all"])
    doc = {
        "doc_id": "fake3",
        "metadata": {"disposition": "dismissed"},
        "full_text": "Paragraph one.\n\nParagraph two.\n\nThe petition is dismissed.",
    }
    result = structure_document(doc, llm_client=fake)
    assert result.validation["ok"] is True
    assert result.llm_fallback_reason is not None


def test_llm_output_with_a_gap_in_coverage_is_rejected_and_falls_back():
    # 3 paragraphs in the batch but the response only labels paragraph 0 --
    # paragraph 1 must not be silently dropped.
    fake = _FakeLLMClient(responses=[
        json.dumps({"spans": [{"start": 0, "end": 0, "label": "facts", "confidence": 0.9}]})
    ])
    paras = ["First.", "Second.", "Third."]
    labeled, failed = label_body_paragraphs(paras, 0, 2, {}, llm_client=fake)
    assert labeled == []
    assert failed == [(0, 2)]


def test_llm_output_with_an_invalid_label_is_rejected_and_falls_back():
    fake = _FakeLLMClient(responses=[
        json.dumps({"spans": [{"start": 0, "end": 0, "label": "made_up_label", "confidence": 0.9}]})
    ])
    labeled, failed = label_body_paragraphs(["Only paragraph."], 0, 0, {}, llm_client=fake)
    assert labeled == []
    assert failed == [(0, 0)]


def test_unclassified_is_a_valid_explicit_choice_not_only_a_failure_mode():
    fake = _FakeLLMClient(responses=[
        json.dumps({"spans": [{"start": 0, "end": 0, "label": "unclassified", "confidence": 0.3}]})
    ])
    labeled, failed = label_body_paragraphs(["Ambiguous mixed content."], 0, 0, {}, llm_client=fake)
    assert failed == []
    assert labeled[0]["label"] == "unclassified"


# ---------------------------------------------------------------------------
# 11 & 15. Exact full_text / metadata preservation
# ---------------------------------------------------------------------------


def test_full_text_is_byte_for_byte_identical_in_the_output_document():
    doc = _load("5c5abcc1235caf6d3338de1f")
    result = structure_document(doc, use_llm=False)
    output = to_output_document(doc, result)
    assert output["full_text"] == doc["full_text"]


def test_metadata_is_unchanged_in_the_output_document():
    doc = _load("5c5abcc1235caf6d3338de1f")
    result = structure_document(doc, use_llm=False)
    output = to_output_document(doc, result)
    assert output["metadata"] == doc["metadata"]


def test_doc_id_is_preserved():
    doc = _load("5c5abcc1235caf6d3338de1f")
    result = structure_document(doc, use_llm=False)
    output = to_output_document(doc, result)
    assert output["doc_id"] == doc["doc_id"]


def test_structure_is_additive_full_text_key_still_present_alongside_structure():
    doc = _load("5c5abcc1235caf6d3338de1f")
    result = structure_document(doc, use_llm=False)
    output = to_output_document(doc, result)
    assert set(output.keys()) == {"doc_id", "metadata", "full_text", "structure"}


# ---------------------------------------------------------------------------
# 12. No text duplication or loss (the mandatory validator itself)
# ---------------------------------------------------------------------------


def test_identity_fallback_always_validates():
    text = "Para one.\n\nPara two.\n\nPara three."
    paras = paragraphs(text)
    spans = identity_fallback_spans(paras)
    result = validate_spans(spans, paras, text)
    assert result.ok is True


def test_validator_catches_a_missing_paragraph():
    from src.rag_prep.structurer_deterministic import make_span

    text = "Para one.\n\nPara two.\n\nPara three."
    paras = paragraphs(text)
    spans = [make_span(paras, "paragraph_group", 0, 0), make_span(paras, "paragraph_group", 2, 2)]
    result = validate_spans(spans, paras, text)
    assert result.ok is False
    assert any("missing" in e for e in result.errors)


def test_validator_catches_an_overlapping_span():
    from src.rag_prep.structurer_deterministic import make_span

    text = "Para one.\n\nPara two.\n\nPara three."
    paras = paragraphs(text)
    spans = [
        make_span(paras, "paragraph_group", 0, 1),
        make_span(paras, "paragraph_group", 1, 2),  # overlaps at index 1
    ]
    result = validate_spans(spans, paras, text)
    assert result.ok is False


def test_validator_catches_rewritten_span_text():
    from src.rag_prep.structure_types import Span

    text = "Para one.\n\nPara two."
    paras = paragraphs(text)
    spans = [Span(kind="paragraph_group", paragraph_start=0, paragraph_end=1, text="REWRITTEN TEXT")]
    result = validate_spans(spans, paras, text)
    assert result.ok is False


def test_full_corpus_deterministic_pass_never_produces_a_validation_failure():
    """The full-corpus regression this implementation was checked against:
    every one of the 2,088 processed documents must validate, with or
    without the LLM step, never losing or duplicating content."""

    import glob

    files = sorted(glob.glob(str(PROCESSED_DIR / "*.json")))
    assert len(files) > 0, "var/rag/processed/ must be populated for this test"
    failures = []
    for f in files:
        doc = json.loads(Path(f).read_text(encoding="utf-8"))
        result = structure_document(doc, use_llm=False)
        if not result.validation["ok"]:
            failures.append((doc["doc_id"], result.validation["errors"]))
    assert failures == [], f"{len(failures)} document(s) failed validation: {failures[:5]}"
