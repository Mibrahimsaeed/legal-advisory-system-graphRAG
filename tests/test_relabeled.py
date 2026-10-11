"""Tests for src.rag_prep.relabeled.

Regression coverage for the TypeError that broke this module:
``label_body_paragraphs() got an unexpected keyword argument 'diagnostics'``
-- caused by importing from ``structurer_llm`` (v1, no ``diagnostics``
parameter) instead of ``structurerLLM`` (v2, "robust", same signature/
return shape plus ``diagnostics``). No real LLM/network client is used
here -- a fake, in-memory client stands in for Gemini/Ollama.
"""

from __future__ import annotations

from src.rag_prep.relabeled import relabel_document


class _FakeLLMClient:
    """Returns pre-scripted JSON responses in order; records every prompt
    it was asked to complete. Never touches the network."""

    def __init__(self, responses: list[str]):
        self._responses = list(responses)
        self.prompts: list[str] = []

    def complete(self, system: str, prompt: str, format: str | None = None) -> str:
        self.prompts.append(prompt)
        return self._responses.pop(0)


def _doc() -> dict:
    full_text = (
        "This is paragraph zero, already labeled.\n\n"
        "This is paragraph one, about the facts of the case.\n\n"
        "This is paragraph two, about the issues raised."
    )
    return {
        "doc_id": "doc1",
        "full_text": full_text,
        "structure": {
            "spans": [
                {"type": "paragraph_group", "paragraph_start": 0, "paragraph_end": 0,
                 "text": "This is paragraph zero, already labeled.", "label": "facts"},
                {"type": "paragraph_group", "paragraph_start": 1, "paragraph_end": 2,
                 "text": ("This is paragraph one, about the facts of the case.\n\n"
                           "This is paragraph two, about the issues raised."),
                 "label": "unclassified"},
            ],
        },
    }


def test_relabel_document_does_not_raise_the_diagnostics_typeerror():
    # This is the exact regression: calling relabel_document() used to
    # raise `TypeError: label_body_paragraphs() got an unexpected
    # keyword argument 'diagnostics'` before the import was corrected.
    client = _FakeLLMClient([
        '{"spans": [{"start": 1, "end": 1, "label": "facts", "confidence": 0.9}, '
        '{"start": 2, "end": 2, "label": "issues", "confidence": 0.85}]}'
    ])
    new = relabel_document(_doc(), llm_client=client)
    assert new["structure"]["relabel"]["still_unclassified"] == 0
    assert new["structure"]["relabel"]["relabeled_paragraphs"] == 2


def test_relabel_document_imports_label_body_paragraphs_from_the_v2_module():
    # Guards against the exact mistake recurring: relabeled.py must use
    # structurerLLM (v2, accepts `diagnostics`), never structurer_llm
    # (v1, does not).
    from src.rag_prep import relabeled

    assert relabeled.label_body_paragraphs.__module__ == "src.rag_prep.structurerLLM"


def test_relabel_document_only_relabels_the_unclassified_run():
    client = _FakeLLMClient([
        '{"spans": [{"start": 1, "end": 1, "label": "facts", "confidence": 0.9}, '
        '{"start": 2, "end": 2, "label": "issues", "confidence": 0.85}]}'
    ])
    doc = _doc()
    new = relabel_document(doc, llm_client=client)
    spans = new["structure"]["spans"]
    assert spans[0]["label"] == "facts"  # already-labeled span passed through unchanged
    assert spans[0]["paragraph_start"] == 0 and spans[0]["paragraph_end"] == 0
    relabeled_spans = [s for s in spans if s["paragraph_start"] in (1, 2)]
    assert {s["label"] for s in relabeled_spans} == {"facts", "issues"}
    # exactly one real LLM call was made, for the one unclassified run
    assert len(client.prompts) == 1


def test_relabel_document_keeps_paragraphs_unclassified_on_llm_failure_without_losing_text():
    client = _FakeLLMClient(["not json at all"] * 4)  # exhausts retry + split attempts
    doc = _doc()
    new = relabel_document(doc, llm_client=client)
    assert new["structure"]["relabel"]["still_unclassified"] == 2
    spans = new["structure"]["spans"]
    # every paragraph is still represented, verbatim, even though labeling failed
    covered_text = "\n\n".join(s["text"] for s in sorted(spans, key=lambda s: s["paragraph_start"]))
    assert covered_text == doc["full_text"]


def test_relabel_document_is_a_noop_when_nothing_is_unclassified():
    doc = _doc()
    doc["structure"]["spans"][1]["label"] = "issues"  # nothing left unclassified
    client = _FakeLLMClient([])  # must never be called
    new = relabel_document(doc, llm_client=client)
    assert new["structure"]["relabel"]["relabeled_paragraphs"] == 0
    assert new["structure"]["relabel"]["still_unclassified"] == 0
    assert client.prompts == []
