"""Focused tests for the OpenAI structuring backend
(src.rag_prep.structurer_llm_openai).

No live network access: openai.OpenAI is replaced with a fake object that
records the request kwargs and returns/raises pre-configured responses.
Covers exactly what the task calls for -- API response parsing, the
structured-output schema, paragraph-label mapping through the shared
pipeline, malformed/missing fields handled safely, source text unchanged,
paragraph ordering preserved, and an OpenAI failure not corrupting output
-- without touching the existing Qwen-path tests in test_structurer.py.
"""

from __future__ import annotations

import json

import openai
import pytest

from src.common.exceptions import RetryExhaustedError
from src.rag_prep.structure_types import SEMANTIC_LABELS
from src.rag_prep.structurer import structure_document
from src.rag_prep.structurer_llm_openai import (
    _RESPONSE_JSON_SCHEMA,
    OpenAIStructuringClient,
    OpenAIStructuringConfig,
    to_rag_structuring_llm_config,
)


def _make_openai_exc(cls, message="boom"):
    """Builds a real instance of an openai.* exception class without going
    through its actual __init__ (which requires a response/body payload we
    have no reason to fabricate) -- isinstance() still sees the real type,
    which is all call_with_retry's retryable-exception check needs."""

    exc = cls.__new__(cls)
    Exception.__init__(exc, message)
    return exc


class _FakeMessage:
    def __init__(self, content):
        self.content = content


class _FakeChoice:
    def __init__(self, content):
        self.message = _FakeMessage(content)


class _FakeUsage:
    def __init__(self, prompt_tokens, completion_tokens):
        self.prompt_tokens = prompt_tokens
        self.completion_tokens = completion_tokens


class _FakeCompletionResponse:
    def __init__(self, content, prompt_tokens=10, completion_tokens=5, with_usage=True):
        self.choices = [_FakeChoice(content)]
        self.usage = _FakeUsage(prompt_tokens, completion_tokens) if with_usage else None


class _FakeCompletions:
    def __init__(self, responses):
        self._responses = list(responses)
        self.calls: list[dict] = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        item = self._responses.pop(0)
        if isinstance(item, BaseException):
            raise item
        return item


class _FakeChatNamespace:
    def __init__(self, completions):
        self.completions = completions


class _FakeOpenAI:
    """Stand-in for openai.OpenAI(api_key=...)."""

    def __init__(self, responses):
        self.chat = _FakeChatNamespace(_FakeCompletions(responses))

    @property
    def calls(self):
        return self.chat.completions.calls


def _client_with(monkeypatch, responses, api_key="sk-test") -> tuple[OpenAIStructuringClient, _FakeOpenAI]:
    fake = _FakeOpenAI(responses)
    monkeypatch.setattr(openai, "OpenAI", lambda api_key: fake)
    client = OpenAIStructuringClient(api_key=api_key)
    return client, fake


VALID_CONTENT = json.dumps({
    "spans": [{"start": 0, "end": 1, "label": "facts", "confidence": 0.9}],
})


# -- config / schema shape ---------------------------------------------------

def test_response_schema_label_enum_matches_semantic_labels():
    enum = _RESPONSE_JSON_SCHEMA["schema"]["properties"]["spans"]["items"]["properties"]["label"]["enum"]
    assert enum == list(SEMANTIC_LABELS)


def test_response_schema_is_strict_and_rejects_extra_fields():
    assert _RESPONSE_JSON_SCHEMA["strict"] is True
    assert _RESPONSE_JSON_SCHEMA["schema"]["additionalProperties"] is False
    assert _RESPONSE_JSON_SCHEMA["schema"]["properties"]["spans"]["items"]["additionalProperties"] is False


def test_to_rag_structuring_llm_config_maps_batching_fields_only():
    cfg = OpenAIStructuringConfig(
        max_paragraphs_per_batch=37, max_chars_per_batch=9999, max_chars_per_paragraph_shown=500,
    )
    mapped = to_rag_structuring_llm_config(cfg)
    assert mapped.max_paragraphs_per_batch == 37
    assert mapped.max_chars_per_batch == 9999
    assert mapped.max_chars_per_paragraph_shown == 500


# -- credential handling ------------------------------------------------------

def test_missing_api_key_raises_before_any_client_is_built(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="OPENAI_API_KEY"):
        OpenAIStructuringClient(api_key=None)


# -- request/response mechanics ------------------------------------------------

def test_complete_parses_response_content_and_tracks_usage(monkeypatch):
    client, fake = _client_with(monkeypatch, [_FakeCompletionResponse(VALID_CONTENT, 123, 45)])

    result = client.complete(system="sys", prompt="prompt text")

    assert result == VALID_CONTENT
    assert client.call_count == 1
    assert client.failed_call_count == 0
    assert client.total_prompt_tokens == 123
    assert client.total_completion_tokens == 45

    sent = fake.calls[0]
    assert sent["model"] == client.cfg.model
    assert sent["response_format"]["json_schema"] is _RESPONSE_JSON_SCHEMA
    assert sent["messages"] == [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "prompt text"},
    ]


def test_complete_handles_missing_usage_without_crashing(monkeypatch):
    client, _ = _client_with(monkeypatch, [_FakeCompletionResponse(VALID_CONTENT, with_usage=False)])

    result = client.complete(system="sys", prompt="prompt")

    assert result == VALID_CONTENT
    assert client.total_prompt_tokens == 0
    assert client.total_completion_tokens == 0


def test_complete_raises_on_null_content_and_records_failure_not_success(monkeypatch):
    client, _ = _client_with(monkeypatch, [_FakeCompletionResponse(None)])

    with pytest.raises(ValueError, match="no message content"):
        client.complete(system="sys", prompt="prompt")

    assert client.call_count == 0
    assert client.failed_call_count == 1


# -- retry / failure behaviour -------------------------------------------------

def test_transient_error_is_retried_then_succeeds(monkeypatch):
    responses = [
        _make_openai_exc(openai.RateLimitError, "rate limited"),
        _FakeCompletionResponse(VALID_CONTENT),
    ]
    client, fake = _client_with(monkeypatch, responses)
    monkeypatch.setattr("time.sleep", lambda _seconds: None)

    result = client.complete(system="sys", prompt="prompt")

    assert result == VALID_CONTENT
    assert len(fake.calls) == 2
    assert client.call_count == 1
    assert client.failed_call_count == 0


def test_persistent_transient_error_exhausts_retries_and_is_recorded_as_failed(monkeypatch):
    responses = [_make_openai_exc(openai.InternalServerError, "down")] * 3
    client, fake = _client_with(monkeypatch, responses)
    monkeypatch.setattr("time.sleep", lambda _seconds: None)

    with pytest.raises(RetryExhaustedError):
        client.complete(system="sys", prompt="prompt")

    assert len(fake.calls) == 3
    assert client.call_count == 0
    assert client.failed_call_count == 1


def test_non_retryable_error_raises_immediately_without_retry(monkeypatch):
    client, fake = _client_with(monkeypatch, [_make_openai_exc(openai.AuthenticationError, "bad key")])

    with pytest.raises(openai.AuthenticationError):
        client.complete(system="sys", prompt="prompt")

    assert len(fake.calls) == 1  # no retry attempted for a non-retryable error
    assert client.failed_call_count == 1


# -- end-to-end through the shared pipeline (structurer.structure_document) --

_FULL_TEXT = (
    "JUDGMENT\n\n"
    "1. This is the first body paragraph stating the facts of the case.\n\n"
    "2. This is a second body paragraph continuing the discussion with no "
    "disposition wording of its own.\n\n"
    "3. The petition is allowed."
)


def _doc():
    return {
        "doc_id": "fake-openai-pilot-doc",
        "metadata": {"disposition": "allowed"},
        "full_text": _FULL_TEXT,
    }


def test_pipeline_labels_paragraphs_preserves_text_and_ordering(monkeypatch):
    # Body is paragraph [1,1] only: paragraph 3 ("the petition is allowed")
    # anchors a genuine disposition match, and paragraph 2 -- sitting in
    # the same 2-paragraph tail scan with no disposition wording of its
    # own -- is swallowed alongside it as an anchored "unclassified" final
    # order (see structurer_deterministic.detect_final_orders), leaving
    # only paragraph 1 for the LLM. The fake response covers index 1.
    content = json.dumps({"spans": [{"start": 1, "end": 1, "label": "facts", "confidence": 0.95}]})
    client, fake = _client_with(monkeypatch, [_FakeCompletionResponse(content)])
    llm_cfg = to_rag_structuring_llm_config(OpenAIStructuringConfig())

    doc = _doc()
    result = structure_document(doc, llm_client=client, llm_cfg=llm_cfg)

    assert result.structure_status == "structured"
    assert result.validation["ok"] is True
    assert result.used_llm is True
    # Source text is never touched -- structure_document only ever adds
    # an additive "structure" key (see to_output_document); the input
    # dict's own full_text is untouched here directly.
    assert doc["full_text"] == _FULL_TEXT
    # Paragraph ordering preserved: sorted spans have strictly increasing,
    # contiguous, non-overlapping paragraph ranges (validate_spans already
    # enforces this; asserting it again here ties it to this backend).
    ordered = sorted(result.spans, key=lambda s: s.paragraph_start)
    for prev, nxt in zip(ordered, ordered[1:]):
        assert nxt.paragraph_start == prev.paragraph_end + 1
    assert len(fake.calls) == 1


def test_openai_call_failure_falls_back_without_corrupting_output(monkeypatch):
    # Every attempt for the sole batch fails -> label_body_paragraphs
    # reports it as a failed batch -> structurer.py must fall that one
    # batch back to per-paragraph "unclassified" rather than raise or
    # fabricate a label, and the document must still validate.
    responses = [_make_openai_exc(openai.APIConnectionError, "down")] * 3
    client, _ = _client_with(monkeypatch, responses)
    monkeypatch.setattr("time.sleep", lambda _seconds: None)
    llm_cfg = to_rag_structuring_llm_config(OpenAIStructuringConfig())

    doc = _doc()
    result = structure_document(doc, llm_client=client, llm_cfg=llm_cfg)

    assert result.validation["ok"] is True
    assert result.llm_fallback_reason is not None
    assert doc["full_text"] == _FULL_TEXT
    labels = {s.label for s in result.spans}
    assert "unclassified" in labels
