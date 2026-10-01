"""Focused tests for the Gemini structuring backend
(src.rag_prep.structurer_llm_gemini).

No live network access: google.genai.Client is replaced with a fake object
that records the request kwargs and returns/raises pre-configured
responses. Mirrors test_structurer_llm_openai.py's coverage -- response
parsing, the structured-output schema, paragraph-label mapping through the
shared pipeline, usage tracking, missing/invalid API key, transient vs.
non-retryable failure handling, and an API failure not corrupting output
-- without touching the existing Qwen or OpenAI backend tests.
"""

from __future__ import annotations

import json
import time

import httpx
import pytest
from google.genai import errors as genai_errors

from src.common.exceptions import RetryExhaustedError
from src.rag_prep.structure_types import SEMANTIC_LABELS
from src.rag_prep.structurer import structure_document
from src.rag_prep.structurer_llm_gemini import (
    _MIN_SECONDS_BETWEEN_REQUESTS,
    _RESPONSE_JSON_SCHEMA,
    GeminiStructuringClient,
    GeminiStructuringConfig,
    _hard_timeout,
    _HardCallTimeout,
    _retry_delay_seconds,
    to_rag_structuring_llm_config,
)
import src.rag_prep.structurer_llm_gemini as gemini_module


def _client_error(code, message="error"):
    return genai_errors.ClientError(code, {"error": {"message": message}})


def _server_error(code=503, message="unavailable"):
    return genai_errors.ServerError(code, {"error": {"message": message}})


class _FakeUsageMetadata:
    def __init__(self, prompt_token_count, candidates_token_count):
        self.prompt_token_count = prompt_token_count
        self.candidates_token_count = candidates_token_count


class _FakeGenerateContentResponse:
    def __init__(self, text, prompt_tokens=10, completion_tokens=5, with_usage=True):
        self._text = text
        self.usage_metadata = (
            _FakeUsageMetadata(prompt_tokens, completion_tokens) if with_usage else None
        )

    @property
    def text(self):
        return self._text


class _FakeModels:
    def __init__(self, responses):
        self._responses = list(responses)
        self.calls: list[dict] = []

    def generate_content(self, **kwargs):
        self.calls.append(kwargs)
        item = self._responses.pop(0)
        if isinstance(item, BaseException):
            raise item
        return item


class _FakeGenaiClient:
    def __init__(self, responses):
        self.models = _FakeModels(responses)

    @property
    def calls(self):
        return self.models.calls


def _client_with(monkeypatch, responses, api_key="fake-key") -> tuple[GeminiStructuringClient, _FakeGenaiClient]:
    fake = _FakeGenaiClient(responses)
    monkeypatch.setattr(gemini_module.genai, "Client", lambda api_key, http_options=None: fake)
    client = GeminiStructuringClient(api_key=api_key)
    return client, fake


VALID_CONTENT = json.dumps({
    "spans": [{"start": 0, "end": 1, "label": "facts", "confidence": 0.9}],
})


# -- config / schema shape ---------------------------------------------------

def test_response_schema_label_enum_matches_semantic_labels():
    enum = _RESPONSE_JSON_SCHEMA["properties"]["spans"]["items"]["properties"]["label"]["enum"]
    assert enum == list(SEMANTIC_LABELS)


def test_response_schema_requires_all_span_fields():
    required = _RESPONSE_JSON_SCHEMA["properties"]["spans"]["items"]["required"]
    assert set(required) == {"start", "end", "label", "confidence"}
    assert _RESPONSE_JSON_SCHEMA["required"] == ["spans"]


def test_to_rag_structuring_llm_config_maps_batching_fields_only():
    cfg = GeminiStructuringConfig(
        max_paragraphs_per_batch=37, max_chars_per_batch=9999, max_chars_per_paragraph_shown=500,
    )
    mapped = to_rag_structuring_llm_config(cfg)
    assert mapped.max_paragraphs_per_batch == 37
    assert mapped.max_chars_per_batch == 9999
    assert mapped.max_chars_per_paragraph_shown == 500


def test_default_batch_target_matches_openai_pilot_not_qwen():
    cfg = GeminiStructuringConfig()
    assert cfg.max_paragraphs_per_batch == 40
    assert cfg.max_chars_per_batch == 12000


# -- credential handling ------------------------------------------------------

def test_missing_api_key_raises_before_any_client_is_built(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="GEMINI_API_KEY"):
        GeminiStructuringClient(api_key=None)


# -- request/response mechanics ------------------------------------------------

def test_complete_parses_response_text_and_tracks_usage(monkeypatch):
    client, fake = _client_with(monkeypatch, [_FakeGenerateContentResponse(VALID_CONTENT, 123, 45)])

    result = client.complete(system="sys", prompt="prompt text")

    assert result == VALID_CONTENT
    assert client.call_count == 1
    assert client.failed_call_count == 0
    assert client.total_prompt_tokens == 123
    assert client.total_completion_tokens == 45

    sent = fake.calls[0]
    assert sent["model"] == client.cfg.model
    assert sent["contents"] == "prompt text"
    assert sent["config"].system_instruction == "sys"
    assert sent["config"].response_mime_type == "application/json"
    assert sent["config"].response_json_schema is _RESPONSE_JSON_SCHEMA


def test_complete_handles_missing_usage_without_crashing(monkeypatch):
    client, _ = _client_with(monkeypatch, [_FakeGenerateContentResponse(VALID_CONTENT, with_usage=False)])

    result = client.complete(system="sys", prompt="prompt")

    assert result == VALID_CONTENT
    assert client.total_prompt_tokens == 0
    assert client.total_completion_tokens == 0


def test_complete_raises_on_null_text_and_records_failure_not_success(monkeypatch):
    client, _ = _client_with(monkeypatch, [_FakeGenerateContentResponse(None)])

    with pytest.raises(ValueError, match="no text content"):
        client.complete(system="sys", prompt="prompt")

    assert client.call_count == 0
    assert client.failed_call_count == 1


# -- retry / failure behaviour -------------------------------------------------

def test_server_error_is_retried_then_succeeds(monkeypatch):
    responses = [_server_error(), _FakeGenerateContentResponse(VALID_CONTENT)]
    client, fake = _client_with(monkeypatch, responses)
    monkeypatch.setattr("time.sleep", lambda _seconds: None)

    result = client.complete(system="sys", prompt="prompt")

    assert result == VALID_CONTENT
    assert len(fake.calls) == 2
    assert client.call_count == 1
    assert client.failed_call_count == 0


def test_rate_limit_client_error_is_retried_then_succeeds(monkeypatch):
    responses = [_client_error(429, "rate limited"), _FakeGenerateContentResponse(VALID_CONTENT)]
    client, fake = _client_with(monkeypatch, responses)
    monkeypatch.setattr("time.sleep", lambda _seconds: None)

    result = client.complete(system="sys", prompt="prompt")

    assert result == VALID_CONTENT
    assert len(fake.calls) == 2


def test_network_timeout_is_retried_then_succeeds(monkeypatch):
    responses = [httpx.TimeoutException("timed out"), _FakeGenerateContentResponse(VALID_CONTENT)]
    client, fake = _client_with(monkeypatch, responses)
    monkeypatch.setattr("time.sleep", lambda _seconds: None)

    result = client.complete(system="sys", prompt="prompt")

    assert result == VALID_CONTENT
    assert len(fake.calls) == 2


def test_persistent_server_error_exhausts_retries_and_is_recorded_as_failed(monkeypatch):
    responses = [_server_error()] * 3
    client, fake = _client_with(monkeypatch, responses)
    monkeypatch.setattr("time.sleep", lambda _seconds: None)

    with pytest.raises(RetryExhaustedError):
        client.complete(system="sys", prompt="prompt")

    assert len(fake.calls) == 3
    assert client.call_count == 0
    assert client.failed_call_count == 1


def test_authentication_error_raises_immediately_without_retry(monkeypatch):
    client, fake = _client_with(monkeypatch, [_client_error(401, "invalid api key")])

    with pytest.raises(genai_errors.ClientError):
        client.complete(system="sys", prompt="prompt")

    assert len(fake.calls) == 1  # no retry for a non-429 client error
    assert client.failed_call_count == 1


def test_bad_request_error_raises_immediately_without_retry(monkeypatch):
    client, fake = _client_with(monkeypatch, [_client_error(400, "malformed request")])

    with pytest.raises(genai_errors.ClientError):
        client.complete(system="sys", prompt="prompt")

    assert len(fake.calls) == 1


# -- server-provided retryDelay + request pacing -------------------------------

def test_retry_delay_seconds_parses_retry_info_from_429_body():
    body = {
        "error": {
            "code": 429, "status": "RESOURCE_EXHAUSTED",
            "details": [{"@type": "type.googleapis.com/google.rpc.RetryInfo", "retryDelay": "7s"}],
        },
    }
    exc = genai_errors.ClientError(429, body)
    assert _retry_delay_seconds(exc) == 7.0


def test_retry_delay_seconds_returns_none_when_absent():
    exc = genai_errors.ClientError(429, {"error": {"message": "rate limited"}})
    assert _retry_delay_seconds(exc) is None


def test_resource_exhausted_429_honors_server_retry_delay_over_default_backoff(monkeypatch):
    body = {
        "error": {
            "code": 429, "status": "RESOURCE_EXHAUSTED",
            "details": [{"@type": "type.googleapis.com/google.rpc.RetryInfo", "retryDelay": "12s"}],
        },
    }
    responses = [genai_errors.ClientError(429, body), _FakeGenerateContentResponse(VALID_CONTENT)]
    client, fake = _client_with(monkeypatch, responses)
    sleeps: list[float] = []
    monkeypatch.setattr(gemini_module.time, "sleep", sleeps.append)

    result = client.complete(system="sys", prompt="prompt")

    assert result == VALID_CONTENT
    assert len(fake.calls) == 2
    # The default backoff policy (base_delay_seconds=1.0, jittered) could
    # never produce ~12s on its own -- this value can only have come from
    # the server's own RetryInfo.retryDelay being honored.
    assert any(11.5 <= s <= 12.5 for s in sleeps), sleeps


def test_pacing_enforces_minimum_interval_between_request_attempts(monkeypatch):
    client, _ = _client_with(monkeypatch, [_FakeGenerateContentResponse(VALID_CONTENT)])
    fake_times = iter([100.0, 100.1, 100.1])
    monkeypatch.setattr(gemini_module.time, "perf_counter", lambda: next(fake_times))
    sleeps: list[float] = []
    monkeypatch.setattr(gemini_module.time, "sleep", sleeps.append)

    client._pace()  # first call: nothing to pace against yet
    client._pace()  # second call: only 0.1s has "elapsed" since the first

    assert len(sleeps) == 1
    assert sleeps[0] == pytest.approx(_MIN_SECONDS_BETWEEN_REQUESTS - 0.1, abs=1e-6)


# -- hard watchdog (defense against a hang below the HTTP layer) --------------

def test_hard_timeout_raises_when_the_block_runs_too_long():
    with pytest.raises(_HardCallTimeout):
        with _hard_timeout(1):
            time.sleep(3)


def test_hard_timeout_does_not_raise_when_the_block_finishes_in_time():
    with _hard_timeout(2):
        time.sleep(0.01)
    # No exception -- reaching here is the assertion.


def test_hard_call_timeout_is_treated_as_a_retryable_failure(monkeypatch):
    # A generate_content() call that blocks past the watchdog must be
    # retried exactly like a network timeout, not left to hang the run --
    # this is the actual bug this watchdog was added to fix (observed: a
    # 3KB/21-paragraph document hung for 45+ minutes even with an HTTP
    # read-timeout configured).
    monkeypatch.setattr("src.rag_prep.structurer_llm_gemini._HARD_CALL_TIMEOUT_SECONDS", 1)

    class _SlowThenFastModels:
        def __init__(self):
            self.calls = 0

        def generate_content(self, **kwargs):
            self.calls += 1
            if self.calls == 1:
                time.sleep(3)  # exceeds the 1s watchdog patched in above
            return _FakeGenerateContentResponse(VALID_CONTENT)

    class _SlowThenFastClient:
        def __init__(self):
            self.models = _SlowThenFastModels()

    fake = _SlowThenFastClient()
    monkeypatch.setattr(gemini_module.genai, "Client", lambda api_key, http_options=None: fake)
    monkeypatch.setattr("time.sleep", lambda _seconds: None)
    client = GeminiStructuringClient(api_key="fake-key")
    # Patching time.sleep globally above would also defeat the watchdog's
    # own timer (SIGALRM is wall-clock, not affected by mocking time.sleep)
    # -- restore a real sleep just for the attempt that must actually block.
    monkeypatch.undo()
    monkeypatch.setattr("src.rag_prep.structurer_llm_gemini._HARD_CALL_TIMEOUT_SECONDS", 1)
    monkeypatch.setattr(gemini_module.genai, "Client", lambda api_key, http_options=None: fake)
    monkeypatch.setattr(gemini_module, "_BACKOFF_POLICY", gemini_module._BACKOFF_POLICY)
    client = GeminiStructuringClient(api_key="fake-key")

    result = client.complete(system="sys", prompt="prompt")

    assert result == VALID_CONTENT
    assert fake.models.calls == 2
    assert client.call_count == 1
    assert client.failed_call_count == 0


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
        "doc_id": "fake-gemini-pilot-doc",
        "metadata": {"disposition": "allowed"},
        "full_text": _FULL_TEXT,
    }


def test_pipeline_labels_paragraphs_preserves_text_and_ordering(monkeypatch):
    # Same tail-window reasoning as the OpenAI backend's equivalent test:
    # paragraph 3 anchors a genuine disposition match, so paragraph 2 (no
    # disposition wording of its own, same 2-paragraph tail scan) is
    # swallowed alongside it, leaving only paragraph 1 for the LLM.
    content = json.dumps({"spans": [{"start": 1, "end": 1, "label": "facts", "confidence": 0.95}]})
    client, fake = _client_with(monkeypatch, [_FakeGenerateContentResponse(content)])
    llm_cfg = to_rag_structuring_llm_config(GeminiStructuringConfig())

    doc = _doc()
    result = structure_document(doc, llm_client=client, llm_cfg=llm_cfg)

    assert result.structure_status == "structured"
    assert result.validation["ok"] is True
    assert result.used_llm is True
    assert doc["full_text"] == _FULL_TEXT
    ordered = sorted(result.spans, key=lambda s: s.paragraph_start)
    for prev, nxt in zip(ordered, ordered[1:]):
        assert nxt.paragraph_start == prev.paragraph_end + 1
    assert len(fake.calls) == 1


def test_gemini_call_failure_falls_back_without_corrupting_output(monkeypatch):
    # Every attempt for the sole batch fails -> label_body_paragraphs
    # reports it as a failed batch -> structurer.py must fall that one
    # batch back to per-paragraph "unclassified" rather than raise or
    # fabricate a label, and the document must still validate.
    responses = [_server_error()] * 3
    client, _ = _client_with(monkeypatch, responses)
    monkeypatch.setattr("time.sleep", lambda _seconds: None)
    llm_cfg = to_rag_structuring_llm_config(GeminiStructuringConfig())

    doc = _doc()
    result = structure_document(doc, llm_client=client, llm_cfg=llm_cfg)

    assert result.validation["ok"] is True
    assert result.llm_fallback_reason is not None
    assert doc["full_text"] == _FULL_TEXT
    labels = {s.label for s in result.spans}
    assert "unclassified" in labels
