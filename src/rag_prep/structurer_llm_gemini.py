"""Gemini backend for Layer 2 semantic labeling -- a second alternative to
the Qwen/Ollama backend in :mod:`structurer_llm`, replacing the OpenAI
pilot attempt (blocked by an exhausted OpenAI credits balance, not by any
flaw in that design -- :mod:`structurer_llm_openai` is left in place
unmodified for a later comparison).

Same architecture as the OpenAI backend: :func:`label_body_paragraphs` in
``structurer_llm.py`` already takes its LLM client via dependency
injection (``llm_client=...``) and only requires an object with a
``complete(system, prompt, max_tokens=None) -> str`` method. This module
adds ONE class satisfying that interface, backed by the Gemini API. Nothing
in ``structurer_llm.py``, ``structurer.py``, or ``structure_validate.py``
is modified: the same prompt builder (``_build_prompt``), response
validator (``_validate_and_parse``), batching logic (``_batches``), and
system prompt (``_SYSTEM_PROMPT``) are imported and reused verbatim.

SDK: the current (non-deprecated) ``google-genai`` package, not
``google-generativeai``.

Credentials: read from the ``GEMINI_API_KEY`` environment variable, never
hardcoded. Reuses the same ``load_dotenv()`` mechanism as the OpenAI
backend rather than inventing a second one.

Structured Outputs: ``response_mime_type="application/json"`` plus
``response_json_schema=...`` (Gemini's direct JSON Schema input, as
opposed to its ``response_schema`` OpenAPI-subset form) constrains the
response to the same ``{"spans": [...]}`` shape ``_validate_and_parse``
already expects -- schema enforcement and the existing downstream
validation are complementary, not a replacement for each other.

Retry: a short, fixed-attempt loop built on :class:`src.common.retry.RetryPolicy`'s
existing backoff math (not a new framework -- see :func:`_retry_delay_seconds`).
Unlike the OpenAI SDK, ``google-genai`` does not expose a dedicated
rate-limit exception class -- transient failures surface as ``ServerError``
(5xx) or as a ``ClientError`` whose ``.code == 429``. Only those are
retried; every other ``ClientError`` (401/403/400 -- bad key, bad request)
is re-raised immediately and never retried. For a 429 whose body carries a
``RetryInfo.retryDelay`` (Google's own suggested wait, e.g. "28s" for
RESOURCE_EXHAUSTED), that server-provided delay is honored instead of the
generic exponential backoff -- :func:`src.common.retry.call_with_retry`
cannot be reused as-is here because it always sleeps its own
policy-computed delay, with no way for the caller to override it per
exception, and this project's free tier can return waits (tens of
seconds) far longer than the default backoff would ever produce.

Rate limiting: the pilot's confirmed free-tier ceiling is 15 requests per
minute with one request in flight at a time (already true -- nothing in
this codebase issues concurrent Gemini calls). :meth:`GeminiStructuringClient.complete`
paces every attempt (including retries) at least ``_MIN_SECONDS_BETWEEN_REQUESTS``
apart, comfortably under that ceiling, since the quota counts each HTTP
request made, not just the successful ones.
"""

from __future__ import annotations

import contextlib
import os
import signal
import time
from dataclasses import dataclass, field

import httpx
from google import genai
from google.genai import errors as genai_errors
from google.genai import types as genai_types

from src.common.exceptions import RetryExhaustedError
from src.common.retry import RetryPolicy
from src.rag_prep.structure_types import SEMANTIC_LABELS

try:
    from dotenv import load_dotenv

    load_dotenv()  # no-op if no .env file is present; never overrides an
    # already-exported GEMINI_API_KEY, so an operator's shell env still wins.
except ImportError:  # pragma: no cover -- python-dotenv is a declared
    # project dependency (requirements.txt); this guard only matters if
    # someone runs this module in an environment where it isn't installed.
    pass

# The exact schema from the task brief -- a plain JSON Schema object, not
# wrapped in OpenAI's {"name", "schema", "strict"} envelope, since
# Gemini's response_json_schema takes the schema directly.
_RESPONSE_JSON_SCHEMA = {
    "type": "object",
    "properties": {
        "spans": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "start": {"type": "integer"},
                    "end": {"type": "integer"},
                    "label": {"type": "string", "enum": list(SEMANTIC_LABELS)},
                    "confidence": {"type": "number"},
                },
                "required": ["start", "end", "label", "confidence"],
            },
        },
    },
    "required": ["spans"],
}


class _RetryableGeminiError(Exception):
    """Marker wrapping a transient Gemini failure (5xx, HTTP 429, or a
    network-level error) -- a bad API key or a malformed request (surfacing
    as a non-429 ClientError) is never wrapped here and therefore
    propagates immediately, unretried. ``retry_delay_seconds``, when set,
    is Google's own suggested wait for a 429 RESOURCE_EXHAUSTED response
    and takes priority over the generic exponential backoff."""

    def __init__(self, message: str, retry_delay_seconds: float | None = None) -> None:
        super().__init__(message)
        self.retry_delay_seconds = retry_delay_seconds


_MAX_ATTEMPTS = 3
_BACKOFF_POLICY = RetryPolicy(max_attempts=_MAX_ATTEMPTS, base_delay_seconds=1.0, max_delay_seconds=15.0)
# Sanity cap on a server-provided retryDelay -- honoring it is the point,
# but an ill-formed or unexpectedly huge value should never hang the pilot.
_MAX_SERVER_RETRY_DELAY_SECONDS = 90.0
# Confirmed free-tier ceiling is 15 requests/minute (one at a time, which
# this codebase already guarantees by never issuing concurrent calls).
# 4.5s comfortably clears that (60/4.5 ~= 13.3/min) with margin for the
# pacing check's own overhead.
_MIN_SECONDS_BETWEEN_REQUESTS = 4.5
# Without an explicit HTTP timeout, a stalled connection can hang
# indefinitely with no exception and no log output -- observed directly
# during the full-corpus run (a call sat idle for 25+ minutes, zero CPU,
# zero response). 90s is generous for this model's normal 2-15s response
# times but still bounded, so a stalled request fails into the existing
# httpx.TimeoutException handling (already treated as retryable below)
# instead of hanging the whole run.
_HTTP_TIMEOUT_MS = 90_000
# Observed directly: even with the http_options timeout above, a single
# generate_content() call on a trivially small (3KB) document still hung
# for 45+ minutes with zero CPU usage and no exception -- ruling out "a
# genuinely large multi-batch document retrying many times" (this backend
# bounds that at _MAX_ATTEMPTS attempts) and pointing instead at something
# below the HTTP layer (DNS resolution / TCP connect) that httpx's own
# read-timeout does not reliably cover. A signal.alarm-based hard
# watchdog bounds the call regardless of WHERE it's stuck, since it
# interrupts the process on a timer rather than relying on the blocked
# call to honor any timeout itself. POSIX-only (fine: this pipeline runs
# on macOS/Linux, never Windows).
_HARD_CALL_TIMEOUT_SECONDS = 100


class _HardCallTimeout(Exception):
    """Raised by the SIGALRM handler -- treated as a retryable failure,
    identically to a network-level timeout."""


@contextlib.contextmanager
def _hard_timeout(seconds: int):
    def _handler(signum, frame):
        raise _HardCallTimeout(f"generate_content() exceeded the {seconds}s hard watchdog")

    previous = signal.signal(signal.SIGALRM, _handler)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, previous)


def _retry_delay_seconds(exc: genai_errors.ClientError) -> float | None:
    """Extracts Google's suggested wait from a 429's RetryInfo detail
    (``{"error": {"details": [{"@type": ".../RetryInfo", "retryDelay": "28s"}]}}``),
    or ``None`` if the response doesn't carry one."""

    body = exc.details if isinstance(exc.details, dict) else {}
    error_body = body.get("error", body) if isinstance(body, dict) else {}
    for item in (error_body.get("details") or []) if isinstance(error_body, dict) else []:
        if isinstance(item, dict) and str(item.get("@type", "")).endswith("RetryInfo"):
            raw = item.get("retryDelay")
            if isinstance(raw, str) and raw.endswith("s"):
                try:
                    return min(float(raw[:-1]), _MAX_SERVER_RETRY_DELAY_SECONDS)
                except ValueError:
                    return None
    return None


@dataclass(frozen=True)
class GeminiStructuringConfig:
    """Mirrors OpenAIStructuringConfig -- see that module's docstring for
    why the Ollama-shaped fields (model/temperature/seed) only matter when
    no explicit llm_client is supplied, which is never the case here."""

    model: str = "gemini-3.5-flash-lite"
    temperature: float = 0.0
    seed: int = 42
    # Same batch target as the OpenAI pilot, per the task's explicit
    # instruction not to shrink back to Qwen's 14/6000 without cause.
    max_paragraphs_per_batch: int = 40
    max_chars_per_batch: int = 12000
    max_chars_per_paragraph_shown: int = 900
    max_output_tokens: int = 4096


@dataclass
class GeminiStructuringClient:
    """Satisfies the LLMClient Protocol (.complete(system, prompt) -> str).

    Tracks its own usage as instance attributes, identically to
    OpenAIStructuringClient, so both pilots produce directly comparable
    per-document call/token/latency records.
    """

    cfg: GeminiStructuringConfig = field(default_factory=GeminiStructuringConfig)
    api_key: str | None = None

    call_count: int = field(default=0, init=False)
    failed_call_count: int = field(default=0, init=False)
    total_prompt_tokens: int = field(default=0, init=False)
    total_completion_tokens: int = field(default=0, init=False)
    total_latency_seconds: float = field(default=0.0, init=False)
    _last_request_at: float | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        key = self.api_key or os.environ.get("GEMINI_API_KEY")
        if not key:
            raise RuntimeError(
                "GEMINI_API_KEY is not set. Export it in your shell or place it in "
                "a .env file at the project root (never commit that file) before "
                "running the Gemini structuring pilot."
            )
        self._client = genai.Client(
            api_key=key,
            http_options=genai_types.HttpOptions(
                timeout=_HTTP_TIMEOUT_MS,
                # The SDK retries 408/429/5xx internally by default (up to
                # 5 attempts, its own exponential backoff) BEFORE this
                # class's own complete()-level retry loop ever sees an
                # exception -- two independent retry layers stacking on
                # top of each other with no coordination between them.
                # Disabled here so this class's retryDelay-aware loop is
                # the only retry mechanism in play, which is what it was
                # built and tested to be.
                retry_options=genai_types.HttpRetryOptions(attempts=1),
            ),
        )

    def _pace(self) -> None:
        """Blocks until at least _MIN_SECONDS_BETWEEN_REQUESTS has passed
        since the previous attempt STARTED -- applied before every attempt,
        not just successful ones, since the free-tier quota counts every
        HTTP request made."""

        if self._last_request_at is not None:
            remaining = _MIN_SECONDS_BETWEEN_REQUESTS - (time.perf_counter() - self._last_request_at)
            if remaining > 0:
                time.sleep(remaining)
        self._last_request_at = time.perf_counter()

    def _single_attempt(self, system: str, prompt: str, max_tokens: int | None) -> str:
        t0 = time.perf_counter()
        config = genai_types.GenerateContentConfig(
            system_instruction=system,
            response_mime_type="application/json",
            response_json_schema=_RESPONSE_JSON_SCHEMA,
            temperature=self.cfg.temperature,
            seed=self.cfg.seed,
            max_output_tokens=max_tokens or self.cfg.max_output_tokens,
        )
        try:
            with _hard_timeout(_HARD_CALL_TIMEOUT_SECONDS):
                response = self._client.models.generate_content(
                    model=self.cfg.model, contents=prompt, config=config,
                )
        except genai_errors.ServerError as exc:
            raise _RetryableGeminiError(str(exc)) from exc
        except genai_errors.ClientError as exc:
            if exc.code == 429:
                raise _RetryableGeminiError(str(exc), retry_delay_seconds=_retry_delay_seconds(exc)) from exc
            raise  # 401/403/400 etc. -- never retried
        except (httpx.TransportError, httpx.TimeoutException, _HardCallTimeout) as exc:
            raise _RetryableGeminiError(str(exc)) from exc

        self.total_latency_seconds += time.perf_counter() - t0
        usage = response.usage_metadata
        if usage is not None:
            self.total_prompt_tokens += usage.prompt_token_count or 0
            self.total_completion_tokens += usage.candidates_token_count or 0
        content = response.text
        if content is None:
            raise ValueError("Gemini response had no text content")
        return content

    def complete(self, system: str, prompt: str, max_tokens: int | None = None) -> str:
        last_exc: BaseException | None = None
        for attempt in range(1, _MAX_ATTEMPTS + 1):
            self._pace()
            try:
                result = self._single_attempt(system, prompt, max_tokens)
            except _RetryableGeminiError as exc:
                last_exc = exc
                if attempt >= _MAX_ATTEMPTS:
                    break
                # A 429's own RetryInfo.retryDelay -- when Google supplies
                # one -- takes priority over the generic backoff, since it
                # can legitimately be tens of seconds (far longer than the
                # default policy would ever wait) and retrying sooner would
                # just draw another RESOURCE_EXHAUSTED response.
                delay = exc.retry_delay_seconds
                if delay is None:
                    delay = _BACKOFF_POLICY.delay_for_attempt(attempt)
                time.sleep(delay)
                continue
            except Exception:
                self.failed_call_count += 1
                raise
            else:
                self.call_count += 1
                return result

        self.failed_call_count += 1
        raise RetryExhaustedError(
            f"gemini_structuring_complete failed after {_MAX_ATTEMPTS} attempt(s)",
            cause=last_exc,
        ) from last_exc

    def usage_summary(self) -> dict:
        return {
            "model": self.cfg.model,
            "call_count": self.call_count,
            "failed_call_count": self.failed_call_count,
            "total_prompt_tokens": self.total_prompt_tokens,
            "total_completion_tokens": self.total_completion_tokens,
            "total_tokens": self.total_prompt_tokens + self.total_completion_tokens,
            "total_latency_seconds": round(self.total_latency_seconds, 2),
        }


def to_rag_structuring_llm_config(cfg: GeminiStructuringConfig):
    """Builds the RagStructuringLLMConfig that label_body_paragraphs()
    actually reads batching parameters from -- see the OpenAI module's
    identically-named helper for why the rest of that dataclass is
    irrelevant here."""

    from src.rag_prep.structurer_llm import RagStructuringLLMConfig

    return RagStructuringLLMConfig(
        max_paragraphs_per_batch=cfg.max_paragraphs_per_batch,
        max_chars_per_batch=cfg.max_chars_per_batch,
        max_chars_per_paragraph_shown=cfg.max_chars_per_paragraph_shown,
    )
