"""OpenAI backend for Layer 2 semantic labeling -- an alternative to the
Qwen/Ollama backend in :mod:`structurer_llm`, for the same task.

Deliberately the smallest possible addition: :func:`label_body_paragraphs`
in ``structurer_llm.py`` already takes its LLM client via dependency
injection (``llm_client=...``) and only requires an object with a
``complete(system, prompt, max_tokens=None) -> str`` method -- the exact
:class:`src.common.llm_client.LLMClient` Protocol shape. This module adds
ONE class satisfying that interface, backed by the OpenAI API instead of
Ollama. Nothing in ``structurer_llm.py``, ``structurer.py``, or
``structure_validate.py`` is modified: the same prompt builder
(``_build_prompt``), the same response validator (``_validate_and_parse``),
the same batching logic (``_batches``), and the same system prompt
(``_SYSTEM_PROMPT``) are imported and reused verbatim, so both backends are
validated identically and asked exactly the same semantic question.

Credentials: read from the ``OPENAI_API_KEY`` environment variable, never
hardcoded. ``python-dotenv`` is already a declared (if previously unused)
project dependency (see ``requirements.txt``) -- reused here via
``load_dotenv()`` rather than inventing a second secrets mechanism, so a
local ``.env`` file works the same way it would anywhere else in the
project, without ever committing a key to the repo.

Structured Outputs: the response is constrained with
``response_format={"type": "json_schema", ..., "strict": True}`` matching
the same ``{"spans": [...]}`` shape ``_validate_and_parse`` already expects
-- schema enforcement and the existing downstream validation are
complementary, not a replacement for each other.

Retry: reuses :func:`src.common.retry.call_with_retry` (the project's
existing retry helper, already used by the Ollama client) rather than
building a second retry framework, with a short, fixed attempt count.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field

from src.common.retry import RetryPolicy, call_with_retry
from src.rag_prep.structure_types import SEMANTIC_LABELS

try:
    from dotenv import load_dotenv

    load_dotenv()  # no-op if no .env file is present; never overrides an
    # already-exported OPENAI_API_KEY, so an operator's shell env still wins.
except ImportError:  # pragma: no cover -- python-dotenv is a declared
    # project dependency (requirements.txt); this guard only matters if
    # someone runs this module in an environment where it isn't installed.
    pass

import openai

_RESPONSE_JSON_SCHEMA = {
    "name": "paragraph_span_labels",
    "strict": True,
    "schema": {
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
                    "additionalProperties": False,
                },
            },
        },
        "required": ["spans"],
        "additionalProperties": False,
    },
}


@dataclass(frozen=True)
class OpenAIStructuringConfig:
    """Mirrors the fields of RagStructuringLLMConfig that actually matter
    once an explicit llm_client is supplied to label_body_paragraphs() --
    the Ollama-specific fields there (model/temperature/seed/think used
    only to build a DEFAULT client) are simply unused on this path."""

    model: str = "gpt-5.4-mini"
    temperature: float = 0.0
    seed: int = 42
    # Larger than Qwen's 14/6000, per the task's instruction to try
    # 30-50 paragraphs where practical, bounded by a char cap so a run of
    # unusually long paragraphs still can't blow the batch out.
    max_paragraphs_per_batch: int = 40
    max_chars_per_batch: int = 12000
    max_chars_per_paragraph_shown: int = 900
    max_output_tokens: int = 4096


@dataclass
class OpenAIStructuringClient:
    """Satisfies the LLMClient Protocol (.complete(system, prompt) -> str).

    Tracks its own usage as instance attributes (call_count,
    total_prompt_tokens, total_completion_tokens, total_latency_seconds)
    rather than changing the shared .complete() return type -- this keeps
    the interface identical to OllamaLLMClient's, so structurer.py and
    structurer_llm.py needed zero changes to accept this client.
    """

    cfg: OpenAIStructuringConfig = field(default_factory=OpenAIStructuringConfig)
    api_key: str | None = None

    call_count: int = field(default=0, init=False)
    failed_call_count: int = field(default=0, init=False)
    total_prompt_tokens: int = field(default=0, init=False)
    total_completion_tokens: int = field(default=0, init=False)
    total_latency_seconds: float = field(default=0.0, init=False)

    def __post_init__(self) -> None:
        key = self.api_key or os.environ.get("OPENAI_API_KEY")
        if not key:
            raise RuntimeError(
                "OPENAI_API_KEY is not set. Export it in your shell or place it in "
                "a .env file at the project root (never commit that file) before "
                "running the OpenAI structuring pilot."
            )
        self._client = openai.OpenAI(api_key=key)

    def complete(self, system: str, prompt: str, max_tokens: int | None = None) -> str:
        def _call() -> str:
            t0 = time.perf_counter()
            response = self._client.chat.completions.create(
                model=self.cfg.model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": prompt},
                ],
                response_format={"type": "json_schema", "json_schema": _RESPONSE_JSON_SCHEMA},
                temperature=self.cfg.temperature,
                seed=self.cfg.seed,
                max_completion_tokens=max_tokens or self.cfg.max_output_tokens,
            )
            self.total_latency_seconds += time.perf_counter() - t0
            if response.usage is not None:
                self.total_prompt_tokens += response.usage.prompt_tokens or 0
                self.total_completion_tokens += response.usage.completion_tokens or 0
            content = response.choices[0].message.content
            if content is None:
                raise ValueError("OpenAI response had no message content")
            return content

        try:
            result = call_with_retry(
                _call,
                policy=RetryPolicy(max_attempts=3, base_delay_seconds=1.0, max_delay_seconds=15.0),
                retryable_exceptions=(
                    openai.APIConnectionError,
                    openai.APITimeoutError,
                    openai.RateLimitError,
                    openai.InternalServerError,
                ),
                operation_name="openai_structuring_complete",
            )
            self.call_count += 1
            return result
        except Exception:
            self.failed_call_count += 1
            raise

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


def to_rag_structuring_llm_config(cfg: OpenAIStructuringConfig):
    """Builds the RagStructuringLLMConfig that label_body_paragraphs()
    actually reads batching parameters from (model/temperature/seed/think
    on that dataclass are irrelevant here since a client is always passed
    explicitly, never constructed from it)."""

    from src.rag_prep.structurer_llm import RagStructuringLLMConfig

    return RagStructuringLLMConfig(
        max_paragraphs_per_batch=cfg.max_paragraphs_per_batch,
        max_chars_per_batch=cfg.max_chars_per_batch,
        max_chars_per_paragraph_shown=cfg.max_chars_per_paragraph_shown,
    )
