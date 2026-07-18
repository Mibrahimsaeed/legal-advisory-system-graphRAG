"""Shared LLM call wrapper.

Was an empty stub through Stage 1; Stage 1.2 is its first consumer
(:mod:`src.clustering.label_clusters`, generating draft domain
definitions from cluster keywords/representative documents). Later
stages that need an LLM call -- GraphRAG entity/relation extraction,
community summarization, cluster labeling during recluster runs -- are
expected to reuse this wrapper rather than each rolling their own
Anthropic client, so retry policy, error typing, and JSON-extraction
helpers stay in one place.

:class:`LLMClient` is a ``Protocol`` (not a concrete base class)
specifically so tests and other callers can inject a fake without
subclassing anything -- see ``tests/test_domain_discovery.py`` for a
``FakeLLMClient`` used to test :mod:`~src.clustering.label_clusters`
without an API key or the ``anthropic`` package installed.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Protocol

from src.common.exceptions import PipelineError
from src.common.logging_utils import get_logger
from src.common.retry import call_with_retry

logger = get_logger(__name__)

try:  # pragma: no cover
    import anthropic
except ImportError:  # pragma: no cover
    anthropic = None  # type: ignore[assignment]


class LLMError(PipelineError):
    """Raised for LLM API/parsing failures. Retryable by default -- most
    failure modes (rate limits, transient 5xxs) are worth retrying; callers
    that need to distinguish "malformed JSON response" from "API down" can
    inspect ``str(exc)``."""

    retryable = True


class LLMClient(Protocol):
    """Minimal interface every LLM client (real or fake) must satisfy."""

    def complete(self, system: str, prompt: str, max_tokens: int | None = None) -> str: ...


@dataclass
class AnthropicLLMClient:
    """Thin wrapper around the ``anthropic`` SDK's Messages API."""

    model: str
    max_tokens: int = 1024
    api_key: str | None = None

    def __post_init__(self) -> None:
        if anthropic is None:
            raise LLMError(
                "The 'anthropic' package is not installed; cannot call the LLM "
                "(see requirements.txt)",
                phase="llm",
            )
        try:
            self._client = (
                anthropic.Anthropic(api_key=self.api_key)
                if self.api_key
                else anthropic.Anthropic()
            )
        except Exception as exc:
            raise LLMError(
                "Failed to initialize Anthropic client "
                "(is ANTHROPIC_API_KEY set?)",
                phase="llm",
                cause=exc,
            ) from exc

    def complete(self, system: str, prompt: str, max_tokens: int | None = None) -> str:
        def _call() -> str:
            try:
                response = self._client.messages.create(
                    model=self.model,
                    max_tokens=max_tokens or self.max_tokens,
                    system=system,
                    messages=[{"role": "user", "content": prompt}],
                )
            except Exception as exc:
                raise LLMError(
                    "Anthropic messages.create() call failed", phase="llm", cause=exc
                ) from exc

            return "".join(
                block.text
                for block in response.content
                if getattr(block, "type", None) == "text"
            )

        return call_with_retry(_call, operation_name="llm_complete")


def get_llm_client(
    model: str, max_tokens: int = 1024, api_key: str | None = None
) -> LLMClient:
    return AnthropicLLMClient(model=model, max_tokens=max_tokens, api_key=api_key)


def extract_json_object(text: str) -> dict[str, Any]:
    """Best-effort extraction of a single JSON object from an LLM response.

    Handles the two common ways a model wraps JSON it was asked to
    return verbatim: a ```json fenced block, or a JSON object preceded/
    followed by explanatory prose. Takes the first ``{`` through the
    last ``}`` in the (defenced) text and parses that span.

    Raises:
        LLMError: if no ``{...}`` span is found, or if the span found
            isn't valid JSON.
    """

    stripped = text.strip()

    if stripped.startswith("```"):
        stripped = stripped.strip("`")
        if stripped.lower().startswith("json"):
            stripped = stripped[4:]
        stripped = stripped.strip()

    start = stripped.find("{")
    end = stripped.rfind("}")
    if start == -1 or end == -1 or end < start:
        raise LLMError(
            f"Could not find a JSON object in LLM response: {text[:200]!r}",
            phase="llm",
        )

    candidate = stripped[start : end + 1]
    try:
        return json.loads(candidate)
    except json.JSONDecodeError as exc:
        raise LLMError(
            f"Failed to parse JSON from LLM response: {exc}",
            phase="llm",
            cause=exc,
        ) from exc