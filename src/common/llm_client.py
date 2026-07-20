from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Protocol

from ollama import Client

from src.common.exceptions import PipelineError
from src.common.logging_utils import get_logger
from src.common.retry import call_with_retry

logger = get_logger(__name__)


class LLMError(PipelineError):
    """Raised for LLM API/parsing failures."""

    retryable = True


class LLMClient(Protocol):
    """Minimal interface every LLM client must satisfy."""

    def complete(
        self,
        system: str,
        prompt: str,
        max_tokens: int | None = None,
    ) -> str:
        ...


@dataclass
class OllamaLLMClient:
    """Thin wrapper around the local Ollama server."""

    model: str
    max_tokens: int = 1024

    def __post_init__(self) -> None:
        try:
            self._client = Client()
        except Exception as exc:
            raise LLMError(
                "Failed to initialize Ollama client.",
                phase="llm",
                cause=exc,
            ) from exc

    def complete(
        self,
        system: str,
        prompt: str,
        max_tokens: int | None = None,
    ) -> str:
        def _call() -> str:
            try:
                response = self._client.chat(
                    model=self.model,
                    messages=[
                        {
                            "role": "system",
                            "content": system,
                        },
                        {
                            "role": "user",
                            "content": prompt,
                        },
                    ],
                    options={
                        "num_predict": max_tokens or self.max_tokens,
                    },
                )

                return response["message"]["content"]

            except Exception as exc:
                raise LLMError(
                    "Ollama chat() call failed.",
                    phase="llm",
                    cause=exc,
                ) from exc

        return call_with_retry(
            _call,
            operation_name="llm_complete",
        )


def get_llm_client(
    model: str,
    max_tokens: int = 1024,
    api_key: str | None = None,
) -> LLMClient:
    # api_key is ignored for Ollama but retained for compatibility.
    return OllamaLLMClient(
        model=model,
        max_tokens=max_tokens,
    )


def extract_json_object(text: str) -> dict[str, Any]:
    """Best-effort extraction of a single JSON object from an LLM response."""

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