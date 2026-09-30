"""Layer 2: Qwen semantic labeling of judgment-body paragraph ranges.

CRITICAL DESIGN RULE, enforced structurally, not just by instruction: the
model is asked ONLY for ``{paragraph_start, paragraph_end, label,
confidence}`` entries. It is never shown the word "rewrite", never asked to
reproduce text, and its response is never used as a text source --
:func:`src.rag_prep.structurer_deterministic.make_span` always assembles the
actual span text by slicing the deterministic paragraph list. If the model
ever DID echo text back, this module doesn't have a code path that would
use it.

Separate configuration, deliberately not sharing anything with
classification's: :class:`RagStructuringLLMConfig` is a local dataclass
with its own hardcoded defaults, not read from ``config/base.yaml`` or
``src.common.config.get_settings()``. The only thing reused from the
classification path is :class:`src.common.llm_client.OllamaLLMClient`
itself -- a generic Ollama transport wrapper, not classification-specific
-- constructed here with its own, independent parameters. Nothing in
``domain_signals.*`` config is read, and nothing here can change what a
classification run does.

Batching: a long judgment body (the audit found documents with 260+ body
paragraphs) cannot be sent in one call -- both the prompt and the expected
JSON response would be unreasonably large. Paragraphs are grouped into
batches bounded by both paragraph count and character budget, each batch
labeled independently, and results merged in paragraph order. A label
response is validated (in-range, non-overlapping, JSON-parseable) before
being trusted; anything that fails validation for a batch falls back to
per-paragraph ``unclassified`` spans for exactly that batch, never for the
whole document -- see :mod:`structurer` for how a fallback at the batch
level still lets the rest of the document structure normally.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

from src.common.llm_client import LLMClient, OllamaLLMClient, extract_json_object
from src.rag_prep.structure_types import SEMANTIC_LABELS


@dataclass(frozen=True)
class RagStructuringLLMConfig:
    """Independent of classification's DomainSignalSettings. Hardcoded
    defaults here rather than in config/base.yaml, per the task's explicit
    "create a separate RAG-structuring configuration" instruction."""

    model: str = "qwen3:14b"
    temperature: float = 0.0
    seed: int = 42
    think: bool = False
    max_tokens: int = 2048
    max_paragraphs_per_batch: int = 14
    max_chars_per_batch: int = 6000
    max_chars_per_paragraph_shown: int = 900  # prompt-display cap only;
    # never affects the text used in the final span (always sliced from
    # the real paragraph list by structurer_deterministic.make_span).


_SYSTEM_PROMPT = (
    "You are labeling the legal FUNCTION of existing paragraphs from a Pakistani "
    "family-law court judgment. You are given a numbered list of paragraphs. "
    "For each paragraph or short contiguous run of paragraphs, assign exactly one "
    "label from this fixed list: " + ", ".join(SEMANTIC_LABELS) + ". "
    "Use \"quoted_material\" when a paragraph quotes another court's order, a "
    "statute, or a precedent verbatim rather than stating the deciding court's own "
    "reasoning. Use \"unclassified\" whenever a paragraph mixes several functions "
    "and cannot be safely assigned one label, or whenever you are not confident. "
    "Do NOT invent a new label. Do NOT reproduce, summarize, or rewrite any "
    "paragraph's text -- respond with paragraph index ranges only. Every paragraph "
    "index in the input must appear in exactly one output range: no gaps, no "
    "overlaps, no paragraph left unlabeled. "
    'Respond with ONLY a JSON object: {"spans": [{"start": <int>, "end": <int>, '
    '"label": <one of the fixed labels>, "confidence": <0-1>}, ...]}.'
)


def _build_prompt(paragraphs: list[str], indices: list[int], quotation_hints: dict[int, int],
                   cfg: RagStructuringLLMConfig) -> str:
    lines = [f"Paragraphs {indices[0]}-{indices[-1]}:"]
    for idx in indices:
        text = paragraphs[idx]
        if len(text) > cfg.max_chars_per_paragraph_shown:
            text = text[: cfg.max_chars_per_paragraph_shown] + " [...]"
        hint = ""
        if idx in quotation_hints:
            hint = f" (contains {quotation_hints[idx]} list-style markers -- possibly quoted material)"
        lines.append(f"[{idx}]{hint} {text}")
    return "\n\n".join(lines)


def _batches(indices: list[int], paragraphs: list[str], cfg: RagStructuringLLMConfig):
    batch: list[int] = []
    chars = 0
    for idx in indices:
        plen = len(paragraphs[idx])
        if batch and (len(batch) >= cfg.max_paragraphs_per_batch or chars + plen > cfg.max_chars_per_batch):
            yield batch
            batch, chars = [], 0
        batch.append(idx)
        chars += plen
    if batch:
        yield batch


def _validate_and_parse(raw: str, expected_indices: set[int]) -> list[dict] | None:
    """Returns a list of {"start","end","label","confidence"} covering
    exactly ``expected_indices`` with no gaps/overlaps, or ``None`` if the
    response fails any check (malformed JSON, bad label, out-of-range
    index, overlap, or incomplete coverage)."""

    try:
        parsed = extract_json_object(raw)
    except Exception:
        return None

    spans = parsed.get("spans")
    if not isinstance(spans, list) or not spans:
        return None

    covered: set[int] = set()
    result = []
    for entry in spans:
        if not isinstance(entry, dict):
            return None
        start, end, label = entry.get("start"), entry.get("end"), entry.get("label")
        if not isinstance(start, int) or not isinstance(end, int) or end < start:
            return None
        if label not in SEMANTIC_LABELS:
            return None
        span_indices = set(range(start, end + 1))
        if not span_indices.issubset(expected_indices):
            return None  # out-of-range for this batch
        if span_indices & covered:
            return None  # overlap
        covered |= span_indices
        confidence = entry.get("confidence")
        if not isinstance(confidence, (int, float)):
            confidence = 0.7
        result.append({"start": start, "end": end, "label": label, "confidence": float(confidence)})

    if covered != expected_indices:
        return None  # gap: not every paragraph in this batch got a label

    return result


def label_body_paragraphs(
    paragraphs: list[str],
    body_start: int,
    body_end: int,
    quotation_hints: dict[int, int],
    llm_client: LLMClient | None = None,
    cfg: RagStructuringLLMConfig | None = None,
) -> tuple[list[dict], list[tuple[int, int]]]:
    """Labels paragraphs ``[body_start, body_end]`` inclusive.

    Returns ``(labeled_spans, failed_batches)``. ``labeled_spans`` covers
    every successfully-labeled paragraph; ``failed_batches`` lists the
    ``(start, end)`` paragraph ranges whose LLM call failed, was malformed,
    or didn't validate -- the caller (structurer.py) is responsible for
    falling back those specific ranges to per-paragraph ``unclassified``
    spans, never for dropping them.
    """

    cfg = cfg or RagStructuringLLMConfig()
    if body_end < body_start:
        return [], []

    client = llm_client or OllamaLLMClient(
        model=cfg.model, max_tokens=cfg.max_tokens, temperature=cfg.temperature,
        seed=cfg.seed, think=cfg.think,
    )

    indices = list(range(body_start, body_end + 1))
    labeled: list[dict] = []
    failed: list[tuple[int, int]] = []

    for batch in _batches(indices, paragraphs, cfg):
        prompt = _build_prompt(paragraphs, batch, quotation_hints, cfg)
        try:
            raw = client.complete(system=_SYSTEM_PROMPT, prompt=prompt)
        except Exception:
            failed.append((batch[0], batch[-1]))
            continue

        parsed = _validate_and_parse(raw, set(batch))
        if parsed is None:
            failed.append((batch[0], batch[-1]))
            continue

        labeled.extend(parsed)

    return labeled, failed
