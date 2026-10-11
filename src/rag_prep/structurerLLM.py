"""Layer 2: Qwen semantic labeling of judgment-body paragraph ranges (v2 -- robust).

CRITICAL DESIGN RULE (unchanged): the model is asked ONLY for
``{start, end, label, confidence}`` entries. Its response is never used as a
text source -- :func:`src.rag_prep.structurer_deterministic.make_span` always
assembles span text by slicing the deterministic paragraph list.

WHY v2
------
v1 rejected a whole batch on the first imperfection (one label spelled
"Court Reasoning", an index sent as the string "15", indices renumbered from 0,
one paragraph skipped, two ranges touching) and silently fell back to
``unclassified`` without recording why. On the audited sample, 11 of 12
documents lost ALL body labels this way.

v2 keeps the same public function and return shape, and:
  1. records WHY every batch failed (exception / bad JSON / bad label / gap ...)
     plus the first 500 chars of the raw response -> logs + optional diagnostics list;
  2. repairs harmless formatting problems instead of rejecting:
       * strips <think>...</think>, accepts a top-level list or {"labels": [...]}
       * "15" -> 15, 15.0 -> 15
       * label normalisation: case, spaces/hyphens, known synonyms
         ("reasoning" -> court_reasoning, "holding" -> findings ...) -- only to
         labels that exist in SEMANTIC_LABELS
       * 0-based / 1-based renumbered indices -> shifted back to absolute indices
       * overlaps -> first range wins
  3. accepts a PARTIAL answer: valid ranges are kept, only the paragraphs the
     model missed become ``unclassified`` (not the whole batch);
  4. retries: one retry with the validation error fed back, then splits the
     batch in half (down to single paragraphs);
  5. asks Ollama for JSON mode when the client supports it (``format="json"``).

A paragraph that still has no label after all of this is reported in
``failed_batches`` exactly as before, so structurer.py needs no change.
"""

from __future__ import annotations

import inspect
import json
import logging
import re
from dataclasses import dataclass

from src.common.llm_client import LLMClient, OllamaLLMClient, extract_json_object
from src.rag_prep.structure_types import SEMANTIC_LABELS

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class RagStructuringLLMConfig:
    """Independent of classification's DomainSignalSettings."""

    model: str = "qwen3:14b"
    temperature: float = 0.0
    seed: int = 42
    think: bool = False
    max_tokens: int = 2048
    max_paragraphs_per_batch: int = 8          # was 14 -- smaller batches fail far less
    max_chars_per_batch: int = 5000
    max_chars_per_paragraph_shown: int = 900   # prompt-display cap only
    max_retries: int = 1                       # retry with error feedback before splitting
    min_split_size: int = 1                    # split failed batches down to this size
    accept_partial: bool = True                # keep valid ranges, gap paragraphs -> unclassified
    circuit_breaker_after: int = 3             # stop calling the LLM for this document after N hard failures


_SYSTEM_PROMPT = (
    "You are labeling the legal FUNCTION of existing paragraphs from a Pakistani "
    "family-law court judgment. You are given a numbered list of paragraphs; the number in "
    "square brackets is the paragraph's ID. "
    "For each paragraph or short contiguous run of paragraphs, assign exactly one "
    "label from this fixed list: " + ", ".join(SEMANTIC_LABELS) + ". "
    "Use \"quoted_material\" when a paragraph quotes another court's order, a "
    "statute, or a precedent verbatim rather than stating the deciding court's own "
    "reasoning. Use \"unclassified\" whenever a paragraph mixes several functions "
    "and cannot be safely assigned one label, or whenever you are not confident. "
    "Do NOT invent a new label. Do NOT reproduce, summarize, or rewrite any "
    "paragraph's text -- respond with paragraph ID ranges only. Use the exact IDs shown "
    "in square brackets (do not renumber from 0 or 1). Every ID must appear in exactly one "
    "range: no gaps, no overlaps. "
    'Respond with ONLY a JSON object: {"spans": [{"start": <int>, "end": <int>, '
    '"label": <one of the fixed labels>, "confidence": <0-1>}, ...]}.'
)


# -- label normalisation ----------------------------------------------------------------
_LABELS = list(SEMANTIC_LABELS)
_LABEL_SET = set(_LABELS)
_SYNONYMS = {
    "reasoning": "court_reasoning", "analysis": "court_reasoning", "discussion": "court_reasoning",
    "ratio": "court_reasoning", "ratio_decidendi": "court_reasoning", "courts_reasoning": "court_reasoning",
    "holding": "findings", "finding": "findings", "conclusion": "findings", "decision": "findings",
    "fact": "facts", "background": "facts", "facts_of_the_case": "facts", "brief_facts": "facts",
    "argument": "arguments", "submissions": "arguments", "contentions": "arguments",
    "petitioner_arguments": "arguments", "respondent_arguments": "arguments",
    "issue": "issues", "questions": "issues", "points_for_determination": "issues",
    "law": "applicable_law", "statute": "applicable_law", "legal_provisions": "applicable_law",
    "history": "procedural_history", "procedure": "procedural_history", "procedural": "procedural_history",
    "quote": "quoted_material", "quotation": "quoted_material", "quoted": "quoted_material",
    "precedent": "quoted_material", "order": "final_order", "operative_order": "final_order",
    "disposition": "final_order", "unknown": "unclassified", "other": "unclassified", "none": "unclassified",
}


def _norm_label(raw) -> str | None:
    if not isinstance(raw, str):
        return None
    key = re.sub(r"[^a-z]+", "_", raw.strip().lower()).strip("_")
    if key in _LABEL_SET:
        return key
    if key.endswith("s") and key[:-1] in _LABEL_SET:
        return key[:-1]
    if key + "s" in _LABEL_SET:
        return key + "s"
    mapped = _SYNONYMS.get(key)
    return mapped if mapped in _LABEL_SET else None


def _to_int(v) -> int | None:
    if isinstance(v, bool):
        return None
    if isinstance(v, int):
        return v
    if isinstance(v, float) and v.is_integer():
        return int(v)
    if isinstance(v, str) and re.fullmatch(r"\s*\[?\s*\d+\s*\]?\s*", v):
        return int(re.sub(r"\D", "", v))
    return None


# -- prompt / batching ---------------------------------------------------------------------
def _build_prompt(paragraphs: list[str], indices: list[int], quotation_hints: dict[int, int],
                  cfg: RagStructuringLLMConfig, feedback: str | None = None) -> str:
    lines = [f"Paragraph IDs {indices[0]}-{indices[-1]} ({len(indices)} paragraphs):"]
    for idx in indices:
        text = paragraphs[idx]
        if len(text) > cfg.max_chars_per_paragraph_shown:
            text = text[: cfg.max_chars_per_paragraph_shown] + " [...]"
        hint = ""
        if idx in quotation_hints:
            hint = f" (contains {quotation_hints[idx]} list-style markers -- possibly quoted material)"
        lines.append(f"[{idx}]{hint} {text}")
    if feedback:
        lines.append(f"Your previous answer was rejected: {feedback}. "
                     f"Return every ID from {indices[0]} to {indices[-1]} exactly once.")
    return "\n\n".join(lines)


def _batches(indices: list[int], paragraphs: list[str], cfg: RagStructuringLLMConfig):
    batch: list[int] = []
    chars = 0
    for idx in indices:
        plen = min(len(paragraphs[idx]), cfg.max_chars_per_paragraph_shown)
        if batch and (len(batch) >= cfg.max_paragraphs_per_batch or chars + plen > cfg.max_chars_per_batch):
            yield batch
            batch, chars = [], 0
        batch.append(idx)
        chars += plen
    if batch:
        yield batch


# -- parsing --------------------------------------------------------------------------------
class _Reject(ValueError):
    pass


def _extract_entries(raw: str) -> list:
    text = re.sub(r"<think>.*?</think>", "", raw or "", flags=re.S).strip()
    parsed = None
    try:
        parsed = extract_json_object(text)
    except Exception:  # noqa: BLE001 -- fall through to our own attempts
        pass
    if parsed is None:
        m = re.search(r"\{.*\}|\[.*\]", text, re.S)
        if not m:
            raise _Reject("no JSON in response")
        try:
            parsed = json.loads(m.group(0))
        except json.JSONDecodeError as exc:
            raise _Reject(f"malformed JSON ({exc.msg})") from exc
    if isinstance(parsed, list):
        return parsed
    if isinstance(parsed, dict):
        for key in ("spans", "labels", "ranges", "paragraphs", "result"):
            if isinstance(parsed.get(key), list):
                return parsed[key]
        if {"start", "end", "label"} <= set(parsed):
            return [parsed]
    raise _Reject("JSON has no 'spans' list")


def _validate_and_parse(raw: str, batch: list[int], accept_partial: bool = True) -> tuple[list[dict], set[int], str | None]:
    """Returns (spans, missing_indices, problem). spans cover only valid entries;
    missing_indices are batch paragraphs the answer did not label validly."""

    expected = set(batch)
    entries = _extract_entries(raw)
    if not entries:
        raise _Reject("empty 'spans' list")

    cleaned = []
    problems = []
    for e in entries:
        if not isinstance(e, dict):
            problems.append("non-object entry")
            continue
        s = _to_int(e.get("start", e.get("paragraph_start", e.get("id"))))
        t = _to_int(e.get("end", e.get("paragraph_end", e.get("id", s))))
        if t is None:
            t = s
        lab = _norm_label(e.get("label"))
        if s is None or t is None or t < s:
            problems.append(f"bad range {e.get('start')!r}-{e.get('end')!r}")
            continue
        if lab is None:
            problems.append(f"unknown label {e.get('label')!r}")
            continue
        conf = e.get("confidence")
        conf = float(conf) if isinstance(conf, (int, float)) and not isinstance(conf, bool) else 0.7
        cleaned.append([s, t, lab, max(0.0, min(1.0, conf))])

    # renumbered indices? (model answered 0..n-1 or 1..n instead of absolute IDs)
    if cleaned and not any(s in expected or t in expected for s, t, *_ in cleaned):
        lo = min(s for s, *_ in cleaned)
        hi = max(t for _, t, *_ in cleaned)
        for base in (0, 1):
            if lo >= base and hi <= base + len(batch) - 1:
                shift = batch[0] - base
                cleaned = [[s + shift, t + shift, lab, c] for s, t, lab, c in cleaned]
                problems.append(f"indices renumbered from {base} (repaired)")
                break

    spans, covered = [], set()
    for s, t, lab, conf in sorted(cleaned, key=lambda x: (x[0], x[1])):
        rng = [i for i in range(s, t + 1) if i in expected and i not in covered]   # clip + first-wins
        if not rng:
            continue
        # keep only the contiguous run starting at the first free index
        run = [rng[0]]
        for i in rng[1:]:
            if i == run[-1] + 1:
                run.append(i)
            else:
                break
        spans.append({"start": run[0], "end": run[-1], "label": lab, "confidence": conf})
        covered.update(run)

    missing = expected - covered
    if missing and not accept_partial:
        raise _Reject(f"{len(missing)} paragraph(s) not labeled")
    if not spans:
        raise _Reject("; ".join(problems) or "no usable span")
    problem = "; ".join(problems) if problems else (f"{len(missing)} paragraph(s) missing" if missing else None)
    return spans, missing, problem


# -- client -------------------------------------------------------------------------------
def _complete(client: LLMClient, system: str, prompt: str) -> str:
    """Ask for JSON mode when the client's complete() accepts it."""
    try:
        params = inspect.signature(client.complete).parameters
    except (TypeError, ValueError):
        params = {}
    if "format" in params:
        return client.complete(system=system, prompt=prompt, format="json")
    if "response_format" in params:
        return client.complete(system=system, prompt=prompt, response_format="json")
    return client.complete(system=system, prompt=prompt)


class _State:
    """Per-document circuit breaker: if the LLM itself is down (timeouts, connection
    errors, empty/garbage output) splitting would only multiply failing calls."""
    def __init__(self, limit: int):
        self.limit = limit
        self.consecutive_hard_failures = 0

    @property
    def open(self) -> bool:
        return self.consecutive_hard_failures >= self.limit


def _label_batch(batch, paragraphs, quotation_hints, client, cfg, diag, state=None) -> tuple[list[dict], list[int]]:
    """Returns (spans, unlabeled_indices) for one batch, with retry + split."""
    state = state or _State(3)
    if state.open:
        return [], list(batch)
    feedback = None
    last_problem = None
    hard = False                      # transport error / no JSON at all -> do not split
    for attempt in range(cfg.max_retries + 1):
        prompt = _build_prompt(paragraphs, batch, quotation_hints, cfg, feedback)
        try:
            raw = _complete(client, _SYSTEM_PROMPT, prompt)
        except Exception as exc:  # noqa: BLE001 -- transport error: record and retry
            last_problem = f"LLM call failed: {type(exc).__name__}: {exc}"
            hard = True
            diag.append({"batch": [batch[0], batch[-1]], "attempt": attempt, "error": last_problem})
            raw = None
        if raw is not None:
            try:
                spans, missing, problem = _validate_and_parse(raw, batch, cfg.accept_partial)
                if problem:
                    diag.append({"batch": [batch[0], batch[-1]], "attempt": attempt, "repaired": problem})
                state.consecutive_hard_failures = 0
                if not missing:
                    return spans, []
                # partial: try once more only for the missing ones (via split below)
                rest = sorted(missing)
                if len(rest) < len(batch):
                    more_spans, still = _split_and_label(rest, paragraphs, quotation_hints, client, cfg, diag, state)
                    return spans + more_spans, still
                last_problem = problem
            except _Reject as exc:
                last_problem = str(exc)
                hard = last_problem.startswith(("no JSON", "malformed JSON"))
                diag.append({"batch": [batch[0], batch[-1]], "attempt": attempt, "error": last_problem,
                             "raw_head": (raw or "")[:500]})
        feedback = last_problem
    log.warning("label batch %s-%s failed after retries: %s", batch[0], batch[-1], last_problem)
    if hard:
        state.consecutive_hard_failures += 1
    elif len(batch) > cfg.min_split_size:          # model answered but wrongly -> smaller batches help
        return _split_and_label(batch, paragraphs, quotation_hints, client, cfg, diag, state)
    return [], list(batch)


def _split_and_label(indices, paragraphs, quotation_hints, client, cfg, diag, state=None):
    """Labels a (possibly non-contiguous) index list by contiguous runs, halving each run;
    _label_batch halves again on failure, down to single paragraphs."""
    runs, cur = [], []
    for i in indices:
        if cur and i != cur[-1] + 1:
            runs.append(cur)
            cur = []
        cur.append(i)
    if cur:
        runs.append(cur)
    spans, unlabeled = [], []
    for run in runs:
        parts = [run] if len(run) == 1 else [run[: len(run) // 2], run[len(run) // 2:]]
        for part in parts:
            s, u = _label_batch(part, paragraphs, quotation_hints, client, cfg, diag, state)
            spans += s
            unlabeled += u
    return spans, unlabeled


# -- public API (unchanged signature and return shape) ----------------------------------------
def label_body_paragraphs(
    paragraphs: list[str],
    body_start: int,
    body_end: int,
    quotation_hints: dict[int, int],
    llm_client: LLMClient | None = None,
    cfg: RagStructuringLLMConfig | None = None,
    diagnostics: list | None = None,
) -> tuple[list[dict], list[tuple[int, int]]]:
    """Labels paragraphs ``[body_start, body_end]`` inclusive.

    Returns ``(labeled_spans, failed_batches)`` exactly as v1 did. ``failed_batches``
    now lists only the contiguous ranges that are STILL unlabeled after repair,
    retry and splitting. Pass a list as ``diagnostics`` to receive one record per
    repaired / rejected attempt (store it in the document's structure block).
    """

    cfg = cfg or RagStructuringLLMConfig()
    diag = diagnostics if diagnostics is not None else []
    if body_end < body_start:
        return [], []

    client = llm_client or OllamaLLMClient(
        model=cfg.model, max_tokens=cfg.max_tokens, temperature=cfg.temperature,
        seed=cfg.seed, think=cfg.think,
    )

    indices = list(range(body_start, body_end + 1))
    labeled: list[dict] = []
    unlabeled: list[int] = []
    state = _State(cfg.circuit_breaker_after)
    for batch in _batches(indices, paragraphs, cfg):
        spans, missing = _label_batch(batch, paragraphs, quotation_hints, client, cfg, diag, state)
        labeled.extend(spans)
        unlabeled.extend(missing)

    # merge adjacent same-label spans created by splitting
    labeled.sort(key=lambda s: s["start"])
    merged: list[dict] = []
    for s in labeled:
        if merged and merged[-1]["label"] == s["label"] and merged[-1]["end"] + 1 == s["start"]:
            merged[-1]["end"] = s["end"]
            merged[-1]["confidence"] = min(merged[-1]["confidence"], s["confidence"])
        else:
            merged.append(dict(s))

    failed: list[tuple[int, int]] = []
    for i in sorted(unlabeled):
        if failed and failed[-1][1] + 1 == i:
            failed[-1] = (failed[-1][0], i)
        else:
            failed.append((i, i))
    if state.open:
        diag.append({"error": "circuit breaker open: LLM unavailable or returning no JSON; "
                              "remaining batches left unclassified"})
    if failed:
        log.warning("paragraphs left unclassified: %s", failed)
    return merged, failed