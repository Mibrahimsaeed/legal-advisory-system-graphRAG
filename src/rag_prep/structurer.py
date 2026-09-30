"""Orchestrator: ties Layers 1 (deterministic), 2 (Qwen), 3 (span assembly),
and 4 (no-content-loss validation) into one call per document.

    processed case (var/rag/processed/<doc_id>.json)
        -> classify_document_kind()          [incomplete_source / non_judgment_text short-circuit]
        -> build_deterministic_spans()       [Layer 1]
        -> label_body_paragraphs()           [Layer 2, Qwen -- only for the body range]
        -> assemble spans                    [Layer 3 -- always slices from `paras`, never LLM text]
        -> validate_spans()                  [Layer 4 -- mandatory]
        -> StructureResult (always content-complete, by construction or by fallback)

The priority order from the task brief is enforced structurally here, not
just by convention: validation failure (of any kind) always degrades to
the identity paragraph-group fallback rather than emitting a result this
module knows to be wrong. A document is never written with spans that
failed :func:`structure_validate.validate_spans`.
"""

from __future__ import annotations

from src.rag_prep.structure_types import (
    STATUS_FALLBACK,
    STATUS_INCOMPLETE_SOURCE,
    STATUS_NON_JUDGMENT_TEXT,
    STATUS_STRUCTURED,
    Span,
    StructureResult,
)
from src.rag_prep.structure_validate import identity_fallback_spans, validate_spans
from src.rag_prep.structurer_deterministic import (
    build_deterministic_spans,
    classify_document_kind,
    make_span,
    numbered_paragraph_fraction,
)
from src.rag_prep.structurer_llm import RagStructuringLLMConfig, label_body_paragraphs


def _final_order_spans(paras: list[str], final_orders: list[dict]) -> list[Span]:
    """One coverage span per distinct paragraph range in final_orders
    (deduplicated -- several disposition sentences can share one
    paragraph, e.g. a multi-matter tail; that richness is preserved in
    StructureResult.final_orders, not by overlapping Span objects here)."""

    seen: set[tuple[int, int]] = set()
    spans = []
    for order in final_orders:
        key = (order["paragraph_start"], order["paragraph_end"])
        if key in seen:
            continue
        seen.add(key)
        spans.append(make_span(paras, "final_order", key[0], key[1], label="final_order"))
    return spans


def structure_document(
    doc: dict,
    llm_client=None,
    llm_cfg: RagStructuringLLMConfig | None = None,
    use_llm: bool = True,
) -> StructureResult:
    """Structures one ``{doc_id, metadata, full_text}`` document.

    ``use_llm=False`` runs Layer 1 + validation only, skipping Qwen
    entirely (paragraph-group fallback for the whole judgment body) --
    used by tests that must not depend on a live Ollama server, and
    available as an operator override.
    """

    full_text = doc["full_text"]
    disposition = doc.get("metadata", {}).get("disposition")

    kind_override = classify_document_kind(full_text, disposition)
    if kind_override is not None:
        paras = full_text.split("\n\n")
        spans = identity_fallback_spans(paras)
        validation = validate_spans(spans, paras, full_text)
        status = STATUS_INCOMPLETE_SOURCE if kind_override == "incomplete_source" else STATUS_NON_JUDGMENT_TEXT
        return StructureResult(
            structure_status=status,
            spans=spans,
            validation=validation.to_dict(),
        )

    det = build_deterministic_spans(full_text)
    paras = det["paras"]

    spans: list[Span] = []

    # -- headnotes + marker first, so their indices are known; then fill
    #    every remaining gap in [0, body_start) with case_caption spans.
    #    A single "one caption span before headnotes" assumption is not
    #    enough: real documents have content AFTER headnotes but BEFORE
    #    the JUDGMENT/ORDER marker too (counsel listings, "Date of
    #    hearing:" -- confirmed directly against 8dd0ca3d14364404d75b76fe,
    #    paragraphs 22-24, which sit exactly in that gap).
    headnotes_detected = det["headnote_span"] is not None
    claimed: set[int] = set()

    if headnotes_detected:
        spans.append(make_span(paras, "headnotes", *det["headnote_span"]))
        claimed |= set(range(det["headnote_span"][0], det["headnote_span"][1] + 1))

    if det["judgment_marker_idx"] is not None:
        spans.append(make_span(
            paras, "judgment_marker", det["judgment_marker_idx"], det["judgment_marker_idx"],
            label=det["judgment_marker"],
        ))
        claimed.add(det["judgment_marker_idx"])

    run_start = None
    for idx in range(det["body_start"]):
        if idx in claimed:
            if run_start is not None:
                spans.append(make_span(paras, "case_caption", run_start, idx - 1))
                run_start = None
        elif run_start is None:
            run_start = idx
    if run_start is not None:
        spans.append(make_span(paras, "case_caption", run_start, det["body_start"] - 1))

    # -- judgment body: Qwen semantic labeling, batched -------------------
    used_llm = False
    llm_fallback_reason = None
    body_start, body_end = det["body_start"], det["body_end"]

    if body_end >= body_start:
        if use_llm:
            labeled, failed_batches = label_body_paragraphs(
                paras, body_start, body_end, det["quotation_hints"],
                llm_client=llm_client, cfg=llm_cfg,
            )
            used_llm = True
            for entry in labeled:
                kind = "quoted_material" if entry["label"] == "quoted_material" else "semantic_section"
                spans.append(make_span(
                    paras, kind, entry["start"], entry["end"],
                    label=entry["label"], confidence=entry["confidence"],
                    attribution=None if kind != "quoted_material" else "unknown",
                ))
            if failed_batches:
                llm_fallback_reason = f"{len(failed_batches)} batch(es) fell back to paragraph_group"
                for f_start, f_end in failed_batches:
                    for idx in range(f_start, f_end + 1):
                        spans.append(make_span(paras, "paragraph_group", idx, idx, label="unclassified"))
        else:
            for idx in range(body_start, body_end + 1):
                spans.append(make_span(paras, "paragraph_group", idx, idx, label="unclassified"))

    # -- final order(s) ----------------------------------------------------
    spans.extend(_final_order_spans(paras, det["final_orders"]))

    # -- validate; fall back to the identity mapping if anything is wrong --
    validation = validate_spans(spans, paras, full_text)
    status = STATUS_STRUCTURED
    if not validation.ok:
        spans = identity_fallback_spans(paras)
        validation = validate_spans(spans, paras, full_text)
        status = STATUS_FALLBACK

    return StructureResult(
        structure_status=status,
        spans=spans,
        headnotes_detected=headnotes_detected,
        judgment_marker=det["judgment_marker"],
        final_orders=det["final_orders"],
        used_llm=used_llm,
        llm_fallback_reason=llm_fallback_reason,
        validation=validation.to_dict(),
    )


def to_output_document(doc: dict, result: StructureResult) -> dict:
    """The final var/rag/structured/<doc_id>.json shape: additive, never
    replacing full_text."""

    return {
        "doc_id": doc["doc_id"],
        "metadata": doc["metadata"],
        "full_text": doc["full_text"],
        "structure": {
            "structure_status": result.structure_status,
            "headnotes_detected": result.headnotes_detected,
            "judgment_marker": result.judgment_marker,
            "final_orders": result.final_orders,
            "used_llm": result.used_llm,
            "llm_fallback_reason": result.llm_fallback_reason,
            "spans": [s.to_dict() for s in sorted(result.spans, key=lambda s: s.paragraph_start)],
            "validation": result.validation,
        },
    }
