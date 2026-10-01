"""Dedicated deterministic chunk audit -- the chunking layer's equivalent
of :mod:`src.rag_prep.structure_validate`'s no-content-loss validator.

:func:`validate_chunk_set` is the full audit (checks A-K from the task
brief) and needs the ORIGINAL structural spans to check coverage (C) and
section consistency (H). :func:`is_valid_existing_output` is a lighter,
self-contained check -- everything EXCEPT C and H, which don't need the
original spans -- meant for a future resumable runner's "is this existing
output actually valid, not just present" gate (never treat mere file
existence as proof of validity).
"""

from __future__ import annotations

from dataclasses import dataclass, field

from src.rag_prep.chunk_types import ALLOWED_SECTIONS, CONTEXT_ONLY_SPAN_KINDS, ParentCase

_SIZE_BUCKETS = (
    ("<250", 0, 249),
    ("250-499", 250, 499),
    ("500-800", 500, 800),
    ("801-1000", 801, 1000),
    (">1000", 1001, None),
)


def _size_bucket(word_count: int) -> str:
    for label, lo, hi in _SIZE_BUCKETS:
        if hi is None and word_count >= lo:
            return label
        if hi is not None and lo <= word_count <= hi:
            return label
    return "unknown"  # unreachable given the bucket list above; kept defensive


@dataclass
class ChunkValidationResult:
    ok: bool
    errors: list[str] = field(default_factory=list)
    diagnostics: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {"ok": self.ok, "errors": self.errors, "diagnostics": self.diagnostics}


def validate_chunk_set(parent: ParentCase, spans: list[dict] | None = None) -> ChunkValidationResult:
    """Runs the full chunk audit against ``parent.chunks``.

    ``spans`` is the ORIGINAL ``structure.spans`` list the chunks were
    built from. When provided, coverage (every retrieval-eligible
    paragraph is represented by at least one chunk) and section
    consistency (a chunk's section traces back to a real span of a
    matching kind covering its paragraph range) are also checked; when
    omitted, those two checks are skipped (used by
    :func:`is_valid_existing_output`, which has no access to the
    original structured source).
    """

    from src.rag_prep.chunker import paragraph_offsets  # local import avoids a cycle

    errors: list[str] = []
    chunks = parent.chunks
    full_text = parent.full_text
    paragraphs = full_text.split("\n\n")
    n_paragraphs = len(paragraphs)
    offsets = paragraph_offsets(full_text)

    if not chunks:
        errors.append("no chunks produced")

    size_histogram: dict[str, int] = {}
    seen_ids: set[str] = set()

    for c in chunks:
        # -- A: schema validity -------------------------------------------------
        if c.parent_id != parent.doc_id:
            errors.append(f"chunk {c.chunk_id}: parent_id {c.parent_id!r} != parent.doc_id {parent.doc_id!r}")
        if c.doc_id != parent.doc_id:
            errors.append(f"chunk {c.chunk_id}: doc_id {c.doc_id!r} != parent.doc_id {parent.doc_id!r}")
        if c.section not in ALLOWED_SECTIONS:
            errors.append(f"chunk {c.chunk_id}: section {c.section!r} outside the allowed vocabulary")
        if c.chunk_id in seen_ids:
            errors.append(f"duplicate chunk_id {c.chunk_id!r}")
        seen_ids.add(c.chunk_id)
        size_histogram[_size_bucket(c.word_count)] = size_histogram.get(_size_bucket(c.word_count), 0) + 1

        # -- F: bounds ------------------------------------------------------------
        if not (0 <= c.source_start < c.source_end <= len(full_text)):
            errors.append(f"chunk {c.chunk_id}: out-of-bounds offsets [{c.source_start},{c.source_end})")
            continue  # text/provenance checks below would be meaningless

        # -- B: source integrity ---------------------------------------------------
        if c.text != full_text[c.source_start:c.source_end]:
            errors.append(f"chunk {c.chunk_id}: text does not match full_text[source_start:source_end]")

        # -- G: paragraph provenance ------------------------------------------------
        if not (0 <= c.paragraph_start <= c.paragraph_end < n_paragraphs):
            errors.append(f"chunk {c.chunk_id}: invalid paragraph range [{c.paragraph_start},{c.paragraph_end}]")
        else:
            expected_start = offsets[c.paragraph_start][0]
            expected_end = offsets[c.paragraph_end][1]
            if (c.source_start, c.source_end) != (expected_start, expected_end):
                errors.append(
                    f"chunk {c.chunk_id}: source offsets do not correspond to its own "
                    f"paragraph_start/paragraph_end"
                )

    # -- I: parent-child consistency (prev/next linkage) --------------------------
    by_id = {c.chunk_id: c for c in chunks}
    for i, c in enumerate(chunks):
        if i == 0 and c.prev_chunk_id is not None:
            errors.append(f"first chunk {c.chunk_id} has a non-null prev_chunk_id")
        if i == len(chunks) - 1 and c.next_chunk_id is not None:
            errors.append(f"last chunk {c.chunk_id} has a non-null next_chunk_id")
        if c.prev_chunk_id is not None:
            prev = by_id.get(c.prev_chunk_id)
            if prev is None or prev.next_chunk_id != c.chunk_id:
                errors.append(f"chunk {c.chunk_id}: prev linkage is not symmetric")
        if c.next_chunk_id is not None:
            nxt = by_id.get(c.next_chunk_id)
            if nxt is None or nxt.prev_chunk_id != c.chunk_id:
                errors.append(f"chunk {c.chunk_id}: next linkage is not symmetric")

    # -- D: ordering + E: duplication-vs-intentional-overlap -----------------------
    ordered = sorted(chunks, key=lambda c: c.chunk_index)
    covered: set[int] = set()
    unintended_duplicates: set[int] = set()
    prev_source_start = -1
    for c in ordered:
        if c.source_start <= prev_source_start:
            errors.append(f"chunk {c.chunk_id}: source_start does not strictly increase in chunk_index order")
        prev_source_start = c.source_start

        span_range = set(range(c.paragraph_start, c.paragraph_end + 1))
        overlap_with_prior = span_range & covered
        if overlap_with_prior and not c.is_overlap:
            unintended_duplicates |= overlap_with_prior
        covered |= span_range

    if unintended_duplicates:
        errors.append(
            f"unintended duplicated paragraph index(es) -- overlap not flagged as intentional: "
            f"{sorted(unintended_duplicates)[:10]}"
        )

    if spans is not None:
        # -- C: coverage ---------------------------------------------------------
        excluded: set[int] = set()
        for s in spans:
            if s.get("type") in CONTEXT_ONLY_SPAN_KINDS:
                excluded |= set(range(s["paragraph_start"], s["paragraph_end"] + 1))
        required = set(range(n_paragraphs)) - excluded
        missing = required - covered
        if missing:
            errors.append(f"missing paragraph index(es) -- not represented by any chunk: {sorted(missing)[:10]}")

        # -- H: section consistency ------------------------------------------------
        for c in chunks:
            matching = [
                s for s in spans
                if s.get("type") == c.span_kind
                and s["paragraph_start"] <= c.paragraph_start
                and c.paragraph_end <= s["paragraph_end"]
            ]
            if not matching:
                errors.append(
                    f"chunk {c.chunk_id}: no original span of kind {c.span_kind!r} "
                    f"covers its paragraph range [{c.paragraph_start},{c.paragraph_end}]"
                )

    diagnostics = {
        "size_histogram": size_histogram,
        "total_chunks": len(chunks),
        "overlap_chunk_count": sum(1 for c in chunks if c.is_overlap),
        "oversized_single_paragraph_count": sum(1 for c in chunks if c.oversized_single_paragraph),
        "coverage_and_section_checks_run": spans is not None,
    }

    return ChunkValidationResult(ok=not errors, errors=errors, diagnostics=diagnostics)


def is_valid_existing_output(doc_dict: dict) -> bool:
    """For a FUTURE resumable chunking runner: is this existing output
    file actually valid, not merely present. Reconstructs a ParentCase
    from a serialized chunk-output dict and re-runs every check that
    doesn't require the original structured source (coverage and
    section-consistency are skipped -- they need the original spans,
    which would mean re-reading the structured_gemini source just to
    decide whether to skip it, defeating the purpose of a skip check).

    Returns False for anything unreadable/malformed rather than raising,
    since a runner's skip-check must never itself crash the run."""

    try:
        parent = ParentCase.from_dict(doc_dict)
    except (KeyError, TypeError, ValueError):
        return False
    try:
        result = validate_chunk_set(parent, spans=None)
    except Exception:  # noqa: BLE001 -- a malformed file must never crash the skip-check
        return False
    return result.ok
