"""Dedicated deterministic audit for statute chunk sets -- the statute
pipeline's equivalent of src.rag_prep.chunk_validate (case-law), built
fully independently (no shared code, no import of it).

:func:`validate_statute_chunk_set` never trusts anything about how the
chunks were produced -- it re-slices every chunk's text directly from
``parent.cleaned_text`` and checks it against the stored ``text``.

:func:`is_valid_existing_output` is the resumability gate: an existing
output file is only treated as "done" if this comes back True, never on
file existence alone. It also catches an interrupted/partial write
(atomic .json.tmp + os.replace() in statute_chunker.py already prevents
a truncated final file, but this function is the semantic completeness
check on top of that).
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field

from src.rag_prep.statute_chunker import StatuteParent, _base_section_number

ALLOWED_CHUNK_TYPES = frozenset({"preamble", "section", "definitions", "schedule", "footnotes"})

# The two literal markers statute_cleaning.py (Stage 1) wraps relocated
# footnotes in. If either shows up inside a non-"footnotes" chunk, the
# footnotes block was swallowed into a substantive chunk instead of
# being carved out into its own -- a hard failure, not a warning, since
# there is no legitimate reason this literal string ever appears in a
# section/definitions/preamble/schedule chunk's own text.
_FOOTNOTE_MARKERS = ("[FOOTNOTES]", "[/FOOTNOTES]")


@dataclass
class StatuteValidationResult:
    ok: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    diagnostics: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {"ok": self.ok, "errors": self.errors, "warnings": self.warnings, "diagnostics": self.diagnostics}


def _valid_reference_schema(references: dict) -> list[str]:
    errors = []
    if not isinstance(references, dict) or set(references.keys()) != {"internal_sections", "external_statutes"}:
        return ["references has an unexpected top-level shape"]
    for item in references.get("internal_sections", []):
        if not isinstance(item.get("section_id"), str) or not isinstance(item.get("source_text"), str):
            errors.append(f"malformed internal_sections entry: {item!r}")
    for item in references.get("external_statutes", []):
        if not isinstance(item.get("name"), str) or not isinstance(item.get("year"), int) \
                or not isinstance(item.get("source_text"), str):
            errors.append(f"malformed external_statutes entry: {item!r}")
    return errors


def validate_statute_chunk_set(parent: StatuteParent) -> StatuteValidationResult:
    errors: list[str] = []
    warnings: list[str] = []
    chunks = parent.chunks
    cleaned_text = parent.cleaned_text

    if not chunks:
        return StatuteValidationResult(ok=False, errors=["no chunks produced"])

    seen_ids: set[str] = set()
    section_id_counts: Counter = Counter()
    schedule_id_counts: Counter = Counter()
    definitions_count = 0
    footnotes_count = 0
    chunk_type_counts: Counter = Counter()
    section_ids_in_order: list[str] = []

    for c in chunks:
        if c.parent_id != parent.doc_id:
            errors.append(f"chunk {c.chunk_id}: parent_id {c.parent_id!r} != parent.doc_id {parent.doc_id!r}")
        if c.doc_id != parent.doc_id:
            errors.append(f"chunk {c.chunk_id}: doc_id {c.doc_id!r} != parent.doc_id {parent.doc_id!r}")
        if c.chunk_type not in ALLOWED_CHUNK_TYPES:
            errors.append(f"chunk {c.chunk_id}: unrecognized chunk_type {c.chunk_type!r}")
        if c.chunk_id in seen_ids:
            errors.append(f"duplicate chunk_id {c.chunk_id!r}")
        seen_ids.add(c.chunk_id)
        chunk_type_counts[c.chunk_type] += 1

        expected_id = f"{parent.doc_id}:{c.chunk_index:04d}"
        if c.chunk_id != expected_id:
            errors.append(f"chunk_id {c.chunk_id!r} != expected deterministic id {expected_id!r}")

        if not (0 <= c.source_start < c.source_end <= len(cleaned_text)):
            errors.append(f"chunk {c.chunk_id}: out-of-bounds offsets [{c.source_start},{c.source_end})")
            continue
        if cleaned_text[c.source_start:c.source_end] != c.text:
            errors.append(f"chunk {c.chunk_id}: text does not match cleaned_text[source_start:source_end]")

        if not isinstance(c.embedding_text, str) or not c.embedding_text:
            errors.append(f"chunk {c.chunk_id}: embedding_text is missing or not a string")

        # -- Check 3: footnote contamination -- a relocated footnotes
        # block must live only in its own dedicated "footnotes" chunk.
        if c.chunk_type != "footnotes":
            for marker in _FOOTNOTE_MARKERS:
                if marker in c.text:
                    errors.append(
                        f"chunk {c.chunk_id} ({c.chunk_type}): contains relocated-footnotes "
                        f"marker {marker!r} -- footnotes leaked into a substantive chunk"
                    )

        if c.chunk_type in ("section", "definitions"):
            if not c.section_id:
                errors.append(f"chunk {c.chunk_id}: {c.chunk_type} chunk is missing section_id")
            else:
                section_id_counts[c.section_id] += 1
                section_ids_in_order.append(c.section_id)
            if c.chunk_type == "definitions":
                definitions_count += 1
        if c.chunk_type == "schedule":
            if not c.schedule_id:
                errors.append(f"chunk {c.chunk_id}: schedule chunk is missing schedule_id")
            else:
                schedule_id_counts[c.schedule_id] += 1
        if c.chunk_type == "footnotes":
            footnotes_count += 1

        # -- Check 4: an external-statute citation's section number must
        # never also appear as one of THIS chunk's internal_sections.
        external_names = {e.get("name", "").lower() for e in c.references.get("external_statutes", [])}
        if external_names:
            for item in c.references.get("internal_sections", []):
                source_text = item.get("source_text", "")
                if any(name and name in source_text.lower() for name in external_names):
                    errors.append(
                        f"chunk {c.chunk_id}: internal_sections entry {item!r} appears to reference "
                        f"an external statute also found in this chunk's external_statutes"
                    )

        errors.extend(f"chunk {c.chunk_id}: {e}" for e in _valid_reference_schema(c.references))

    if footnotes_count > 1:
        errors.append(f"more than one footnotes chunk found ({footnotes_count})")

    # -- Check 2: section completeness (non-blocking) -- a gap of more
    # than 1 between consecutive DISTINCT base section numbers is worth
    # surfacing for manual review, since it is the shape a parser-
    # swallowed section would take. This is deliberately a warning, not
    # a hard failure: a real statute's numbering is not guaranteed to be
    # perfectly continuous (see module docstring / Check 2 discussion),
    # and an omitted/repealed section still produces its own chunk (with
    # body text like "[Omitted]"), so it is never mistaken for a gap.
    distinct_bases = sorted({_base_section_number(sid) for sid in section_ids_in_order})
    for prev, nxt in zip(distinct_bases, distinct_bases[1:]):
        if nxt - prev > 1:
            warnings.append(
                f"possible swallowed section(s): base section numbers jump from {prev} to {nxt} "
                f"with nothing in between"
            )

    # -- ordering + exact, gapless, non-overlapping coverage of the WHOLE
    # cleaned_text -- statute chunking (unlike case-law's) has no
    # context-only exclusions, so every character must be covered by
    # exactly one chunk.
    ordered = sorted(chunks, key=lambda c: c.chunk_index)
    cursor = 0
    for c in ordered:
        if c.source_start != cursor:
            if c.source_start > cursor:
                errors.append(f"gap in coverage before chunk {c.chunk_id}: "
                               f"[{cursor},{c.source_start}) is not covered by any chunk")
            else:
                errors.append(f"chunk {c.chunk_id} overlaps the previous chunk "
                               f"(starts at {c.source_start}, previous ended at {cursor})")
        cursor = max(cursor, c.source_end)
    if cursor != len(cleaned_text):
        errors.append(f"coverage ends at {cursor}, but cleaned_text has length {len(cleaned_text)} "
                       f"-- trailing content is not covered by any chunk")

    chunk_indexes = [c.chunk_index for c in ordered]
    if chunk_indexes != list(range(len(chunks))):
        errors.append(f"chunk_index values are not sequential starting at 0: {chunk_indexes}")

    duplicated_sections = {sid: n for sid, n in section_id_counts.items() if n > 1}
    if duplicated_sections:
        errors.append(f"section_id(s) represented by more than one chunk: {duplicated_sections}")
    duplicated_schedules = {sid: n for sid, n in schedule_id_counts.items() if n > 1}
    if duplicated_schedules:
        errors.append(f"schedule_id(s) represented by more than one chunk: {duplicated_schedules}")
    if definitions_count > 1:
        errors.append(f"more than one definitions chunk found ({definitions_count})")

    diagnostics = {
        "total_chunks": len(chunks),
        "chunk_type_counts": dict(chunk_type_counts),
        "section_count": len(section_id_counts),
        "schedule_count": len(schedule_id_counts),
        "has_definitions_chunk": definitions_count == 1,
        "has_preamble_chunk": parent.preamble_present,
        "has_footnotes_chunk": footnotes_count == 1,
    }

    return StatuteValidationResult(ok=not errors, errors=errors, warnings=warnings, diagnostics=diagnostics)


def is_valid_existing_output(doc_dict: dict) -> bool:
    """Resumability gate for a future runner: is this existing chunked-
    statute output file actually valid, not merely present. Returns
    False (never raises) for anything unreadable/malformed."""

    try:
        parent_dict = doc_dict["parent"]
        parent = StatuteParent.from_dict(parent_dict)
    except (KeyError, TypeError, ValueError):
        return False
    try:
        result = validate_statute_chunk_set(parent)
    except Exception:  # noqa: BLE001 -- a malformed file must never crash the skip-check
        return False
    return result.ok
