"""Deterministic, structure-aware, span-aware parent-child chunking
engine for case-law documents.

    structured case (var/rag/structured_gemini/<doc_id>.json)
        -> paragraph_offsets()            [exact char offsets, length-based, never .find()]
        -> chunk_document()               [per-span paragraph grouping + conditional overlap]
        -> ParentCase(chunks=[ChildChunk, ...])

No LLM calls anywhere in this module. The structuring stage has already
decided what each paragraph IS (facts/issues/reasoning/...); this module
only decides how to GROUP already-labeled paragraphs into retrieval-sized
chunks, never re-classifying anything.

``full_text`` is authoritative throughout: every chunk's ``text`` is
always ``full_text[source_start:source_end]``, and those offsets are
always derived from paragraph LENGTHS (see :func:`paragraph_offsets`),
never from searching for a paragraph's text inside ``full_text`` -- a
judgment that quotes the same statutory language or a lower court's
order more than once would make a text-search-based offset silently
point at the wrong occurrence.
"""

from __future__ import annotations

from dataclasses import replace

from src.rag_prep.chunk_types import (
    CHUNKABLE_SPAN_KINDS,
    CONTEXT_ONLY_SPAN_KINDS,
    INHERITABLE_METADATA_FIELDS,
    OVERLAP_ELIGIBLE_SECTIONS,
    OVERLAP_TARGET_WORDS,
    SOFT_MAX_WORDS,
    ChildChunk,
    ChunkingError,
    ParentCase,
)


def paragraph_offsets(full_text: str) -> list[tuple[int, int]]:
    """Exact ``(char_start, char_end)`` range for every paragraph in
    ``full_text.split("\\n\\n")``, computed purely from paragraph
    lengths. This is the ONLY way offsets are derived anywhere in this
    module -- never by searching for a paragraph's text inside
    full_text, which would silently return the wrong location for any
    paragraph whose text recurs elsewhere in the document.

    Relies on the same round-trip property the structuring layer already
    depends on: ``"\\n\\n".join(full_text.split("\\n\\n")) == full_text``.
    """

    paragraphs = full_text.split("\n\n")
    offsets = []
    cursor = 0
    for p in paragraphs:
        start = cursor
        end = start + len(p)
        offsets.append((start, end))
        cursor = end + 2  # the "\n\n" separator
    return offsets


def _word_count(text: str) -> int:
    return len(text.split())


def _section_for_span(span: dict) -> str:
    label = span.get("label")
    return label if label is not None else span.get("type", "unclassified")


def _validate_span(span: dict, n_paragraphs: int) -> None:
    for field_name in ("type", "paragraph_start", "paragraph_end"):
        if field_name not in span:
            raise ChunkingError(f"span is missing required field '{field_name}': {span!r}")
    span_type = span["type"]
    if span_type not in CONTEXT_ONLY_SPAN_KINDS and span_type not in CHUNKABLE_SPAN_KINDS:
        raise ChunkingError(
            f"span has an unrecognized type '{span_type}' -- not in CONTEXT_ONLY_SPAN_KINDS "
            f"or CHUNKABLE_SPAN_KINDS: {span!r}"
        )
    start, end = span["paragraph_start"], span["paragraph_end"]
    if not isinstance(start, int) or not isinstance(end, int) or end < start:
        raise ChunkingError(f"span has an invalid paragraph range [{start},{end}]: {span!r}")
    if start < 0 or end >= n_paragraphs:
        raise ChunkingError(
            f"span paragraph range [{start},{end}] is out of bounds for a document "
            f"with {n_paragraphs} paragraphs: {span!r}"
        )


def _group_paragraphs(paragraph_indices: list[int], paragraphs: list[str]) -> list[list[int]]:
    """Greedy paragraph grouping: accumulate consecutive paragraphs until
    adding the next one would push the running chunk's word count past
    SOFT_MAX_WORDS, then start a new chunk.

    Never splits a single paragraph: a paragraph that is itself >=
    SOFT_MAX_WORDS simply becomes its own (flagged, see
    ChildChunk.oversized_single_paragraph) chunk rather than being
    discarded, compressed, or sentence-split.

    This directly implements the task's worked example: paragraphs of
    180+160+210+170=720 words accumulate into one chunk; a 5th paragraph
    that would push the total to 1,150 (> SOFT_MAX_WORDS) starts a new
    chunk instead of being force-added.
    """

    groups: list[list[int]] = []
    current: list[int] = []
    current_words = 0
    for idx in paragraph_indices:
        p_words = _word_count(paragraphs[idx])
        if current and current_words + p_words > SOFT_MAX_WORDS:
            groups.append(current)
            current = [idx]
            current_words = p_words
        else:
            current.append(idx)
            current_words += p_words
    if current:
        groups.append(current)
    return groups


def _build_overlap_prefix(prev_group: list[int], paragraphs: list[str]) -> list[int]:
    """Trailing whole paragraphs of ``prev_group`` totalling at most
    ``OVERLAP_TARGET_WORDS``. Overlap stays paragraph-atomic, exactly
    like the primary grouping -- never a partial paragraph.

    Unlike ``_group_paragraphs``, this never force-includes a paragraph
    that alone exceeds the budget: primary grouping must never lose
    text, so a single oversized paragraph has to go somewhere, but
    overlap is a best-effort context primer, not a text-preservation
    guarantee -- dragging in a 600-word paragraph to satisfy a 125-word
    target would defeat the point of "overlap," not fulfil it. If even
    the single closest paragraph is too big, this returns no overlap at
    all for that boundary.

    Also caps at ``len(prev_group) - 1`` paragraphs: pulling in the
    ENTIRE previous group (e.g. when it's a single short paragraph)
    would make the next chunk start at the exact same source position
    as the previous one, violating the required strict
    ``chunk_n.source_start < chunk_n+1.source_start`` ordering. At
    least the previous group's own first paragraph must always stay
    exclusively its own.
    """

    max_count = len(prev_group) - 1
    if max_count <= 0:
        return []

    prefix: list[int] = []
    words = 0
    for idx in reversed(prev_group[-max_count:]):
        p_words = _word_count(paragraphs[idx])
        if words + p_words > OVERLAP_TARGET_WORDS:
            break
        prefix.insert(0, idx)
        words += p_words
    return prefix


def _span_chunks(
    doc_id: str,
    span: dict,
    paragraphs: list[str],
    offsets: list[tuple[int, int]],
    full_text: str,
    start_index: int,
) -> list[ChildChunk]:
    """All ChildChunks for one structural span, with prev/next left
    unset -- filled in by chunk_document() once the whole document's
    chunk order is known."""

    section = _section_for_span(span)
    span_kind = span["type"]
    para_start, para_end = span["paragraph_start"], span["paragraph_end"]
    indices = list(range(para_start, para_end + 1))

    groups = _group_paragraphs(indices, paragraphs)
    # Overlap: only within long continuous reasoning/narrative sections,
    # and only when this span produced more than one chunk (nothing to
    # overlap with otherwise). Never applied across different spans/
    # sections -- each span is chunked independently, so overlap can
    # only ever duplicate paragraphs that belong to the SAME section.
    overlap_allowed = section in OVERLAP_ELIGIBLE_SECTIONS and len(groups) > 1

    chunks: list[ChildChunk] = []
    chunk_index = start_index
    for g_i, group in enumerate(groups):
        overlap_prefix: list[int] = []
        if overlap_allowed and g_i > 0:
            overlap_prefix = _build_overlap_prefix(groups[g_i - 1], paragraphs)

        full_group = overlap_prefix + group
        chunk_para_start, chunk_para_end = full_group[0], full_group[-1]
        source_start = offsets[chunk_para_start][0]
        source_end = offsets[chunk_para_end][1]
        text = full_text[source_start:source_end]

        oversized = len(group) == 1 and _word_count(paragraphs[group[0]]) > SOFT_MAX_WORDS

        chunks.append(ChildChunk(
            chunk_id=f"{doc_id}:{chunk_index:04d}",
            parent_id=doc_id,
            doc_id=doc_id,
            chunk_index=chunk_index,
            section=section,
            span_kind=span_kind,
            paragraph_start=chunk_para_start,
            paragraph_end=chunk_para_end,
            source_start=source_start,
            source_end=source_end,
            text=text,
            word_count=_word_count(text),
            is_overlap=bool(overlap_prefix),
            overlap_paragraph_count=len(overlap_prefix),
            oversized_single_paragraph=oversized,
        ))
        chunk_index += 1

    return chunks


def chunk_document(doc: dict) -> ParentCase:
    """Chunks one structured document (the ``to_output_document()``
    shape from :mod:`src.rag_prep.structurer`: ``{doc_id, metadata,
    full_text, structure: {spans: [...], ...}}``) into a
    :class:`ParentCase` with its ordered, linked :class:`ChildChunk` list.

    ``case_caption``/``judgment_marker`` spans never produce chunks (they
    stay accessible only via the parent's ``full_text``); every other
    span kind (semantic_section/quoted_material/paragraph_group/
    final_order/headnotes) is chunked identically via the same
    paragraph-grouping rule, since the structuring stage has already
    decided what each one IS -- this function only decides how to group
    already-labeled paragraphs into retrieval-sized pieces.

    Raises :class:`ChunkingError` if a span is structurally malformed
    (missing fields or an invalid/out-of-range paragraph range) --
    offsets are never guessed for input that can't be trusted.
    """

    doc_id = doc["doc_id"]
    metadata = doc["metadata"]
    full_text = doc["full_text"]
    spans = sorted(doc.get("structure", {}).get("spans", []), key=lambda s: s.get("paragraph_start", -1))

    paragraphs = full_text.split("\n\n")
    offsets = paragraph_offsets(full_text)

    for span in spans:
        _validate_span(span, len(paragraphs))

    all_chunks: list[ChildChunk] = []
    chunk_index = 0
    for span in spans:
        if span["type"] in CONTEXT_ONLY_SPAN_KINDS:
            continue
        span_chunks = _span_chunks(doc_id, span, paragraphs, offsets, full_text, chunk_index)
        all_chunks.extend(span_chunks)
        chunk_index += len(span_chunks)

    # Link prev/next across the WHOLE document's chunk sequence. This
    # never crosses parent cases -- chunk_document() only ever sees one
    # document's spans, so every id referenced here belongs to this doc.
    linked: list[ChildChunk] = []
    for i, c in enumerate(all_chunks):
        prev_id = all_chunks[i - 1].chunk_id if i > 0 else None
        next_id = all_chunks[i + 1].chunk_id if i < len(all_chunks) - 1 else None
        linked.append(replace(c, prev_chunk_id=prev_id, next_chunk_id=next_id))

    return ParentCase(doc_id=doc_id, metadata=metadata, full_text=full_text, chunks=linked)


def denormalize_chunk(chunk: ChildChunk, parent: ParentCase) -> dict:
    """A self-contained view of one chunk with the parent's purposeful
    metadata fields inherited onto it -- for a downstream store (e.g. a
    vector DB) that needs each record to carry its own context. The
    primary ParentCase/ChildChunk representation deliberately does NOT
    duplicate this data internally (see INHERITABLE_METADATA_FIELDS)."""

    d = chunk.to_dict()
    for field_name in INHERITABLE_METADATA_FIELDS:
        d[field_name] = parent.metadata.get(field_name)
    return d
