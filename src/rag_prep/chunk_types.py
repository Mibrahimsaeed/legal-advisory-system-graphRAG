"""Shared types for the case-law parent-child chunking layer.

Parent-child model: the PARENT is the complete legal case (``doc_id``,
full metadata, the complete authoritative ``full_text``); CHILD chunks
are retrieval-sized, paragraph-atomic slices of that same ``full_text``,
each carrying an exact ``(source_start, source_end)`` character offset
into it.

``full_text`` is always the single source of truth -- a chunk's ``text``
is only ever assembled by slicing ``full_text[source_start:source_end]``,
mirroring the same "never trust/reproduce, always slice" discipline the
structuring layer already uses for its own spans (see
:mod:`src.rag_prep.structure_types`'s module docstring). This module
defines the data shapes only; see :mod:`src.rag_prep.chunker` for how
they're built and :mod:`src.rag_prep.chunk_validate` for how they're
audited.
"""

from __future__ import annotations

from dataclasses import dataclass, field

# Structural span kinds that are deliberately NOT turned into retrieval
# child chunks -- they remain accessible only through the parent's
# full_text, per the "treat structural navigation material as
# context/metadata, not ordinary substantive retrieval chunks" rule.
CONTEXT_ONLY_SPAN_KINDS = ("case_caption", "judgment_marker")

# Every other span kind from the structuring layer is chunkable --
# headnotes/quoted_material/final_order follow the identical short-keep /
# long-split rule as ordinary semantic_section/paragraph_group spans.
CHUNKABLE_SPAN_KINDS = ("semantic_section", "quoted_material", "paragraph_group", "final_order", "headnotes")

# Allowed chunk "section" values: the 13 semantic labels the structuring
# layer can assign, plus "headnotes" (a structural kind with no semantic
# label of its own).
ALLOWED_SECTIONS = frozenset({
    "procedural_history", "facts", "issues", "arguments", "applicable_law",
    "evidence", "court_reasoning", "findings", "authorities_cited",
    "lower_court_orders", "final_order", "quoted_material", "unclassified",
    "headnotes",
})

# Chunk-size targets, in words -- retrieval-quality guidance, not hard
# mathematical constraints.
TARGET_MIN_WORDS = 500
TARGET_MAX_WORDS = 800
SOFT_MAX_WORDS = 1000

# Conditional overlap: only within long continuous reasoning/narrative
# sections, never for short/disposition-type sections, and never when a
# span produced only one chunk (nothing to overlap with).
OVERLAP_ELIGIBLE_SECTIONS = frozenset({
    "facts", "court_reasoning", "evidence", "procedural_history", "quoted_material",
})
OVERLAP_TARGET_WORDS = 125  # midpoint of the requested 100-150 word range

# Case-level metadata fields a chunk may inherit when denormalized for a
# downstream store that needs self-contained records (see
# chunker.denormalize_chunk()) -- kept purposeful/minimal, not a blanket
# copy of the whole metadata dict, per the "don't duplicate huge
# unnecessary parent data into every child" instruction.
INHERITABLE_METADATA_FIELDS = (
    "case_title", "citation", "court", "court_location", "decision_date",
    "judges", "primary_domain", "disposition", "statutes_cited", "provisions_cited",
)


@dataclass(frozen=True)
class ChildChunk:
    """One retrieval-sized, paragraph-atomic, exact-offset slice of a
    parent case's ``full_text``."""

    chunk_id: str
    parent_id: str
    doc_id: str
    chunk_index: int
    section: str              # semantic label, or structural kind when no label exists (e.g. "headnotes")
    span_kind: str            # the ORIGINAL structural span kind this chunk was derived from
    paragraph_start: int
    paragraph_end: int
    source_start: int
    source_end: int
    text: str
    word_count: int
    prev_chunk_id: str | None = None
    next_chunk_id: str | None = None
    is_overlap: bool = False             # True if this chunk's leading paragraph(s) duplicate the tail
    overlap_paragraph_count: int = 0     # of the PREVIOUS chunk -- intentional, auditable overlap
    oversized_single_paragraph: bool = False  # a single paragraph alone exceeded SOFT_MAX_WORDS

    def to_dict(self) -> dict:
        return {
            "chunk_id": self.chunk_id,
            "parent_id": self.parent_id,
            "doc_id": self.doc_id,
            "chunk_index": self.chunk_index,
            "section": self.section,
            "span_kind": self.span_kind,
            "paragraph_start": self.paragraph_start,
            "paragraph_end": self.paragraph_end,
            "source_start": self.source_start,
            "source_end": self.source_end,
            "text": self.text,
            "word_count": self.word_count,
            "prev_chunk_id": self.prev_chunk_id,
            "next_chunk_id": self.next_chunk_id,
            "is_overlap": self.is_overlap,
            "overlap_paragraph_count": self.overlap_paragraph_count,
            "oversized_single_paragraph": self.oversized_single_paragraph,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "ChildChunk":
        return cls(
            chunk_id=d["chunk_id"],
            parent_id=d["parent_id"],
            doc_id=d["doc_id"],
            chunk_index=d["chunk_index"],
            section=d["section"],
            span_kind=d["span_kind"],
            paragraph_start=d["paragraph_start"],
            paragraph_end=d["paragraph_end"],
            source_start=d["source_start"],
            source_end=d["source_end"],
            text=d["text"],
            word_count=d["word_count"],
            prev_chunk_id=d.get("prev_chunk_id"),
            next_chunk_id=d.get("next_chunk_id"),
            is_overlap=d.get("is_overlap", False),
            overlap_paragraph_count=d.get("overlap_paragraph_count", 0),
            oversized_single_paragraph=d.get("oversized_single_paragraph", False),
        )


@dataclass
class ParentCase:
    """The complete legal case -- ``doc_id``, full metadata, and the
    complete authoritative ``full_text``, retained in full regardless of
    how (or whether) its child chunks were produced."""

    doc_id: str
    metadata: dict
    full_text: str
    chunks: list[ChildChunk] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "doc_id": self.doc_id,
            "metadata": self.metadata,
            "full_text": self.full_text,
            "chunks": [c.to_dict() for c in self.chunks],
        }

    @classmethod
    def from_dict(cls, d: dict) -> "ParentCase":
        return cls(
            doc_id=d["doc_id"],
            metadata=d["metadata"],
            full_text=d["full_text"],
            chunks=[ChildChunk.from_dict(c) for c in d["chunks"]],
        )


class ChunkingError(ValueError):
    """Raised when the input structure is genuinely malformed -- a span
    missing required fields, or paragraph indices that are out of range
    or inverted -- such that an exact source offset cannot be derived
    safely. Per the task's "fail validation rather than guessing" rule:
    this is raised rather than silently producing a best-effort (and
    potentially wrong) offset."""
