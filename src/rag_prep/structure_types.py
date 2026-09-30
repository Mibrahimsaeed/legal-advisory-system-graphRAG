"""Shared types for the case-law structuring layer.

Paragraph-index coordinate system: every span in a structured document is
``(paragraph_start, paragraph_end)`` -- inclusive, 0-based indices into
``full_text.split("\\n\\n")``. This split/join round-trip was verified
against all 2,088 processed documents before this module was written
(``"\\n\\n".join(text.split("\\n\\n")) == text`` held for all of them, with
no empty paragraphs) -- it is the property the whole no-content-loss
validator in :mod:`structure_validate` depends on.

Every span's ``text`` field is always assembled by literally slicing
``paragraphs[start:end+1]`` and rejoining with ``"\\n\\n"`` -- never text
echoed back by the LLM. See :mod:`structurer_llm`'s module docstring for
why.
"""

from __future__ import annotations

from dataclasses import dataclass, field

# Fixed semantic-label vocabulary the LLM (Layer 2) is allowed to choose
# from, per the task's "judgment_body" sub-sections. "headnotes" and
# "case_caption" are NOT in this list -- both are detected deterministically
# (Layer 1), never assigned by the model.
SEMANTIC_LABELS = (
    "procedural_history",
    "facts",
    "issues",
    "arguments",
    "applicable_law",
    "evidence",
    "court_reasoning",
    "findings",
    "authorities_cited",
    "lower_court_orders",
    "final_order",
    "quoted_material",
    "unclassified",
)

# Disposition-verb vocabulary reused from src.rag_prep.case_cleaner's
# already-validated tail-matching patterns, applied per-sentence here
# instead of per-tail so a multi-matter paragraph (e.g. "the appeal is
# allowed... C.M.A. ... is dismissed... C.M.A. ... is disposed of
# accordingly") yields one label per matter instead of one label for the
# whole document.
DISPOSITION_LABELS = (
    "dismissed", "allowed", "partly_allowed", "partly_dismissed",
    "remanded", "transferred", "returned", "withdrawn", "abated",
    "infructuous", "disposed_accordingly", "converted", "objection_sustained",
    "set_aside",
)

# structure_status values.
STATUS_STRUCTURED = "structured"
STATUS_FALLBACK = "fallback_paragraph_groups"
STATUS_INCOMPLETE_SOURCE = "incomplete_source"
STATUS_NON_JUDGMENT_TEXT = "non_judgment_text"


@dataclass(frozen=True)
class Span:
    """One labeled region, always paragraph-index-addressed."""

    kind: str  # "case_caption" | "headnotes" | "judgment_marker" |
               # one of SEMANTIC_LABELS | "paragraph_group" | "quoted_material"
    paragraph_start: int
    paragraph_end: int
    text: str
    label: str | None = None       # semantic label, for judgment-body spans
    confidence: float | None = None
    attribution: str | None = None  # for quoted_material: "lower_court" | "precedent" | "statute" | None

    def to_dict(self) -> dict:
        d = {
            "type": self.kind,
            "paragraph_start": self.paragraph_start,
            "paragraph_end": self.paragraph_end,
            "text": self.text,
        }
        if self.label is not None:
            d["label"] = self.label
        if self.confidence is not None:
            d["confidence"] = self.confidence
        if self.kind == "quoted_material":
            d["attribution"] = self.attribution
        return d


@dataclass
class StructureResult:
    structure_status: str
    spans: list[Span] = field(default_factory=list)
    headnotes_detected: bool = False
    judgment_marker: str | None = None  # "JUDGMENT" | "ORDER" | None
    final_orders: list[dict] = field(default_factory=list)  # multi-matter aware
    used_llm: bool = False
    llm_fallback_reason: str | None = None
    validation: dict | None = None
