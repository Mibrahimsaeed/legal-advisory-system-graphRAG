"""Layer 1: deterministic structural detection (no LLM).

Everything here is regex/rule-based on things the structure audit
(``var/rag/audits/``) found to be reliable without semantic judgment:
paragraph boundaries, the lettered-headnote convention, the JUDGMENT/ORDER
marker convention, numbered-paragraph bodies, and the disposition-verb
vocabulary already validated in :mod:`src.rag_prep.case_cleaner`.

This module does NOT attempt semantic classification of facts/issues/
arguments/reasoning -- the audit found those are frequently interleaved
within a single paragraph, which is exactly the judgment call Layer 2
(Qwen) is for. Layer 1's job is narrower: carve out the regions that ARE
reliably identifiable by pattern alone, and hand everything else to Layer 2
as a clean paragraph-index range.
"""

from __future__ import annotations

import re

from src.rag_prep.structure_types import DISPOSITION_LABELS, Span

HEADNOTE_LETTER_RE = re.compile(r"^\([a-z]\)\s+\S")
NUMBERED_PARA_RE = re.compile(r"^\d{1,3}\.\s+\S")
ROMAN_SUBITEM_RE = re.compile(r"\((?:i|ii|iii|iv|v|vi|vii|viii|ix|x|xi|xii)\)", re.IGNORECASE)
LETTER_SUBITEM_RE = re.compile(r"\([a-z]\)")

_JUDGMENT_OR_ORDER_RE = re.compile(r"^(JUDGMENT|ORDER)[.:]?$")

# Disposition-verb patterns, priority-ordered (compound before plain),
# identical vocabulary to case_cleaner.extract_disposition but applied
# PER SENTENCE here -- the mechanism that fixes the multi-matter bug the
# structure audit found (var/rag/audits/ Section 10: 04139076ff36792b38b72f8b
# has three distinct outcomes in one tail paragraph; a single tail-scan
# picked the wrong one).
_DISPOSITION_PATTERNS: list[tuple[str, re.Pattern]] = [
    ("partly_allowed", re.compile(r"\bpart(?:ly|ially)\b.{0,25}\b(allowed|accepted|granted)\b")),
    ("partly_dismissed", re.compile(r"\bpart(?:ly|ially)\b.{0,25}\bdismissed\b")),
    ("dismissed", re.compile(r"\b(dismiss(?:ed)?|rejected|refused|disallowed|dropped)\b")),
    ("allowed", re.compile(r"\b(allowed|accepted|granted|decreed)\b")),
    ("remanded", re.compile(r"\bremand(?:ed)?\b")),
    ("transferred", re.compile(r"\btransferred\b")),
    ("returned", re.compile(r"\breturned\b")),
    ("withdrawn", re.compile(r"\bwithdrawn\b")),
    ("abated", re.compile(r"\babated\b")),
    ("infructuous", re.compile(r"\binfructuous\b")),
    ("objection_sustained", re.compile(r"\bobjection\b.{0,15}\bsustained\b")),
    ("set_aside", re.compile(r"\border\b.{0,10}\bset aside\b")),
    ("disposed_accordingly", re.compile(r"\bdisposed\b|\baccordingly\b")),
    ("converted", re.compile(r"\bconverted\b")),
]

_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.;])\s+")

# The empirically verified length gap (see task report): the genuinely
# incomplete documents found in the corpus are <=4,411 chars; the
# non-judgment academic-article documents found alongside them are all
# >=7,950 chars. 6,000 sits in the middle of that gap -- NOT at either
# boundary value itself (an earlier version of this threshold was set to
# 8,000, which is *below* 7,950's own actual length and misclassified that
# one real document as incomplete_source instead of non_judgment_text;
# caught by running this check against every disposition=None document in
# the corpus before trusting it, not assumed).
_NON_JUDGMENT_LENGTH_FLOOR = 6000

_COUNSEL_TAIL_RE = re.compile(
    r"(advocate|counsel)[^.]*\bfor\s+(petitioner|respondent|appellant)s?\.?\s*$",
    re.IGNORECASE,
)
_DATE_OF_HEARING_TAIL_RE = re.compile(r"^date of hearing\s*[:.]", re.IGNORECASE)
_FOOTNOTE_TAIL_RE = re.compile(
    r"\bibid\.?\b|\bsupra\b|^\s*\d{1,3}\s+[A-Z]", re.IGNORECASE
)


def paragraphs(text: str) -> list[str]:
    """The paragraph list every span index refers to. Reconstructable by
    ``"\\n\\n".join(paragraphs(text)) == text`` -- verified corpus-wide
    before this module was written; never filter/strip an entry here or
    that guarantee breaks."""

    return text.split("\n\n")


def classify_document_kind(full_text: str, disposition: str | None) -> str | None:
    """Returns ``"incomplete_source"``, ``"non_judgment_text"``, or ``None``
    (proceed with normal structuring).

    Both special cases share one precondition -- the cleaning stage's
    already-validated disposition extractor (98.5% coverage corpus-wide)
    found nothing -- then split by length, per the verified gap above.
    A long ``disposition=None`` document ending in numbered footnotes is
    an academic article that happened to be swept into this corpus by
    classification, not a truncated judgment; forcing it into
    "incomplete_source" would misreport what's actually wrong with it.
    """

    if disposition is not None:
        return None

    if len(full_text) >= _NON_JUDGMENT_LENGTH_FLOOR:
        return "non_judgment_text"
    return "incomplete_source"


def find_judgment_marker(paras: list[str]) -> tuple[str | None, int | None]:
    """The paragraph index of a standalone "JUDGMENT" or "ORDER" line, if any.

    Searches in document order and returns the first match -- the audit
    found these are mutually exclusive structural boundary markers, not a
    sequence, so only one is ever expected.
    """

    for idx, p in enumerate(paras):
        stripped = p.strip().rstrip(".:-").strip().upper()
        m = _JUDGMENT_OR_ORDER_RE.match(stripped)
        if m:
            return m.group(1), idx
    return None, None


def find_headnote_span(paras: list[str], stop_idx: int | None) -> tuple[int, int] | None:
    """The contiguous run of lettered headnote paragraphs, if any, bounded
    above by ``stop_idx`` (the judgment/order marker, or None = search the
    whole document). Includes trailing "... rel." / "... ref." authority
    lines immediately following a lettered paragraph, since those are part
    of the same reporter-authored headnote, not judgment-body text."""

    limit = stop_idx if stop_idx is not None else len(paras)
    first = last = None
    for idx in range(limit):
        p = paras[idx]
        if HEADNOTE_LETTER_RE.match(p):
            if first is None:
                first = idx
            last = idx
        elif first is not None and (p.rstrip().endswith("rel.") or p.rstrip().endswith("ref.")):
            last = idx
        elif first is not None and idx > last + 1:
            # A non-headnote, non-authority paragraph more than one away
            # from the last headnote paragraph ends the run.
            break
    if first is None:
        return None
    return first, last


def numbered_paragraph_fraction(paras: list[str], start: int, end: int) -> float:
    if end < start:
        return 0.0
    span = paras[start : end + 1]
    if not span:
        return 0.0
    numbered = sum(1 for p in span if NUMBERED_PARA_RE.match(p))
    return numbered / len(span)


def detect_final_orders(paras: list[str], body_start: int) -> list[dict]:
    """Deterministic, sentence-level disposition detection over the final
    1-2 paragraphs of the judgment body.

    Returns a list (never a single scalar) because the audit found real
    cases with more than one distinct disposition for more than one
    matter in the same document. Each entry is
    ``{"paragraph_start", "paragraph_end", "sentence", "label"}``.
    """

    if body_start >= len(paras):
        return []

    # Look at the last paragraph, and the one before it if the last
    # paragraph alone doesn't carry a recognizable disposition sentence
    # (covers the common "12. This petition stands disposed of..." +
    # separate reporter-tag-line pattern).
    tail_start = max(body_start, len(paras) - 2)
    orders: list[dict] = []
    unmatched_indices: list[int] = []
    for idx in range(tail_start, len(paras)):
        para = paras[idx]
        para_matched = False
        for sentence in _SENTENCE_SPLIT_RE.split(para):
            sentence = sentence.strip()
            if not sentence:
                continue
            lowered = sentence.lower()
            for label, pattern in _DISPOSITION_PATTERNS:
                if pattern.search(lowered):
                    orders.append({
                        "paragraph_start": idx,
                        "paragraph_end": idx,
                        "sentence": sentence,
                        "label": label,
                    })
                    para_matched = True
                    break
        if not para_matched:
            unmatched_indices.append(idx)

    # A paragraph inside the scanned tail window that matched no
    # disposition pattern is only treated as part of the "final order"
    # region -- rather than left as ordinary body content for Layer 2 to
    # label -- when at least one SIBLING paragraph in the same tail scan
    # DID match. That anchoring is what distinguishes "this is the tail,
    # and this one paragraph's disposition word just happens to be
    # truncated at the source" (verified directly: one such document's
    # source case.html literally ends "...Petition\ndismisse<o:p>",
    # missing the final "d.") from "this short document's body simply
    # doesn't state an outcome in its last two paragraphs" -- in the
    # latter case there is no confirmed disposition to anchor to, so the
    # paragraph(s) stay in the body range for normal semantic labeling
    # instead of being swallowed into final_orders as an empty guess.
    if orders:
        for idx in unmatched_indices:
            orders.append({
                "paragraph_start": idx, "paragraph_end": idx,
                "sentence": paras[idx], "label": "unclassified",
            })

    return orders


def quotation_density_hint(paras: list[str], start: int, end: int) -> dict[int, int]:
    """Per-paragraph count of roman/letter sub-item markers, for paragraphs
    in ``[start, end]``. A paragraph with several such markers in sequence
    is very likely a quoted schedule/order -- passed to Layer 2 as a
    SIGNAL (not a final decision) to strengthen its quoted_material calls,
    exactly the "deterministic hint -> LLM confirms" architecture."""

    hints = {}
    for idx in range(start, end + 1):
        n = len(ROMAN_SUBITEM_RE.findall(paras[idx])) + len(LETTER_SUBITEM_RE.findall(paras[idx]))
        if n >= 3:
            hints[idx] = n
    return hints


def build_deterministic_spans(full_text: str) -> dict:
    """Everything Layer 1 can establish without semantic judgment.

    Returns a dict with: paras, judgment_marker, judgment_marker_idx,
    headnote_span, body_start, body_end (exclusive of final-order
    paragraphs), final_orders, quotation_hints.
    """

    paras = paragraphs(full_text)
    marker, marker_idx = find_judgment_marker(paras)
    headnote_span = find_headnote_span(paras, marker_idx)

    if marker_idx is not None:
        body_start = marker_idx + 1
    elif headnote_span is not None:
        body_start = headnote_span[1] + 1
    else:
        body_start = 0

    final_orders = detect_final_orders(paras, body_start)
    body_end = len(paras) - 1
    if final_orders:
        body_end = min(o["paragraph_start"] for o in final_orders) - 1

    quotation_hints = (
        quotation_density_hint(paras, body_start, body_end) if body_end >= body_start else {}
    )

    return {
        "paras": paras,
        "judgment_marker": marker,
        "judgment_marker_idx": marker_idx,
        "headnote_span": headnote_span,
        "body_start": body_start,
        "body_end": body_end,
        "final_orders": final_orders,
        "quotation_hints": quotation_hints,
    }


def make_span(paras: list[str], kind: str, start: int, end: int, **kwargs) -> Span:
    text = "\n\n".join(paras[start : end + 1])
    return Span(kind=kind, paragraph_start=start, paragraph_end=end, text=text, **kwargs)
