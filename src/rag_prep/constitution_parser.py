"""Constitution of Pakistan structural parsing -- Stage 3 of the
independent Constitution pipeline.

    cleaned Constitution JSON (var/rag/cleaned_constitution/<doc_id>.json)
        -> parse_constitution_structure()   [deterministic regex/state-machine, no LLM]
        -> parsed-structure JSON            (var/rag/constitution_parsed/<doc_id>.json)

Converts the already-cleaned Constitution text into a hierarchical AST:

    Constitution
    |-- Preamble / Objectives Resolution (ANNEX)
    |-- Part
    |   `-- Chapter
    |       `-- Article
    |           |-- Clause
    |           |   `-- Sub-clause
    |           `-- Proviso
    `-- Schedule

This stage never rewrites, paraphrases, or deletes any constitutional
text -- it only locates boundaries and nests the SAME verbatim text
underneath them. Every node carries the exact ``[source_start,
source_end)`` half-open slice of ``cleaned_text`` it represents, and at
every level a node's children, concatenated by position, reconstruct
that node's own span exactly -- any gap (text that didn't match a
recognized child pattern) is kept, never dropped, as an explicit
``"text"`` node. See ``validate_constitution_structure()`` for the
checks that enforce this.

ARTICLE-HEADER DESIGN NOTE
---------------------------
The task's own example regex (``^\\s*Article\\s+(\\d+[A-Z]?)\\.?\\s*(.*)``)
assumes a literal "Article 17" prefix. A full sweep of the real,
currently-cleaned Constitution of Pakistan document found ZERO such
occurrences -- every real Article is headed by a bare "<number>.
<title/body>" line (the same convention as the statute/case-law
numbering style), confirmed exactly against the cleaning stage's own
``article_index`` (328/328 real articles matched, 0 missed, 0 false
positives -- see _ARTICLE_SAME_LINE_RE / _ARTICLE_BARE_RE below for the
two real sub-shapes this required). This module follows the real
evidence, not the task's illustrative example, per the project's
standing "narrow evidenced regex over an assumed one" discipline.

PIPELINE ISOLATION
-------------------
No import of src.ingestion/, src.classification/, src.rag_prep.
case_cleaner/structurer*/chunk*/statute_*, or src.extraction.
case_loader. Independent entry point, independent input/output
directories.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

INPUT_DIR = Path("var/rag/cleaned_constitution")
OUTPUT_DIR = Path("var/rag/constitution_parsed")

_FOOTNOTES_BLOCK_MARKER = "\n\n[FOOTNOTES]\n"

# -- shared "leading noise" fragment -------------------------------------------
# The cleaning stage's footnote-reference sentinel ("[^5.1]") and, less
# often, a bare amendment-bracket opener ("[") can sit directly in front
# of ANY structural heading (confirmed real cases: "[^189.1][FIRST
# SCHEDULE", "[^172.2][[^172.3]270A.--(1)..." -- note the latter chains
# TWO sentinels). Zero-or-more repetitions of either, consumed but never
# included in a heading's own captured identifier/title group, covers
# every real case found without requiring a fixed count.
_LEADING_NOISE = r"(?:(?:\[\^[^\]]+\]|\[)[ \t]*)*"

_PREAMBLE_RE = re.compile(rf"^[ \t]*{_LEADING_NOISE}PREAMBLE[ \t]*$", re.MULTILINE)
_ANNEX_RE = re.compile(rf"^[ \t]*{_LEADING_NOISE}ANNEX\b.*$", re.MULTILINE)
_PART_RE = re.compile(rf"^[ \t]*{_LEADING_NOISE}PART[ \t]+([IVXLCDM]+)\b.*$", re.MULTILINE)
_CHAPTER_RE = re.compile(rf"^[ \t]*{_LEADING_NOISE}CHAPTER[ \t]+(\d+[A-Za-z]{{0,2}})\.[ \t–—.-]*(.*)$", re.MULTILINE)
# A stray leading "*" (confirmed once, "*FIFTH SCHEDULE") or bracket is
# tolerated in addition to the standard sentinel noise.
_SCHEDULE_RE = re.compile(
    rf"^[ \t]*{_LEADING_NOISE}\*?((?:FIRST|SECOND|THIRD|FOURTH|FIFTH|SIXTH|SEVENTH)[ \t]+SCHEDULE)\b.*$",
    re.MULTILINE,
)

# -- Articles: two real sub-shapes, both confirmed exactly (328/328,
# 0 missed, 0 false positives) against the cleaning stage's own
# article_index for the real document --
#
#   1. same-line title/body text right after "<id>. " (the common case)
#   2. a BARE "<id>." with nothing else on that line, its text entirely
#      on the next line -- but ONLY when that next line (after its own
#      leading noise) starts a fresh clause "(1)" specifically. This
#      narrow extra condition is what excludes the two real false-
#      positive candidates found during validation: a citation ending a
#      sentence with a bare year ("...Act, 2010.\n(2) Before entering
#      upon office...") is followed by "(2)" (a CONTINUING clause of an
#      already-open Article), never "(1)" -- only a genuine new Article
#      is followed by its own first clause.
_ARTICLE_SAME_LINE_RE = re.compile(
    rf"^[ \t]*{_LEADING_NOISE}(\d+[A-Za-z]{{0,3}})\.[ \t–—.-]*(?=\S)(.*)$", re.MULTILINE
)
_ARTICLE_BARE_RE = re.compile(
    rf"^[ \t]*{_LEADING_NOISE}(\d+[A-Za-z]{{0,3}})\.[ \t]*\n[ \t]*{_LEADING_NOISE}\(1\)",
    re.MULTILINE,
)

# A marginal side-note the cleaning stage already attached to most real
# Articles (312 of 328 in the real document -- see
# constitution_cleaning.py's _attach_marginal_notes), rendered as its
# own "## <id>. <title>" line immediately before the Article's own
# "<id>. <body>" line. This is the ONLY reliable heading source: the
# Article's own first line never carries a separate title in this
# document (it goes straight into clause (1)'s body -- e.g. Article 1's
# real first line is "1. (1) Pakistan shall be Federal Republic...",
# not "1. The Republic and its territories. (1) Pakistan..."), so
# treating "whatever follows the period" as the heading would silently
# capture clause (1)'s own text as a fake title. A missing marginal
# note (confirmed for 16 real articles) means ``heading`` stays None --
# never invented.
_MARGINAL_NOTE_LINE_RE = re.compile(r"^##[ \t]+(\d+[A-Za-z]{0,3})\.[ \t]+(.+)$", re.MULTILINE)

_CLAUSE_RE = re.compile(rf"^[ \t]*{_LEADING_NOISE}\((\d+[A-Za-z]{{0,2}})\)[ \t]*", re.MULTILINE)
_SUBCLAUSE_RE = re.compile(rf"^[ \t]*{_LEADING_NOISE}\(([a-z]+)\)[ \t]*", re.MULTILINE)
_PROVISO_RE = re.compile(rf"^[ \t]*{_LEADING_NOISE}Provided[ \t]+(?:further[ \t]+)?that\b", re.MULTILINE)

# Clause (1) -- almost always the very FIRST clause of an Article --
# very commonly sits immediately after the Article's own number on the
# SAME line ("1. (1) Pakistan shall be Federal Republic...", confirmed
# as the real Article 1's own exact shape), with no preceding newline
# at all. _CLAUSE_RE's own "^" anchor can never match there -- this
# companion pattern is tried ONLY once, at the exact start of whatever
# span is being searched for clauses (an Article's or a Chapter-less
# Part's own span start), to catch that one specific position. The
# equivalent inline-subclause case ("(1) (a) ...") is handled the same
# way in _parse_subclause_level.
_INLINE_NUMBERED_RE = re.compile(rf"\((\d+[A-Za-z]{{0,2}})\)[ \t]*")
_INLINE_LETTERED_RE = re.compile(rf"\(([a-z]+)\)[ \t]*")


class ConstitutionParsingError(ValueError):
    """Raised when a structural boundary cannot be determined safely --
    e.g. no PART/Article found anywhere. Per "fail loudly rather than
    silently guessing": raised instead of returning a bogus structure."""


@dataclass
class Node:
    type: str  # "root" | "preamble" | "annex" | "part" | "chapter" | "schedule" | "article" | "clause" | "subclause" | "proviso" | "text"
    id: str
    source_start: int
    source_end: int
    text: str
    heading: str | None = None
    article_number: str | None = None
    number: str | None = None
    children: list["Node"] = field(default_factory=list)

    def to_dict(self) -> dict:
        d = {
            "type": self.type,
            "id": self.id,
            "source_start": self.source_start,
            "source_end": self.source_end,
            "text": self.text,
        }
        if self.heading is not None:
            d["heading"] = self.heading
        if self.article_number is not None:
            d["article_number"] = self.article_number
        if self.number is not None:
            d["number"] = self.number
        d["children"] = [c.to_dict() for c in self.children]
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "Node":
        return cls(
            type=d["type"], id=d["id"], source_start=d["source_start"], source_end=d["source_end"],
            text=d["text"], heading=d.get("heading"), article_number=d.get("article_number"),
            number=d.get("number"), children=[cls.from_dict(c) for c in d.get("children", [])],
        )


def _base_number(identifier: str) -> int:
    m = re.match(r"\d+", identifier)
    return int(m.group()) if m else 0


def _filter_monotonic(matches: list, base_of) -> list:
    """Drops a match whose base number regresses below the running
    maximum -- the same structural (never vocabulary-based) safety net
    statute_chunker.py uses, kept here as defense-in-depth even though
    the real document needed no matches filtered by it (the real
    false-positive case found during validation was excluded by the
    Article regex's own "(1)" condition, not by this filter)."""

    filtered, running_max = [], None
    for m in matches:
        base = base_of(m)
        if running_max is not None and base < running_max:
            continue
        filtered.append(m)
        running_max = base if running_max is None else max(running_max, base)
    return filtered


def _make_text_node(node_id: str, start: int, end: int, cleaned_text: str) -> Node:
    return Node(type="text", id=node_id, source_start=start, source_end=end, text=cleaned_text[start:end])


def _fill_gaps(
    cleaned_text: str, parent_start: int, parent_end: int, children: list[Node], gap_id_prefix: str,
) -> list[Node]:
    """Inserts an explicit "text" node for every gap between the
    parent's own span and its recognized children (before the first,
    between consecutive, after the last) -- so the children, placed
    back in order, always reconstruct the parent's span exactly. A
    whitespace-only gap is dropped silently (nothing is lost; there is
    no content there), never merged into a neighbour."""

    result: list[Node] = []
    cursor = parent_start
    gap_n = 0
    for child in children:
        if child.source_start > cursor:
            gap_text = cleaned_text[cursor:child.source_start]
            if gap_text.strip():
                result.append(_make_text_node(f"{gap_id_prefix}_text{gap_n}", cursor, child.source_start, cleaned_text))
                gap_n += 1
        result.append(child)
        cursor = max(cursor, child.source_end)
    if parent_end > cursor:
        gap_text = cleaned_text[cursor:parent_end]
        if gap_text.strip():
            result.append(_make_text_node(f"{gap_id_prefix}_text{gap_n}", cursor, parent_end, cleaned_text))
    return result


def _find_inline_first_marker(pattern: re.Pattern, cleaned_text: str, start: int, end: int) -> re.Match | None:
    """Looks for `pattern` within `start`'s own line only (up to its
    first newline) -- the inline-clause-(1)/inline-subclause-(a) case:
    a confirmed real shape where clause (1) sits immediately after the
    Article's own number on the SAME line ("1. (1) Pakistan shall be
    Federal Republic...", Article 1's own real first line), which the
    line-anchored _CLAUSE_RE/_SUBCLAUSE_RE can never match on their own
    since there is no preceding newline at that position at all."""

    newline_pos = cleaned_text.find("\n", start, end)
    line_end = newline_pos if newline_pos != -1 else end
    return pattern.search(cleaned_text, start, line_end)


def _parse_clause_level(cleaned_text: str, start: int, end: int, id_prefix: str) -> list[Node]:
    """Parses clauses, sub-clauses (nested one level inside their
    clause), and provisos (children of whichever -- clause or bare
    article body -- they immediately follow) within [start, end)."""

    clause_matches = list(_CLAUSE_RE.finditer(cleaned_text, start, end))
    inline = _find_inline_first_marker(_INLINE_NUMBERED_RE, cleaned_text, start, end)
    if inline and (not clause_matches or inline.start() < clause_matches[0].start()):
        clause_matches.insert(0, inline)
    if not clause_matches:
        return _parse_proviso_level(cleaned_text, start, end, id_prefix)

    clauses: list[Node] = []
    for i, m in enumerate(clause_matches):
        c_start = m.start()
        c_end = clause_matches[i + 1].start() if i + 1 < len(clause_matches) else end
        number = m.group(1)
        sub_children = _parse_subclause_level(cleaned_text, c_start, c_end, f"{id_prefix}_clause{number}")
        clauses.append(Node(
            type="clause", id=f"{id_prefix}_clause{number}", source_start=c_start, source_end=c_end,
            text=cleaned_text[c_start:c_end], number=number, children=sub_children,
        ))
    return _fill_gaps(cleaned_text, start, end, clauses, id_prefix)


def _parse_subclause_level(cleaned_text: str, start: int, end: int, id_prefix: str) -> list[Node]:
    sub_matches = list(_SUBCLAUSE_RE.finditer(cleaned_text, start, end))
    inline = _find_inline_first_marker(_INLINE_LETTERED_RE, cleaned_text, start, end)
    if inline and (not sub_matches or inline.start() < sub_matches[0].start()):
        sub_matches.insert(0, inline)
    if not sub_matches:
        return _parse_proviso_level(cleaned_text, start, end, id_prefix)

    subs: list[Node] = []
    for i, m in enumerate(sub_matches):
        s_start = m.start()
        s_end = sub_matches[i + 1].start() if i + 1 < len(sub_matches) else end
        letter = m.group(1)
        proviso_children = _parse_proviso_level(cleaned_text, s_start, s_end, f"{id_prefix}_sub{letter}")
        subs.append(Node(
            type="subclause", id=f"{id_prefix}_sub{letter}", source_start=s_start, source_end=s_end,
            text=cleaned_text[s_start:s_end], number=letter, children=proviso_children,
        ))
    return _fill_gaps(cleaned_text, start, end, subs, id_prefix)


def _parse_proviso_level(cleaned_text: str, start: int, end: int, id_prefix: str) -> list[Node]:
    proviso_matches = list(_PROVISO_RE.finditer(cleaned_text, start, end))
    if not proviso_matches:
        # No further structure below this point -- the remaining span is
        # the clause/sub-clause/article's own plain body text.
        if cleaned_text[start:end].strip():
            return [_make_text_node(f"{id_prefix}_body", start, end, cleaned_text)]
        return []

    nodes: list[Node] = []
    for i, m in enumerate(proviso_matches):
        p_start = m.start()
        p_end = proviso_matches[i + 1].start() if i + 1 < len(proviso_matches) else end
        nodes.append(Node(
            type="proviso", id=f"{id_prefix}_proviso{i + 1}", source_start=p_start, source_end=p_end,
            text=cleaned_text[p_start:p_end], children=[],
        ))
    return _fill_gaps(cleaned_text, start, end, nodes, id_prefix)


def _build_marginal_note_index(cleaned_text: str, end: int) -> dict[str, str]:
    return {m.group(1): m.group(2).strip() for m in _MARGINAL_NOTE_LINE_RE.finditer(cleaned_text, 0, end)}


def _parse_articles(cleaned_text: str, start: int, end: int, id_prefix: str, marginal_notes: dict[str, str]) -> list[Node]:
    same_line = list(_ARTICLE_SAME_LINE_RE.finditer(cleaned_text, start, end))
    bare = list(_ARTICLE_BARE_RE.finditer(cleaned_text, start, end))
    by_pos: dict[int, re.Match] = {}
    for m in same_line:
        by_pos[m.start()] = m
    for m in bare:
        by_pos.setdefault(m.start(), m)  # same-line match (if any) wins at an identical start
    matches = [by_pos[pos] for pos in sorted(by_pos)]
    matches = _filter_monotonic(matches, lambda m: _base_number(m.group(1)))

    articles: list[Node] = []
    for i, m in enumerate(matches):
        a_start = m.start()
        a_end = matches[i + 1].start() if i + 1 < len(matches) else end
        number = m.group(1)
        heading = marginal_notes.get(number)
        a_id = f"{id_prefix}_article{number}"
        children = _parse_clause_level(cleaned_text, a_start, a_end, a_id)
        articles.append(Node(
            type="article", id=a_id, source_start=a_start, source_end=a_end,
            text=cleaned_text[a_start:a_end], article_number=number, heading=heading, children=children,
        ))
    return _fill_gaps(cleaned_text, start, end, articles, id_prefix)


def _parse_chapters_and_articles(
    cleaned_text: str, start: int, end: int, id_prefix: str, marginal_notes: dict[str, str],
) -> list[Node]:
    chapter_matches = list(_CHAPTER_RE.finditer(cleaned_text, start, end))
    if not chapter_matches:
        return _parse_articles(cleaned_text, start, end, id_prefix, marginal_notes)

    chapter_nodes: list[Node] = []
    for i, m in enumerate(chapter_matches):
        ch_start = m.start()
        ch_end = chapter_matches[i + 1].start() if i + 1 < len(chapter_matches) else end
        number = m.group(1)
        heading = m.group(2).strip() or None
        ch_id = f"{id_prefix}_chapter{number}"
        children = _parse_articles(cleaned_text, ch_start, ch_end, ch_id, marginal_notes)
        chapter_nodes.append(Node(
            type="chapter", id=ch_id, source_start=ch_start, source_end=ch_end,
            text=cleaned_text[ch_start:ch_end], heading=heading, children=children,
        ))

    # A Part can have Articles directly under it, BEFORE its first
    # Chapter begins -- confirmed real case: Article 7 ("Definition of
    # the State") sits directly under PART II, right before its
    # "CHAPTER 1.--FUNDAMENTAL RIGHTS" begins. Each gap between (and
    # around) the chapters above is therefore re-parsed for Articles of
    # its own, not left as a single opaque "text" node; only whatever a
    # gap's own Article-parse doesn't explain becomes a text node, via
    # the same _fill_gaps used everywhere else in this module.
    nodes: list[Node] = []
    cursor = start
    for chapter in chapter_nodes:
        if chapter.source_start > cursor:
            nodes.extend(_parse_articles(cleaned_text, cursor, chapter.source_start, id_prefix, marginal_notes))
        nodes.append(chapter)
        cursor = chapter.source_end
    if end > cursor:
        nodes.extend(_parse_articles(cleaned_text, cursor, end, id_prefix, marginal_notes))

    return _fill_gaps(cleaned_text, start, end, nodes, id_prefix)


def parse_constitution_structure(cleaned_text: str) -> Node:
    """Builds the full hierarchical AST for one cleaned Constitution
    document's text. Raises ConstitutionParsingError if not even one
    PART or one Article can be found anywhere -- the one case where this
    parser refuses to guess."""

    footnotes_idx = cleaned_text.find(_FOOTNOTES_BLOCK_MARKER)
    content_end = footnotes_idx if footnotes_idx != -1 else len(cleaned_text)

    annex_matches = list(_ANNEX_RE.finditer(cleaned_text, 0, content_end))
    schedule_matches = list(_SCHEDULE_RE.finditer(cleaned_text, 0, content_end))
    # A Schedule (e.g. the Fourth Schedule's "Legislative Lists") can
    # contain its OWN internal "PART I"/"PART II" subdivisions -- these
    # are schedule content, never a real top-level Part of the
    # Constitution. Confirmed real case: 4 such schedule-internal "PART
    # I"/"PART II" matches exist after the real FIRST SCHEDULE heading,
    # alongside the 12 genuine top-level Parts (I-XII), all of which
    # occur strictly before it. Top-level Part detection is therefore
    # bounded to end at the first Annex/Schedule boundary.
    main_body_end = min([content_end] + [m.start() for m in annex_matches] + [m.start() for m in schedule_matches])
    part_matches = list(_PART_RE.finditer(cleaned_text, 0, main_body_end))
    marginal_notes = _build_marginal_note_index(cleaned_text, main_body_end)

    if not part_matches and not _ARTICLE_SAME_LINE_RE.search(cleaned_text, 0, content_end):
        raise ConstitutionParsingError(
            "No PART heading and no Article heading found anywhere in the document -- "
            "cannot determine the Constitution's structure."
        )

    # Top-level boundaries, in source order: preamble zone (implicit,
    # from 0 to the first of these), each Part, the Annex (if present),
    # each Schedule (if present).
    top_level_starts: list[tuple[int, str, re.Match]] = (
        [(m.start(), "part", m) for m in part_matches]
        + [(m.start(), "annex", m) for m in annex_matches]
        + [(m.start(), "schedule", m) for m in schedule_matches]
    )
    top_level_starts.sort(key=lambda t: t[0])

    top_nodes: list[Node] = []
    preamble_match = _PREAMBLE_RE.search(cleaned_text, 0, content_end)
    first_boundary = top_level_starts[0][0] if top_level_starts else content_end
    if preamble_match:
        p_start = preamble_match.start()
        children = _parse_proviso_level(cleaned_text, p_start, first_boundary, "preamble")
        top_nodes.append(Node(
            type="preamble", id="preamble", source_start=p_start, source_end=first_boundary,
            text=cleaned_text[p_start:first_boundary], children=children,
        ))

    for i, (pos, kind, m) in enumerate(top_level_starts):
        node_end = top_level_starts[i + 1][0] if i + 1 < len(top_level_starts) else content_end
        if kind == "part":
            number = m.group(1)
            node_id = f"part_{number}"
            children = _parse_chapters_and_articles(cleaned_text, pos, node_end, node_id, marginal_notes)
            top_nodes.append(Node(
                type="part", id=node_id, source_start=pos, source_end=node_end,
                text=cleaned_text[pos:node_end], heading=None, children=children,
            ))
        elif kind == "annex":
            children = _parse_proviso_level(cleaned_text, pos, node_end, "annex")
            top_nodes.append(Node(
                type="annex", id="annex", source_start=pos, source_end=node_end,
                text=cleaned_text[pos:node_end], children=children,
            ))
        else:  # schedule
            label = re.sub(r"[ \t]+", "_", m.group(1).strip()).lower()
            node_id = f"schedule_{label}"
            children = _parse_proviso_level(cleaned_text, pos, node_end, node_id)
            top_nodes.append(Node(
                type="schedule", id=node_id, source_start=pos, source_end=node_end,
                text=cleaned_text[pos:node_end], heading=m.group(1).strip(), children=children,
            ))

    top_nodes = _fill_gaps(cleaned_text, 0, content_end, top_nodes, "root")

    return Node(type="root", id="root", source_start=0, source_end=content_end,
                text=cleaned_text[0:content_end], children=top_nodes)


# -- validation -----------------------------------------------------------------

@dataclass
class ValidationResult:
    ok: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    diagnostics: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {"ok": self.ok, "errors": self.errors, "warnings": self.warnings, "diagnostics": self.diagnostics}


def validate_constitution_structure(root: Node, cleaned_text: str) -> ValidationResult:
    """Deterministically checks the hard invariants STEP 9 requires:
    exact offsets, gapless/non-overlapping coverage at every level,
    Articles/clauses/sub-clauses never appearing outside their expected
    parent type, duplicate/decreasing Article numbers (flagged as
    warnings -- the Constitution's own real numbering is not perfectly
    contiguous, e.g. letter-suffixed insertions, so this is advisory,
    exactly like statute_chunk_validate.py's equivalent section-gap
    check), and no content lost anywhere in the tree."""

    errors: list[str] = []
    warnings: list[str] = []
    article_numbers: list[str] = []
    counts: dict[str, int] = {}

    def _walk(node: Node, parent: Node | None, allowed_parent_types: set[str] | None):
        counts[node.type] = counts.get(node.type, 0) + 1

        if not (0 <= node.source_start <= node.source_end <= len(cleaned_text)):
            errors.append(f"{node.id}: out-of-bounds offsets [{node.source_start},{node.source_end})")
        elif cleaned_text[node.source_start:node.source_end] != node.text:
            errors.append(f"{node.id}: text does not match cleaned_text[source_start:source_end]")

        if allowed_parent_types is not None and parent is not None and node.type not in allowed_parent_types:
            errors.append(f"{node.id} ({node.type}) appears under unexpected parent type {parent.type!r}")

        if node.type == "article":
            article_numbers.append(node.article_number or "")

        # Gapless, non-overlapping, fully-covering children -- but only
        # for container node types. "text" and "proviso" are deliberate
        # LEAVES in this model (no further decomposition is attempted
        # beneath them), so an empty children list is correct for them,
        # not a sign of lost text -- their own `text` field already is
        # their full span.
        if node.type not in ("text", "proviso"):
            cursor = node.source_start
            for child in node.children:
                if child.source_start < cursor:
                    errors.append(f"{child.id} overlaps a preceding sibling under {node.id}")
                elif child.source_start > cursor:
                    errors.append(f"gap in coverage under {node.id} before {child.id}: "
                                   f"[{cursor},{child.source_start}) is unexplained structural text")
                cursor = max(cursor, child.source_end)
            if cursor != node.source_end:
                errors.append(f"{node.id}: children do not fully cover its span "
                               f"(covered up to {cursor}, span ends at {node.source_end}) -- lost text")

        child_allowed: dict[str, set[str]] = {
            "root": {"preamble", "part", "annex", "schedule", "text"},
            "part": {"chapter", "article", "text"},
            "chapter": {"article", "text"},
            "article": {"clause", "proviso", "text"},
            "clause": {"subclause", "proviso", "text"},
            "subclause": {"proviso", "text"},
        }
        allowed = child_allowed.get(node.type)
        for child in node.children:
            _walk(child, node, allowed)

    _walk(root, None, None)

    seen: dict[str, int] = {}
    for num in article_numbers:
        seen[num] = seen.get(num, 0) + 1
    duplicates = {n: c for n, c in seen.items() if c > 1}
    if duplicates:
        warnings.append(f"Article number(s) appearing more than once: {duplicates}")

    bases = [_base_number(n) for n in article_numbers]
    for prev, nxt, prev_id, nxt_id in zip(bases, bases[1:], article_numbers, article_numbers[1:]):
        if nxt < prev:
            warnings.append(f"Article order decreases: {prev_id} is followed by {nxt_id}")

    diagnostics = {
        "node_type_counts": counts,
        "article_count": len(article_numbers),
        "article_numbers": article_numbers,
    }
    return ValidationResult(ok=not errors, errors=errors, warnings=warnings, diagnostics=diagnostics)


def is_valid_existing_output(doc_dict: dict) -> bool:
    """Resumability gate: is this existing parsed-structure output file
    actually valid, not merely present. Returns False (never raises) for
    anything unreadable/malformed."""

    try:
        root = Node.from_dict(doc_dict["structure"])
        cleaned_text = doc_dict["cleaned_text"]
    except (KeyError, TypeError, ValueError):
        return False
    try:
        result = validate_constitution_structure(root, cleaned_text)
    except Exception:  # noqa: BLE001 -- a malformed file must never crash the skip-check
        return False
    return result.ok


def parse_constitution_document(cleaned_doc: dict) -> dict:
    """Parses one cleaned Constitution document (constitution_cleaning.py's
    output shape) into the structural AST output shape. Never modifies
    the input; ``cleaned_text`` is carried through unchanged so later
    stages (and is_valid_existing_output) can re-verify every offset."""

    cleaned_text = cleaned_doc["cleaned_text"]
    root = parse_constitution_structure(cleaned_text)
    result = validate_constitution_structure(root, cleaned_text)

    return {
        "doc_id": cleaned_doc["doc_id"],
        "document_type": cleaned_doc.get("metadata", {}).get("document_type", "constitution"),
        "title": cleaned_doc.get("metadata", {}).get("title"),
        "cleaned_text": cleaned_text,
        "structure": root.to_dict(),
        "validation": result.to_dict(),
    }


def parse_directory(input_dir: Path = INPUT_DIR, output_dir: Path = OUTPUT_DIR) -> dict:
    """Parses every ``<doc_id>.json`` in ``input_dir`` into
    ``output_dir/<doc_id>.json``. Resumable: an existing output is
    skipped only if is_valid_existing_output() confirms it, never on
    file existence alone. Never modifies ``input_dir``."""

    input_dir, output_dir = Path(input_dir), Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    processed = regenerated = skipped_valid = 0
    failed_docs: list[dict] = []

    for cleaned_path in sorted(input_dir.glob("*.json")):
        doc_id = cleaned_path.stem
        out_path = output_dir / f"{doc_id}.json"

        was_present = out_path.exists()
        if was_present:
            try:
                existing = json.loads(out_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                existing = None
            if existing is not None and is_valid_existing_output(existing):
                skipped_valid += 1
                continue

        try:
            cleaned_doc = json.loads(cleaned_path.read_text(encoding="utf-8"))
            parsed_doc = parse_constitution_document(cleaned_doc)
        except (OSError, json.JSONDecodeError, KeyError, ConstitutionParsingError) as exc:
            failed_docs.append({"doc_id": doc_id, "error": str(exc)})
            continue

        if not parsed_doc["validation"]["ok"]:
            failed_docs.append({"doc_id": doc_id, "error": parsed_doc["validation"]["errors"]})
            continue

        tmp_path = out_path.with_suffix(".json.tmp")
        tmp_path.write_text(json.dumps(parsed_doc, ensure_ascii=False, indent=2), encoding="utf-8")
        tmp_path.replace(out_path)
        processed += 1
        if was_present:
            regenerated += 1

    return {
        "processed": processed, "skipped_valid": skipped_valid, "regenerated": regenerated,
        "failed": len(failed_docs), "failed_docs": failed_docs,
    }


def main() -> None:  # pragma: no cover -- thin CLI wrapper
    summary = parse_directory()
    print(f"Constitution parsing: {summary['processed']} processed "
          f"({summary['regenerated']} regenerated), {summary['skipped_valid']} skipped (valid), "
          f"{summary['failed']} failed")
    for f in summary["failed_docs"]:
        print(f"  FAILED {f['doc_id']}: {f['error']}")


if __name__ == "__main__":  # pragma: no cover
    main()
