"""Section-aware statute chunking -- Stage 2 of the independent statute
pipeline.

    cleaned statute JSON (var/rag/cleaned_statutes/<doc_id>.json)
        -> parse_statute_structure()   [preamble / sections / schedules / footnotes, by regex anchor]
        -> build_statute_chunks()      [one chunk per section/definitions/preamble/schedule-part/footnotes]
        -> chunked statute JSON        (var/rag/statutes_chunked/<doc_id>.json)

Unlike case-law chunking (src/rag_prep/chunker.py -- paragraph-grouping
toward a word-count target, because a judgment's prose has no reliable
atomic legal unit smaller than "paragraph"), a statute's atomic retrieval
unit is the SECTION: a numbered section (with all its subsections,
clauses, provisos, explanations, and attached amendment footnotes) is
never split, and never merged with another section, regardless of
length. This module is completely independent of chunker.py/
chunk_types.py/chunk_validate.py and never imports them.

Design was validated directly against the 9 real cleaned statutes in
var/rag/cleaned_statutes/ before being written (not assumed) -- see the
regex comments below for the specific real-document shapes each pattern
was built to handle (e.g. "25-A" vs "21A" section-ID forms both existing
in the same real Act; a SCHEDULE with "Part I"/"PART II" subdivisions;
a schedule whose entire content is "[ENACTMENTS REPEALED.]").

PIPELINE ISOLATION
-------------------
No import of src.rag_prep.chunker/chunk_types/chunk_validate (case-law
chunking), src.rag_prep.case_cleaner/structurer*/statute_cleaner (the
case-law citation extractor), src.ingestion, or src.classification.
Independent entry point, independent output directory. Never touches
the case-law pipeline's var/rag/ directories or var/metadata.db, and
never processes Constitution documents (out of scope; a separate future
pipeline).
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

INPUT_DIR = Path("var/rag/cleaned_statutes")
OUTPUT_DIR = Path("var/rag/statutes_chunked")

INHERITABLE_METADATA_FIELDS = (
    "law_name", "year", "document_type", "domain", "domain_source", "source_file",
)


class StatuteChunkingError(ValueError):
    """Raised when a structural boundary cannot be determined confidently
    -- e.g. no section header found at all. Per the task's explicit
    "do not silently generate potentially incorrect offsets" rule: this
    is raised rather than guessing where sections start."""


# -- section headers --------------------------------------------------------------
# "1. Short title and extent.__(1) This Act..." / "6A. Matters pertaining to..."
# / "25-A. Transfer of cases" (both the plain "21A" and hyphenated "25-A"
# alphanumeric-suffix forms are real, observed in the same real Act side
# by side). Anchored at the start of a line; requires a literal period
# right after the id so amendment-footnote lines like "1Rep. by..." or
# "2Ins. by..." (digit directly prefixing a word, no period) never match.
#
# The leading "(?:\d+\[)?" is an OPTIONAL amendment-insertion prefix --
# "2[9. No Court shall take cognizance..." is how a section that was
# wholly INSERTED or SUBSTITUTED by a later amendment is rendered (the
# "2[" marks the amendment; "9." is the section's own real number).
# Confirmed as a real, currently-missing section in the real corpus: doc
# 470c476c4195cbc08e94a6da's chunk list jumped straight from section 8 to
# section 10 before this fix, because "2[9. No Court shall..." didn't
# match at all and was silently swallowed into section 8's body -- the
# exact "parser-swallowed section" failure mode this prefix exists to
# prevent. It cannot collide with an amendment marker on a SUBSECTION/
# CLAUSE/PROVISO/PHRASE ("1[(2) It extends...", "5[Provided that...",
# "4[or is about to be]", all real, all left alone) because the group
# right after the prefix must itself be a bare digit run -- any of those
# four real forms has "(" or a letter there instead, so the whole
# alternative simply fails to match and the line is correctly ignored.
#
# The leading "[ \t]*" tolerates a single stray leading space/tab before
# the header -- a real PDF-extraction artifact, confirmed causing an
# actual swallowed section in two different real documents (" 1[19.
# Court fee..." in doc 24520c344d2285555aa10988/its duplicate, and
# " 29. Discharge or variation of orders..." / " 62. Power to make
# rules..." in doc 8251b5f4daa92e5a22319fc4 -- the latter two meant
# section 62, the Act's LAST section, was silently absent altogether).
# Confirmed safe to allow generally (not just for the amendment-prefixed
# branch): a full sweep of all 9 real cleaned statutes for ANY
# leading-whitespace + "<digits>[suffix]." line found only these four
# genuine, swallowed section headers and zero false positives (e.g. no
# indented numbered sub-list item anywhere takes this exact shape).
#
# NOTE -- a further, real artifact is known and deliberately NOT fixed
# here: a bare "<id>." with its title pushed to the next line entirely
# ("18.\nAppearance through agents...") swallows 3 sections across 2
# real documents. Relaxing the trailing "[ \t]+(.+)$" to allow an empty
# same-line title was tried and reverted -- it also matched a bare
# citation-year line ("1882.") as a bogus "section 1882" in a different
# real document, which is worse than the 3 known swallows it fixed.
# Per "prefer a narrow documented exception over an unsafe broad regex":
# left as a known gap, surfaced via statute_chunk_validate.py's
# non-blocking "possible swallowed section(s)" warning rather than
# patched with a regex that trades one real bug for another.
_SECTION_HEADER_RE = re.compile(r"^[ \t]*(?:\d+\[)?(\d+-?[A-Za-z]{0,2})\.[ \t]+(.+)$", re.MULTILINE)

# A section's TITLE is the text up to the first sentence-ending period
# (statute section titles never contain an internal period); whatever
# follows -- including a "__" extraction artifact for an em-dash, or the
# body text directly -- is the section's substantive content, not part
# of the title.
_TITLE_BODY_SPLIT_RE = re.compile(r"^(.*?)\.(?:__)?\s*(.*)$")

# Matches only when the WHOLE section title is "Definitions" (trailing
# punctuation aside) -- not merely containing the word, which would
# wrongly tag something like a later "Power to amend definitions" section.
_DEFINITIONS_TITLE_RE = re.compile(r"^definitions?[\s.]*$", re.IGNORECASE)

# A standalone structural SCHEDULE heading -- deliberately case-SENSITIVE
# (all-caps "SCHEDULE"/"THE SCHEDULE" is the real heading convention
# observed; an inline cross-reference like "and Schedule V" or "s. 156
# and Schedule." uses Title-case "Schedule" and must NOT match).
_SCHEDULE_HEADING_RE = re.compile(r"^(?:THE\s+)?SCHEDULE\b.*$", re.MULTILINE)
_SCHEDULE_PART_RE = re.compile(r"^\d*\[?PART\s+([IVXLCDM]+)\]?", re.MULTILINE | re.IGNORECASE)

# -- cross-reference extraction ---------------------------------------------------
# "Section 9(1)(b)", "section 2", "Sections 5 and 7", "Section 17A" --
# captures the whole reference phrase (possibly a list) as one match;
# the phrase is split on ","/"and" afterward to emit one entry per
# section id, all sharing the same source_text (the literal phrase that
# produced them).
_SECTION_REF_PHRASE_RE = re.compile(
    r"\bSections?\s+"
    r"(\d+-?[A-Za-z]{0,2}(?:\s*\(\s*\d+\s*\))?(?:\s*\(\s*[a-z]\s*\))?"
    r"(?:\s*(?:,|and)\s*(?:and\s+)?\d+-?[A-Za-z]{0,2}"
    r"(?:\s*\(\s*\d+\s*\))?(?:\s*\(\s*[a-z]\s*\))?)*)",
    re.IGNORECASE,
)
_SECTION_REF_ITEM_RE = re.compile(
    r"^(\d+-?[A-Za-z]{0,2})(?:\s*\(\s*(\d+)\s*\))?(?:\s*\(\s*([a-z])\s*\))?$"
)
_SECTION_REF_SPLIT_RE = re.compile(r"\s*(?:,|and)\s*", re.IGNORECASE)

# "Guardians and Wards Act, 1890", "the Code of Civil Procedure, 1908" --
# name anchored on the Act/Ordinance/Code keyword, year a plain trailing
# "<keyword>, <YYYY>" (the dominant citation style actually observed in
# these statutes' own cross-references, as opposed to case-law's
# "(<N> of <YYYY>)" parenthetical style -- both anchor on the same
# keyword, so a parenthetical citation immediately following is simply
# left as part of the surrounding text, not separately parsed/duplicated).
_EXTERNAL_STATUTE_RE = re.compile(
    r"\b([A-Z][A-Za-z.,&'\-]*(?:[ \t]+[A-Za-z.,&'\-]+){0,6}?[ \t]+(?:Act|Ordinance|Code))"
    r",?[ \t]+(\d{4})\b"
)
# Common abbreviations the task explicitly asks to support -- a fixed,
# small, non-invented map (never guessed beyond this list).
_KNOWN_ABBREVIATIONS = {
    "CPC": "Code of Civil Procedure",
    "PPC": "Pakistan Penal Code",
    "CRPC": "Code of Criminal Procedure",
}
_ABBREVIATION_RE = re.compile(r"\b(CPC|PPC|CrPC)[ \t]+(\d{4})\b", re.IGNORECASE)

# A "Section N" (or list of them) immediately followed by "of <External
# Act Name>, <year>" belongs to the EXTERNAL statute being cited, not to
# the current one -- "section 25 of the Guardians and Wards Act, 1890"
# must never become an internal reference to *this* statute's own
# section 25. Matched as a short, fixed connective phrase right after
# the section-reference phrase; the external statute itself is still
# picked up separately by _EXTERNAL_STATUTE_RE/_ABBREVIATION_RE over the
# whole text, so nothing is lost -- only the wrong internal pairing is
# suppressed. A genuine self-reference ("of this Act", "of the said
# Act") is never affected: _EXTERNAL_STATUTE_RE requires a capitalized
# proper-noun-style name, which "this"/"the said" never satisfies.
_SECTION_REF_EXTERNAL_FOLLOWUP_RE = re.compile(r"[ \t]+of[ \t]+(?:the[ \t]+)?")

# -- footnotes block ----------------------------------------------------------------
# statute_cleaning.py (Stage 1) relocates page-bottom amendment/footnote
# lines out of the substantive body and appends them, once, as this
# exact literal block at the very end of cleaned_text. The chunker must
# carve this region out into its own dedicated chunk BEFORE section/
# schedule parsing even starts -- otherwise the last section (or last
# schedule, if present) would silently swallow it, which is precisely
# the "footnote contamination" failure mode statute_chunk_validate.py's
# audit checks for.
_FOOTNOTES_BLOCK_MARKER = "\n\n[FOOTNOTES]\n"

_NUMERIC_BRACKET_PREFIX_RE = re.compile(r"\d+\[([^\[\]]*)\]")


def _derive_embedding_text(raw_text: str) -> str:
    """Strips ONLY the numeric amendment-bracket prefix/suffix pairs
    ("2[Magistrate]" -> "Magistrate") from a chunk's raw_text, for later
    embedding. Never paraphrases, summarizes, or removes substantive
    wording -- every character inside the brackets survives untouched,
    only the "<digits>[" ... "]" wrapper itself is removed. Applied
    repeatedly (innermost-first) so a nested amendment -- a real,
    observed shape, e.g. "2[9. ... 3[except on a complaint made by the
    Union Council] ... ]" -- is fully unwrapped rather than leaving an
    outer bracket in place because an inner one was in the way."""

    text = raw_text
    while True:
        new_text = _NUMERIC_BRACKET_PREFIX_RE.sub(r"\1", text)
        if new_text == text:
            return new_text
        text = new_text


@dataclass(frozen=True)
class StatuteChunk:
    chunk_id: str
    parent_id: str
    doc_id: str
    chunk_index: int
    chunk_type: str  # "preamble" | "section" | "definitions" | "schedule" | "footnotes"
    section_id: str | None
    section_title: str | None
    schedule_id: str | None
    source_start: int
    source_end: int
    text: str  # exact slice: parent.cleaned_text[source_start:source_end] -- the "raw_text"
    embedding_text: str = ""  # derived from `text`; see _derive_embedding_text()
    references: dict = field(default_factory=lambda: {"internal_sections": [], "external_statutes": []})

    def to_dict(self) -> dict:
        return {
            "chunk_id": self.chunk_id, "parent_id": self.parent_id, "doc_id": self.doc_id,
            "chunk_index": self.chunk_index, "chunk_type": self.chunk_type,
            "section_id": self.section_id, "section_title": self.section_title,
            "schedule_id": self.schedule_id, "source_start": self.source_start,
            "source_end": self.source_end, "text": self.text,
            "embedding_text": self.embedding_text, "references": self.references,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "StatuteChunk":
        return cls(
            chunk_id=d["chunk_id"], parent_id=d["parent_id"], doc_id=d["doc_id"],
            chunk_index=d["chunk_index"], chunk_type=d["chunk_type"],
            section_id=d.get("section_id"), section_title=d.get("section_title"),
            schedule_id=d.get("schedule_id"), source_start=d["source_start"],
            source_end=d["source_end"], text=d["text"],
            embedding_text=d.get("embedding_text", d["text"]),
            references=d.get("references", {"internal_sections": [], "external_statutes": []}),
        )


@dataclass
class StatuteParent:
    doc_id: str
    metadata: dict
    cleaned_text: str
    chunks: list[StatuteChunk] = field(default_factory=list)
    preamble_present: bool = True

    def to_dict(self) -> dict:
        return {
            "doc_id": self.doc_id, "metadata": self.metadata, "cleaned_text": self.cleaned_text,
            "preamble_present": self.preamble_present,
            "chunks": [c.to_dict() for c in self.chunks],
        }

    @classmethod
    def from_dict(cls, d: dict) -> "StatuteParent":
        return cls(
            doc_id=d["doc_id"], metadata=d["metadata"], cleaned_text=d["cleaned_text"],
            preamble_present=d.get("preamble_present", True),
            chunks=[StatuteChunk.from_dict(c) for c in d["chunks"]],
        )


def _is_followed_by_external_statute_citation(text: str, pos: int) -> bool:
    fm = _SECTION_REF_EXTERNAL_FOLLOWUP_RE.match(text, pos)
    if not fm:
        return False
    start = fm.end()
    return bool(_EXTERNAL_STATUTE_RE.match(text, start) or _ABBREVIATION_RE.match(text, start))


def extract_references(text: str) -> dict:
    """Deterministic, regex-only cross-reference extraction -- no LLM.
    Deduplicates byte-identical ``source_text`` occurrences within this
    one chunk; keeps distinct wording as separate entries."""

    internal: list[dict] = []
    seen_internal: set[tuple] = set()
    for m in _SECTION_REF_PHRASE_RE.finditer(text):
        if _is_followed_by_external_statute_citation(text, m.end()):
            continue  # "section 25 OF the Guardians and Wards Act, 1890" -- not internal
        source_text = m.group(0)
        pieces = _SECTION_REF_SPLIT_RE.split(m.group(1))
        for piece in pieces:
            item = _SECTION_REF_ITEM_RE.match(piece.strip())
            if not item:
                continue
            key = (item.group(1), item.group(2), item.group(3), source_text)
            if key in seen_internal:
                continue
            seen_internal.add(key)
            internal.append({
                "section_id": item.group(1), "subsection": item.group(2),
                "clause": item.group(3), "source_text": source_text,
            })

    external: list[dict] = []
    seen_external: set[str] = set()
    for m in _EXTERNAL_STATUTE_RE.finditer(text):
        source_text = m.group(0).strip()
        if source_text in seen_external:
            continue
        seen_external.add(source_text)
        name = re.sub(r"\s+", " ", m.group(1)).strip().rstrip(",")
        external.append({"name": name, "year": int(m.group(2)), "source_text": source_text})
    for m in _ABBREVIATION_RE.finditer(text):
        source_text = m.group(0).strip()
        if source_text in seen_external:
            continue
        seen_external.add(source_text)
        external.append({
            "name": _KNOWN_ABBREVIATIONS[m.group(1).upper()], "year": int(m.group(2)),
            "source_text": source_text,
        })

    return {"internal_sections": internal, "external_statutes": external}


def _split_title_body(raw_rest: str) -> str:
    m = _TITLE_BODY_SPLIT_RE.match(raw_rest)
    return m.group(1).strip() if m else raw_rest.strip()


def _base_section_number(section_id: str) -> int:
    return int(re.match(r"\d+", section_id).group())


def _filter_amendment_footnotes_by_sequence(raw_matches: list) -> list:
    """Drops a section-header-SHAPED match whose number regresses below
    the highest genuine section number already confirmed earlier in the
    document.

    This is a purely structural rule, not a vocabulary one: real statute
    section numbers are always non-decreasing in source order (1, 2, 3,
    ..., 47, even across alphanumeric suffixes like 6, 6A, 6B, 7). An
    amendment/annotation footnote embedded inside a later section's body
    (e.g. "2. The word \"District\" rep. by s. 4 of ...", explaining a
    footnote marker that appeared earlier IN SECTION 47's text) happens
    to share the "<number>. " shape, but its number is strictly smaller
    than the running maximum -- confirmed directly against the real
    document that exposed this (doc 42769a60dc59c5473affbc32: sections
    1 through 47 appear in strict increasing order, then a "2." shows up
    at position 42123, deep inside section 47's body).

    A vocabulary-based rule (rejecting "rep."/"subs."/"ins."/"repealed")
    was deliberately NOT used: a genuinely repealed section is real and
    common (e.g. "6. [Repealed]." in another real document, and
    "2. [Repeal.] Rep. by the Repealing Act, 1938..." in THIS SAME
    document, at its correct, forward position as the real Section 2) --
    only the out-of-sequence POSITION distinguishes the footnote from a
    genuine section, never the words it contains.
    """

    filtered = []
    running_max_base: int | None = None
    for m in raw_matches:
        base = _base_section_number(m.group(1))
        if running_max_base is not None and base < running_max_base:
            continue  # amendment-footnote/annotation noise, not a section boundary
        filtered.append(m)
        running_max_base = base if running_max_base is None else max(running_max_base, base)
    return filtered


def parse_statute_structure(cleaned_text: str) -> dict:
    """Locates (never slices yet) the preamble, each section, each
    schedule/schedule-part, and the relocated footnotes block (if any)
    in ``cleaned_text``, as half-open character ranges. Raises
    StatuteChunkingError if no section header can be found at all --
    the one case where this parser refuses to guess."""

    footnotes_region_start = cleaned_text.find(_FOOTNOTES_BLOCK_MARKER)
    content_end = footnotes_region_start if footnotes_region_start != -1 else len(cleaned_text)

    # Sections/schedules are only ever searched for within the real
    # document content -- never inside the relocated footnotes block,
    # which is plain citation prose that could otherwise coincidentally
    # resemble a section/schedule heading.
    schedule_heading_matches = list(_SCHEDULE_HEADING_RE.finditer(cleaned_text, 0, content_end))
    schedule_region_start = schedule_heading_matches[0].start() if schedule_heading_matches else content_end

    body_text = cleaned_text[:schedule_region_start]
    raw_section_matches = list(_SECTION_HEADER_RE.finditer(body_text))
    section_matches = _filter_amendment_footnotes_by_sequence(raw_section_matches)
    if not section_matches:
        raise StatuteChunkingError(
            "No section header pattern (e.g. '1. Short title...') found anywhere in the "
            "document body -- cannot determine where the preamble ends and sections begin."
        )

    preamble_end = section_matches[0].start()

    sections: list[dict] = []
    for i, m in enumerate(section_matches):
        section_id = m.group(1)
        end = section_matches[i + 1].start() if i + 1 < len(section_matches) else len(body_text)
        title = _split_title_body(m.group(2))
        sections.append({
            "section_id": section_id,
            "section_title": title,
            "start": m.start(),
            "end": end,
            "is_definitions": bool(_DEFINITIONS_TITLE_RE.search(title)),
        })

    schedules: list[dict] = []
    for s_i, sh in enumerate(schedule_heading_matches):
        schedule_end = (
            schedule_heading_matches[s_i + 1].start()
            if s_i + 1 < len(schedule_heading_matches) else content_end
        )
        schedule_span = cleaned_text[sh.start():schedule_end]
        part_matches = list(_SCHEDULE_PART_RE.finditer(schedule_span))
        label_prefix = f"Schedule {s_i + 1} - " if len(schedule_heading_matches) > 1 else ""
        if not part_matches:
            schedules.append({
                "schedule_id": (label_prefix.rstrip(" -") or "Schedule"),
                "start": sh.start(), "end": schedule_end,
            })
        else:
            for p_i, pm in enumerate(part_matches):
                p_start = sh.start() + pm.start()
                p_end = sh.start() + (part_matches[p_i + 1].start() if p_i + 1 < len(part_matches) else len(schedule_span))
                schedules.append({
                    "schedule_id": f"{label_prefix}Part {pm.group(1)}",
                    "start": p_start, "end": p_end,
                })
            # Any schedule text BEFORE its first Part marker (e.g. a
            # "[see SECTION 5]" reference line right under the heading)
            # belongs to that schedule too -- prepend it to the first part
            # by widening that part's own start back to the heading.
            schedules[-len(part_matches)]["start"] = sh.start()

    footnotes = (
        {"start": footnotes_region_start, "end": len(cleaned_text)}
        if footnotes_region_start != -1 else None
    )

    return {
        "preamble": {"start": 0, "end": preamble_end},
        "sections": sections,
        "schedules": schedules,
        "footnotes": footnotes,
    }


def build_statute_chunks(doc: dict) -> StatuteParent:
    """Builds the full, ordered, linked chunk set for one cleaned statute
    document (the statute_cleaning.py output shape: doc_id, metadata,
    cleaned_text, ...). Chunk order: preamble -> sections (source order)
    -> schedules/parts (source order) -> footnotes (if any) -- matching
    the document's own physical layout, which is already legal/source
    order (the relocated footnotes block, when present, is always the
    very last thing in cleaned_text -- see statute_cleaning.py)."""

    doc_id = doc["doc_id"]
    cleaned_text = doc["cleaned_text"]
    structure = parse_statute_structure(cleaned_text)

    chunks: list[StatuteChunk] = []
    idx = 0

    def _append(chunk_type, section_id, section_title, schedule_id, start, end):
        nonlocal idx
        text = cleaned_text[start:end]
        chunks.append(StatuteChunk(
            chunk_id=f"{doc_id}:{idx:04d}", parent_id=doc_id, doc_id=doc_id, chunk_index=idx,
            chunk_type=chunk_type, section_id=section_id, section_title=section_title,
            schedule_id=schedule_id, source_start=start, source_end=end, text=text,
            embedding_text=_derive_embedding_text(text), references=extract_references(text),
        ))
        idx += 1

    preamble_present = structure["preamble"]["end"] > structure["preamble"]["start"]
    if preamble_present:
        _append("preamble", None, None, None, structure["preamble"]["start"], structure["preamble"]["end"])

    for s in structure["sections"]:
        _append(
            "definitions" if s["is_definitions"] else "section",
            s["section_id"], s["section_title"], None, s["start"], s["end"],
        )

    for sch in structure["schedules"]:
        _append("schedule", None, None, sch["schedule_id"], sch["start"], sch["end"])

    if structure["footnotes"] is not None:
        _append("footnotes", None, None, None, structure["footnotes"]["start"], structure["footnotes"]["end"])

    return StatuteParent(
        doc_id=doc_id, metadata=doc["metadata"], cleaned_text=cleaned_text,
        chunks=chunks, preamble_present=preamble_present,
    )


def denormalize_chunk(chunk: StatuteChunk, parent: StatuteParent) -> dict:
    d = chunk.to_dict()
    for field_name in INHERITABLE_METADATA_FIELDS:
        d[field_name] = parent.metadata.get(field_name)
    return d


def chunk_directory(input_dir: Path = INPUT_DIR, output_dir: Path = OUTPUT_DIR) -> dict:
    """Chunks every ``<doc_id>.json`` in ``input_dir`` into
    ``output_dir/<doc_id>.json``. Resumable: an existing output is
    skipped only if ``is_valid_existing_output()`` (see
    statute_chunk_validate.py) confirms it, never on file existence
    alone. Never modifies ``input_dir``."""

    from src.rag_prep.statute_chunk_validate import is_valid_existing_output, validate_statute_chunk_set

    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    processed = regenerated = skipped_valid = 0
    failed_docs: list[dict] = []

    for cleaned_path in sorted(input_dir.glob("*.json")):
        doc_id = cleaned_path.stem
        out_path = output_dir / f"{doc_id}.json"

        if out_path.exists():
            try:
                existing = json.loads(out_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                existing = None
            if existing is not None and is_valid_existing_output(existing):
                skipped_valid += 1
                continue
            was_present = True
        else:
            was_present = False

        try:
            doc = json.loads(cleaned_path.read_text(encoding="utf-8"))
            parent = build_statute_chunks(doc)
            result = validate_statute_chunk_set(parent)
        except (OSError, json.JSONDecodeError, KeyError, StatuteChunkingError) as exc:
            failed_docs.append({"doc_id": doc_id, "error": str(exc)})
            continue

        if not result.ok:
            failed_docs.append({"doc_id": doc_id, "error": result.errors})
            continue

        tmp_path = out_path.with_suffix(".json.tmp")
        tmp_path.write_text(
            json.dumps({"parent": parent.to_dict(), "validation": result.to_dict()},
                       ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        tmp_path.replace(out_path)
        processed += 1
        if was_present:
            regenerated += 1

    return {
        "processed": processed, "skipped_valid": skipped_valid, "regenerated": regenerated,
        "failed": len(failed_docs), "failed_docs": failed_docs,
    }


def main() -> None:  # pragma: no cover -- thin CLI wrapper
    summary = chunk_directory()
    print(f"Statute chunking: {summary['processed']} processed "
          f"({summary['regenerated']} regenerated), {summary['skipped_valid']} skipped (valid), "
          f"{summary['failed']} failed")
    for f in summary["failed_docs"]:
        print(f"  FAILED {f['doc_id']}: {f['error']}")


if __name__ == "__main__":  # pragma: no cover
    main()
