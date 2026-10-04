"""Statute cleaning -- Stage 1 of the independent statute pipeline.

    raw statute JSON (var/rag/statutes_ingested/<doc_id>.json)
        -> clean_statute_text()      [conservative noise removal + whitespace normalization]
        -> cleaned statute JSON      (var/rag/cleaned_statutes/<doc_id>.json)

This stage removes PDF-extraction/layout noise (first-page table-of-
contents listings, "Page N of M" footers, generation-date stamps, stray
footnote-glyph lines, page-break hyphenation) while leaving every
substantive legal word untouched. It never summarizes, rewrites,
paraphrases, classifies, interprets, or renumbers anything -- see
``CONSERVATIVE CLEANING RULE`` below.

Deliberately named ``statute_cleaning.py``, NOT ``statute_cleaner.py``:
that filename is already taken by an existing, actively-used case-law
module (``src/rag_prep/statute_cleaner.py`` extracts statute *citations*
out of judgment text for ``case_cleaner.py`` -- a completely different,
case-law-pipeline purpose). Overwriting it would have broken the case-law
cleaning pipeline and its existing tests; this module is independent of
it and never imports it.

CONSERVATIVE CLEANING RULE
---------------------------
    legal content preservation  >  noise removal  >  formatting normalization

Every pattern removed here requires an unambiguous, narrowly-scoped
anchor (an exact "Page N of M" shape, a literal "CONTENTS" heading
followed by a recognizable enacting-clause/Act-citation anchor, a bare
footnote glyph alone on its own line). When the anchor for a rule isn't
found, that rule is a no-op for that document -- text is never removed
on a guess. This was validated directly against the 9 real ingested
statute documents in var/rag/statutes_ingested/ before being written
(not assumed), and deliberately does NOT attempt riskier heuristics
such as deduplicating repeated ALL-CAPS title lines, since a genuine
short legal heading could coincidentally repeat and the harm of wrongly
deleting one outweighs the cosmetic benefit of removing a harmless
duplicate.

PIPELINE ISOLATION
-------------------
Independent entry point, independent input/output directories, no
import of anything under src.ingestion/, src.classification/,
src.rag_prep.case_cleaner/structurer*/chunk*, or src.rag_prep.statute_cleaner
(the case-law citation extractor). Never touches the case-law pipeline's
var/rag/ directories or var/metadata.db.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

INPUT_DIR = Path("var/rag/statutes_ingested")
OUTPUT_DIR = Path("var/rag/cleaned_statutes")

# If cleaning removes more than this fraction of the raw text's length,
# the result is flagged for human review rather than silently accepted
# (task requirement: "flag it for review rather than silently accepting
# it"). This is a diagnostic flag, not a hard failure -- the cleaned
# text is still written.
LARGE_REMOVAL_FLAG_THRESHOLD = 0.15

# -- TOC removal ---------------------------------------------------------------
# Real statutes in this corpus open with: Act title -> "CONTENTS" -> a
# numbered short-title listing (sometimes spanning a page break, with
# page-footer noise interleaved) -> the real Act citation/enacting
# clause -> the actual Section 1 body. The listing itself carries no
# legal force (it's a reporter-style index), so it is removed; anything
# at or after the first recognizable end-of-TOC anchor is kept intact.
_TOC_START_RE = re.compile(r"^[ \t]*CONTENTS[ \t]*$", re.IGNORECASE | re.MULTILINE)
_TOC_END_RES = [
    # "ACT No. VIII OF 1939", "(W.P. Act XXXV of 1964)", "Ordinance No. X of 1961"
    re.compile(r"\(?\s*(?:W\.?P\.?\s*)?(?:Act|Ordinance)\s+(?:No\.?\s*)?[IVXLCDM0-9]+\s+of\s+\d{4}\s*\)?",
               re.IGNORECASE),
    re.compile(r"\bIt is hereby enacted\b", re.IGNORECASE),
    re.compile(r"\bWHEREAS\b", re.IGNORECASE),
    re.compile(r"\bPreamble\s*[:.\-]", re.IGNORECASE),
]
# Bounds how far past "CONTENTS" to search for an end anchor -- a
# generous window for a multi-section listing, but not unbounded (an
# anchor found implausibly far away is more likely a coincidental match
# than the real end of this TOC).
_TOC_SEARCH_WINDOW_CHARS = 6000

# -- unambiguous extraction/layout noise, matched as whole lines --------------
_PAGE_MARKER_RE = re.compile(r"^[ \t]*Page[ \t]+\d+[ \t]+of[ \t]+\d+[ \t]*$", re.IGNORECASE | re.MULTILINE)
_DATE_STAMP_RE = re.compile(r"^[ \t]*Date:[ \t]*\d{1,2}-\d{1,2}-\d{4}[ \t]*$", re.MULTILINE)
_FINAL_COPY_STAMP_RE = re.compile(r"^[ \t]*FINAL COPY\b.*$", re.IGNORECASE | re.MULTILINE)
_LONE_FOOTNOTE_GLYPH_RE = re.compile(r"^[ \t]*[*†‡][ \t]*$", re.MULTILINE)

# -- page-break hyphenation: "appoint-\nment" -> "appointment". Only
# joins when the line break falls immediately after the hyphen between
# two lowercase letters -- a deliberate hyphenated compound word
# ("non-resident") never has a line break inserted mid-hyphen by a human
# author, so this is a safe, narrow signature of a PDF line-wrap artifact.
_HYPHEN_LINEBREAK_RE = re.compile(r"([a-z])-[ \t]*\n[ \t]*([a-z])")

# -- generic whitespace normalization (never touches letters/digits) ---------
_CRLF_RE = re.compile(r"\r\n?")
_MULTI_SPACE_RE = re.compile(r"[ \t]{2,}")
_MULTI_BLANK_LINE_RE = re.compile(r"\n{3,}")
_TRAILING_LINE_WS_RE = re.compile(r"[ \t]+\n")
# Stray spaces PDF extraction sometimes inserts inside a subsection/
# clause marker -- "( 1 )" -> "(1)", "( a )" -> "(a)". Only touches the
# marker's own parentheses/spacing, never the number or letter itself,
# so the legal numbering is unchanged.
_PAREN_MARKER_SPACE_RE = re.compile(r"\(\s+([0-9a-zA-Z]{1,4})\s+\)")


@dataclass(frozen=True)
class CleaningDiagnostics:
    raw_length: int
    cleaned_length: int
    chars_removed: int
    pct_removed: float
    toc_removed: bool
    metadata_law_name_cleared: bool
    large_removal_flag: bool

    def to_dict(self) -> dict:
        return {
            "raw_length": self.raw_length,
            "cleaned_length": self.cleaned_length,
            "chars_removed": self.chars_removed,
            "pct_removed": self.pct_removed,
            "toc_removed": self.toc_removed,
            "metadata_law_name_cleared": self.metadata_law_name_cleared,
            "large_removal_flag": self.large_removal_flag,
        }


def _remove_toc(text: str) -> tuple[str, bool]:
    """Removes the first-page CONTENTS listing, if a recognizable
    end-of-TOC anchor is found; otherwise returns ``text`` unchanged
    (conservative: never guesses where legal text "probably" starts)."""

    start_match = _TOC_START_RE.search(text)
    if not start_match:
        return text, False

    window_end = min(len(text), start_match.end() + _TOC_SEARCH_WINDOW_CHARS)
    window = text[start_match.end():window_end]

    best_offset: int | None = None
    for pattern in _TOC_END_RES:
        end_match = pattern.search(window)
        if end_match and (best_offset is None or end_match.start() < best_offset):
            best_offset = end_match.start()

    if best_offset is None:
        return text, False

    toc_end = start_match.end() + best_offset
    return text[:start_match.start()] + text[toc_end:], True


def _remove_layout_noise(text: str) -> str:
    text = _PAGE_MARKER_RE.sub("", text)
    text = _DATE_STAMP_RE.sub("", text)
    text = _FINAL_COPY_STAMP_RE.sub("", text)
    text = _LONE_FOOTNOTE_GLYPH_RE.sub("", text)
    return text


def _join_hyphenated_linebreaks(text: str) -> str:
    return _HYPHEN_LINEBREAK_RE.sub(r"\1\2", text)


def _normalize_whitespace(text: str) -> str:
    text = _CRLF_RE.sub("\n", text)
    text = _TRAILING_LINE_WS_RE.sub("\n", text)
    text = _MULTI_SPACE_RE.sub(" ", text)
    text = _MULTI_BLANK_LINE_RE.sub("\n\n", text)
    text = _PAREN_MARKER_SPACE_RE.sub(r"(\1)", text)
    return text.strip()


_PAGE_MARKER_WHOLE_RE = re.compile(r"^Page\s+\d+\s+of\s+\d+$", re.IGNORECASE)


def clean_statute_text(full_text: str) -> tuple[str, CleaningDiagnostics]:
    """Cleans one statute's raw extracted text. Returns ``(cleaned_text, diagnostics)``.

    Order matters: TOC removal and layout-noise removal operate on the
    raw line structure first; hyphenation joining and whitespace
    normalization run last, since they would otherwise interfere with
    the line-anchored patterns above (e.g. collapsing the blank lines
    that separate a "Page N of M" footer from its surrounding text
    before that pattern has had a chance to match it as its own line).
    """

    raw_length = len(full_text)

    text, toc_removed = _remove_toc(full_text)
    text = _remove_layout_noise(text)
    text = _join_hyphenated_linebreaks(text)
    text = _normalize_whitespace(text)

    cleaned_length = len(text)
    chars_removed = raw_length - cleaned_length
    pct_removed = round(chars_removed / raw_length, 4) if raw_length else 0.0

    diagnostics = CleaningDiagnostics(
        raw_length=raw_length,
        cleaned_length=cleaned_length,
        chars_removed=chars_removed,
        pct_removed=pct_removed,
        toc_removed=toc_removed,
        metadata_law_name_cleared=False,  # set by clean_statute_document()
        large_removal_flag=pct_removed > LARGE_REMOVAL_FLAG_THRESHOLD,
    )
    return text, diagnostics


def clean_statute_document(raw_doc: dict) -> dict:
    """Cleans one raw ingestion document (the statute_ingestion.py output
    shape) into the cleaned-statute output shape. ``full_text`` (the raw
    extracted source) is carried through unchanged as the authoritative
    record; ``cleaned_text`` is the new, additive field."""

    full_text = raw_doc["full_text"]
    cleaned_text, diagnostics = clean_statute_text(full_text)

    metadata = dict(raw_doc["metadata"])
    law_name_cleared = False
    law_name = metadata.get("law_name")
    if isinstance(law_name, str) and _PAGE_MARKER_WHOLE_RE.match(law_name.strip()):
        # A clearly extraction-related artifact (the ingestion stage
        # picked up a page-footer string as the PDF's "title" metadata,
        # not a real law name) -- cleared to null rather than guessed
        # at, per "if metadata is unavailable, use null".
        metadata["law_name"] = None
        law_name_cleared = True

    diagnostics_dict = diagnostics.to_dict()
    diagnostics_dict["metadata_law_name_cleared"] = law_name_cleared

    return {
        "doc_id": raw_doc["doc_id"],
        "metadata": metadata,
        "full_text": full_text,
        "cleaned_text": cleaned_text,
        "cleaning_diagnostics": diagnostics_dict,
    }


def clean_directory(input_dir: Path = INPUT_DIR, output_dir: Path = OUTPUT_DIR) -> dict:
    """Cleans every ``<doc_id>.json`` in ``input_dir`` into
    ``output_dir/<doc_id>.json``. Never modifies ``input_dir``.

    Returns {"processed", "failed", "large_removal_flagged", "failed_docs"}.
    One document's failure (malformed/unreadable raw JSON) is recorded
    and does not stop the rest of the batch.
    """

    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    processed = 0
    large_removal_flagged = 0
    failed_docs: list[dict] = []

    for raw_path in sorted(input_dir.glob("*.json")):
        try:
            raw_doc = json.loads(raw_path.read_text(encoding="utf-8"))
            cleaned_doc = clean_statute_document(raw_doc)
        except (OSError, json.JSONDecodeError, KeyError) as exc:
            failed_docs.append({"source_file": raw_path.name, "error": str(exc)})
            continue

        out_path = output_dir / f"{cleaned_doc['doc_id']}.json"
        tmp_path = out_path.with_suffix(".json.tmp")
        tmp_path.write_text(json.dumps(cleaned_doc, ensure_ascii=False, indent=2), encoding="utf-8")
        tmp_path.replace(out_path)
        processed += 1
        if cleaned_doc["cleaning_diagnostics"]["large_removal_flag"]:
            large_removal_flagged += 1

    return {
        "processed": processed,
        "failed": len(failed_docs),
        "large_removal_flagged": large_removal_flagged,
        "failed_docs": failed_docs,
    }


def main() -> None:  # pragma: no cover -- thin CLI wrapper
    summary = clean_directory()
    print(f"Statute cleaning: {summary['processed']} processed, {summary['failed']} failed, "
          f"{summary['large_removal_flagged']} flagged for large removal")
    for f in summary["failed_docs"]:
        print(f"  FAILED {f['source_file']}: {f['error']}")


if __name__ == "__main__":  # pragma: no cover
    main()
