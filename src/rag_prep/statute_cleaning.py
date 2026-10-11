# """Statute cleaning -- Stage 1 of the independent statute pipeline.

#     raw statute JSON (var/rag/statutes_ingested/<doc_id>.json)
#         -> clean_statute_text()      [conservative noise removal + whitespace normalization]
#         -> cleaned statute JSON      (var/rag/cleaned_statutes/<doc_id>.json)

# This stage removes PDF-extraction/layout noise (first-page table-of-
# contents listings, "Page N of M" footers, generation-date stamps, stray
# footnote-glyph lines, horizontal-rule dividers, page-bottom amendment
# footnotes, page-break hyphenation, soft-wrapped clause/definition
# headers) while leaving every substantive legal word untouched. It never
# summarizes, rewrites, paraphrases, classifies, interprets, or renumbers
# anything -- see ``CONSERVATIVE CLEANING RULE`` below.

# Deliberately named ``statute_cleaning.py``, NOT ``statute_cleaner.py``:
# that filename is already taken by an existing, actively-used case-law
# module (``src/rag_prep/statute_cleaner.py`` extracts statute *citations*
# out of judgment text for ``case_cleaner.py`` -- a completely different,
# case-law-pipeline purpose). Overwriting it would have broken the case-law
# cleaning pipeline and its existing tests; this module is independent of
# it and never imports it.

# CONSERVATIVE CLEANING RULE
# ---------------------------
#     legal content preservation  >  noise removal  >  formatting normalization

# Every pattern removed here requires an unambiguous, narrowly-scoped
# anchor (an exact "Page N of M" shape, a literal "CONTENTS" heading
# followed by a recognizable enacting-clause/Act-citation anchor, a bare
# footnote glyph alone on its own line). When the anchor for a rule isn't
# found, that rule is a no-op for that document -- text is never removed
# on a guess. This was validated directly against the 9 real ingested
# statute documents in var/rag/statutes_ingested/ before being written
# (not assumed), and deliberately does NOT attempt riskier heuristics
# such as deduplicating repeated ALL-CAPS title lines, since a genuine
# short legal heading could coincidentally repeat and the harm of wrongly
# deleting one outweighs the cosmetic benefit of removing a harmless
# duplicate.

# PIPELINE ISOLATION
# -------------------
# Independent entry point, independent input/output directories, no
# import of anything under src.ingestion/, src.classification/,
# src.rag_prep.case_cleaner/structurer*/chunk*, or src.rag_prep.statute_cleaner
# (the case-law citation extractor). Never touches the case-law pipeline's
# var/rag/ directories or var/metadata.db.
# """

# from __future__ import annotations

# import json
# import re
# from dataclasses import dataclass
# from pathlib import Path

# INPUT_DIR = Path("var/rag/statutes_ingested")
# OUTPUT_DIR = Path("var/rag/cleaned_statutes")

# # If cleaning removes more than this fraction of the raw text's length,
# # the result is flagged for human review rather than silently accepted
# # (task requirement: "flag it for review rather than silently accepting
# # it"). This is a diagnostic flag, not a hard failure -- the cleaned
# # text is still written.
# LARGE_REMOVAL_FLAG_THRESHOLD = 0.15

# # -- TOC removal ---------------------------------------------------------------
# # Real statutes in this corpus open with: Act title -> "CONTENTS" -> a
# # numbered short-title listing (sometimes spanning a page break, with
# # page-footer noise interleaved) -> the real Act citation/enacting
# # clause -> the actual Section 1 body. The listing itself carries no
# # legal force (it's a reporter-style index), so it is removed; anything
# # at or after the first recognizable end-of-TOC anchor is kept intact.
# _TOC_START_RE = re.compile(r"^[ \t]*CONTENTS[ \t]*$", re.IGNORECASE | re.MULTILINE)
# _TOC_END_RES = [
#     # "ACT No. VIII OF 1939", "(W.P. Act XXXV of 1964)", "Ordinance No. X of 1961"
#     re.compile(r"\(?\s*(?:W\.?P\.?\s*)?(?:Act|Ordinance)\s+(?:No\.?\s*)?[IVXLCDM0-9]+\s+of\s+\d{4}\s*\)?",
#                re.IGNORECASE),
#     re.compile(r"\bIt is hereby enacted\b", re.IGNORECASE),
#     re.compile(r"\bWHEREAS\b", re.IGNORECASE),
#     re.compile(r"\bPreamble\s*[:.\-]", re.IGNORECASE),
# ]
# # Bounds how far past "CONTENTS" to search for an end anchor -- a
# # generous window for a multi-section listing, but not unbounded (an
# # anchor found implausibly far away is more likely a coincidental match
# # than the real end of this TOC).
# _TOC_SEARCH_WINDOW_CHARS = 6000

# # -- unambiguous extraction/layout noise, matched as whole lines --------------
# _PAGE_MARKER_RE = re.compile(r"^[ \t]*Page[ \t]+\d+[ \t]+of[ \t]+\d+[ \t]*$", re.IGNORECASE | re.MULTILINE)
# # "Date: 06-05-2024" and, confirmed in 2 of 9 real documents, the same
# # generation-date stamp prefixed with "RGN " ("RGN Date: 06-05-2024"),
# # one of which also has a leading horizontal-rule-style underscore run
# # before it ("________    RGN Date: 24-03-2025"). Never seen anywhere
# # else in the corpus in either form, so this is treated as the two known
# # variants of this one exact stamp, not a generic "any prefix" or
# # "strip underscores" rule -- the anchor is still always the literal
# # "Date: <D>-<D>-<YYYY>" at the end of the line.
# _DATE_STAMP_RE = re.compile(r"^[ \t_]*(?:RGN[ \t]+)?Date:[ \t]*\d{1,2}-\d{1,2}-\d{4}[ \t]*$", re.MULTILINE)
# _FINAL_COPY_STAMP_RE = re.compile(r"^[ \t]*FINAL COPY\b.*$", re.IGNORECASE | re.MULTILINE)
# _LONE_FOOTNOTE_GLYPH_RE = re.compile(r"^[ \t]*[*†‡][ \t]*$", re.MULTILINE)
# # A standalone horizontal-rule-style divider (e.g. a line of underscores
# # separating a Schedule's chapters, or separating body text from the
# # page-bottom footnote block below it). Confirmed in the real corpus to
# # always be pure layout decoration -- the substantive heading text next
# # to it (e.g. "II.--JURISDICTION") lives on its own separate line and is
# # never part of the divider line itself, so removing the divider line
# # never removes adjacent heading/body text.
# _DIVIDER_LINE_RE = re.compile(r"^[ \t]*_{5,}[ \t]*$", re.MULTILINE)

# # -- page-break hyphenation: "appoint-\nment" -> "appointment". Only
# # joins when the line break falls immediately after a hyphen, with the
# # next line starting lowercase (a deliberate hyphenated compound word
# # such as "non-resident" never has a line break inserted mid-hyphen by a
# # human author, so a hyphen immediately followed by a line break is a
# # safe, narrow signature of a PDF line-wrap artifact -- but the hyphen
# # itself may or may not be meaningful, see _KNOWN_SINGLE_WORD_LINE_BREAKS
# # below and _join_hyphenated_linebreaks's docstring).
# _HYPHEN_LINEBREAK_RE = re.compile(r"([A-Za-z]+)-[ \t]*\n[ \t]*([a-z][A-Za-z]*)")

# # Real statute text in this corpus hyphenates many deliberate compounds
# # ("sub-section", "non-Muslim", "Mujtahid-e-Alam", "dwelling-house",
# # "legatee-in-enjoyment" -- "sub-section" alone appears 84 times on a
# # single line elsewhere in the corpus, always hyphenated, confirming it
# # is the author's spelling, not a PDF artifact). A blind "always strip
# # the hyphen" rule is therefore wrong far more often than it is right.
# # This allowlist is the single, narrow exception: pairs confirmed, either
# # from the real corpus or an existing test fixture, to be a genuine
# # single word that a human author never hyphenates (so the line-wrap
# # hyphen here is purely a PDF artifact and must be removed, not kept).
# # Anything NOT in this list keeps its hyphen -- the newline is still
# # removed (the two line-fragments are still joined into one line), but
# # as "prefix-suffix", never as a guessed, dictionary-corrected single word.
# _KNOWN_SINGLE_WORD_LINE_BREAKS = frozenset({
#     "education",  # confirmed real corpus: "education" appears 6 times unhyphenated
#     "appointment",  # pre-existing test fixture (src/rag_prep/statute_cleaning.py tests)
# })

# # -- generic whitespace normalization (never touches letters/digits) ---------
# _CRLF_RE = re.compile(r"\r\n?")
# _MULTI_SPACE_RE = re.compile(r"[ \t]{2,}")
# _MULTI_BLANK_LINE_RE = re.compile(r"\n{3,}")
# _TRAILING_LINE_WS_RE = re.compile(r"[ \t]+\n")
# # Stray spaces PDF extraction sometimes inserts inside a subsection/
# # clause marker -- "( 1 )" -> "(1)", "( a )" -> "(a)". Only touches the
# # marker's own parentheses/spacing, never the number or letter itself,
# # so the legal numbering is unchanged.
# _PAREN_MARKER_SPACE_RE = re.compile(r"\(\s+([0-9a-zA-Z]{1,4})\s+\)")

# # -- page-bottom amendment/footnote lines -------------------------------------
# # PDF text extraction loses the superscript formatting of a footnote
# # reference number, leaving it glued directly to the footnote's own text
# # with no period or space in between ("1Subs. by the Central Laws...").
# # A genuine section/clause number is never glued to its title this way --
# # it is always "<number>. <title>" with a period-and-space (see
# # _SECTION_HEADER_LIKE_RE below, and statute_chunker.py's own
# # _SECTION_HEADER_RE) -- so "digit(s) glued directly to a letter" is a
# # safe, narrow signature of a footnote reference, PROVIDED the word
# # glued to the digit is itself unambiguously a footnote/amendment-note
# # lead word. The vocabulary below was built by enumerating every such
# # glued-digit line across all 9 real ingested statutes and keeping only
# # the words that never collided with real heading/title text; two
# # candidates found in that scan ("THE", "ACT") were deliberately
# # EXCLUDED after discovering they also occur in a genuine Act title line
# # with a footnote-reference digit glued to the front (e.g. "1THE FAMILY
# # COURTS ACT, 1964" -- the Act's own real heading, not disposable noise),
# # and three more ("Succession", "Code", "Repealing") were excluded because
# # they are just the coincidental first word of whichever external Act a
# # given footnote happens to cite, not a reliable structural signal.
# # Under-matching here is safe (the line is simply left in place, exactly
# # today's behavior); over-matching is not, so the vocabulary stays
# # deliberately narrow rather than exhaustive.
# _FOOTNOTE_LEAD_WORD_RE = re.compile(
#     r"^[ \t]*\d{1,2}(?:Subs|Sub|Ins|Omitted|Deleted|Added|Renumbered|Rep|Proviso|Clause|This|For|Please)\b"
# )
# # A genuine section/sub-section header -- "17A. Interim order...",
# # "4A.  ", "6. [Repealed]." -- always has a period followed by whitespace
# # right after the (optional one-or-two-letter-suffixed) number. Used to
# # make sure a footnote-lead match can never be read as cutting into one
# # of these (belt-and-suspenders: the closed vocabulary above already
# # can't collide with this shape, since "A"/"B" aren't in it).
# _SECTION_HEADER_LIKE_RE = re.compile(r"^[ \t]*\d+-?[A-Za-z]{0,2}\.[ \t]")
# # An inline amendment marker resuming the substantive body ("2[9. No
# # Court shall...", "1[(2) It extends...") -- a footnote entry's
# # continuation lines never start like this in the real corpus, so seeing
# # one ends the footnote-continuation absorption (see _relocate_footnotes).
# _INLINE_AMENDMENT_MARKER_START_RE = re.compile(r"^[ \t]*\d+\[")

# # -- soft-wrapped clause/definition header: "(b)\n'Chairman' means..." --
# # A defined-term clause marker, (a)/(b)/(c)/..., sometimes has its quoted
# # term pushed to the next extracted line because the quoted text didn't
# # fit the PDF's line width. Confirmed widely in the real corpus (every
# # lettered definitions clause in 5 of the 9 real statutes wraps this way).
# # Narrowly anchored: only fires when the marker is the LAST thing on its
# # line and the very next line opens with a quote/bracket character --
# # an ordinary paragraph break after a clause marker, followed by normal
# # prose, never satisfies both conditions at once.
# _SOFT_WRAPPED_CLAUSE_RE = re.compile(
#     r"(\([a-zA-Z]{1,3}\))[ \t]*\n[ \t]*(?=[\"'“‘(\[])",
#     re.MULTILINE,
# )


# @dataclass(frozen=True)
# class CleaningDiagnostics:
#     raw_length: int
#     cleaned_length: int
#     chars_removed: int
#     pct_removed: float
#     toc_removed: bool
#     metadata_law_name_cleared: bool
#     large_removal_flag: bool
#     footnotes_relocated_count: int = 0

#     def to_dict(self) -> dict:
#         return {
#             "raw_length": self.raw_length,
#             "cleaned_length": self.cleaned_length,
#             "chars_removed": self.chars_removed,
#             "pct_removed": self.pct_removed,
#             "toc_removed": self.toc_removed,
#             "metadata_law_name_cleared": self.metadata_law_name_cleared,
#             "large_removal_flag": self.large_removal_flag,
#             "footnotes_relocated_count": self.footnotes_relocated_count,
#         }


# def _remove_toc(text: str) -> tuple[str, bool]:
#     """Removes the first-page CONTENTS listing, if a recognizable
#     end-of-TOC anchor is found; otherwise returns ``text`` unchanged
#     (conservative: never guesses where legal text "probably" starts)."""

#     start_match = _TOC_START_RE.search(text)
#     if not start_match:
#         return text, False

#     window_end = min(len(text), start_match.end() + _TOC_SEARCH_WINDOW_CHARS)
#     window = text[start_match.end():window_end]

#     best_offset: int | None = None
#     for pattern in _TOC_END_RES:
#         end_match = pattern.search(window)
#         if end_match and (best_offset is None or end_match.start() < best_offset):
#             best_offset = end_match.start()

#     if best_offset is None:
#         return text, False

#     toc_end = start_match.end() + best_offset
#     return text[:start_match.start()] + text[toc_end:], True


# def _remove_layout_noise(text: str) -> str:
#     text = _PAGE_MARKER_RE.sub("", text)
#     text = _DATE_STAMP_RE.sub("", text)
#     text = _FINAL_COPY_STAMP_RE.sub("", text)
#     text = _LONE_FOOTNOTE_GLYPH_RE.sub("", text)
#     text = _DIVIDER_LINE_RE.sub("", text)
#     return text


# def _relocate_footnotes(text: str) -> tuple[str, list[str]]:
#     """Pulls page-bottom amendment/footnote lines out of the substantive
#     body and returns them separately, in their original order, so the
#     caller can append them as one coherent ``[FOOTNOTES]`` block instead
#     of leaving them scattered between definitions/subsections.

#     A footnote "entry" is one footnote-lead line plus any immediately
#     following non-blank lines that are themselves neither a new
#     footnote, a section-header-like line, nor a resumed inline-amendment
#     marker -- i.e. its own PDF-wrapped continuation (see module-level
#     regex docstrings). Absorption stops at the first blank line, which
#     is how every real footnote block in the corpus is bounded.
#     """

#     lines = text.split("\n")
#     body_lines: list[str] = []
#     footnotes: list[str] = []
#     i, n = 0, len(lines)

#     while i < n:
#         line = lines[i]
#         if _FOOTNOTE_LEAD_WORD_RE.match(line) and not _SECTION_HEADER_LIKE_RE.match(line):
#             entry_parts = [line.strip()]
#             i += 1
#             while i < n:
#                 nxt = lines[i]
#                 if (
#                     nxt.strip() == ""
#                     or _FOOTNOTE_LEAD_WORD_RE.match(nxt)
#                     or _SECTION_HEADER_LIKE_RE.match(nxt)
#                     or _INLINE_AMENDMENT_MARKER_START_RE.match(nxt)
#                 ):
#                     break
#                 entry_parts.append(nxt.strip())
#                 i += 1
#             footnotes.append(_MULTI_SPACE_RE.sub(" ", " ".join(entry_parts)).strip())
#             continue
#         body_lines.append(line)
#         i += 1

#     return "\n".join(body_lines), footnotes


# def _repair_soft_wrapped_clause_headers(text: str) -> str:
#     return _SOFT_WRAPPED_CLAUSE_RE.sub(r"\1 ", text)


# def _join_hyphenated_linebreaks(text: str) -> str:
#     """Closes a PDF line-wrap gap around a hyphen. The hyphen itself is
#     only dropped (producing one merged word) for the small, evidenced
#     allowlist in _KNOWN_SINGLE_WORD_LINE_BREAKS; every other case keeps
#     the hyphen (producing "prefix-suffix"), since that is what a
#     deliberately hyphenated legal compound requires -- see the
#     allowlist's docstring for the real-corpus evidence behind this."""

#     def _repl(m: re.Match) -> str:
#         prefix, suffix = m.group(1), m.group(2)
#         if (prefix + suffix).lower() in _KNOWN_SINGLE_WORD_LINE_BREAKS:
#             return prefix + suffix
#         return f"{prefix}-{suffix}"

#     return _HYPHEN_LINEBREAK_RE.sub(_repl, text)


# def _normalize_whitespace(text: str) -> str:
#     text = _CRLF_RE.sub("\n", text)
#     text = _TRAILING_LINE_WS_RE.sub("\n", text)
#     text = _MULTI_SPACE_RE.sub(" ", text)
#     text = _MULTI_BLANK_LINE_RE.sub("\n\n", text)
#     text = _PAREN_MARKER_SPACE_RE.sub(r"(\1)", text)
#     return text.strip()


# _PAGE_MARKER_WHOLE_RE = re.compile(r"^Page\s+\d+\s+of\s+\d+$", re.IGNORECASE)


# def clean_statute_text(full_text: str) -> tuple[str, CleaningDiagnostics]:
#     """Cleans one statute's raw extracted text. Returns ``(cleaned_text, diagnostics)``.

#     Order matters, matching the pipeline's conceptual flow (layout
#     cleanup -> footnote extraction/relocation -> line/wrap repair ->
#     whitespace normalization): TOC removal and layout-noise removal
#     (including divider-line removal) operate on the raw line structure
#     first, since later steps would interfere with their line-anchored
#     patterns (e.g. collapsing the blank lines that separate a "Page N of
#     M" footer from its surrounding text before that pattern has had a
#     chance to match it as its own line). Footnote relocation runs next,
#     while blank lines still mark each footnote block's real boundary.
#     Hyphenation joining and the soft-wrapped-clause repair run after
#     that, and whitespace normalization runs last of all. The relocated
#     footnotes are appended, as one ``[FOOTNOTES]`` block, only after the
#     rest of the body has been fully cleaned and normalized.
#     """

#     raw_length = len(full_text)

#     text, toc_removed = _remove_toc(full_text)
#     text = _remove_layout_noise(text)
#     text, footnote_entries = _relocate_footnotes(text)
#     text = _repair_soft_wrapped_clause_headers(text)
#     text = _join_hyphenated_linebreaks(text)
#     text = _normalize_whitespace(text)

#     if footnote_entries:
#         footnote_block = "\n".join(footnote_entries)
#         text = f"{text}\n\n[FOOTNOTES]\n{footnote_block}\n[/FOOTNOTES]"

#     cleaned_length = len(text)
#     chars_removed = raw_length - cleaned_length
#     pct_removed = round(chars_removed / raw_length, 4) if raw_length else 0.0

#     diagnostics = CleaningDiagnostics(
#         raw_length=raw_length,
#         cleaned_length=cleaned_length,
#         chars_removed=chars_removed,
#         pct_removed=pct_removed,
#         toc_removed=toc_removed,
#         metadata_law_name_cleared=False,  # set by clean_statute_document()
#         large_removal_flag=pct_removed > LARGE_REMOVAL_FLAG_THRESHOLD,
#         footnotes_relocated_count=len(footnote_entries),
#     )
#     return text, diagnostics


# def clean_statute_document(raw_doc: dict) -> dict:
#     """Cleans one raw ingestion document (the statute_ingestion.py output
#     shape) into the cleaned-statute output shape. ``full_text`` (the raw
#     extracted source) is carried through unchanged as the authoritative
#     record; ``cleaned_text`` is the new, additive field."""

#     full_text = raw_doc["full_text"]
#     cleaned_text, diagnostics = clean_statute_text(full_text)

#     metadata = dict(raw_doc["metadata"])
#     law_name_cleared = False
#     law_name = metadata.get("law_name")
#     if isinstance(law_name, str) and _PAGE_MARKER_WHOLE_RE.match(law_name.strip()):
#         # A clearly extraction-related artifact (the ingestion stage
#         # picked up a page-footer string as the PDF's "title" metadata,
#         # not a real law name) -- cleared to null rather than guessed
#         # at, per "if metadata is unavailable, use null".
#         metadata["law_name"] = None
#         law_name_cleared = True

#     diagnostics_dict = diagnostics.to_dict()
#     diagnostics_dict["metadata_law_name_cleared"] = law_name_cleared

#     return {
#         "doc_id": raw_doc["doc_id"],
#         "metadata": metadata,
#         "full_text": full_text,
#         "cleaned_text": cleaned_text,
#         "cleaning_diagnostics": diagnostics_dict,
#     }


# def clean_directory(input_dir: Path = INPUT_DIR, output_dir: Path = OUTPUT_DIR) -> dict:
#     """Cleans every ``<doc_id>.json`` in ``input_dir`` into
#     ``output_dir/<doc_id>.json``. Never modifies ``input_dir``.

#     Returns {"processed", "failed", "large_removal_flagged", "failed_docs"}.
#     One document's failure (malformed/unreadable raw JSON) is recorded
#     and does not stop the rest of the batch.
#     """

#     input_dir = Path(input_dir)
#     output_dir = Path(output_dir)
#     output_dir.mkdir(parents=True, exist_ok=True)

#     processed = 0
#     large_removal_flagged = 0
#     failed_docs: list[dict] = []

#     for raw_path in sorted(input_dir.glob("*.json")):
#         try:
#             raw_doc = json.loads(raw_path.read_text(encoding="utf-8"))
#             cleaned_doc = clean_statute_document(raw_doc)
#         except (OSError, json.JSONDecodeError, KeyError) as exc:
#             failed_docs.append({"source_file": raw_path.name, "error": str(exc)})
#             continue

#         out_path = output_dir / f"{cleaned_doc['doc_id']}.json"
#         tmp_path = out_path.with_suffix(".json.tmp")
#         tmp_path.write_text(json.dumps(cleaned_doc, ensure_ascii=False, indent=2), encoding="utf-8")
#         tmp_path.replace(out_path)
#         processed += 1
#         if cleaned_doc["cleaning_diagnostics"]["large_removal_flag"]:
#             large_removal_flagged += 1

#     return {
#         "processed": processed,
#         "failed": len(failed_docs),
#         "large_removal_flagged": large_removal_flagged,
#         "failed_docs": failed_docs,
#     }


# def main() -> None:  # pragma: no cover -- thin CLI wrapper
#     summary = clean_directory()
#     print(f"Statute cleaning: {summary['processed']} processed, {summary['failed']} failed, "
#           f"{summary['large_removal_flagged']} flagged for large removal")
#     for f in summary["failed_docs"]:
#         print(f"  FAILED {f['source_file']}: {f['error']}")


# if __name__ == "__main__":  # pragma: no cover
#     main()



"""Statute cleaning -- Stage 1 of the independent statute pipeline (v2, page-aware).

    raw statute JSON (var/rag/statutes_ingested/<doc_id>.json)
        -> clean_statute_document()      [page-aware noise removal, footnote relocation,
                                           paragraph repair, letter-preservation check]
        -> cleaned statute JSON          (var/rag/cleaned_statutes/<doc_id>.json)
           (documents failing the quality gate go to var/rag/cleaned_statutes/_rejected/)

What changed from v1 and why
----------------------------
v1 found page-bottom footnotes anywhere in the text with a word list ("1Subs", "1Ins", ...).
That missed every footnote written "1 Subs." (space), "1\\nSubs." (number on its own line),
"1.Subs." or "5S. 4A ins." -- which then stayed in the legal text and, in the Guardians and
Wards Act, became a fake "section 5S". Footnote numbers also restart on every page, so v1's
"3[...]" markers could not be linked to any footnote.

v2 works page by page, like the Constitution cleaner:
  1. The raw text is split into its pages ("Page N of M" / blank-line page breaks).
  2. Page furniture is removed per page (header stamp, "Page N of M", date stamp, divider
     lines, isolated print codes such as "135171").
  3. The footnote zone of each page is found STRUCTURALLY: the block at the bottom of the
     page that is a run of entries numbered 1, 2, 3 ... in order, with nothing but blank
     lines after it. Any entry shape is accepted ("1Subs.", "1 Subs.", "1\\nSubs.",
     "1.Subs.", "2. The word ...", "5S. 4A ins.") because position + numbering, not
     vocabulary, identifies it.
  4. Inline markers in that page's text are rewritten to page-scoped references:
     "3[Pakistan]" on page 4 -> "[^4.3][Pakistan]", so every marker links to exactly one
     footnote ("[^4.3]: Subs. by Act No. I of 1969, s. 2.").
  5. Wrapped lines are rejoined into paragraphs (one section / sub-section / clause /
     proviso per line). A section number or clause label left alone on its own line
     ("18." / "(a)") is joined to its text -- the cause of "section 18 swallowed by 17B".
  6. The Act's name and year are read from its own title line when the metadata value is
     missing or is an extraction artifact ("FINAL COPY Updated till ...").
  7. Verification: every letter of the raw text (minus removed furniture / TOC) must be
     present in the cleaned text + footnotes, and no letter may be invented. A quality gate
     fails the document instead of silently writing a bad result.

CONSERVATIVE CLEANING RULE (unchanged)
---------------------------------------
    legal content preservation  >  noise removal  >  formatting normalization
Nothing is summarised, paraphrased, renumbered or "corrected". Removed text is limited to
page furniture, the CONTENTS listing and the page-bottom footnotes (which are moved, not
deleted). The letter-preservation check enforces this mechanically.

PIPELINE ISOLATION
-------------------
Independent entry point, independent input/output directories, no import of anything under
src.ingestion/, src.classification/, src.rag_prep.case_cleaner/structurer*/chunk*, or
src.rag_prep.statute_cleaner (the case-law citation extractor).
"""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

INPUT_DIR = Path("var/rag/statutes_ingested")
OUTPUT_DIR = Path("var/rag/cleaned_statutes")

LARGE_REMOVAL_FLAG_THRESHOLD = 0.15

# -- TOC removal (unchanged anchors) -----------------------------------------------------
_TOC_START_RE = re.compile(r"^[ \t]*CONTENTS[ \t]*$", re.IGNORECASE | re.MULTILINE)
_TOC_END_RES = [
    re.compile(r"\(?\s*(?:W\.?P\.?\s*)?(?:Act|Ordinance)\s+(?:No\.?\s*)?[IVXLCDM0-9]+\s+of\s+\d{4}\s*\)?",
               re.IGNORECASE),
    re.compile(r"\bIt is hereby enacted\b", re.IGNORECASE),
    re.compile(r"\bWHEREAS\b", re.IGNORECASE),
    re.compile(r"\bPreamble\s*[:.\-]", re.IGNORECASE),
]
_TOC_SEARCH_WINDOW_CHARS = 6000

# -- page furniture (whole lines only) ---------------------------------------------------
_PAGE_MARKER_RE = re.compile(r"^[ \t]*Page[ \t]+(\d+)[ \t]+of[ \t]+\d+[ \t]*$", re.IGNORECASE)
_DATE_STAMP_RE = re.compile(r"^[ \t_]*(?:RGN[ \t]+)?Date:[ \t]*\d{1,2}-\d{1,2}-\d{4}[ \t]*$")
_FINAL_COPY_STAMP_RE = re.compile(r"^[ \t]*FINAL COPY\b.*$", re.IGNORECASE)
_DIVIDER_LINE_RE = re.compile(r"^[ \t]*_{5,}[ \t]*$")
_PRINT_CODE_RE = re.compile(r"^[ \t]*\d{5,}[ \t]*$")              # "135171": no legal text is a bare 5+ digit line
_LONE_GLYPH_RE = re.compile(r"^[ \t]*[†‡][ \t]*$")
_STAR_LINE_RE = re.compile(r"^[ \t]*\*[ \t]*$")                    # "* * *" omission markers printed one per line

# -- footnotes ----------------------------------------------------------------------------
# Entry start: footnote number, then the note. Shapes seen in the corpus:
#   "1Subs. by", "1 Subs. by", "1" (note on the next line), "1.Subs. by", "2. The word",
#   "5S. 4A ins. by" ("S." = section), "1For notification ...".
_FN_ENTRY_RE = re.compile(r"^[ \t]*(\d{1,2}|l(?=[A-Z][a-z]))(?:[ \t]*$|\.?[ \t]*(?=[A-Za-z‘“\"'(]))")
# Text that marks a block as amendment/editorial notes (checked on the whole zone, not per line).
_FN_VOCAB_RE = re.compile(
    r"\b(?:by|ibid|Subs|subs|Ins|ins|Omitted|omitted|Added|added|Deleted|Rep|rep|Renumbered|Gazette|"
    r"Act|Ordinance|Order|A\.\s?O\.|Sch|See|see|definition|notification|word|words)\b")
_INLINE_MARKER_RE = re.compile(r"(?<![\w.\]])(\d{1,2}|l)(?=\[)")       # "3[Pakistan]"; "l[" = a "1" misread as "l"
_GLUED_WORD_MARKER_RE = re.compile(                                       # "2notification", "1good faith", "1ACT No."
    r"(?<![\w.,/\-\[(])(\d{1,2})(?=[A-Za-z]{2,})(?!(?:st|nd|rd|th)\b)")
_TRAILING_MARKER_RE = re.compile(r"(?<=[A-Za-z]{3})(\d{1,2})(?=[ \t]*[.,;:)\]]|[ \t]*$)")   # "Procedure1."
_STAR_MARKER_RE = re.compile(r"(?<![\w.])(\d{1,2})(?=[ \t]*\*)")           # "2* * *"
_GLUED_TITLE_MARKER_RE = re.compile(r"^([ \t]*)(\d{1,2})(?=[A-Z]{2,})")  # "1THE FAMILY COURTS ACT"
_STRAY_FOOTNOTE_RE = re.compile(                                          # used only to VERIFY the body
    r"^[ \t]*(?:\d{1,2}|l)(?:[ \t]+|\.)?(?:S\.[ \t]*\d|Subs\.|Sub\.|Ins\.|Ins,|Omitted by|Deleted by|Added by|"
    r"Renumbered|Rep\. by|The words?\b|This Act was|For notification|For definition|Please see)")

# -- paragraph repair ----------------------------------------------------------------------
_REF = r"(?:\[\^[^\]]+\])"
_BARE_LABEL_RE = re.compile(                                               # "18." / "(a)" alone on a line
    rf"^[ \t]*{_REF}*\[?(?:\d{{1,3}}[A-Z]{{0,2}}(?:-[A-Z])?\.|\((?:\d{{1,3}}[A-Za-z]{{0,2}}|[a-z]{{1,3}})\))[ \t]*$")
_PARA_START_RE = re.compile(
    rf"^[ \t]*{_REF}*\[*(?:"
    r"\d{1,3}[A-Z]{0,2}(?:-[A-Z])?\.[ \t\[(]"                               # section header "17A. ", "25-A. "
    r"|\((?:\d{1,3}[A-Za-z]{0,2}|[a-z]{1,3}|[ivxl]{1,5})\)"                 # (1) (1a) (a) (bb) (iv)
    r"|Provided\b|Explanation\b|Illustrations?\b|CHAPTER\b|PART\b|(?:THE[ \t]+)?SCHEDULE\b|\*)")
_ALLCAPS_LINE_RE = re.compile(r"^[ \t]*[^a-z\n]{2,80}$")                  # headings: "CHAPTER II", "PRELIMINARY"
_HYPHEN_END_RE = re.compile(r"([A-Za-z]+)-$")

_KNOWN_SINGLE_WORD_LINE_BREAKS = frozenset({"education", "appointment"})

# -- law name ----------------------------------------------------------------------------------
_TITLE_LINE_RE = re.compile(
    rf"^[ \t]*{_REF}*[ \t]*(?:THE[ \t]+)?([A-Z][A-Z .,'&()\-]*?(?:ACT|ORDINANCE|ORDER|CODE|RULES|REGULATIONS?))"
    r"[ \t]*,?[ \t]*(\d{4})[ \t]*$")
_ARTIFACT_NAME_RE = re.compile(r"^\s*(?:Page\s+\d+\s+of\s+\d+|FINAL COPY|Updated till|\d)", re.IGNORECASE)
_SMALL_WORDS = {"and", "of", "the", "in", "for", "on", "to", "or", "by"}

_MULTI_SPACE_RE = re.compile(r"[ \t]{2,}")
_PAGE_MARKER_WHOLE_RE = re.compile(r"^Page\s+\d+\s+of\s+\d+$", re.IGNORECASE)


@dataclass
class Footnote:
    id: str
    page: int
    number: int
    text: str
    referenced: bool = False

    def to_dict(self) -> dict:
        return {"id": self.id, "page": self.page, "number": self.number, "text": self.text,
                "referenced": self.referenced}


@dataclass
class CleaningDiagnostics:
    raw_length: int
    cleaned_length: int = 0
    chars_removed: int = 0
    pct_removed: float = 0.0
    toc_removed: bool = False
    metadata_law_name_cleared: bool = False
    large_removal_flag: bool = False
    footnotes_relocated_count: int = 0
    pages: int = 0
    pages_with_footnotes: int = 0
    markers_linked: int = 0
    unresolved_markers: list = field(default_factory=list)
    misread_markers_fixed: list = field(default_factory=list)   # footnote "1" printed as "l" ("l[provincial ...")
    unreferenced_footnotes: list = field(default_factory=list)
    print_codes_removed: list = field(default_factory=list)
    bare_labels_joined: int = 0
    max_section_number: int = 0
    pct_removed_non_whitespace: float = 0.0
    law_name_source: str | None = None
    letter_preservation_ok: bool | None = None
    letter_preservation_diff: dict = field(default_factory=dict)
    paren_counts: dict = field(default_factory=dict)          # {"(": [raw, cleaned], ")": [raw, cleaned]}
    stray_footnote_lines: list = field(default_factory=list)
    bare_label_lines_left: list = field(default_factory=list)
    quality_gate_errors: list = field(default_factory=list)

    def to_dict(self) -> dict:
        return dict(self.__dict__)


# -- pages / TOC -------------------------------------------------------------------------
def _find_toc_span(text: str) -> tuple[int, int] | None:
    start_match = _TOC_START_RE.search(text)
    if not start_match:
        return None
    window = text[start_match.end():min(len(text), start_match.end() + _TOC_SEARCH_WINDOW_CHARS)]
    best = None
    for pattern in _TOC_END_RES:
        m = pattern.search(window)
        if m and (best is None or m.start() < best):
            best = m.start()
    if best is None:
        return None
    end = start_match.end() + best
    while end < len(text) and text[end] in " \t\r\n":           # the anchor match may start with whitespace
        end += 1
    if text[end:end + 1] != "(":
        k = end                                                   # "1ACT No. ..." -> keep a footnote marker glued
        while k > start_match.end() and text[k - 1].isdigit() and end - k < 2:   # to the anchor (1-2 digits, no space)
            k -= 1
        if k < end and not text[k - 1].isdigit():
            end = k
    return start_match.start(), end


def _split_pages(full_text: str) -> list[tuple[int, int, int]]:
    """(start, end, page_number) spans covering full_text. Ingestion joins pages with a blank
    triple newline; the "Page N of M" line inside each page gives its number."""
    spans, pos = [], 0
    for m in re.finditer(r"\n\n\n", full_text):
        spans.append((pos, m.start()))
        pos = m.end()
    spans.append((pos, len(full_text)))
    markers = len(re.findall(r"(?mi)^[ \t]*Page[ \t]+\d+[ \t]+of[ \t]+\d+[ \t]*$", full_text))
    if markers and len(spans) != markers:                       # fall back to the page markers themselves
        starts = [m.start() for m in re.finditer(r"(?mi)^[ \t]*(?:FINAL COPY[^\n]*\n[ \t]*)?Page[ \t]+\d+[ \t]+of[ \t]+\d+",
                                                 full_text)]
        if starts:
            starts[0] = 0
            spans = [(s, starts[i + 1] if i + 1 < len(starts) else len(full_text)) for i, s in enumerate(starts)]
    out = []
    for i, (s, e) in enumerate(spans):
        m = re.search(r"(?mi)^[ \t]*Page[ \t]+(\d+)[ \t]+of[ \t]+\d+", full_text[s:e])
        out.append((s, e, int(m.group(1)) if m else i + 1))
    return out


def _remove_furniture(lines: list[str], diag: CleaningDiagnostics) -> list[str]:
    out: list[str] = []
    i = 0
    while i < len(lines):
        line = lines[i]
        if (_PAGE_MARKER_RE.match(line) or _DATE_STAMP_RE.match(line) or _FINAL_COPY_STAMP_RE.match(line)
                or _DIVIDER_LINE_RE.match(line) or _LONE_GLYPH_RE.match(line)):
            i += 1
            continue
        if _PRINT_CODE_RE.match(line):
            diag.print_codes_removed.append(line.strip())
            i += 1
            continue
        if _STAR_LINE_RE.match(line):                       # collapse a run of "*" lines into one "* * *"
            j = i
            while j < len(lines) and (_STAR_LINE_RE.match(lines[j]) or not lines[j].strip()):
                j += 1
            run = sum(1 for k in range(i, j) if _STAR_LINE_RE.match(lines[k]))
            out.append("* * *" if run > 1 else "*")
            i = j
            continue
        out.append(line)
        i += 1
    return out


def _split_footnote_zone(lines: list[str]) -> tuple[list[str], list[tuple[int, list[str]]]]:
    """Returns (body_lines, [(number, entry_lines)]). The zone is the LAST run at the bottom of
    the page that starts with entry 1 and continues 2, 3, ... to the end of the page."""
    nonblank = [i for i, l in enumerate(lines) if l.strip()]
    if not nonblank:
        return lines, []
    candidates = [i for i in nonblank if (m := _FN_ENTRY_RE.match(lines[i])) and m.group(1) in ("1", "l")]
    for z in reversed(candidates):
        entries: list[tuple[int, list[str]]] = []
        expected, ok = 1, True
        for i in range(z, len(lines)):
            l = lines[i]
            if not l.strip():
                continue
            m = _FN_ENTRY_RE.match(l)
            if m and (1 if m.group(1) == "l" else int(m.group(1))) == expected:
                entries.append((expected, [l]))
                expected += 1
            elif entries:
                entries[-1][1].append(l)
            else:
                ok = False
                break
        if not ok or not entries:
            continue
        # every entry must carry text (on its own line or the next), and the block must read as notes
        if any(not re.sub(r"^[ \t]*(?:\d{1,2}|l)\.?", "", " ".join(e)).strip() for _, e in entries):
            continue
        if not _FN_VOCAB_RE.search(" ".join(" ".join(e) for _, e in entries)):
            continue
        body = lines[:z]
        while body and not body[-1].strip():
            body.pop()
        return body, entries
    return lines, []


def _entry_text(entry_lines: list[str]) -> str:
    first = re.sub(r"^[ \t]*(?:\d{1,2}|l(?=[A-Z]))\.?[ \t]*", "", entry_lines[0], count=1)
    parts = [p.strip() for p in [first] + entry_lines[1:] if p.strip()]
    return _MULTI_SPACE_RE.sub(" ", " ".join(parts)).strip()


def _link_markers(line: str, page: int, numbers: set[int], used: set[int], unresolved: list,
                  misread: list) -> str:
    """Rewrites footnote markers to page-scoped references. Only numbers that HAVE a footnote
    on this page are touched, so ordinary numbers in the legal text are never altered."""
    def inline(m):
        n = 1 if m.group(1) == "l" else int(m.group(1))
        if n in numbers:
            used.add(n)
            if m.group(1) == "l":
                misread.append(f"p{page}: l[ -> [^{page}.1][")
            return f"[^{page}.{n}]"
        unresolved.append(f"p{page}:{m.group(0)}[")
        return m.group(0)
    line = _INLINE_MARKER_RE.sub(inline, line)

    def glued(m):
        n = int(m.group(2))
        if n in numbers:
            used.add(n)
            return f"{m.group(1)}[^{page}.{n}]"
        return m.group(0)
    line = _GLUED_TITLE_MARKER_RE.sub(glued, line)

    def simple(m):
        n = int(m.group(1))
        if n in numbers:
            used.add(n)
            return f"[^{page}.{n}]"
        return m.group(0)
    line = _GLUED_WORD_MARKER_RE.sub(simple, line)
    line = _TRAILING_MARKER_RE.sub(lambda m: f"{simple(m)}", line)
    return _STAR_MARKER_RE.sub(simple, line)


# -- paragraph repair -----------------------------------------------------------------------
class _HyphenEvidence:
    def __init__(self, text: str):
        self.hyph = Counter(m.lower() for m in re.findall(r"\b[A-Za-z]+-[A-Za-z]+\b", text.replace("-\n", "~")))
        # unhyphenated spellings the document itself uses (line-wrapped occurrences NOT pre-joined,
        # otherwise the occurrence being decided would count as evidence for itself)
        self.joined = Counter(w.lower() for w in re.findall(r"\b[A-Za-z]+\b", text.replace("-\n", " ")))

    def keep(self, pre: str, suf: str) -> bool:
        if suf[:1].isupper():
            return True
        if (pre + suf).lower() in _KNOWN_SINGLE_WORD_LINE_BREAKS:
            return False
        h, j = self.hyph.get(f"{pre}-{suf}".lower(), 0), self.joined.get((pre + suf).lower(), 0)
        # keep the hyphen unless the unhyphenated word is clearly the document's own spelling
        return not (j > 0 and h == 0)


_SECTION_LINE_RE = re.compile(rf"^[ \t]*{_REF}*\[?(\d{{1,3}})[A-Z]{{0,2}}(?:-[A-Z])?\.[ \t]+\S")
_BARE_NUMBER_RE = re.compile(rf"^[ \t]*{_REF}*\[?(\d{{1,3}})[A-Z]{{0,2}}(?:-[A-Z])?\.[ \t]*$")


def _max_section_number(lines: list[str]) -> int:
    nums = [int(m.group(1)) for l in lines if (m := _SECTION_LINE_RE.match(l))]
    return max(nums) if nums else 0


def _is_label(line: str, max_section: int) -> bool:
    """A bare "N." is a section label only when N fits the Act's numbering; "186." in a court form
    ("The day of ___, 186.") is a date, not section 186 of a 62-section Act."""
    if not _BARE_LABEL_RE.match(line):
        return False
    m = _BARE_NUMBER_RE.match(line)
    return not (m and max_section and int(m.group(1)) > max_section + 5)


def _join_bare_labels(lines: list[str], diag: CleaningDiagnostics) -> list[str]:
    max_section = _max_section_number(lines)
    diag.max_section_number = max_section
    out: list[str] = []
    i = 0
    while i < len(lines):
        line = lines[i]
        if _is_label(line, max_section):
            j = i + 1
            while j < len(lines) and not lines[j].strip() and j - i <= 2:
                j += 1
            if j < len(lines) and lines[j].strip() and not _is_label(lines[j], max_section):
                out.append(f"{line.rstrip()} {lines[j].strip()}")
                diag.bare_labels_joined += 1
                i = j + 1
                continue
        out.append(line)
        i += 1
    return out


def _join(parts: list[str], ev: _HyphenEvidence) -> str:
    out = ""
    for p in parts:
        p = _MULTI_SPACE_RE.sub(" ", p).strip()
        if not p:
            continue
        m = _HYPHEN_END_RE.search(out)
        if m and re.match(r"[A-Za-z]", p):
            suf = re.match(r"[A-Za-z]+", p).group(0)
            out = (out if ev.keep(m.group(1), suf) else out[:-1]) + p
        else:
            out = f"{out} {p}" if out else p
    return out


def _unwrap(lines: list[str], ev: _HyphenEvidence) -> list[str]:
    paras: list[list[str]] = []
    for line in lines:
        if not line.strip():
            if paras and paras[-1]:
                paras.append([])
            continue
        heading = bool(_ALLCAPS_LINE_RE.match(line)) and bool(re.search(r"[A-Z]{3}", line))
        if not paras or not paras[-1] or heading or _PARA_START_RE.match(line):
            paras.append([line])
            if heading:
                paras.append([])
        else:
            paras[-1].append(line)
    return [_join(p, ev) for p in paras if p]


# -- metadata ---------------------------------------------------------------------------------
def _title_case(name: str) -> str:
    words = name.lower().split()
    return " ".join(w if (i and w in _SMALL_WORDS) else w[:1].upper() + w[1:] for i, w in enumerate(words))


_TITLE_NO_YEAR_RE = re.compile(
    rf"^[ \t]*{_REF}*[ \t]*(?:THE[ \t]+)?([A-Z][A-Z .,'&()\-]*?(?:ACT|ORDINANCE|ORDER|CODE|RULES|REGULATIONS?))[ \t]*,?[ \t]*$")
_YEAR_LINE_RE = re.compile(r"(?:\b(?:ACT|ORDINANCE)\b[^\n]*?\bOF[ \t]+(\d{4})|\[[^\]]*?(\d{4})\])", re.IGNORECASE)


def _extract_law_name(body: str) -> tuple[str | None, int | None, str | None]:
    lines = body.split("\n")[:8]
    for i, line in enumerate(lines):
        m = _TITLE_LINE_RE.match(line)
        if m:
            return _title_case(m.group(1).strip(" ,")), int(m.group(2)), line.strip()
        m = _TITLE_NO_YEAR_RE.match(line)
        if m and not re.match(r"^[ \t]*ACT[ \t]+No", line, re.I):
            year = None
            for nxt in lines[i + 1:i + 4]:
                ym = _YEAR_LINE_RE.search(nxt)
                if ym:
                    year = int(ym.group(1) or ym.group(2))
                    break
            return _title_case(m.group(1).strip(" ,")), year, line.strip()
    return None, None, None


# -- verification ---------------------------------------------------------------------------
def _letters(s: str) -> Counter:
    return Counter(re.sub(r"[^a-z]", "", s.lower()))


# -- main --------------------------------------------------------------------------------------
def clean_statute_text_full(full_text: str) -> tuple[str, CleaningDiagnostics, list[Footnote]]:
    full_text = full_text.replace("\r\n", "\n").replace("\r", "\n")
    diag = CleaningDiagnostics(raw_length=len(full_text))
    ev = _HyphenEvidence(full_text)

    toc = _find_toc_span(full_text)
    diag.toc_removed = toc is not None
    pages = _split_pages(full_text)
    diag.pages = len(pages)

    body_lines: list[str] = []
    expected_text: list[str] = []                 # raw text minus furniture/TOC, for verification
    footnotes: list[Footnote] = []
    for s, e, page_no in pages:
        text = full_text[s:e]
        if toc:                                   # drop the part of this page inside the CONTENTS listing
            ts, te = max(toc[0], s), min(toc[1], e)
            if ts < te:
                text = full_text[s:ts] + full_text[te:e]
        lines = _remove_furniture(text.split("\n"), diag)
        expected_text.append("\n".join(lines))
        page_body, entries = _split_footnote_zone(lines)
        numbers = {n for n, _ in entries}
        if entries:
            diag.pages_with_footnotes += 1
        used: set[int] = set()
        page_body = [_link_markers(l, page_no, numbers, used, diag.unresolved_markers, diag.misread_markers_fixed)
                     for l in page_body]
        body_lines.extend(page_body)
        for n, ent in entries:
            if re.match(r"^[ \t]*l(?=[A-Z])", ent[0]):
                diag.misread_markers_fixed.append(f"p{page_no}: footnote number 'l' -> 1")
            footnotes.append(Footnote(id=f"{page_no}.{n}", page=page_no, number=n, text=_entry_text(ent),
                                      referenced=n in used))

    body_lines = _join_bare_labels(body_lines, diag)
    paras = _unwrap(body_lines, ev)
    body = "\n".join(paras)
    body = re.sub(r"\(\s+([0-9a-zA-Z]{1,4})\s+\)", r"(\1)", body)          # "( 1 )" -> "(1)"
    body = re.sub(r"(?<=[.,:])_{3,}(?=\S)", "—", body)                      # "commencement.___(1)" -> "commencement.—(1)"
    body = _MULTI_SPACE_RE.sub(" ", body).strip()

    diag.markers_linked = len(re.findall(r"\[\^[^\]]+\]", body))
    diag.footnotes_relocated_count = len(footnotes)
    diag.unreferenced_footnotes = [f.id for f in footnotes if not f.referenced]
    diag.stray_footnote_lines = [l[:90] for l in body.split("\n") if _STRAY_FOOTNOTE_RE.match(l)]
    diag.bare_label_lines_left = [l for l in body.split("\n") if _is_label(l, diag.max_section_number)]

    # letter preservation: nothing lost, nothing invented
    a = _letters("\n".join(expected_text))
    a["l"] -= len(diag.misread_markers_fixed)      # each "l[" marker is a misread "1", not a legal letter
    a = +a
    b = _letters(body + "".join(f.text for f in footnotes))
    missing, extra = dict(a - b), dict(b - a)
    diag.letter_preservation_ok = not missing and not extra
    # punctuation that carries legal structure: clause labels "(a)", "(1)" and parenthetical citations
    exp_txt, got_txt = "\n".join(expected_text), body + "".join(f.text for f in footnotes)
    diag.paren_counts = {c: [exp_txt.count(c), got_txt.count(c)] for c in "()"}
    diag.letter_preservation_diff = {"missing": missing, "extra": extra}

    block = "\n".join(f"[^{f.id}]: {f.text}" for f in footnotes)
    cleaned = f"{body}\n\n[FOOTNOTES]\n{block}\n[/FOOTNOTES]" if footnotes else body
    diag.cleaned_length = len(cleaned)
    diag.chars_removed = diag.raw_length - diag.cleaned_length
    diag.pct_removed = round(diag.chars_removed / diag.raw_length, 4) if diag.raw_length else 0.0
    ns = lambda t: len(re.sub(r"\s", "", t))
    raw_ns = ns(full_text)
    diag.pct_removed_non_whitespace = round((raw_ns - ns(cleaned)) / raw_ns, 4) if raw_ns else 0.0
    diag.large_removal_flag = diag.pct_removed_non_whitespace > LARGE_REMOVAL_FLAG_THRESHOLD
    return cleaned, diag, footnotes


def clean_statute_text(full_text: str) -> tuple[str, CleaningDiagnostics]:
    """Backward-compatible API (v1 signature): returns (cleaned_text, diagnostics)."""
    cleaned, diag, _ = clean_statute_text_full(full_text)
    return cleaned, diag


def quality_gate(diag: CleaningDiagnostics) -> list[str]:
    errors = []
    if not diag.letter_preservation_ok:
        errors.append(f"letter preservation failed: {diag.letter_preservation_diff}")
    bad = {c: v for c, v in diag.paren_counts.items() if v[0] != v[1]}
    if bad:
        errors.append(f"parentheses not preserved (raw vs cleaned): {bad}")
    if diag.stray_footnote_lines:
        errors.append(f"footnote text left inside the legal text: {diag.stray_footnote_lines[:5]}")
    if diag.bare_label_lines_left:
        errors.append(f"section/clause labels left alone on a line: {diag.bare_label_lines_left[:5]}")
    if diag.large_removal_flag:
        errors.append(f"removed {diag.pct_removed_non_whitespace:.1%} of the raw text's non-space characters")
    return errors


def clean_statute_document(raw_doc: dict) -> dict:
    full_text = raw_doc["full_text"]
    cleaned_text, diag, footnotes = clean_statute_text_full(full_text)

    metadata = dict(raw_doc["metadata"])
    body = cleaned_text.split("\n\n[FOOTNOTES]\n", 1)[0]
    name, year, title_line = _extract_law_name(body)
    current = metadata.get("law_name")
    if isinstance(current, str) and _PAGE_MARKER_WHOLE_RE.match(current.strip()):
        diag.metadata_law_name_cleared = True
        current = None
    if name and (not current or _ARTIFACT_NAME_RE.match(str(current))):
        metadata["law_name_raw"] = raw_doc["metadata"].get("law_name")
        metadata["law_name"] = name
        diag.law_name_source = "title_line"
    elif current:
        metadata["law_name"] = current
        diag.law_name_source = "ingestion_metadata"
    else:
        metadata["law_name"] = None
    if metadata.get("year") is None and year:
        metadata["year"] = year
    metadata["law_title"] = title_line

    diag.quality_gate_errors = quality_gate(diag)
    return {
        "doc_id": raw_doc["doc_id"],
        "metadata": metadata,
        "full_text": full_text,
        "cleaned_text": cleaned_text,
        "footnotes": [f.to_dict() for f in footnotes],
        "cleaning_diagnostics": diag.to_dict(),
    }


def clean_directory(input_dir: Path = INPUT_DIR, output_dir: Path = OUTPUT_DIR) -> dict:
    """Cleans every <doc_id>.json in input_dir. Duplicate documents (same content hash, e.g. a
    PDF downloaded twice) are cleaned once; later copies are skipped and reported. Documents
    that fail the quality gate are written to output_dir/_rejected/ for review."""
    input_dir, output_dir = Path(input_dir), Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    processed = large_removal_flagged = 0
    failed_docs, rejected_docs, duplicates = [], [], []
    seen_hashes: dict[str, str] = {}

    for raw_path in sorted(input_dir.glob("*.json")):
        try:
            raw_doc = json.loads(raw_path.read_text(encoding="utf-8"))
            h = (raw_doc.get("metadata") or {}).get("content_hash") or \
                hashlib.sha256(raw_doc["full_text"].encode("utf-8")).hexdigest()
            if h in seen_hashes:
                duplicates.append({"source_file": raw_path.name, "duplicate_of": seen_hashes[h]})
                continue
            seen_hashes[h] = raw_doc["doc_id"]
            cleaned_doc = clean_statute_document(raw_doc)
        except (OSError, json.JSONDecodeError, KeyError) as exc:
            failed_docs.append({"source_file": raw_path.name, "error": str(exc)})
            continue

        errors = cleaned_doc["cleaning_diagnostics"]["quality_gate_errors"]
        target = output_dir
        if errors:
            target = output_dir / "_rejected"
            target.mkdir(parents=True, exist_ok=True)
            rejected_docs.append({"source_file": raw_path.name, "errors": errors})
        out_path = target / f"{cleaned_doc['doc_id']}.json"
        tmp_path = out_path.with_suffix(".json.tmp")
        tmp_path.write_text(json.dumps(cleaned_doc, ensure_ascii=False, indent=2), encoding="utf-8")
        tmp_path.replace(out_path)
        processed += 1
        if cleaned_doc["cleaning_diagnostics"]["large_removal_flag"]:
            large_removal_flagged += 1

    return {"processed": processed, "failed": len(failed_docs), "rejected": len(rejected_docs),
            "duplicates_skipped": len(duplicates), "large_removal_flagged": large_removal_flagged,
            "failed_docs": failed_docs, "rejected_docs": rejected_docs, "duplicates": duplicates}


def main() -> None:  # pragma: no cover -- thin CLI wrapper
    summary = clean_directory()
    print(f"Statute cleaning: {summary['processed']} processed, {summary['failed']} failed, "
          f"{summary['rejected']} rejected by quality gate, {summary['duplicates_skipped']} duplicates skipped, "
          f"{summary['large_removal_flagged']} flagged for large removal")
    for f in summary["failed_docs"]:
        print(f"  FAILED {f['source_file']}: {f['error']}")
    for r in summary["rejected_docs"]:
        print(f"  REJECTED {r['source_file']}:")
        for e in r["errors"]:
            print(f"    - {e}")
    for d in summary["duplicates"]:
        print(f"  DUPLICATE {d['source_file']} (same content as {d['duplicate_of']})")
    if summary["failed"] or summary["rejected"]:
        raise SystemExit(1)


if __name__ == "__main__":  # pragma: no cover
    main()