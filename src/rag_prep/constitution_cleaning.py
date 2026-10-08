# """Constitution of Pakistan cleaning -- Stage 2 of the independent
# Constitution pipeline.

#     raw ingestion JSON (var/rag/constitution_ingested/<doc_id>.json)
#         -> clean_constitution_text()    [conservative noise removal + whitespace normalization]
#         -> cleaned constitution JSON    (var/rag/cleaned_constitution/<doc_id>.json)

# This stage removes PDF-extraction/publication artifacts (cover/title/
# preface/TOC front matter, repeated running headers, page numbers,
# printer metadata, decorative dividers, page-bottom amendment-history
# footnotes, page-break hyphenation) while leaving every substantive
# constitutional word -- including every amendment marker, every
# "[Omitted]"/"* * * *" status, and every bracket that is not a bare
# amendment-footnote reference wrapper -- untouched. It never summarizes,
# rewrites, paraphrases, classifies, interprets, parses Articles, or
# chunks anything -- see ``CONSERVATIVE CLEANING RULE`` below.

# Deliberately its own module, NOT a reuse of statute_cleaning.py: the
# Constitution is one document with its own distinct publication format
# (a different publisher, different running-header/footnote conventions,
# and -- critically -- the opposite statistical default for page-break
# hyphenation; see _join_hyphenated_linebreaks's docstring). Reusing the
# statute cleaner's rules verbatim would import assumptions that do not
# hold for this source. It is modelled on statute_cleaning.py's
# architecture (same conservative philosophy, same raw/cleaned-text
# separation, same "narrow evidenced regex, no-op when the anchor isn't
# found" discipline) purely for consistency, never imported from it.

# CONSERVATIVE CLEANING RULE
# ---------------------------
#     legal content preservation  >  noise removal  >  formatting normalization

# Every pattern removed here requires an unambiguous, narrowly-scoped
# anchor, validated directly against the real, single Constitution of
# Pakistan document in var/rag/constitution_ingested/ (not assumed).
# Several task-requested transformations (Arabic-script decorative
# invocations, Article/marginal-note association, typographic bracket
# normalization around bare clause numbering with no amendment digit)
# were investigated against the real document and found to have no
# real-world instance to safely act on; their detection logic exists (so
# they are not silently unimplemented) but is a documented no-op on this
# corpus rather than a guess.

# PIPELINE ISOLATION
# -------------------
# No import of src.ingestion/, src.classification/, src.rag_prep.
# case_cleaner/structurer*/chunk*/statute_*, or src.extraction.
# case_loader. Independent entry point, independent input/output
# directories. Never touches the case-law or statute pipelines' var/rag/
# directories or var/metadata.db.
# """

# from __future__ import annotations

# import json
# import re
# from dataclasses import dataclass
# from pathlib import Path

# INPUT_DIR = Path("var/rag/constitution_ingested")
# OUTPUT_DIR = Path("var/rag/cleaned_constitution")

# # Same diagnostic-flag convention as statute_cleaning.py.
# LARGE_REMOVAL_FLAG_THRESHOLD = 0.15

# # -- front/back matter (cover, title page, preface, TOC) ---------------------
# # The real document's front matter is bounded by a unique, reliable
# # anchor: a line that is exactly "PREAMBLE" (the Constitution's own
# # first substantive line, confirmed to occur exactly once in the real
# # document). Everything before it -- cover page, title page, printer
# # metadata, PREFACE, and the multi-page roman-numeral-paginated
# # Table of Contents -- carries no legal force and is removed. If this
# # anchor is not found, nothing is removed (never guess where the real
# # text "probably" starts).
# _PREAMBLE_ANCHOR_RE = re.compile(r"^[ \t]*PREAMBLE[ \t]*$", re.MULTILINE)

# # -- printer/publisher metadata -----------------------------------------------
# # "2221(25)L&J---by Waleed---PC-2 (Quark 10)" -- a one-off print-shop job
# # tag on the cover page (confirmed: occurs exactly once in the real
# # document). Already inside the front-matter span removed above in
# # practice, but kept as its own narrowly-anchored rule in case a future
# # edition of this same publication places it elsewhere.
# _PRINTER_METADATA_RE = re.compile(
#     r"^[ \t]*\d+\(\d+\)[A-Za-z&]+---by[ \t]+\S+---PC-\d+[ \t]*\([^)\n]*\)[ \t]*$",
#     re.MULTILINE,
# )

# # -- running header + page number ---------------------------------------------
# # "CONSTITUTION OF PAKISTAN" is the running header on every body page.
# # Confirmed real layouts (all four combined into one rule):
# #   1. bare, own line, no number  -- front-matter pages (paired with a
# #      separate roman-numeral page marker -- see group 2 below)
# #   2. "<page_num>   CONSTITUTION OF PAKISTAN" / the reverse, SAME line
# #      (recto/verso alternation, confirmed pages ~1-65)
# #   3. "CONSTITUTION OF PAKISTAN" alone, then a page number ALONE on the
# #      very next line (confirmed from ~page 67 onward -- a PDF text-
# #      extraction-order artifact, not a different header)
# #   4. "CONSTITUTION OF PAKISTAN" with a stray digit GLUED directly onto
# #      "PAKISTAN" (confirmed once, page 1 only -- the same kind of
# #      lost-superscript artifact as the amendment-footnote references
# #      handled by _FOOTNOTE_LEAD_WORD_RE, just landing on the header
# #      instead of a word) -- "\d*" right after PAKISTAN absorbs it.
# # The second line's own page-number shape (arabic 1-3 digits, or a
# # bare/parenthesised lowercase roman numeral, confirmed as both "(i)"
# # in front matter and, generalised here, a bare "i" per the task's own
# # example) is consumed together with the header line in ONE match, so
# # the two are only ever removed as a pair -- never a bare number alone
# # (which would risk the real Article-51/106 seat-count table data, each
# # number of which also sits alone on its own line; those numbers are
# # never adjacent to this header, so they are never matched here).
# _RUNNING_HEADER_RE = re.compile(
#     r"^[ \t]*\d{0,3}[ \t]*CONSTITUTION OF PAKISTAN\d*[ \t]*\d{0,3}[ \t]*\n"
#     r"(?:[ \t]*\(?[ivxlcdm]{1,6}\)?[ \t]*\n|[ \t]*\d{1,3}[ \t]*\n)?",
#     re.MULTILINE,
# )

# # -- decorative dividers --------------------------------------------------------
# # A standalone horizontal-rule line, in either of the two real styles
# # observed: a run of underscores, or a run of en/em dashes (U+2010-
# # U+2015). Never touches the "* * * *" omission marker (asterisks are
# # outside this character class entirely) or an inline em-dash used as
# # ordinary punctuation within a sentence (this only matches when the
# # WHOLE line is dash characters and nothing else).
# _DIVIDER_LINE_RE = re.compile(r"^[ \t]*(?:_{3,}|[‐-―]{2,})[ \t]*$", re.MULTILINE)

# # -- page-bottom amendment/footnote lines --------------------------------------
# # Same lost-superscript signature as statute_cleaning.py's footnote
# # relocation (a footnote-reference number glued directly to its own
# # footnote text, with no period/space in between -- a genuine Article/
# # clause number is always "<id>.<ws>" instead). This vocabulary was
# # built the same way: by enumerating every such glued-digit line across
# # the real document and keeping only words that never collided with
# # real heading/title text. One real collision was found and excluded:
# # "1CHAPTER 3A.--FEDERAL SHARIAT COURT" is a genuine Chapter heading
# # with a stray reference digit glued to its front, not a footnote --
# # "CHAPTER" is therefore deliberately NOT in this vocabulary. Bare
# # ordinal date suffixes ("10th", "6th") were also excluded for the same
# # reason a vocabulary word must never be just 1-2 letters that could
# # coincidentally be an ordinal suffix rather than a real lead word.
# _FOOTNOTE_LEAD_WORD_RE = re.compile(
#     r"^[ \t]*\d{1,2}(?:Subs|Sub|lns|Ins|Omitted|Added|Rep|Proviso|Provisos|Provisios|New|"
#     r"See|For|The|Clause|Clauses|Article|Articles|Paragraph|Paragraphs|Explanation|Re|"
#     r"Entry|Entries|Certain|Existing|Order|First|Sixth|Concurrent)\b"
# )
# # A genuine Article/clause/sub-clause header -- "51. (1)", "9A.", "106."
# # -- always has a period followed by whitespace right after the
# # (optional letter-suffixed) number. Belt-and-suspenders, mirroring
# # statute_chunker.py's own _SECTION_HEADER_LIKE_RE: the vocabulary above
# # already can't collide with this shape on its own.
# _ARTICLE_HEADER_LIKE_RE = re.compile(r"^[ \t]*\d+[A-Za-z]{0,2}\.[ \t]")
# # A resumed inline amendment marker ("2[(2) The territories...") ends a
# # footnote entry's continuation-line absorption, same rationale as the
# # statute pipeline's equivalent check.
# _INLINE_AMENDMENT_MARKER_START_RE = re.compile(r"^[ \t]*\d+\[")

# # -- amendment-bracket normalization -------------------------------------------
# # Every real amendment bracket in this document is "<digits>[...]" (a
# # lost-superscript reference number glued directly to an opening
# # bracket -- the same artifact as the footnote lead-words above, just
# # wrapping a bracket instead of starting a footnote line). A full sweep
# # of the real document found exactly two shapes:
# #   * ~93% (400/431): a short inserted/substituted WORD OR PHRASE, e.g.
# #     "4[or is about to be]", "2[Balochistan]" -- the brackets are the
# #     substantive, permanent "this text was amended here" marking and
# #     must survive; only the reference digit (which exists purely to
# #     point at a footnote the main text no longer displays inline) is
# #     dropped. Task rule #5's own example ("¹[commission of]" ->
# #     "[commission of]") is exactly this shape.
# #   * ~7% (31/431): the bracket's own content STARTS with the clause's
# #     real identifying number, e.g. "2[(2) The territories of Pakistan
# #     shall comprise..." -- here the whole numbered clause (or whole
# #     Article, e.g. "2[106. (1) Each Provincial Assembly...") was
# #     substituted, and the outer "<digit>[...]" is a redundant wrapper
# #     around a clause/Article that already carries its own proper
# #     number. Stripping the ENTIRE wrapper, not just the digit, leaves
# #     a clean "(2) The territories..." that reads as an ordinary clause
# #     -- task rule #9's own example ("[(2) The territories...]" ->
# #     "(2) The territories...").
# # Applied innermost-first (same technique as statute_chunker.py's
# # _derive_embedding_text) so a nested amendment unwraps fully.
# _AMENDMENT_BRACKET_RE = re.compile(r"\d+\[([^\[\]]*)\]")
# _BRACKET_CONTENT_IS_CLAUSE_NUMBER_RE = re.compile(r"^(?:\(\d+[A-Za-z]{0,2}\)|\([a-zA-Z0-9]{1,4}\)|\d+[A-Za-z]{0,2}\.)")

# # -- page-break hyphenation -----------------------------------------------------
# # Unlike the statute corpus (where most real line-wrapped hyphens were
# # genuine compounds, e.g. "sub-section"), a full sweep of every real
# # hyphen-linebreak in THIS document found the opposite distribution:
# # 84 distinct prefix/suffix pairs, of which only ~9 are genuine
# # compounds that must keep their hyphen (verified against this
# # document's OWN single-line usage elsewhere, e.g. "sub-paragraph"
# # appears hyphenated 8 times and "subparagraph" 0 times) -- the other
# # ~75 (accor-dance, Funda-mental, educa-tional, Govern-ment, ...) are
# # ordinary single English words broken purely by the PDF's line width,
# # which must be rejoined WITHOUT a hyphen. The default here is therefore
# # inverted from statute_cleaning.py's: join removing the hyphen, except
# # for this small, evidenced preserve-list.
# _HYPHEN_LINEBREAK_RE = re.compile(r"([A-Za-z]+)-[ \t]*\n[ \t]*([a-z][A-Za-z]*)")
# _KNOWN_HYPHENATED_COMPOUNDS = frozenset({
#     "majlise",  # "Majlis-e-Shoora" -- confirmed 173x hyphenated in-document
#     "nondiscrimination",  # standard English compound, never one word
#     "selfincrimination",  # standard fixed legal term, never one word
#     "subparagraph",  # confirmed 8x hyphenated, 0x as one word, in-document
#     "noconfidence",  # confirmed 11x hyphenated, 0x as one word, in-document
#     "wellbeing",  # confirmed 13x hyphenated, 0x as one word, in-document
#     "hydroelectric",  # confirmed 6x hyphenated, 0x as one word, in-document
#     "fullstop",  # both "full stop" and "full-stop" coexist in-document;
#                  # hyphenating the wrap is the closer of the two to "not
#                  # inventing a new word" (never "fullstop" as one word)
#     "twentythird",  # spelled-out ordinals are always hyphenated in English
# })

# # -- generic whitespace normalization (never touches letters/digits) ---------
# _CRLF_RE = re.compile(r"\r\n?")
# _MULTI_SPACE_RE = re.compile(r"[ \t]{2,}")
# _MULTI_BLANK_LINE_RE = re.compile(r"\n{3,}")
# _TRAILING_LINE_WS_RE = re.compile(r"[ \t]+\n")
# _PAREN_MARKER_SPACE_RE = re.compile(r"\(\s+([0-9a-zA-Z]{1,4})\s+\)")

# # -- marginal/side-note association (see docstring on _associate_marginal_notes) --
# # A marginal note, if present, would be a short Title-Case label line
# # immediately followed by the matching numbered Article's own text
# # repeating that same label. Investigated directly against the real
# # document: no such paired "label line, then matching Article text"
# # shape was found anywhere in the extracted body (the PDF's marginal-
# # note column, if it exists in the source layout, was not captured as a
# # distinguishable separate line by text-layer extraction). This
# # detector therefore exists, and is tested, but is a documented no-op
# # on the real corpus -- never a guessed/invented heading.
# _MARGINAL_NOTE_CANDIDATE_RE = re.compile(
#     r"^([A-Z][A-Za-z ,.'—-]{3,80})\n[ \t]*\n?[ \t]*(\d+[A-Za-z]{0,2})\.[ \t]+\1\b",
#     re.MULTILINE,
# )


# @dataclass(frozen=True)
# class ConstitutionCleaningDiagnostics:
#     raw_length: int
#     cleaned_length: int
#     chars_removed: int
#     pct_removed: float
#     front_matter_removed: bool
#     footnotes_relocated_count: int
#     large_removal_flag: bool

#     def to_dict(self) -> dict:
#         return {
#             "raw_length": self.raw_length,
#             "cleaned_length": self.cleaned_length,
#             "chars_removed": self.chars_removed,
#             "pct_removed": self.pct_removed,
#             "front_matter_removed": self.front_matter_removed,
#             "footnotes_relocated_count": self.footnotes_relocated_count,
#             "large_removal_flag": self.large_removal_flag,
#         }


# def _remove_front_matter(text: str) -> tuple[str, bool]:
#     """Removes everything before the unique "PREAMBLE" anchor (cover,
#     title page, printer metadata, preface, TOC). No-op if the anchor is
#     not found -- never guesses where the real text starts."""

#     m = _PREAMBLE_ANCHOR_RE.search(text)
#     if not m:
#         return text, False
#     return text[m.start():], True


# def _remove_layout_noise(text: str) -> str:
#     text = _PRINTER_METADATA_RE.sub("", text)
#     text = _RUNNING_HEADER_RE.sub("", text)
#     text = _DIVIDER_LINE_RE.sub("", text)
#     return text


# def _relocate_footnotes(text: str) -> tuple[str, list[str]]:
#     """Identical strategy to statute_cleaning.py's footnote relocation
#     (see that module for the full state-machine rationale): pulls
#     page-bottom amendment/footnote lines out of the substantive body and
#     returns them separately, in original order, so the caller can append
#     them as one coherent ``[FOOTNOTES]`` block."""

#     lines = text.split("\n")
#     body_lines: list[str] = []
#     footnotes: list[str] = []
#     i, n = 0, len(lines)

#     while i < n:
#         line = lines[i]
#         if _FOOTNOTE_LEAD_WORD_RE.match(line) and not _ARTICLE_HEADER_LIKE_RE.match(line):
#             entry_parts = [line.strip()]
#             i += 1
#             while i < n:
#                 nxt = lines[i]
#                 if (
#                     nxt.strip() == ""
#                     or _FOOTNOTE_LEAD_WORD_RE.match(nxt)
#                     or _ARTICLE_HEADER_LIKE_RE.match(nxt)
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


# def _normalize_amendment_brackets(text: str) -> str:
#     """See _AMENDMENT_BRACKET_RE's module-level docstring for the two
#     real shapes this handles and the evidence behind each. Applied
#     repeatedly (innermost bracket first) so nested amendments fully
#     unwrap."""

#     def _repl(m: re.Match) -> str:
#         inner = m.group(1)
#         if _BRACKET_CONTENT_IS_CLAUSE_NUMBER_RE.match(inner):
#             return inner  # whole wrapper (digit + both brackets) dropped
#         return f"[{inner}]"  # only the reference digit dropped; brackets kept

#     while True:
#         new_text = _AMENDMENT_BRACKET_RE.sub(_repl, text)
#         if new_text == text:
#             return new_text
#         text = new_text


# def _join_hyphenated_linebreaks(text: str) -> str:
#     def _repl(m: re.Match) -> str:
#         prefix, suffix = m.group(1), m.group(2)
#         if (prefix + suffix).lower() in _KNOWN_HYPHENATED_COMPOUNDS:
#             return f"{prefix}-{suffix}"
#         return prefix + suffix

#     return _HYPHEN_LINEBREAK_RE.sub(_repl, text)


# def _associate_marginal_notes(text: str) -> str:
#     """Promotes a short label line immediately followed by a matching
#     numbered Article heading into a single "### Article <id>. <label>"
#     line, ONLY when the Article's own text starts by literally repeating
#     that exact label (an unambiguous, narrow structural match -- never
#     an invented or inferred heading). See the module-level regex
#     docstring: this is a documented no-op on the real Constitution of
#     Pakistan corpus (no such paired shape was found there), kept so the
#     capability exists and is tested rather than silently missing."""

#     def _repl(m: re.Match) -> str:
#         label, article_id = m.group(1).strip(), m.group(2)
#         return f"### Article {article_id}. {label}\n\n{article_id}. {label}"

#     return _MARGINAL_NOTE_CANDIDATE_RE.sub(_repl, text)


# def _normalize_whitespace(text: str) -> str:
#     text = _CRLF_RE.sub("\n", text)
#     text = _TRAILING_LINE_WS_RE.sub("\n", text)
#     text = _MULTI_SPACE_RE.sub(" ", text)
#     text = _MULTI_BLANK_LINE_RE.sub("\n\n", text)
#     text = _PAREN_MARKER_SPACE_RE.sub(r"(\1)", text)
#     return text.strip()


# def clean_constitution_text(raw_text: str) -> tuple[str, ConstitutionCleaningDiagnostics]:
#     """Cleans the Constitution's raw extracted text. Returns
#     ``(cleaned_text, diagnostics)``.

#     Order mirrors statute_cleaning.py's own pipeline, for the same
#     reasons: front-matter removal and layout-noise removal operate on
#     the raw line structure first; footnote relocation runs while blank
#     lines still mark each footnote block's real boundary; bracket
#     normalization, hyphenation joining, and marginal-note association
#     run next; whitespace normalization runs last; the relocated
#     footnotes are appended, as one ``[FOOTNOTES]`` block, only after the
#     rest of the body has been fully cleaned and normalized.
#     """

#     raw_length = len(raw_text)

#     text, front_matter_removed = _remove_front_matter(raw_text)
#     text = _remove_layout_noise(text)
#     text, footnote_entries = _relocate_footnotes(text)
#     text = _normalize_amendment_brackets(text)
#     text = _join_hyphenated_linebreaks(text)
#     text = _associate_marginal_notes(text)
#     text = _normalize_whitespace(text)

#     if footnote_entries:
#         footnote_block = "\n".join(footnote_entries)
#         text = f"{text}\n\n[FOOTNOTES]\n{footnote_block}\n[/FOOTNOTES]"

#     cleaned_length = len(text)
#     chars_removed = raw_length - cleaned_length
#     pct_removed = round(chars_removed / raw_length, 4) if raw_length else 0.0

#     diagnostics = ConstitutionCleaningDiagnostics(
#         raw_length=raw_length,
#         cleaned_length=cleaned_length,
#         chars_removed=chars_removed,
#         pct_removed=pct_removed,
#         front_matter_removed=front_matter_removed,
#         footnotes_relocated_count=len(footnote_entries),
#         large_removal_flag=pct_removed > LARGE_REMOVAL_FLAG_THRESHOLD,
#     )
#     return text, diagnostics


# def clean_constitution_document(raw_doc: dict) -> dict:
#     """Cleans one raw ingestion document (constitution_ingestion.py's
#     output shape: doc_id, source_file, metadata, pages, raw_text) into
#     the cleaned-constitution output shape. ``raw_text`` is carried
#     through unchanged as the authoritative record; ``cleaned_text`` is
#     the new, additive field. ``pages`` is intentionally not carried
#     forward -- cleaning operates on the combined raw_text, and no
#     requirement in this stage calls for page-level cleaned output."""

#     raw_text = raw_doc["raw_text"]
#     cleaned_text, diagnostics = clean_constitution_text(raw_text)

#     return {
#         "doc_id": raw_doc["doc_id"],
#         "metadata": dict(raw_doc["metadata"]),
#         "raw_text": raw_text,
#         "cleaned_text": cleaned_text,
#         "cleaning_diagnostics": diagnostics.to_dict(),
#     }


# def clean_directory(input_dir: Path = INPUT_DIR, output_dir: Path = OUTPUT_DIR) -> dict:
#     """Cleans every ``<doc_id>.json`` in ``input_dir`` into
#     ``output_dir/<doc_id>.json``. Never modifies ``input_dir``.

#     Returns {"processed", "failed", "large_removal_flagged", "failed_docs"}.
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
#             cleaned_doc = clean_constitution_document(raw_doc)
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
#     print(f"Constitution cleaning: {summary['processed']} processed, {summary['failed']} failed, "
#           f"{summary['large_removal_flagged']} flagged for large removal")
#     for f in summary["failed_docs"]:
#         print(f"  FAILED {f['source_file']}: {f['error']}")


# if __name__ == "__main__":  # pragma: no cover
#     main()



"""Constitution of Pakistan cleaning -- Stage 2 of the independent
Constitution pipeline (v2: page-aware).

    raw ingestion JSON (var/rag/constitution_ingested/<doc_id>.json)
        -> clean_constitution_document()
        -> cleaned constitution JSON (var/rag/cleaned_constitution/<doc_id>.json)

WHY v2 WORKS PAGE BY PAGE
--------------------------
v1 cleaned the concatenated ``raw_text``. That throws away the one thing
that makes footnotes and marginal notes recoverable: the page. In this
PDF's text layer the footnote block is NOT always at the bottom of the
page -- on some pages it comes first (p.152), or in the middle with body
text after it (p.100, p.114). v1's "absorb continuation lines until a
blank line" loop therefore swallowed real constitutional text into
``[FOOTNOTES]`` (Art. 160(4)-(7), Art. 175A(2) table cells, Art. 212(1)(b)),
and its footnote numbers (which restart on every page) could no longer be
linked back to their markers.

v2 processes each page of ``raw_doc["pages"]``:

  1. drop running header / page number / printer tags, remember the
     printed page number;
  2. split the page's footnote zone off (anchored on the page's own
     footnote "1", or a separator rule followed by a footnote), and put
     back any body text the extractor emitted *after* the footnote zone;
  3. turn every lost-superscript footnote marker on that page into a
     page-scoped reference ``[^<printed_page>.<n>]`` that resolves to a
     footnote with the same id -- including markers glued to Article
     numbers ("11." -> "[^5.1]1.", "154." -> "[^32.1]54."), to words
     ("1constitute"), to "* * *" omissions, and to amendment brackets;
  4. find Article headings (validated against the order of the Table of
     Contents) and move each Article's marginal note out of the body
     into a heading line placed directly before the Article;
  5. repair the genuine layout tables (Art. 51 and Art. 106 seat tables,
     Art. 175A(2) membership table, Fifth Schedule pension tables);
  6. on PDF pages 224-251 restore capitals that the PDF font stores as
     lowercase ("Prime minister" -> "Prime Minister", "rs." -> "Rs."), and
     everywhere repair garbled small caps ("COUrT" -> "COURT").

Then, on the whole body: amendment-bracket normalisation (v1 rules, but
the reference is kept), hyphenation joining (evidence-based, incl. across
page breaks and inside footnotes), unwrapping of PDF line breaks into
paragraphs (section sub-headings such as "Financial Procedure", Schedule
header lines and oath invocations stay on their own lines; multi-line
all-caps headings are joined), whitespace normalisation.

``verify_letter_preservation`` compares letters case-insensitively, because
step 6 changes case (never letters); ``capitalisation_repairs`` counts it.

CONSERVATIVE CLEANING RULE (unchanged)
---------------------------------------
    legal content preservation  >  noise removal  >  formatting normalization

Nothing is summarised, paraphrased or invented. Every word in the
cleaned output comes from the raw text; marginal notes are moved (not
rewritten); footnote markers are converted (not dropped). The companion
check ``verify_letter_preservation`` proves this on every run: the
multiset of letters in (body + footnotes) must equal the multiset of
letters in the raw body pages minus the removed headers (ignoring case).

PIPELINE ISOLATION (unchanged)
-------------------------------
No import of other pipelines. Independent input/output directories.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from pathlib import Path

INPUT_DIR = Path("var/rag/constitution_ingested")
OUTPUT_DIR = Path("var/rag/cleaned_constitution")

LARGE_REMOVAL_FLAG_THRESHOLD = 0.15

# Private-use sentinels for a footnote reference while the text is still
# being processed (so later regexes never confuse it with real digits or
# brackets). Rendered as "[^<id>]" at the very end.
_REF_OPEN, _REF_CLOSE = "", ""
_REF_SENTINEL_RE = re.compile(f"{_REF_OPEN}([^{_REF_CLOSE}]*){_REF_CLOSE}")


def _ref(fid: str) -> str:
    return f"{_REF_OPEN}{fid}{_REF_CLOSE}"


# --------------------------------------------------------------------------
# Line-level patterns
# --------------------------------------------------------------------------
_PREAMBLE_ANCHOR_RE = re.compile(r"^[ \t]*PREAMBLE[ \t]*$", re.MULTILINE)

# "92   CONSTITUTION OF PAKISTAN", "CONSTITUTION OF PAKISTAN   11",
# "CONSTITUTION OF PAKISTAN" (number on the next line), "...PAKISTAN1".
_HEADER_RE = re.compile(r"^[ \t]*(\d{1,3})?[ \t]*CONSTITUTION OF PAKISTAN\d*[ \t]*(\d{1,3})?[ \t]*$")
_LONE_PAGE_NUMBER_RE = re.compile(r"^[ \t]*(?:\(?[ivxlcdm]{1,6}\)?|\d{1,3})[ \t]*$")

# Cover-page job tag and back-cover print line.
_PRINTER_LINE_RE = re.compile(
    r"^[ \t]*(?:\d+\(\d+\)[A-Za-z&]+---by[ \t]+\S+---PC-\d+[ \t]*\([^)\n]*\)"
    r"|PCPPI\S*.*\(PC-\d+\))[ \t]*$"
)

# Footnote separator rule (single long rule, or two rules with a gap).
_SEPARATOR_RE = re.compile(r"^[ \t]*[_‐-―-]{15,}(?:[ \t]+[_‐-―-]{10,})?[ \t]*$")
# Short decorative rules ("––––" end of Part, "______" end of Article).
_DIVIDER_LINE_RE = re.compile(r"^[ \t]*(?:_{3,}(?:[ \t]+_{3,})?|[‐-―]{2,})[ \t]*$")

# A footnote entry starts with its (lost-superscript) number glued to the
# text: "1Subs.", "4.New clause", "4 Ins." ("12 of 1975" is excluded: a
# space is only accepted before a capital letter).
_FN_START_RE = re.compile(r"^[ \t]*(\d{1,2})(?:\.(?=[A-Za-z])|[ \t](?=[A-Z])|(?=[A-Za-z‘’“\"']))")

# Line-final abbreviations that do NOT end a footnote sentence.
_ABBREV_END_RE = re.compile(
    r"\b(?:Amdt|subs|ins|Subs|Ins|s|ss|Art|Arts|No|Nos|Pt|Ext|Gaz|Pak|p|pp|Sch|viz|Ord|O|P|"
    r"Govt|Vol|Para|Cl|Ch|Notfn|Min|Deptt|Rs|w\.e\.f|i\.e|e\.g)\.$"
)

# Article heading at line start (after optional ref sentinel and "[").
_ARTICLE_HEAD_RE = re.compile(
    rf"^(?P<ind>[ \t]*)(?P<pre>(?:(?:{_REF_OPEN}[^{_REF_CLOSE}]*{_REF_CLOSE})?\[)*"
    rf"(?:{_REF_OPEN}[^{_REF_CLOSE}]*{_REF_CLOSE})?)"
    # "90. (1)", "161." (alone), "170.5[(1)]", "270A.–(1)"
    rf"(?P<num>\d{{1,4}}[A-Z]{{0,3}})\.(?=[ \t]|$|[–—-]|\d{{1,2}}\[|{_REF_OPEN})"
)

_ASTERISK_LINE_RE = re.compile(rf"^[ \t]*(?:{_REF_OPEN}[^{_REF_CLOSE}]*{_REF_CLOSE})?[ \t]*\*(?:[ \t]*\*)*[ \t]*$")

_KNOWN_HYPHENATED_COMPOUNDS = frozenset({
    "majlise", "nondiscrimination", "selfincrimination", "subparagraph", "noconfidence",
    "wellbeing", "hydroelectric", "fullstop", "twentythird",
})

_BRACKET_CONTENT_IS_CLAUSE_NUMBER_RE = re.compile(
    r"^(?:\(\d+[A-Za-z]{0,2}\)|\([a-zA-Z0-9]{1,4}\)|\d+[A-Za-z]{0,2}\.)"
)


def _norm_letters(s: str) -> str:
    return re.sub(r"[^a-z]", "", s.lower())


def _collapse(s: str) -> str:
    return re.sub(r"[ \t]{2,}", " ", s).strip()


def _is_terminated(line: str) -> bool:
    s = line.rstrip()
    return bool(re.search(r"[.;:][’”'\"]*$", s)) and not _ABBREV_END_RE.search(s)


# --------------------------------------------------------------------------
# Data holders
# --------------------------------------------------------------------------
@dataclass
class Footnote:
    id: str
    pdf_page: int
    printed_page: str
    number: int
    text: str
    see_also: str | None = None
    referenced: bool = False

    def to_dict(self) -> dict:
        return {
            "id": self.id, "pdf_page": self.pdf_page, "printed_page": self.printed_page,
            "number": self.number, "text": self.text, "see_also": self.see_also,
            "referenced": self.referenced,
        }


@dataclass
class Diagnostics:
    raw_length: int = 0
    cleaned_length: int = 0
    pages_processed: int = 0
    footnotes: int = 0
    footnote_refs: int = 0
    unresolved_markers: list = field(default_factory=list)   # [(page, number)] marker w/o footnote
    unreferenced_footnotes: list = field(default_factory=list)  # footnote ids with no marker
    displaced_body_blocks_restored: list = field(default_factory=list)  # pdf pages
    glued_article_numbers_fixed: list = field(default_factory=list)  # ("11"->"1", page)
    articles_found: int = 0
    articles_in_toc: int = 0
    articles_missing: list = field(default_factory=list)
    marginal_notes_attached: int = 0
    articles_without_marginal_note: list = field(default_factory=list)
    table_repairs: list = field(default_factory=list)
    capitalisation_repairs: int = 0
    letter_preservation_ok: bool | None = None
    letter_preservation_diff: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        d = dict(self.__dict__)
        d["chars_removed"] = self.raw_length - self.cleaned_length
        d["pct_removed"] = round(d["chars_removed"] / self.raw_length, 4) if self.raw_length else 0.0
        d["large_removal_flag"] = d["pct_removed"] > LARGE_REMOVAL_FLAG_THRESHOLD
        return d


# --------------------------------------------------------------------------
# Table of contents -> ordered Article ids + titles (used only to validate
# Article headings and to recognise marginal notes; never emitted as text)
# --------------------------------------------------------------------------
def _parse_toc(front_pages: list[str]) -> list[tuple[str, str]]:
    toc: list[list[str]] = []
    cur = None
    for text in front_pages:
        for line in text.split("\n"):
            m = re.match(r"^\s*(\d+[A-Z]{0,3})\.\s+(\S.*)$", line)
            if m:
                cur = [m.group(1), m.group(2).strip()]
                toc.append(cur)
                continue
            if cur and re.match(r"^\s{12,}\S", line) and not re.search(
                r"PART|CHAPTER|SCHEDULE|ARTICLES|PAGES|\d+–\d+", line
            ):
                cur[1] += " " + line.strip()
            elif line.strip() == "" or re.search(r"PART|CHAPTER|––––|CONSTITUTION", line):
                cur = None
    return [(a, t) for a, t in toc]


# --------------------------------------------------------------------------
# Per-page steps
# --------------------------------------------------------------------------
def _strip_page_furniture(lines: list[str]) -> tuple[list[str], str | None]:
    """Removes running header (+ its page number line) and printer tags.
    Returns (lines, printed_page_number_or_None)."""
    out: list[str] = []
    printed = None
    i = 0
    while i < len(lines):
        line = lines[i]
        m = _HEADER_RE.match(line)
        if m:
            printed = m.group(1) or m.group(2)
            if not printed and i + 1 < len(lines) and _LONE_PAGE_NUMBER_RE.match(lines[i + 1]):
                printed = lines[i + 1].strip().strip("()")
                i += 1
            i += 1
            continue
        if _PRINTER_LINE_RE.match(line):
            i += 1
            continue
        out.append(line)
        i += 1
    return out, printed


def _split_footnote_zone(lines: list[str]):
    """Returns (body_lines, [(number, [entry lines])], displaced_body_lines).

    Zone start: the LAST line that starts footnote number 1 (a page's
    footnotes always restart at 1); if the page has none, a separator rule
    followed by a footnote start. Inside the zone, entries must number
    consecutively; anything else is a continuation line. After the last
    entry, a block of short lines that follows a finished sentence is body
    text the extractor emitted after the footnotes (pp. 100, 114, 152) and
    is handed back as displaced body.
    """
    starts = [(i, int(m.group(1))) for i, l in enumerate(lines) if (m := _FN_START_RE.match(l))]
    ones = [i for i, n in starts if n == 1]
    z = ones[-1] if ones else None
    if z is None:
        for i, l in enumerate(lines):
            if _SEPARATOR_RE.match(l):
                j = next((k for k in range(i + 1, len(lines)) if lines[k].strip()), None)
                if j is not None and _FN_START_RE.match(lines[j]):
                    z = j
    if z is None:
        return lines, [], []

    body = lines[:z]
    while body and (not body[-1].strip() or _SEPARATOR_RE.match(body[-1])):
        body.pop()

    entries: list[tuple[int, list[str]]] = []
    exp = None
    for idx in range(z, len(lines)):
        l = lines[idx]
        m = _FN_START_RE.match(l)
        if m and (exp is None or int(m.group(1)) == exp):
            entries.append((int(m.group(1)), [l]))
            exp = int(m.group(1)) + 1
        elif not l.strip() or _SEPARATOR_RE.match(l):
            continue
        else:
            entries[-1][1].append(l)

    # displaced body after the last entry
    displaced: list[str] = []
    num, ent = entries[-1]
    for j in range(1, len(ent)):
        rest = [_collapse(x) for x in ent[j:] if x.strip()]
        if not _is_terminated(ent[j - 1]) or not rest:
            continue
        short_block = len(rest) >= 3 and all(len(x) <= 72 for x in rest)
        margin_block = len(rest) >= 2 and all(len(x) <= 30 for x in rest)
        if short_block or margin_block:
            displaced = ent[j:]
            entries[-1] = (num, ent[:j])
            break
    return body, entries, displaced


def _convert_markers(line: str, page_id: str, fn_numbers: set[int], used: set[int]) -> str:
    """Lost-superscript footnote markers -> page-scoped ref sentinels.
    Only numbers that exist as a footnote on this page are converted."""

    def conv(m):
        n = int(m.group(1))
        if n in fn_numbers:
            used.add(n)
            return _ref(f"{page_id}.{n}")
        return m.group(0)

    # 170.5[(1)]   marker right after an Article number's period
    line = re.sub(r"^([ \t]*\d{1,3}[A-Z]{0,3}\.)(\d{1,2})(?=\[)",
                  lambda m: m.group(1) + conv(re.match(r"(\d+)", m.group(2))), line)
    # 2[...]   amendment bracket
    line = re.sub(r"(?<![\d.])(\d{1,2})(?=\[)", conv, line)  # also "]9[", "jurisdiction3["
    # 7* * *   omission marker
    line = re.sub(r"(?<![\w.])(\d{1,2})(?=[ \t]*\*)", conv, line)
    # 1constitute / 1CHAPTER   glued before a word (not ordinals 1st/2nd/4th)
    line = re.sub(r"(?<![\w.,/\-])(\d{1,2})(?=[A-Za-z]{2,})(?!(?:st|nd|rd|th)\b)", conv, line)
    # law4 extends   glued after a word
    line = re.sub(r"(?<=[A-Za-z]{2})(\d{1,2})(?=[\s,;:.)\]]|$)", conv, line)
    # Order,1 declare   glued after punctuation
    line = re.sub(r"(?<=[,;:])(\d{1,2})(?=[ \t])", conv, line)
    return line


# --------------------------------------------------------------------------
# Article headings + marginal notes
# --------------------------------------------------------------------------
class _ArticleTracker:
    def __init__(self, toc: list[tuple[str, str]]):
        self.ids = [a for a, _ in toc]
        self.titles = dict(toc)
        self.ptr = 0
        self.found: list[str] = []

    def match(self, line: str, page_id: str, fn_numbers: set[int], used: set[int], diag: Diagnostics):
        """Returns (possibly rewritten line, article_id) or (line, None)."""
        if self.ptr >= len(self.ids):
            return line, None
        m = _ARTICLE_HEAD_RE.match(line)
        if not m:
            return line, None
        num = m.group("num")
        window = self.ids[self.ptr:self.ptr + 6]
        art = None
        if num in window:
            art = num
        else:
            for k in (1, 2):  # a 1- or 2-digit footnote marker glued in front
                head, tail = num[:k], num[k:]
                if tail and tail in window and int(head) in fn_numbers:
                    used.add(int(head))
                    diag.glued_article_numbers_fixed.append(
                        {"raw": num, "article": tail, "printed_page": page_id})
                    line = (m.group("ind") + m.group("pre") + _ref(f"{page_id}.{int(head)}")
                            + tail + line[m.end("num"):])
                    art = tail
                    break
        if art is None:
            return line, None
        self.ptr = self.ids.index(art, self.ptr) + 1
        self.found.append(art)
        return line, art


def _attach_marginal_notes(lines: list[str], heads: list[tuple[int, str]], titles: dict[str, str],
                           diag: Diagnostics, ev: "_HyphenEvidence | None" = None) -> list[str]:
    """For each Article starting on this page, finds the run of short lines
    whose letters spell the Article's TOC title (the marginal note as
    printed in the margin), removes those lines from the body and inserts
    their own text as a heading line right before the Article."""
    if not heads:
        return lines
    # The margin column is narrow: no printed marginal-note line is wider
    # than 20 characters (measured on this PDF), so body sub-headings such as
    # "Amendment of Constitution" (Part XI title, 25 chars) are never taken.
    cand = [i for i, l in enumerate(lines)
            if l.strip() and len(_collapse(_REF_SENTINEL_RE.sub("", l))) <= 22
            and not _ARTICLE_HEAD_RE.match(l) and not _DIVIDER_LINE_RE.match(l)
            and not _SEPARATOR_RE.match(l)]
    cand_set = set(cand)
    taken: set[int] = set()
    headings: dict[int, str] = {}

    def runs_from(i):
        acc, idxs = "", []
        j = i
        while j in cand_set and j not in taken:
            acc += _norm_letters(lines[j])
            idxs.append(j)
            yield acc, list(idxs)
            j += 1

    # Pass 1 (exact): marginal notes are printed in the same order as their
    # Articles, so each match must start after the previous one -- this keeps
    # "Principles of Policy." (Art. 29) from stealing the tail of
    # "Responsibility with respect to Principles of Policy." (Art. 30).
    # Pass 2 (fuzzy, ratio >= 0.85) for notes worded slightly differently
    # from the Table of Contents.
    for exact in (True, False):
        min_start = 0
        for line_idx, art in heads:
            if line_idx in headings or art not in titles:
                continue
            target = _norm_letters(titles[art])
            if not target:
                continue
            best = None
            for i in cand:
                if i in taken or (exact and i < min_start):
                    continue
                for acc, idxs in runs_from(i):
                    if len(acc) > len(target) * 1.3:
                        break
                    if exact and acc == target:
                        best = (1.0, idxs)
                        break
                    if not exact and abs(len(acc) - len(target)) <= max(4, len(target) * 0.2):
                        r = SequenceMatcher(None, acc, target, autojunk=False).ratio()
                        if r >= 0.85 and (best is None or r > best[0]):
                            best = (r, idxs)
                if exact and best:
                    break
            if best:
                idxs = best[1]
                taken.update(idxs)
                min_start = idxs[-1] + 1
                # same hyphen evidence as the body: "well-|being" -> "well-being", "jurisdic-|tion" -> "jurisdiction"
                headings[line_idx] = _join_wrapped([lines[k] for k in idxs], ev)

    out: list[str] = []
    for i, l in enumerate(lines):
        if i in taken:
            continue
        if i in headings:
            art = dict(heads)[i]
            out.append(f"## {art}. {headings[i]}")
            diag.marginal_notes_attached += 1
        out.append(l)
    return out


# --------------------------------------------------------------------------
# Tables
# --------------------------------------------------------------------------
def _repair_seat_table(lines: list[str], diag: Diagnostics, page: int) -> list[str]:
    """Art. 106: header cells and each row's cells are one-per-line in the
    text layer. Rebuild as a markdown table. No-op unless the exact shape
    (4 header cells, rows of name + 4 integers) is found."""
    hdr = ["General seats", "Women", "Non-Muslims", "Total"]
    for i in range(len(lines) - 4):
        if [lines[i + k].strip() for k in range(4)] != hdr:
            continue
        j = i + 4
        while j < len(lines) and (not lines[j].strip() or _DIVIDER_LINE_RE.match(lines[j])):
            j += 1
        rows = []
        while j + 4 < len(lines) + 1:
            name = lines[j].strip() if j < len(lines) else ""
            nums = [lines[j + k].strip() for k in range(1, 5)] if j + 4 < len(lines) else []
            if name and len(nums) == 4 and all(re.fullmatch(r"\d+\]?", x) for x in nums) \
                    and not re.fullmatch(r"\d+", name):
                closing = "]" if nums[-1].endswith("]") else ""
                nums[-1] = nums[-1].rstrip("]")
                rows.append(f"| {_collapse(name)}{closing} | " + " | ".join(nums) + " |")
                j += 5
            else:
                break
        if len(rows) < 2:
            continue
        while j < len(lines) and _DIVIDER_LINE_RE.match(lines[j]):
            j += 1
        table = ["", "|  | " + " | ".join(hdr) + " |", "|---|---|---|---|---|", *rows, ""]
        diag.table_repairs.append({"pdf_page": page, "table": "Art. 106 seat table", "rows": len(rows)})
        start = i
        while start > 0 and _DIVIDER_LINE_RE.match(lines[start - 1]):
            start -= 1
        return lines[:start] + table + lines[j:]
    return lines


def _repair_pension_table(lines: list[str], diag: Diagnostics, page: int) -> list[str]:
    """Fifth Schedule pension tables: header "Judge | Minimum amount. |
    Maximum amount." and rows "Chief Justice | rs. 7,000 | rs. 8,000" come
    out one cell per line. No-op unless that exact shape is found."""
    val = re.compile(r"^rs\.\s*[\d,]+\.?\]?$")
    nb = [(i, l.strip()) for i, l in enumerate(lines) if l.strip() and not _DIVIDER_LINE_RE.match(l)]
    for k in range(len(nb) - 5):
        if [t for _, t in nb[k:k + 5]] != ["Judge", "Minimum", "Maximum", "amount.", "amount."]:
            continue
        rows, r = [], k + 5
        while r + 2 < len(nb) + 0 and r + 2 <= len(nb) - 1 and not val.match(nb[r][1]) \
                and val.match(nb[r + 1][1]) and val.match(nb[r + 2][1]):
            rows.append(f"| {nb[r][1]} | {nb[r + 1][1]} | {nb[r + 2][1]} |")
            r += 3
        if len(rows) < 2:
            continue
        first, last = nb[k][0], nb[r - 1][0]
        while first > 0 and _DIVIDER_LINE_RE.match(lines[first - 1]):
            first -= 1
        table = ["", "| Judge | Minimum amount. | Maximum amount. |", "|---|---|---|", *rows, ""]
        diag.table_repairs.append({"pdf_page": page, "table": "Fifth Schedule pension table", "rows": len(rows)})
        return lines[:first] + table + lines[last + 1:]
    return lines


def _repair_column_table(lines: list[str], diag: Diagnostics, page: int) -> list[str]:
    """Art. 51(3): a table laid out with wide column gaps ("Balochistan   16   4   20").
    Unwrapping would run it into one line, so it is rebuilt as markdown.
    Only fires on >= 3 consecutive gap-separated lines whose data rows are
    numeric (or "–") after the first cell."""
    cells = [re.split(r"[ \t]{3,}", l.strip()) if l.strip() else [] for l in lines]
    numeric = lambda c: all(re.fullmatch(r"(?:[\d,]+|[–-])\]?", x) for x in c[1:]) and len(c) >= 3
    i = 0
    while i < len(lines):
        if len(cells[i]) >= 3 and all(len(x) <= 20 for x in cells[i]) and not numeric(cells[i]):
            j = i + 1
            rows = []
            while j < len(lines) and numeric(cells[j]) and len(cells[j]) == len(cells[i]) + 1:
                rows.append(cells[j])
                j += 1
            if len(rows) >= 2:
                hdr = ["", *cells[i]]
                table = ["", "| " + " | ".join(hdr) + " |", "|" + "---|" * len(hdr),
                         *["| " + " | ".join(r) + " |" for r in rows], ""]
                diag.table_repairs.append({"pdf_page": page, "table": "column table", "rows": len(rows)})
                return lines[:i] + table + lines[j:]
        i += 1
    return lines


# --------------------------------------------------------------------------
# Broken capitals (PDF font encoding)
# --------------------------------------------------------------------------
# On PDF pages 224-251 (Third to Fifth Schedules) the body font subset maps
# the capitals M, R, L, V and Y to lowercase glyph codes: the page prints
# "Military", "Prime Minister", "Rs." but the text layer says "military",
# "Prime minister", "rs.". (Verified on this PDF: that font subset contains
# no uppercase M/R/L/V/Y at all across 39k characters.) Small-caps headings
# anywhere in the document show the same fault ("COUrT", "lEGISlATIvE").
BROKEN_CAPITALS_PDF_PAGES = range(224, 252)
_BROKEN_INITIALS = "mrlvy"


class _CapitalsRepair:
    def __init__(self, good_text: str):
        self.cap = Counter()
        self.low = Counter()
        for w in re.findall(r"\b[A-Za-z][a-z]+\b", good_text):
            (self.cap if w[0].isupper() else self.low)[w.lower()] += 1
        # two-word names that are always capitalised elsewhere ("Legislative List")
        bi = Counter(re.findall(r"\b([A-Za-z][a-z]+ [A-Za-z][a-z]+)\b", good_text))
        self.cap_bigrams = {k.lower(): k for k, v in bi.items()
                            if v >= 2 and k.split()[0][0].isupper() and k.split()[1][0].isupper()
                            and k.split()[1][0].lower() in _BROKEN_INITIALS
                            and bi.get(k.lower(), 0) == 0 and bi.get(k.split()[0] + " " + k.split()[1].lower(), 0) == 0}
        self.changes = 0

    def _fix_word(self, m: re.Match) -> str:
        w = m.group(0)
        k = w.lower()
        c, l = self.cap.get(k, 0), self.low.get(k, 0)
        if c >= 3 and c >= 9 * l:  # this word is (almost) always capitalised elsewhere in the document
            self.changes += 1
            return w[0].upper() + w[1:]
        return w

    def fix_small_caps(self, line: str) -> str:
        def up(m):
            self.changes += 1
            return m.group(0).upper()
        # "COUrT", "mINISTEr", "lEGISlATIvE": >= 2 capitals, lowercase only from the broken set
        line = re.sub(rf"\b(?=[A-Za-z]*[A-Z][A-Za-z]*[A-Z])(?=[A-Za-z]*[{_BROKEN_INITIALS}])[A-Z{_BROKEN_INITIALS}]{{3,}}\b",
                      up, line)

        # then a small-caps "OR" between capitals: "FEDERAL MINISTER Or MINISTER OF STATE"
        def or_up(m):
            self.changes += 1
            return m.group(1) + "OR"
        return re.sub(rf"((?:[A-Z]{{2,}}|\]|\[|{_REF_CLOSE})[ \t]+|\[|{_REF_CLOSE})Or(?=[ \t]+[A-Z\[]|[ \t]*$)",
                      or_up, line)

    def fix_page_line(self, line: str, starts_paragraph: bool, is_heading: bool) -> str:
        line = re.sub(rf"\b[{_BROKEN_INITIALS}][a-z]+\b", self._fix_word, line)

        def bigram(m):
            fixed = self.cap_bigrams.get(m.group(0).lower())
            if fixed and fixed != m.group(0):
                self.changes += 1
                return fixed
            return m.group(0)
        line = re.sub(r"\b[A-Za-z][a-z]+ [A-Za-z][a-z]+\b", bigram, line)

        def cap(m):
            self.changes += 1
            return m.group(1) + m.group(2).upper()
        ref = rf"(?:{_REF_OPEN}[^{_REF_CLOSE}]*{_REF_CLOSE}|[\[(])*"
        # a numbered list item / paragraph starts with a capital ("2. military," -> "2. Military,")
        if starts_paragraph:
            line = re.sub(rf"^([ \t]*{ref}(?:\d+[A-Z]?\.[ \t]+{ref})?)([{_BROKEN_INITIALS}])(?=[a-z]+\b)", cap, line)
        # a new sentence inside the line ("... my duties. may Allah" -> "May Allah")
        line = re.sub(rf"([.?!][’”'\"\]]?[ \t]+{ref})([{_BROKEN_INITIALS}])(?=[a-z]+\b)", cap, line)
        if is_heading:  # "Federal legislative list"
            line = re.sub(rf"(\b)(?!(?:{'|'.join(_SMALL_WORDS)})\b)([{_BROKEN_INITIALS}])(?=[a-z]+\b)", cap, line)
        n = len(re.findall(r"\brs\.", line))
        if n:
            self.changes += n
            line = re.sub(r"\brs\.", "Rs.", line)
        return line


_175A_ANCHORS = {
    "iiia": "a Judge of the Federal Constitutional",
    "vi": "an advocate having not less than",
    "vii": "two members from the Senate and two",
    "viii": "a woman or non Muslim or a",
}


def _repair_175a_table(lines: list[str], displaced: list[str], diag: Diagnostics, page: int):
    """Art. 175A(2): in the text layer the description cells of rows
    (iiia), (vi), (vii), (viii) come out after the page's footnotes, and the
    rows themselves keep only their label and "Member". Puts each cell back
    on its row. No-op unless every row and every anchor is found exactly
    once and every displaced line is consumed."""
    row_re = re.compile(rf"^(?P<ind>[ \t]*)(?P<pre>(?:{_REF_OPEN}[^{_REF_CLOSE}]*{_REF_CLOSE})?\[?)"
                        r"\((?P<lab>iiia|vi|vii|viii)\)[ \t]+(?P<role>Members?;?\]?)[ \t]*$")
    rows = {m.group("lab"): (i, m) for i, l in enumerate(lines) if (m := row_re.match(l))}
    text = [_collapse(x) for x in displaced if x.strip()]
    starts = {}
    for lab, anchor in _175A_ANCHORS.items():
        hits = [k for k, t in enumerate(text) if t.startswith(anchor)]
        if len(hits) != 1:
            return lines, displaced
        starts[lab] = hits[0]
    if set(rows) != set(_175A_ANCHORS):
        return lines, displaced
    order = sorted(starts.items(), key=lambda kv: kv[1])
    segs = {}
    for n, (lab, s) in enumerate(order):
        e = order[n + 1][1] if n + 1 < len(order) else len(text)
        segs[lab] = text[s:e]
    if order[0][1] != 0:
        return lines, displaced
    out = list(lines)
    for lab, (i, m) in rows.items():
        seg = segs[lab]
        prov = next((k for k, t in enumerate(seg) if t.startswith("Provided that")), None)
        main, proviso = (seg[:prov], seg[prov:]) if prov is not None else (seg, [])
        new = [f"{m.group('ind')}{m.group('pre')}({lab}) {_join_wrapped(main)} {m.group('role')}"]
        if proviso:
            new.append("        " + _join_wrapped(proviso))
        out[i] = "\n".join(new)
    diag.table_repairs.append({"pdf_page": page, "table": "Art. 175A(2) membership table", "rows": 4})
    return "\n".join(out).split("\n"), []


# --------------------------------------------------------------------------
# Hyphenation, unwrapping, brackets
# --------------------------------------------------------------------------
class _HyphenEvidence:
    """Decides "accor-|dance" -> "accordance" vs "self-|incrimination" ->
    "self-incrimination" from the document's own single-line usage."""

    def __init__(self, full_text: str):
        flat = re.sub(r"-[ \t]*\n[ \t]*", "", full_text)  # exclude wrapped forms from evidence
        self.hyph = Counter(m.lower() for m in re.findall(r"\b[A-Za-z]+-[A-Za-z]+\b", full_text.replace("-\n", "~")))
        self.joined = Counter(w.lower() for w in re.findall(r"\b[A-Za-z]+\b", flat))

    def keep_hyphen(self, prefix: str, suffix: str) -> bool:
        if suffix[:1].isupper():  # Majlis-e-|Shoora, non-|Muslims
            return True
        p, s = prefix.lower(), suffix.lower()
        if (p + s) in _KNOWN_HYPHENATED_COMPOUNDS:
            return True
        h, j = self.hyph.get(f"{p}-{s}", 0), self.joined.get(p + s, 0)
        return h > 0 and h >= j


def _join_wrapped(parts: list[str], ev: "_HyphenEvidence | None" = None) -> str:
    out = ""
    for p in parts:
        p = _collapse(p)
        if not p:
            continue
        if out.endswith("-") and re.search(r"[A-Za-z]-$", out) and re.match(r"[A-Za-z]", p):
            pre = re.search(r"([A-Za-z]+)-$", out).group(1)
            suf = re.match(r"[A-Za-z]+", p).group(0)
            keep = ev.keep_hyphen(pre, suf) if ev else (suf[:1].isupper())
            out = (out if keep else out[:-1]) + p
        else:
            out = f"{out} {p}" if out else p
    return out


def _is_heading_like(s: str) -> bool:
    letters = re.sub(r"[^A-Za-z]", "", s)
    return len(letters) >= 3 and letters.isupper()


_CLAUSE_LABEL_RE = re.compile(
    rf"^(?:{_REF_OPEN}[^{_REF_CLOSE}]*{_REF_CLOSE}|\[)*"
    r"\((?:\d+[A-Z]{0,2}|[a-z]{1,2}|[ivxl]+[a-z]?|[A-Z])\)(?=[ \t]|$)"
)
_PARA_OPENER_RE = re.compile(
    rf"^(?:{_REF_OPEN}[^{_REF_CLOSE}]*{_REF_CLOSE}|\[)*(?:Provided|Explanation|Note|Illustration)\b"
)


_SMALL_WORDS = {"of", "and", "the", "by", "for", "to", "in", "on", "a", "an", "or", "from", "under", "with"}
# "[Article 41 (3)]", "(Article 2A)", "(In the name of Allah, ...)" -- Schedule/Annex header lines
_STANDALONE_LINE_RE = re.compile(r"^(?:\[Article[^\]]*\]|\(Article[^)]*\)|\(In the name of Allah[^)]*\))$")


def _prev_done(p: str) -> bool:
    return bool(re.search(r"[.;:—–\-][’”'\"\]]*$", p)) or p.endswith(("––", "—"))


def _is_subheading(s: str, prev: str | None) -> bool:
    """A centred section sub-heading printed between Articles ("Procedure
    Generally", "Financial Procedure", "II. Regulations"): a short
    title-case line with no closing punctuation, following a finished
    sentence. A line-wrap continuation never qualifies, because a wrap only
    happens inside an unfinished sentence."""
    s = _REF_SENTINEL_RE.sub("", s).strip()
    p = _REF_SENTINEL_RE.sub("", prev or "").strip()
    # the previous line must END A SENTENCE (a trailing hyphen is a word break, not an end)
    if not p or not re.search(r"[.;:][’”'\"\])]*$", p):
        return False
    if (not s or len(s) > 70 or re.search(r"[.;:,]", re.sub(r"^[IVX]+\.\s+", "", s))
            or _ARTICLE_HEAD_RE.match(s) or _CLAUSE_LABEL_RE.match(s) or _PARA_OPENER_RE.match(s)):
        return False
    if _ARTICLE_HEAD_RE.match(_REF_SENTINEL_RE.sub("", prev).strip()) and re.fullmatch(
            r"(?:\[)*\d+[A-Z]{0,3}\.", _REF_SENTINEL_RE.sub("", prev).strip()):
        return False  # "270BB." alone on a line, then its first word
    words = re.sub(r"^[IVX]+\.\s+", "", s).split()
    if not words or len(words) > 10 or any(re.search(r"\d", w) for w in words) or not words[0][0].isupper():
        return False
    return all(w[0].isupper() or w.lower() in _SMALL_WORDS or w[0] in "‘“([" for w in words)


def _starts_block(line: str, prev: str | None) -> bool:
    """Does ``line`` start a new paragraph, or is it a PDF line-wrap
    continuation of ``prev``? Continuation lines sit at column 0 in the text
    layer; clause/paragraph starts are indented -- except inside tables,
    where continuation lines are indented too, so indentation alone is not
    enough: it must come with a clause label, a "Provided"/"Explanation"
    opener, or a finished previous sentence."""
    if prev is None or not prev.strip():
        return True
    s = line.strip()
    p = prev.strip()
    if s.startswith(("##", "|")) or p.startswith(("##", "|")):
        return True
    if _ARTICLE_HEAD_RE.match(line) or _ASTERISK_LINE_RE.match(line) or _ASTERISK_LINE_RE.match(prev):
        return True
    if _is_heading_like(s) or _is_heading_like(p):
        return True
    if _STANDALONE_LINE_RE.match(s) or _STANDALONE_LINE_RE.match(p):
        return True
    if re.search(r"[Mm]erciful\.\)$", p):  # "(In the name of Allah, the most Beneficent,| the most Merciful.)"
        return True
    indented = bool(re.match(r"^[ \t]{2,}\S", line))
    prev_done = _prev_done(p)
    lab = _CLAUSE_LABEL_RE.match(s)
    if lab and (indented or prev_done or not s[lab.end():].strip() or re.search(r"[;,]\s*(?:or|and)$", p)):
        return True
    if _PARA_OPENER_RE.match(s) and (indented or prev_done):
        return True
    if indented and prev_done and re.match(rf"^(?:{_REF_OPEN}|[\[“‘\"'A-Z])", s):
        return True
    return False


# Role column of the Judicial Commission tables ("...each of the Federal   Members")
_ROLE_OPEN, _ROLE_CLOSE = "\ue002", "\ue003"
_ROLE_COLUMN_RE = re.compile(r"^(?P<row>.*\S)[ \t]{3,}(?P<role>(?:Chairperson|Members?);?\]?)[ \t]*$")


def _stash_role_column(line: str) -> str:
    m = _ROLE_COLUMN_RE.match(line)
    if not m or not _CLAUSE_LABEL_RE.match(m.group("row").strip()):
        return line
    return f"{m.group('row')}{_ROLE_OPEN}{m.group('role')}{_ROLE_CLOSE}"


def _place_role_column(para: str) -> str:
    m = re.search(f"{_ROLE_OPEN}([^{_ROLE_CLOSE}]*){_ROLE_CLOSE}", para)
    if not m:
        return para
    return (para[:m.start()] + para[m.end():]).rstrip() + " " + m.group(1)


def _unwrap(lines: list[str], ev: _HyphenEvidence) -> list[str]:
    paras: list[list[str]] = []
    prev = None
    prev_heading = False
    for line in lines:
        if not line.strip():
            if paras and paras[-1]:
                paras.append([])
            prev = line
            prev_heading = False
            continue
        if _ASTERISK_LINE_RE.match(line) and prev is not None and _ASTERISK_LINE_RE.match(prev) and paras and paras[-1]:
            paras[-1].append(line)  # one "* * * *" omission spread over several lines
        else:
            heading = _is_subheading(line, prev)
            p = _REF_SENTINEL_RE.sub("", prev or "").strip()
            s = _REF_SENTINEL_RE.sub("", line).strip()
            structural = r"^(?:\[|\(|PART\b|CHAPTER\b|[A-Z]+ SCHEDULE\b|ANNEX\b|PREAMBLE\b)"
            continues_heading = bool(paras and paras[-1]) and (
                ((prev_heading or _is_heading_like(p))
                 and re.search(r"\b(?:of|the|and|or|OF|THE|AND|OR|A)$", p)
                 and len(s) <= 70 and not re.search(r"[.;:]$", s) and s[:1].isupper())
                # "...JUDGE OF THE FEDERAL" + "CONSTITUTIONAL COURT OR OF THE SUPREME COURT OR OF A" + "HIGH COURT."
                or (_is_heading_like(p) and _is_heading_like(s) and not re.search(r"[.\]]$", p)
                    and not re.match(structural, s) and not re.match(structural, re.sub(r"^\W+", "", p))
                    and len(paras[-1]) < 6)
            )
            if continues_heading:  # "V. Ordinances Promulgated by the Governor of" + "Former Province of West Pakistan"
                paras[-1].append(line)
                heading = prev_heading
            elif heading or prev_heading or _starts_block(line, prev) or not paras or not paras[-1]:
                paras.append([line])
            else:
                paras[-1].append(line)
            prev_heading = heading
        prev = line
    out = []
    for para in paras:
        if not para:
            out.append("")
            continue
        if all(_ASTERISK_LINE_RE.match(x) for x in para):
            refs = "".join(m.group(0) for m in _REF_SENTINEL_RE.finditer("".join(para)))
            out.append(f"{refs}* * * *")
        else:
            out.append(_place_role_column(_join_wrapped(para, ev)))
    return out


def _normalize_amendment_brackets(text: str) -> str:
    """v1's rule, but the footnote reference survives:
    "<ref>[Balochistan]" stays; "<ref>[(2) The territories ...]" -> "<ref>(2) The territories ..."."""
    opener = re.compile(f"{_REF_OPEN}[^{_REF_CLOSE}]*{_REF_CLOSE}\\[")
    result: list[str] = []
    i, n = 0, len(text)
    while i < n:
        m = opener.match(text, i)
        if m:
            depth, j = 1, m.end()
            start_inner = j
            while j < n and depth > 0:
                if text[j] == "[":
                    depth += 1
                elif text[j] == "]":
                    depth -= 1
                j += 1
            if depth == 0:
                ref = text[m.start():m.end() - 1]
                inner = _normalize_amendment_brackets(text[start_inner:j - 1])
                if _BRACKET_CONTENT_IS_CLAUSE_NUMBER_RE.match(inner):
                    result.append(ref + inner)
                else:
                    result.append(f"{ref}[{inner}]")
                i = j
                continue
        result.append(text[i])
        i += 1
    return "".join(result)


# --------------------------------------------------------------------------
# Main entry points
# --------------------------------------------------------------------------
def clean_constitution_pages(pages: list[dict]) -> tuple[str, list[Footnote], list[dict], Diagnostics]:
    diag = Diagnostics()
    pages = sorted(pages, key=lambda p: p["page_number"])
    full_raw = "\n\n".join(p["text"] for p in pages)
    diag.raw_length = len(full_raw)

    start = next((k for k, p in enumerate(pages) if _PREAMBLE_ANCHOR_RE.search(p["text"])), None)
    if start is None:
        raise ValueError("PREAMBLE anchor not found -- refusing to guess where the Constitution starts")
    toc = _parse_toc([p["text"] for p in pages[:start]])
    titles = dict(toc)
    tracker = _ArticleTracker(toc)
    diag.articles_in_toc = len(toc)
    ev = _HyphenEvidence(full_raw)
    caps = _CapitalsRepair("\n".join(p["text"] for p in pages[start:]
                                      if p["page_number"] not in BROKEN_CAPITALS_PDF_PAGES))

    body_lines: list[str] = []
    footnotes: list[Footnote] = []
    article_index: list[dict] = []
    prev_printed = None

    for p in pages[start:]:
        n = p["page_number"]
        lines, printed = _strip_page_furniture(p["text"].split("\n"))
        if printed is None:
            printed = str(int(prev_printed) + 1) if prev_printed and prev_printed.isdigit() else "1"
        prev_printed = printed
        if not any(l.strip() for l in lines):
            continue
        diag.pages_processed += 1

        body, entries, displaced = _split_footnote_zone(lines)
        fn_numbers = {num for num, _ in entries}
        used: set[int] = set()

        body = [_convert_markers(l, printed, fn_numbers, used) for l in body]
        displaced = [_convert_markers(l, printed, fn_numbers, used) for l in displaced]
        if displaced:
            body, displaced = _repair_175a_table(body, displaced, diag, n)
        if displaced:
            diag.displaced_body_blocks_restored.append(n)
            body = body + displaced

        body = [_stash_role_column(l) for l in body]
        heads = []
        for i, l in enumerate(body):
            new, art = tracker.match(l, printed, fn_numbers, used, diag)
            if art:
                body[i] = new
                heads.append((i, art))
                article_index.append({"article": art, "title_toc": titles.get(art),
                                      "printed_page": printed, "pdf_page": n})
        body = _attach_marginal_notes(body, heads, titles, diag, ev)
        body = _repair_seat_table(body, diag, n)
        body = _repair_pension_table(body, diag, n)
        body = _repair_column_table(body, diag, n)
        body = [caps.fix_small_caps(l) for l in body]
        if n in BROKEN_CAPITALS_PDF_PAGES:
            fixed = []
            for k, l in enumerate(body):
                prev = body[k - 1] if k else (body_lines[-1] if body_lines else None)
                ls = l.strip()
                under_caps_heading = (prev is not None and _is_heading_like(_REF_SENTINEL_RE.sub("", prev).strip())
                                      and 0 < len(ls) <= 70 and not re.search(r"[.;:,]$", ls) and ls[:1].isupper())
                fixed.append(caps.fix_page_line(l, _starts_block(l, prev), _is_subheading(l, prev) or under_caps_heading))
            body = fixed
            entries = [(num, [caps.fix_page_line(x, k == 0, False) for k, x in enumerate(ent)])
                       for num, ent in entries]
        body = [l for l in body if not _DIVIDER_LINE_RE.match(l) and not _SEPARATOR_RE.match(l)]
        while body and not body[0].strip():
            body.pop(0)
        while body and not body[-1].strip():
            body.pop()
        body_lines.extend(body)

        for num, ent in entries:
            first = _FN_START_RE.sub("", ent[0], count=1)
            text = _join_wrapped([first] + ent[1:], ev)
            text = re.sub(r"^\.\s*", "", text)
            fid = f"{printed}.{num}"
            see = re.search(r"See\s*footnote\s+(\d+)\s+on\s+page\s+(\d+)", text)
            footnotes.append(Footnote(id=fid, pdf_page=n, printed_page=printed, number=num, text=text,
                                      see_also=f"{see.group(2)}.{see.group(1)}" if see else None,
                                      referenced=num in used))

    diag.capitalisation_repairs = caps.changes
    found = set(tracker.found)
    diag.articles_found = len(tracker.found)
    diag.articles_missing = [a for a, _ in toc if a not in found]

    # whole-body passes
    paras = _unwrap(body_lines, ev)
    text = "\n".join(paras)
    text = _normalize_amendment_brackets(text)
    text = re.sub(r"\(\s+([0-9a-zA-Z]{1,4})\s+\)", r"(\1)", text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    text = re.sub(r"[ \t]+\n", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text).strip()

    # articles that got no marginal-note heading
    headed = set(re.findall(r"^## (\d+[A-Z]{0,3})\. ", text, flags=re.M))
    diag.articles_without_marginal_note = [a for a in tracker.found if a not in headed]

    text = _REF_SENTINEL_RE.sub(lambda m: f"[^{m.group(1)}]", text)
    diag.footnote_refs = len(re.findall(r"\[\^[^\]]+\]", text))
    referenced = set(re.findall(r"\[\^([^\]]+)\]", text))
    for f in footnotes:
        f.referenced = f.id in referenced
    diag.footnotes = len(footnotes)
    diag.unreferenced_footnotes = [f.id for f in footnotes if not f.referenced]
    diag.unresolved_markers = sorted(referenced - {f.id for f in footnotes})
    return text, footnotes, article_index, diag


def verify_letter_preservation(pages: list[dict], body: str, footnotes: list[Footnote]) -> tuple[bool, dict]:
    """Every letter of the raw body pages (minus running headers / printer
    tags / page numbers) must appear in body + footnotes, and nothing else."""
    pages = sorted(pages, key=lambda p: p["page_number"])
    start = next(k for k, p in enumerate(pages) if _PREAMBLE_ANCHOR_RE.search(p["text"]))
    raw = []
    for p in pages[start:]:
        lines, _ = _strip_page_furniture(p["text"].split("\n"))
        raw.append("\n".join(lines))
    a = Counter(re.sub(r"[^a-z]", "", "\n".join(raw).lower()))
    body = re.sub(r"^## \d+[A-Z]{0,3}\. ", "", body, flags=re.M)  # heading's id is a copy of the Article number
    cleaned = re.sub(r"\[\^[^\]]+\]", "", body) + "".join(f.text for f in footnotes)
    b = Counter(re.sub(r"[^a-z]", "", cleaned.lower()))
    missing, extra = dict(a - b), dict(b - a)
    return (not missing and not extra), {"missing": missing, "extra": extra}


def clean_constitution_document(raw_doc: dict) -> dict:
    body, footnotes, article_index, diag = clean_constitution_pages(raw_doc["pages"])
    ok, d = verify_letter_preservation(raw_doc["pages"], body, footnotes)
    diag.letter_preservation_ok, diag.letter_preservation_diff = ok, d

    block = "\n".join(f"[^{f.id}]: {f.text}" for f in footnotes)
    cleaned_text = f"{body}\n\n[FOOTNOTES]\n{block}\n[/FOOTNOTES]" if footnotes else body
    diag.cleaned_length = len(cleaned_text)
    return {
        "doc_id": raw_doc["doc_id"],
        "metadata": dict(raw_doc["metadata"]),
        "raw_text": raw_doc["raw_text"],
        "cleaned_text": cleaned_text,
        "body_text": body,
        "footnotes": [f.to_dict() for f in footnotes],
        "article_index": article_index,
        "cleaning_diagnostics": diag.to_dict(),
    }


def strip_footnote_refs(text: str) -> str:
    """For embedding text: drop "[^5.2]" references."""
    return re.sub(r"\[\^[^\]]+\]", "", text)


def clean_directory(input_dir: Path = INPUT_DIR, output_dir: Path = OUTPUT_DIR) -> dict:
    input_dir, output_dir = Path(input_dir), Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    processed, flagged, failed_docs = 0, 0, []
    for raw_path in sorted(input_dir.glob("*.json")):
        try:
            raw_doc = json.loads(raw_path.read_text(encoding="utf-8"))
            cleaned_doc = clean_constitution_document(raw_doc)
        except (OSError, json.JSONDecodeError, KeyError, ValueError) as exc:
            failed_docs.append({"source_file": raw_path.name, "error": str(exc)})
            continue
        out_path = output_dir / f"{cleaned_doc['doc_id']}.json"
        tmp_path = out_path.with_suffix(".json.tmp")
        tmp_path.write_text(json.dumps(cleaned_doc, ensure_ascii=False, indent=2), encoding="utf-8")
        tmp_path.replace(out_path)
        processed += 1
        if cleaned_doc["cleaning_diagnostics"]["large_removal_flag"]:
            flagged += 1
    return {"processed": processed, "failed": len(failed_docs),
            "large_removal_flagged": flagged, "failed_docs": failed_docs}


def main() -> None:  # pragma: no cover
    summary = clean_directory()
    print(f"Constitution cleaning: {summary['processed']} processed, {summary['failed']} failed, "
          f"{summary['large_removal_flagged']} flagged for large removal")
    for f in summary["failed_docs"]:
        print(f"  FAILED {f['source_file']}: {f['error']}")


if __name__ == "__main__":  # pragma: no cover
    main()