"""Tests for the statute cleaning stage (src.rag_prep.statute_cleaning).

Synthetic fixtures only -- no real statute PDFs or ingested JSON required.
Mirrors the real noise patterns this module was validated against (the
9 real documents in var/rag/statutes_ingested/), but every fixture here
is hand-constructed.
"""

from __future__ import annotations

import json

from src.rag_prep.statute_cleaning import (
    LARGE_REMOVAL_FLAG_THRESHOLD,
    clean_directory,
    clean_statute_document,
    clean_statute_text,
)


def _raw_doc(full_text: str, **metadata_overrides) -> dict:
    metadata = {
        "law_name": "Some Act", "year": None, "document_type": "statute",
        "domain": "family_law", "domain_source": "curated", "source_file": "x.pdf",
        "page_count": 1, "extraction_engine": "fitz", "content_hash": "abc",
    }
    metadata.update(metadata_overrides)
    return {"doc_id": "doc1", "metadata": metadata, "full_text": full_text}


_SAMPLE_WITH_TOC = """THE SAMPLE FAMILY ACT, 1970

CONTENTS
1.
Short title and extent
2.
Definitions
3.
Jurisdiction

THE SAMPLE FAMILY ACT, 1970
ACT No. XII OF 1970
[1st January, 1970]

Preamble. WHEREAS it is expedient to provide for family matters;
It is hereby enacted as follows:--

1. Short title and extent. (1) This Act may be called the Sample Family Act, 1970.
(2) It extends to the whole of Pakistan.

2. Definitions. In this Act, unless the context otherwise requires--
(a) "Court" means a Family Court;
(b) "minor" means a person under the age of eighteen years.

3. Jurisdiction. The Family Court shall have exclusive jurisdiction in all matters
relating to maintenance, dissolution of marriage, and custody of children."""


# -- 1/2. TOC removal -----------------------------------------------------------

def test_toc_is_removed(tmp_path):
    cleaned, diag = clean_statute_text(_SAMPLE_WITH_TOC)
    assert diag.toc_removed is True
    assert "CONTENTS" not in cleaned
    # The TOC's bare numbered listing must be gone...
    assert "1.\nShort title and extent" not in cleaned


def test_toc_followed_by_substantive_section_is_preserved():
    cleaned, _ = clean_statute_text(_SAMPLE_WITH_TOC)
    # ...but the REAL Section 1, with its actual legal body, survives intact.
    assert "1. Short title and extent. (1) This Act may be called the Sample Family Act, 1970." in cleaned
    assert "2. Definitions. In this Act, unless the context otherwise requires" in cleaned
    assert "3. Jurisdiction. The Family Court shall have exclusive jurisdiction" in cleaned
    assert "Court" in cleaned and "Family Court" in cleaned
    assert "maintenance, dissolution of marriage, and custody of children" in cleaned


def test_conservative_no_removal_when_no_end_anchor_found():
    # "CONTENTS" present but no recognizable Act-citation/enacting-clause
    # anchor anywhere after it -- must NOT guess; text stays untouched
    # except for ordinary whitespace normalization.
    text = "CONTENTS\n\nSome ambiguous trailing text with no statutory anchor at all."
    cleaned, diag = clean_statute_text(text)
    assert diag.toc_removed is False
    assert "CONTENTS" in cleaned
    assert "Some ambiguous trailing text" in cleaned


def test_no_contents_marker_means_no_toc_removal_attempted():
    text = "1. Short title. This Act may be called the Sample Act.\n\n2. Definitions. ..."
    cleaned, diag = clean_statute_text(text)
    assert diag.toc_removed is False
    assert "Short title" in cleaned


# -- 3. header/footer/page-number removal ---------------------------------------

def test_page_number_markers_removed():
    text = "Some text.\n\nPage 1 of 4\n\nMore text after the page break."
    cleaned, _ = clean_statute_text(text)
    assert "Page 1 of 4" not in cleaned
    assert "Some text." in cleaned
    assert "More text after the page break." in cleaned


def test_date_generation_stamp_removed():
    text = "1. Short title.\n\nDate: 29-05-2024"
    cleaned, _ = clean_statute_text(text)
    assert "Date: 29-05-2024" not in cleaned
    assert "1. Short title." in cleaned


def test_rgn_prefixed_date_stamp_removed():
    # Confirmed in 2 of 9 real documents: "RGN Date: DD-MM-YYYY",
    # sometimes with a leading horizontal-rule-style underscore run.
    text = "1. Short title.\n\nRGN Date: 06-05-2024\n\n1Omitted by some Ordinance."
    cleaned, _ = clean_statute_text(text)
    assert "RGN" not in cleaned
    assert "1. Short title." in cleaned
    assert "1Omitted by some Ordinance." in cleaned


def test_underscore_prefixed_rgn_date_stamp_removed():
    text = "6. [Repealed].\n\n________                    RGN Date: 24-03-2025\n\n1Omitted by West Pakistan Act."
    cleaned, _ = clean_statute_text(text)
    assert "RGN" not in cleaned
    assert "6. [Repealed]." in cleaned
    assert "1Omitted by West Pakistan Act." in cleaned


def test_final_copy_stamp_removed():
    text = "FINAL COPY Updated till 17.1.2025\n\n1. Short title. Real content here."
    cleaned, _ = clean_statute_text(text)
    assert "FINAL COPY" not in cleaned
    assert "1. Short title. Real content here." in cleaned


def test_legitimate_text_near_a_page_boundary_is_not_removed():
    # Only the exact "Page N of M" line is removed -- surrounding
    # substantive text, even immediately adjacent, must survive.
    text = "5. Rights to dower.\n\nPage 3 of 4\n\nNothing in this Act shall affect any right."
    cleaned, _ = clean_statute_text(text)
    assert "5. Rights to dower." in cleaned
    assert "Nothing in this Act shall affect any right." in cleaned


# -- 4. page-break hyphenation ---------------------------------------------------

def test_hyphenated_linebreak_word_is_joined():
    text = "The Court shall make an appoint-\nment of a guardian without delay."
    cleaned, _ = clean_statute_text(text)
    assert "appointment" in cleaned
    assert "appoint-\nment" not in cleaned


def test_legitimate_compound_hyphen_on_one_line_is_untouched():
    text = "This applies to a non-resident husband under this section."
    cleaned, _ = clean_statute_text(text)
    assert "non-resident" in cleaned


# -- 5/footnote markers ----------------------------------------------------------

def test_lone_footnote_glyph_line_removed():
    text = "Some clause text.\n\n*\n\nFurther clause text."
    cleaned, _ = clean_statute_text(text)
    assert "Some clause text." in cleaned
    assert "Further clause text." in cleaned
    lines = [l.strip() for l in cleaned.split("\n")]
    assert "*" not in lines


def test_subsection_and_clause_markers_are_preserved():
    text = "1. Short title. (1) This Act may be called X Act.\n(2) It extends to Pakistan.\n(a) first; (b) second; (i) item one; (ii) item two."
    cleaned, _ = clean_statute_text(text)
    for marker in ("(1)", "(2)", "(a)", "(b)", "(i)", "(ii)"):
        assert marker in cleaned


def test_amendment_annotation_markers_are_preserved_not_stripped():
    # "1[(2) ... ]" -- a footnote-reference-style amendment marker that
    # carries real legal meaning (something was inserted/amended here)
    # and must never be treated as disposable footnote noise.
    text = "1. Short title.\n1[(2) It extends to the whole of Pakistan.]\n\n1Ins. by the Amending Act, 1980."
    cleaned, _ = clean_statute_text(text)
    assert "1[(2) It extends to the whole of Pakistan.]" in cleaned
    assert "1Ins. by the Amending Act, 1980." in cleaned


# -- 6. whitespace normalization -------------------------------------------------

def test_excessive_blank_lines_collapsed_but_paragraphs_preserved():
    text = "First section text.\n\n\n\n\n\nSecond section text."
    cleaned, _ = clean_statute_text(text)
    assert "\n\n\n" not in cleaned
    assert "First section text." in cleaned
    assert "Second section text." in cleaned


def test_output_is_not_flattened_into_a_single_paragraph():
    cleaned, _ = clean_statute_text(_SAMPLE_WITH_TOC)
    assert cleaned.count("\n\n") >= 2


def test_repeated_spaces_collapsed():
    text = "This   Act     extends   to   Pakistan."
    cleaned, _ = clean_statute_text(text)
    assert "   " not in cleaned
    assert "This Act extends to Pakistan." in cleaned


# -- 7. section/subsection marker normalization ----------------------------------

def test_stray_space_inside_marker_parentheses_normalized():
    text = "This Act applies as per clause ( 1 ) and sub-clause ( a )."
    cleaned, _ = clean_statute_text(text)
    assert "(1)" in cleaned
    assert "(a)" in cleaned
    assert "( 1 )" not in cleaned


def test_legal_numbering_and_hierarchy_unchanged():
    text = "5. Rights.\n(1) First.\n(a) Sub.\n(i) Item."
    cleaned, _ = clean_statute_text(text)
    assert "5. Rights." in cleaned
    assert "(1) First." in cleaned
    assert "(a) Sub." in cleaned
    assert "(i) Item." in cleaned


# -- 8. schedule content preserved -----------------------------------------------

def test_schedule_content_preserved():
    text = "26. Power to make rules.\n\nSCHEDULE\n\n(a) The Guardians and Wards Act, 1890;\n(b) The Punjab Act, 1948."
    cleaned, _ = clean_statute_text(text)
    assert "SCHEDULE" in cleaned
    assert "The Guardians and Wards Act, 1890" in cleaned
    assert "The Punjab Act, 1948" in cleaned


# -- 9/13. metadata preservation --------------------------------------------------

def test_metadata_preserved_unchanged_when_not_noise(tmp_path):
    raw = _raw_doc(_SAMPLE_WITH_TOC, law_name="The Sample Family Act", year="1970")
    cleaned = clean_statute_document(raw)
    assert cleaned["metadata"]["law_name"] == "The Sample Family Act"
    assert cleaned["metadata"]["year"] == "1970"
    assert cleaned["metadata"]["domain"] == "family_law"
    assert cleaned["metadata"]["domain_source"] == "curated"
    assert cleaned["metadata"]["document_type"] == "statute"
    assert cleaned["metadata"]["source_file"] == "x.pdf"


def test_page_marker_law_name_is_cleared_to_null():
    raw = _raw_doc(_SAMPLE_WITH_TOC, law_name="Page 1 of 4")
    cleaned = clean_statute_document(raw)
    assert cleaned["metadata"]["law_name"] is None
    assert cleaned["cleaning_diagnostics"]["metadata_law_name_cleared"] is True


# -- 12. raw input unchanged / output non-empty -----------------------------------

def test_raw_full_text_is_preserved_unchanged_in_output():
    raw = _raw_doc(_SAMPLE_WITH_TOC)
    cleaned = clean_statute_document(raw)
    assert cleaned["full_text"] == raw["full_text"]


def test_doc_id_preserved():
    raw = _raw_doc(_SAMPLE_WITH_TOC)
    cleaned = clean_statute_document(raw)
    assert cleaned["doc_id"] == raw["doc_id"]


def test_cleaned_output_is_non_empty():
    raw = _raw_doc(_SAMPLE_WITH_TOC)
    cleaned = clean_statute_document(raw)
    assert cleaned["cleaned_text"].strip() != ""


def test_clean_directory_does_not_modify_raw_input_files(tmp_path):
    input_dir = tmp_path / "in"
    output_dir = tmp_path / "out"
    input_dir.mkdir()
    raw = _raw_doc(_SAMPLE_WITH_TOC)
    raw_path = input_dir / "doc1.json"
    raw_path.write_text(json.dumps(raw), encoding="utf-8")
    original_bytes = raw_path.read_bytes()

    summary = clean_directory(input_dir, output_dir)

    assert summary["processed"] == 1
    assert raw_path.read_bytes() == original_bytes


# -- 10/11. content-preservation validation / large-removal flag -----------------

def test_large_removal_is_flagged_not_silently_accepted():
    # A pathological case: almost the whole document IS the TOC region
    # relative to a tiny amount of real body text after the anchor.
    text = "CONTENTS\n" + "\n".join(f"{i}.\nSection {i} title" for i in range(1, 40)) + \
        "\n\nACT No. I OF 1970\nIt is hereby enacted as follows:\n1. Short title. Real text."
    cleaned, diag = clean_statute_text(text)
    assert diag.toc_removed is True
    assert diag.pct_removed > LARGE_REMOVAL_FLAG_THRESHOLD
    assert diag.large_removal_flag is True
    # Still not silently discarded -- the real section text survives.
    assert "1. Short title. Real text." in cleaned


def test_small_removal_is_not_flagged():
    cleaned, diag = clean_statute_text(_SAMPLE_WITH_TOC)
    assert diag.large_removal_flag is False


def test_no_unexpected_large_deletion_for_text_with_no_noise_patterns():
    text = "1. Short title.\n\n2. Definitions.\n\n3. Jurisdiction of the Family Court."
    cleaned, diag = clean_statute_text(text)
    assert diag.pct_removed < 0.05
    assert "1. Short title." in cleaned
    assert "2. Definitions." in cleaned
    assert "3. Jurisdiction of the Family Court." in cleaned


# -- inline amendment markers / amendment notes / section-looking text ------------
# (confirmed against the real Child Marriage Restraint Act text, which
# exposed the RGN-stamp gap this task was specifically about)

def test_inline_amendment_markers_preserved_verbatim():
    text = ('1. Short title. X.\n\n'
            '2. Definitions. A "child" is under 3[sixteen] years. '
            '1[* * *] 4[or is about to be] 2[Magistrate of the first class].')
    cleaned, _ = clean_statute_text(text)
    for marker in ('3[sixteen]', '1[* * *]', '4[or is about to be]', '2[Magistrate of the first class]'):
        assert marker in cleaned


def test_amendment_markers_are_never_interpreted_or_stripped():
    # The cleaner must not transform "2[Magistrate of the first class]"
    # into "Magistrate of the first class" -- that is a retrieval-
    # specific representation that belongs to a later stage, not cleaning.
    text = '1. Short title. No Court other than a 2[Magistrate of the first class] shall act.'
    cleaned, _ = clean_statute_text(text)
    assert "2[Magistrate of the first class]" in cleaned
    assert cleaned.count("Magistrate of the first class]") == cleaned.count("2[Magistrate of the first class]")


def test_amendment_notes_preserved_as_a_coherent_block():
    text = ('4. Punishment. Whoever contracts a marriage shall be punishable.\n\n'
            '1Subs. by the Repealing and Amending Act No. VIII of 1930, s.2 and 1st Sch.\n'
            '2Subs. by the Central Laws (Statute Reforms) Ordinance No. XXI of 1960, s.3.\n'
            '3Sub. and Omitted by the Muslim Family Laws Ordinance No. VIII of 1961, s.12.\n'
            '4Ins. by Act No. XIX of 1938, s.2.')
    cleaned, _ = clean_statute_text(text)
    assert "1Subs. by the Repealing and Amending Act No. VIII of 1930, s.2 and 1st Sch." in cleaned
    assert "2Subs. by the Central Laws (Statute Reforms) Ordinance No. XXI of 1960, s.3." in cleaned
    assert "3Sub. and Omitted by the Muslim Family Laws Ordinance No. VIII of 1961, s.12." in cleaned
    assert "4Ins. by Act No. XIX of 1938, s.2." in cleaned


def test_amendment_notes_at_the_bottom_of_a_page_are_not_deleted():
    # Page position alone is not evidence of disposability -- amendment
    # notes immediately following substantive text, right where a page
    # break would fall, must survive exactly like any other.
    text = ('4. Punishment for male adult marrying a child. Whoever, being a male, '
            'contracts a child marriage shall be punishable with simple imprisonment.\n\n'
            '1Subs. by the Federal Laws (Revision and Declaration) Ordinance, 1981.\n'
            '2Ins. by the Guardians and Wards (Amdt.) Act, 1926.')
    cleaned, _ = clean_statute_text(text)
    assert "1Subs. by the Federal Laws (Revision and Declaration) Ordinance, 1981." in cleaned
    assert "2Ins. by the Guardians and Wards (Amdt.) Act, 1926." in cleaned


def test_section_looking_amendment_prefixed_text_left_textually_intact():
    # The cleaner must NOT "fix" "2[9." into "9." or otherwise touch it --
    # that structural interpretation belongs entirely to the chunker.
    text = ('8. Jurisdiction under this Act. No Court shall take cognizance.\n\n'
            '2[9. No Court shall take cognizance of any offence under this Act '
            '3[except on a complaint made by the Union Council] after one year.]')
    cleaned, _ = clean_statute_text(text)
    assert '2[9. No Court shall take cognizance of any offence under this Act ' \
           '3[except on a complaint made by the Union Council] after one year.]' in cleaned
    assert "\n9. No Court shall" not in cleaned  # never rewritten to a bare "9."


def test_repealed_and_omitted_language_untouched():
    text = ('3.\n3[Omitted.]\n\n'
            '6. 1[Repealed].\n\n'
            '1Rep. by the Repealing and Amending Act, 1942 (XXV of 1942), s. 2 and 1st Sch.')
    cleaned, _ = clean_statute_text(text)
    assert "3[Omitted.]" in cleaned
    assert "1[Repealed]" in cleaned
    assert "1Rep. by the Repealing and Amending Act, 1942 (XXV of 1942), s. 2 and 1st Sch." in cleaned


# -- footnote relocation -----------------------------------------------------------

def test_footnote_lines_removed_from_their_original_location():
    text = ('4. Punishment. Whoever contracts a marriage shall be punishable.\n\n'
            '1Subs. by the Repealing and Amending Act No. VIII of 1930, s.2 and 1st Sch.\n\n'
            '5. Next section. Something else entirely.')
    cleaned, _ = clean_statute_text(text)
    body = cleaned.split("[FOOTNOTES]")[0]
    assert "1Subs. by the Repealing and Amending Act No. VIII of 1930, s.2 and 1st Sch." not in body
    assert "4. Punishment." in body
    assert "5. Next section." in body


def test_footnotes_appear_exactly_once_inside_a_single_footnotes_block():
    text = ('4. Punishment. Something.\n\n'
            '1Subs. by the Repealing and Amending Act No. VIII of 1930, s.2 and 1st Sch.\n'
            '2Ins. by Act No. XIX of 1938, s.2.')
    cleaned, diag = clean_statute_text(text)
    assert cleaned.count("[FOOTNOTES]") == 1
    assert cleaned.count("[/FOOTNOTES]") == 1
    assert cleaned.index("[FOOTNOTES]") < cleaned.index("1Subs.")
    assert cleaned.index("1Subs.") < cleaned.index("[/FOOTNOTES]")
    assert diag.footnotes_relocated_count == 2


def test_footnote_order_is_preserved():
    text = ('4. Punishment. Something.\n\n'
            '1Subs. by the Repealing and Amending Act No. VIII of 1930, s.2 and 1st Sch.\n'
            '2Ins. by Act No. XIX of 1938, s.2.\n'
            '3Omitted by Act No. X of 1996, s.2.')
    cleaned, _ = clean_statute_text(text)
    footnote_block = cleaned.split("[FOOTNOTES]")[1]
    pos_1 = footnote_block.index("1Subs.")
    pos_2 = footnote_block.index("2Ins.")
    pos_3 = footnote_block.index("3Omitted")
    assert pos_1 < pos_2 < pos_3


def test_no_empty_footnotes_block_when_there_are_no_footnotes():
    cleaned, diag = clean_statute_text(_SAMPLE_WITH_TOC)
    assert "[FOOTNOTES]" not in cleaned
    assert diag.footnotes_relocated_count == 0


def test_legitimate_numbered_provisions_are_not_classified_as_footnotes():
    text = ('1. Short title. This Act may be called the Sample Act.\n\n'
            '2. Definitions. In this Act--\n(a) "Court" means a Family Court.\n\n'
            '1A. Saving. Nothing in this Act shall affect pending proceedings.')
    cleaned, diag = clean_statute_text(text)
    assert diag.footnotes_relocated_count == 0
    assert "1. Short title." in cleaned
    assert "2. Definitions." in cleaned
    assert "1A. Saving." in cleaned
    assert "[FOOTNOTES]" not in cleaned


def test_footnote_continuation_line_absorbed_into_the_same_entry():
    # Real-corpus-shaped: a footnote's citation wraps onto a second,
    # lowercase-starting line with no digit prefix of its own.
    text = ('1. Short title.\n\n'
            '1This Act was passed by the West Pakistan Assembly on 30th June, 1964; and,\n'
            '  published in the West Pakistan Gazette on 18th July, 1964.\n\n'
            '2. Definitions.')
    cleaned, diag = clean_statute_text(text)
    assert diag.footnotes_relocated_count == 1
    footnote_block = cleaned.split("[FOOTNOTES]")[1]
    assert "published in the West Pakistan Gazette on 18th July, 1964." in footnote_block
    assert "1This Act was passed by the West Pakistan Assembly on 30th June, 1964; and," in footnote_block


def test_footnote_continuation_stops_at_a_new_footnote_not_absorbed_into_prior_entry():
    text = ('1Subs. by the Central Laws (Statute Reform) Ordinance, 1960.\n'
            '2Ins. by the Muslim Family Laws Ordinance, 1961.')
    cleaned, diag = clean_statute_text(text)
    assert diag.footnotes_relocated_count == 2
    footnote_block = cleaned.split("[FOOTNOTES]")[1]
    assert "1Subs. by the Central Laws (Statute Reform) Ordinance, 1960." in footnote_block
    assert "2Ins. by the Muslim Family Laws Ordinance, 1961." in footnote_block


def test_footnote_continuation_stops_at_resumed_inline_amendment_marker():
    text = ('1Omitted by A.O., 1949.\n'
            '2[9. No Court shall take cognizance of any offence under this Act.]')
    cleaned, diag = clean_statute_text(text)
    assert diag.footnotes_relocated_count == 1
    body, footnote_block = cleaned.split("[FOOTNOTES]")
    assert "2[9. No Court shall take cognizance of any offence under this Act.]" in body
    assert "2[9." not in footnote_block


def test_real_affected_document_footnotes_relocated_without_losing_content():
    # Guard against the real bug class this requirement targets: an
    # amendment footnote sitting between two definitions/subsections.
    # Skips if the real fixture isn't present (CI/sandbox without var/).
    import json
    from pathlib import Path

    fixture = Path("var/rag/statutes_ingested/0425160ffedaefd8c79b3241.json")
    if not fixture.exists():
        import pytest
        pytest.skip("real fixture not present in this environment")
    raw = json.loads(fixture.read_text(encoding="utf-8"))
    cleaned = clean_statute_document(raw)
    assert cleaned["full_text"] == raw["full_text"]
    assert cleaned["cleaning_diagnostics"]["footnotes_relocated_count"] > 0
    assert "[FOOTNOTES]" in cleaned["cleaned_text"]


# -- divider-line removal -----------------------------------------------------------

def test_standalone_underscore_divider_removed():
    text = "3. Interpretation.\n\n________\n\nII.—JURISDICTION\n\n4. Something."
    cleaned, _ = clean_statute_text(text)
    assert "________" not in cleaned
    assert "3. Interpretation." in cleaned
    assert "II.—JURISDICTION" in cleaned
    assert "4. Something." in cleaned


def test_long_underscore_divider_removed():
    text = "Some text.\n\n" + ("_" * 60) + "\n\nMore text."
    cleaned, _ = clean_statute_text(text)
    assert "_" * 60 not in cleaned
    assert "Some text." in cleaned
    assert "More text." in cleaned


def test_short_underscore_run_in_substantive_text_preserved():
    # Only a STANDALONE line of 5+ underscores is layout noise; a short
    # run embedded in real text (e.g. a fill-in-the-blank form field)
    # must survive untouched.
    text = "The applicant's name is ___ and address is ___."
    cleaned, _ = clean_statute_text(text)
    assert "___" in cleaned


# -- soft-wrapped clause/definition header repair ------------------------------------

def test_soft_wrapped_clause_marker_with_curly_double_quote_joined():
    text = '2. Definitions. In this Act--\n(b)\n‘Chairman’ means the Chairman of the Union Council.'
    cleaned, _ = clean_statute_text(text)
    assert "(b) ‘Chairman’ means the Chairman of the Union Council." in cleaned
    assert "(b)\n" not in cleaned


def test_soft_wrapped_clause_marker_with_curly_single_quote_joined():
    text = '2. Definitions.\n(d)\n“Prescribed authority” means the authority prescribed by rules.'
    cleaned, _ = clean_statute_text(text)
    assert "(d) “Prescribed authority” means the authority prescribed by rules." in cleaned
    assert "(d)\n" not in cleaned


def test_ordinary_paragraph_newline_after_clause_text_is_unaffected():
    # A normal paragraph break -- the clause marker is NOT the last thing
    # on its line, so this must not be touched at all.
    text = "(a) “Court” means a Family Court;\n(b) “minor” means a person under eighteen."
    cleaned, _ = clean_statute_text(text)
    assert "(a) “Court” means a Family Court;" in cleaned
    assert "(b) “minor” means a person under eighteen." in cleaned


def test_clause_marker_followed_by_ordinary_prose_on_next_line_unaffected():
    # The next line does NOT start with a quote/bracket -- must not be
    # joined; this is an ordinary (if slightly unusual) line wrap, not
    # the specific soft-wrap pattern this rule targets.
    text = "(c)\nGovernment means the Provincial Government."
    cleaned, _ = clean_statute_text(text)
    assert "(c)\nGovernment means the Provincial Government." in cleaned


# -- conservative hyphenation repair (allowlist-based) -------------------------------

def test_known_broken_token_mujtahid_e_alam_is_repaired():
    text = "The Court shall maintain a panel of Mujtahid-\ne-Alam having the prescribed qualifications."
    cleaned, _ = clean_statute_text(text)
    assert "Mujtahid-e-Alam" in cleaned
    assert "Mujtahid-\ne-Alam" not in cleaned


def test_dwelling_house_is_not_collapsed_into_one_word():
    text = "When my wife left my dwelling-\nhouse on the day of the marriage."
    cleaned, _ = clean_statute_text(text)
    assert "dwelling-house" in cleaned
    assert "dwellinghouse" not in cleaned


def test_unknown_hyphenated_compound_line_break_keeps_its_hyphen():
    # "sub-section" is always hyphenated elsewhere in real statute text
    # (never "subsection") -- an unknown broken token defaults to
    # keeping its hyphen rather than guessing it should be removed.
    text = "The evidence referred to in sub-\nsection (1) shall be recorded."
    cleaned, _ = clean_statute_text(text)
    assert "sub-section (1)" in cleaned
    assert "subsection" not in cleaned


def test_known_single_word_break_education_is_still_fully_joined():
    text = "Every child has a right to educa-\ntion under this Act."
    cleaned, _ = clean_statute_text(text)
    assert "education" in cleaned
    assert "educa-tion" not in cleaned
    assert "educa-\ntion" not in cleaned


# -- pipeline isolation ------------------------------------------------------------

def test_statute_cleaning_module_does_not_import_case_law_or_statute_cleaner_code():
    import ast

    import src.rag_prep.statute_cleaning as mod

    tree = ast.parse(open(mod.__file__, encoding="utf-8").read())
    imported_modules = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported_modules.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported_modules.append(node.module)

    forbidden_prefixes = (
        "src.ingestion",
        "src.classification",
        "src.rag_prep.case_cleaner",
        "src.rag_prep.structurer",
        "src.rag_prep.chunk",
        "src.rag_prep.statute_cleaner",  # the existing, unrelated case-law citation extractor
        "src.extraction.case_loader",
    )
    for name in imported_modules:
        assert not name.startswith(forbidden_prefixes), f"statute_cleaning.py must not import {name!r}"
