"""Tests for the Constitution cleaning stage
(src.rag_prep.constitution_cleaning).

Synthetic fixtures mirror the real noise patterns this module was
validated against (the real Constitution of Pakistan document in
var/rag/constitution_ingested/), but every fixture here is hand-
constructed -- no real ingested JSON is required for these tests.
"""

from __future__ import annotations

import json

from src.rag_prep.constitution_cleaning import (
    LARGE_REMOVAL_FLAG_THRESHOLD,
    clean_constitution_document,
    clean_constitution_text,
    clean_directory,
)


def _raw_doc(raw_text: str, **metadata_overrides) -> dict:
    metadata = {"document_type": "constitution", "title": "Constitution of Pakistan"}
    metadata.update(metadata_overrides)
    return {
        "doc_id": "doc1",
        "source_file": "/x/constitution.pdf",
        "metadata": metadata,
        "pages": [{"page_number": 1, "text": raw_text}],
        "raw_text": raw_text,
    }


_SAMPLE_WITH_FRONT_MATTER = """THE
CONSTITUTION
OF THE
ISLAMIC REPUBLIC
OF
PAKISTAN
2221(25)L&J---by Waleed---PC-2 (Quark 10)


PREFACE
This Fourteenth Edition incorporates all amendments made till date.

____
CONTENTS
____
       ARTICLES
       1.    The Republic and its territories.

PREAMBLE
 WHEREAS sovereignty over the entire Universe belongs
to Almighty Allah alone;

                             CONSTITUTION OF PAKISTAN                          1

PART I
Introductory
1. (1) Pakistan shall be Federal Republic to be known as
the Islamic Republic of Pakistan."""


# -- 1. front matter / TOC removal ---------------------------------------------

def test_front_matter_and_toc_removed_up_to_preamble():
    cleaned, diag = clean_constitution_text(_SAMPLE_WITH_FRONT_MATTER)
    assert diag.front_matter_removed is True
    assert "PREFACE" not in cleaned
    assert "CONTENTS" not in cleaned
    assert cleaned.startswith("PREAMBLE")


def test_preamble_and_body_survive_front_matter_removal():
    cleaned, _ = clean_constitution_text(_SAMPLE_WITH_FRONT_MATTER)
    assert "WHEREAS sovereignty over the entire Universe belongs" in cleaned
    assert "Pakistan shall be Federal Republic" in cleaned


def test_no_front_matter_removed_when_preamble_anchor_absent():
    text = "1. (1) Pakistan shall be Federal Republic to be known as the Islamic Republic."
    cleaned, diag = clean_constitution_text(text)
    assert diag.front_matter_removed is False
    assert "Pakistan shall be Federal Republic" in cleaned


# -- 2. repeated running header removal ----------------------------------------

def test_running_header_combined_with_pagenum_on_same_line_removed():
    text = "PREAMBLE\nSome text.\n\n                             CONSTITUTION OF PAKISTAN                          3\n\nMore text."
    cleaned, _ = clean_constitution_text(text)
    assert "CONSTITUTION OF PAKISTAN" not in cleaned
    assert "Some text." in cleaned
    assert "More text." in cleaned


def test_running_header_with_pagenum_before_header_removed():
    text = "PREAMBLE\nSome text.\n\n66                          CONSTITUTION OF PAKISTAN\n\nMore text."
    cleaned, _ = clean_constitution_text(text)
    assert "CONSTITUTION OF PAKISTAN" not in cleaned
    assert "66" not in cleaned
    assert "Some text." in cleaned
    assert "More text." in cleaned


def test_running_header_split_across_two_lines_removed_with_its_pagenum():
    # Confirmed real layout artifact: header alone, page number on the
    # very next line (not combined on one line).
    text = "PREAMBLE\nSome text.\n\n                             CONSTITUTION OF PAKISTAN\n67\n\nMore text."
    cleaned, _ = clean_constitution_text(text)
    assert "CONSTITUTION OF PAKISTAN" not in cleaned
    assert "\n67\n" not in cleaned
    assert "Some text." in cleaned
    assert "More text." in cleaned


# -- 3. standalone page-number removal (via the header pairing above) ---------

def test_page_number_only_removed_when_paired_with_the_header():
    # A bare number elsewhere (e.g. seat-count table data) must NOT be
    # removed just because it looks like a page number.
    text = "PREAMBLE\n51\n11\n3\n65\nMore text."
    cleaned, _ = clean_constitution_text(text)
    assert "51" in cleaned and "11" in cleaned and "3" in cleaned and "65" in cleaned


# -- 4. printer metadata removal ------------------------------------------------

def test_printer_metadata_line_removed():
    text = "PREAMBLE\n2221(25)L&J---by Waleed---PC-2 (Quark 10)\nReal content here."
    cleaned, _ = clean_constitution_text(text)
    assert "L&J---by Waleed" not in cleaned
    assert "Real content here." in cleaned


# -- 5. RGN/date-style artifact handling ----------------------------------------

def test_date_like_text_is_not_mistaken_for_an_artifact():
    # No RGN/date-stamp artifact pattern was found in the real
    # Constitution corpus (unlike the statute corpus) -- this confirms
    # ordinary dates in body/footnote text are simply left alone.
    text = "PREAMBLE\n1The provisions were brought into force with effect from 10th March, 1985."
    cleaned, _ = clean_constitution_text(text)
    assert "10th March, 1985" in cleaned.split("[FOOTNOTES]")[1] if "[FOOTNOTES]" in cleaned else "10th March, 1985" in cleaned


# -- 6. decorative dividers -------------------------------------------------------

def test_standalone_underscore_divider_removed():
    text = "PREAMBLE\nHigh treason.\n\n______________________________\n\nPART II"
    cleaned, _ = clean_constitution_text(text)
    assert "______________________________" not in cleaned
    assert "High treason." in cleaned
    assert "PART II" in cleaned


def test_standalone_dash_divider_removed():
    text = "PREAMBLE\nenact and give to ourselves, this Constitution.\n––––––\nPART I"
    cleaned, _ = clean_constitution_text(text)
    assert "––––––" not in cleaned
    assert "PART I" in cleaned


def test_fill_in_the_blank_underscores_inside_a_sentence_preserved():
    # A real, substantive oath-form blank -- "I, ______, do solemnly
    # swear" -- must never be touched; it is not a standalone divider.
    text = 'PREAMBLE\n(In the name of Allah, the most Beneficent.)\n I, ______________________, do solemnly swear that I am a Muslim.'
    cleaned, _ = clean_constitution_text(text)
    assert "I, ______________________, do solemnly swear" in cleaned


def test_asterisk_omission_marker_is_never_touched_by_divider_removal():
    text = "PREAMBLE\n (2) There shall be no discrimination on the basis of sex\n8*          *           *           *           *\n (3) Nothing in this Article."
    cleaned, _ = clean_constitution_text(text)
    assert "*" in cleaned
    body = cleaned.split("[FOOTNOTES]")[0]
    assert "*" in body


# -- 7. superscript/footnote callout normalization ------------------------------

def test_superscript_digit_removed_but_brackets_and_content_kept():
    text = 'PREAMBLE\nNo Court other than a 4[commission of] shall act.'
    cleaned, _ = clean_constitution_text(text)
    assert "[commission of]" in cleaned
    assert "4[commission of]" not in cleaned


def test_short_phrase_bracket_content_never_rewritten():
    text = "PREAMBLE\nthe Provinces of 3[Balochistan], the Punjab and 5[Sindh]."
    cleaned, _ = clean_constitution_text(text)
    assert "[Balochistan]" in cleaned
    assert "[Sindh]" in cleaned


# -- 8. preservation of constitutional brackets / clause-level unwrap ----------

def test_clause_level_amendment_wrapper_unwrapped_but_clause_number_kept():
    text = "PREAMBLE\n1[51. (1)    There shall be 2[three hundred and thirty-six] seats for members.]"
    cleaned, _ = clean_constitution_text(text)
    assert "51. (1)" in cleaned
    assert "three hundred and thirty-six" in cleaned
    assert "1[51." not in cleaned


def test_nested_amendment_brackets_fully_unwrapped():
    text = ("PREAMBLE\n"
            "1[2. (1) Pakistan shall comprise the Provinces of 2[Balochistan], the "
            "3[Khyber Pakhtunkhwa], the Punjab and 4[Sindh].]")
    cleaned, _ = clean_constitution_text(text)
    assert "2. (1) Pakistan shall comprise" in cleaned
    assert "1[2." not in cleaned
    assert "[Balochistan]" in cleaned
    assert "[Khyber Pakhtunkhwa]" in cleaned


def test_statutory_reference_bracket_not_globally_removed():
    text = "PREAMBLE\nsubject to 2[the Code of Civil Procedure, 1908 (Act V of 1908)]."
    cleaned, _ = clean_constitution_text(text)
    assert "[the Code of Civil Procedure, 1908 (Act V of 1908)]" in cleaned


# -- 9. preservation of omitted/repealed provisions ------------------------------

def test_omitted_article_status_text_preserved():
    text = ("PREAMBLE\n246. Something.\n\n"
            "247.\n[Administration of Tribal Areas.] Omitted by the\n"
            "Constitution (Twenty-fifth Amdt.) Act, 2018 (37 of 2018), s. 9.\n\n"
            "CHAPTER 4.--GENERAL")
    cleaned, _ = clean_constitution_text(text)
    assert "[Administration of Tribal Areas.]" in cleaned
    assert "Omitted by the" in cleaned
    assert "CHAPTER 4" in cleaned


def test_repealed_amendment_history_preserved_in_footnotes():
    text = ("PREAMBLE\n140. Something.\n\n"
            "6Sixth Schedule and Seventh Schedule omitted by the Constitution "
            "(Eighteenth Amdt.) Act, 2010.")
    cleaned, _ = clean_constitution_text(text)
    assert "Sixth Schedule and Seventh Schedule omitted" in cleaned


# -- 10. preservation of "* * * *" omission -------------------------------------

def test_multi_asterisk_omission_marker_preserved_verbatim():
    text = ("PREAMBLE\nhe is not declared by a competent court to be of unsound mind.\n"
            "        4*          *           *           *           *        \n"
            "        (2)    A person shall be entitled to vote.")
    cleaned, _ = clean_constitution_text(text)
    assert "*" in cleaned
    assert "(2)    A person shall be entitled to vote." in cleaned or "(2) A person shall be entitled to vote." in cleaned


def test_single_asterisk_omission_marker_preserved():
    text = "PREAMBLE\nno discrimination on the basis of sex\n1*.\n(3) Nothing in this Article."
    cleaned, _ = clean_constitution_text(text)
    assert "1*." in cleaned.split("[FOOTNOTES]")[0] or "*." in cleaned.split("[FOOTNOTES]")[0]


# -- 11. amendment-history footnote handling -------------------------------------

def test_amendment_footnote_relocated_to_footnotes_block():
    text = ("PREAMBLE\n9A. Clean and healthy environment.\n\n"
            "1Ins. by the Constitution (Eighteenth Amdt.) Act, 2010 (10 of 2010), s. 2.")
    cleaned, diag = clean_constitution_text(text)
    assert diag.footnotes_relocated_count == 1
    body, footnote_block = cleaned.split("[FOOTNOTES]")
    assert "1Ins. by the Constitution" not in body
    assert "1Ins. by the Constitution (Eighteenth Amdt.) Act, 2010 (10 of 2010), s. 2." in footnote_block


def test_inline_amendment_information_not_removed():
    text = "PREAMBLE\n9. Security of 2[person] shall not be denied."
    cleaned, diag = clean_constitution_text(text)
    assert diag.footnotes_relocated_count == 0
    assert "[person]" in cleaned


def test_genuine_chapter_heading_with_glued_digit_not_treated_as_footnote():
    # Confirmed real collision risk: "1CHAPTER 3A.--FEDERAL SHARIAT
    # COURT" is a genuine heading, not a footnote -- "CHAPTER" is
    # deliberately excluded from the footnote-lead vocabulary.
    text = "PREAMBLE\n1CHAPTER 3A.—FEDERAL SHARIAT COURT\n\n203. Something."
    cleaned, diag = clean_constitution_text(text)
    assert diag.footnotes_relocated_count == 0
    assert "CHAPTER 3A" in cleaned


# -- 12. hyphenated line-break repair --------------------------------------------

def test_ordinary_word_wrap_joined_without_hyphen():
    text = "PREAMBLE\nthe provisions of the Constitution except those of Articles 6, 8 to 28, (both inclu-\nsive)."
    cleaned, _ = clean_constitution_text(text)
    assert "inclusive" in cleaned
    assert "inclu-" not in cleaned


def test_known_compound_majlis_e_shoora_keeps_its_hyphen():
    text = "PREAMBLE\nThe Parliament shall maintain a panel of Majlis-\ne-Shoora (Parliament)."
    cleaned, _ = clean_constitution_text(text)
    assert "Majlis-e-Shoora" in cleaned


def test_known_compound_sub_paragraph_keeps_its_hyphen():
    text = "PREAMBLE\nreferred to in sub-\nparagraph (ii) of paragraph (b)."
    cleaned, _ = clean_constitution_text(text)
    assert "sub-paragraph" in cleaned
    assert "subparagraph" not in cleaned


def test_unknown_wrap_defaults_to_joining_not_preserving_hyphen():
    text = "PREAMBLE\nin accor-\ndance with law."
    cleaned, _ = clean_constitution_text(text)
    assert "accordance" in cleaned
    assert "accor-dance" not in cleaned


# -- 13. whitespace normalization -------------------------------------------------

def test_excessive_blank_lines_collapsed_but_paragraphs_preserved():
    text = "PREAMBLE\nFirst clause.\n\n\n\n\n\nSecond clause."
    cleaned, _ = clean_constitution_text(text)
    assert "\n\n\n" not in cleaned
    assert "First clause." in cleaned
    assert "Second clause." in cleaned


def test_repeated_spaces_collapsed():
    text = "PREAMBLE\nThis   Constitution     extends   to   Pakistan."
    cleaned, _ = clean_constitution_text(text)
    assert "   " not in cleaned
    assert "This Constitution extends to Pakistan." in cleaned


def test_document_not_flattened_into_a_single_paragraph():
    cleaned, _ = clean_constitution_text(_SAMPLE_WITH_FRONT_MATTER)
    assert cleaned.count("\n\n") >= 1


# -- 14. marginal/side-note association ------------------------------------------

def test_marginal_note_associated_when_label_exactly_precedes_matching_article():
    text = "PREAMBLE\nThe Republic and its territories\n\n1. The Republic and its territories shall be as follows."
    cleaned, _ = clean_constitution_text(text)
    assert "### Article 1. The Republic and its territories" in cleaned


def test_no_marginal_note_invented_when_no_matching_label_present():
    # Confirmed real-corpus behavior: this detector is a documented
    # no-op when the source has no distinguishable marginal-label line.
    text = "PREAMBLE\nPART I\nIntroductory\n1. (1) Pakistan shall be Federal Republic to be known as Pakistan."
    cleaned, _ = clean_constitution_text(text)
    assert "### Article" not in cleaned
    assert "PART I" in cleaned
    assert "Introductory" in cleaned


# -- raw_text preservation / document-level behavior ----------------------------

def test_raw_text_is_preserved_unchanged_in_output():
    raw = _raw_doc(_SAMPLE_WITH_FRONT_MATTER)
    cleaned = clean_constitution_document(raw)
    assert cleaned["raw_text"] == raw["raw_text"]


def test_doc_id_and_metadata_preserved():
    raw = _raw_doc(_SAMPLE_WITH_FRONT_MATTER, title="Constitution of Pakistan")
    cleaned = clean_constitution_document(raw)
    assert cleaned["doc_id"] == raw["doc_id"]
    assert cleaned["metadata"]["document_type"] == "constitution"
    assert cleaned["metadata"]["title"] == "Constitution of Pakistan"


def test_cleaned_output_is_non_empty():
    raw = _raw_doc(_SAMPLE_WITH_FRONT_MATTER)
    cleaned = clean_constitution_document(raw)
    assert cleaned["cleaned_text"].strip() != ""


def test_clean_directory_does_not_modify_raw_input_files(tmp_path):
    input_dir = tmp_path / "in"
    output_dir = tmp_path / "out"
    input_dir.mkdir()
    raw = _raw_doc(_SAMPLE_WITH_FRONT_MATTER)
    raw_path = input_dir / "doc1.json"
    raw_path.write_text(json.dumps(raw), encoding="utf-8")
    original_bytes = raw_path.read_bytes()

    summary = clean_directory(input_dir, output_dir)

    assert summary["processed"] == 1
    assert raw_path.read_bytes() == original_bytes


def test_large_removal_is_flagged_not_silently_accepted():
    text = "PREFACE\n" + ("Filler front matter text line.\n" * 60) + "\nPREAMBLE\nReal content."
    cleaned, diag = clean_constitution_text(text)
    assert diag.front_matter_removed is True
    assert diag.pct_removed > LARGE_REMOVAL_FLAG_THRESHOLD
    assert diag.large_removal_flag is True
    assert "Real content." in cleaned


# -- 16. deterministic repeated execution ----------------------------------------

def test_cleaning_is_deterministic_across_repeated_runs():
    first, _ = clean_constitution_text(_SAMPLE_WITH_FRONT_MATTER)
    second, _ = clean_constitution_text(_SAMPLE_WITH_FRONT_MATTER)
    assert first == second


def test_document_cleaning_is_deterministic_across_repeated_runs():
    raw = _raw_doc(_SAMPLE_WITH_FRONT_MATTER)
    first = clean_constitution_document(raw)
    second = clean_constitution_document(raw)
    assert first == second


# -- pipeline isolation ------------------------------------------------------------

def test_constitution_cleaning_does_not_import_statute_or_case_law_code():
    import ast

    import src.rag_prep.constitution_cleaning as mod

    tree = ast.parse(open(mod.__file__, encoding="utf-8").read())
    imported = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.extend(a.name for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.append(node.module)

    forbidden_prefixes = (
        "src.ingestion", "src.classification",
        "src.rag_prep.case_cleaner", "src.rag_prep.structurer",
        "src.rag_prep.chunk", "src.rag_prep.statute_",
        "src.extraction.case_loader",
    )
    for name in imported:
        assert not name.startswith(forbidden_prefixes), f"constitution_cleaning.py must not import {name!r}"
