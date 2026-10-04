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
