"""Tests for the Constitution ingestion stage
(src.rag_prep.constitution_ingestion).

Synthetic PDFs only, built on the fly with PyMuPDF (fitz) -- the real
Constitution of Pakistan PDF is never required or used here.
"""

from __future__ import annotations

import json

import fitz
import pytest

from src.rag_prep.constitution_ingestion import (
    DOCUMENT_TYPE,
    TITLE,
    ConstitutionIngestionError,
    doc_id_for_constitution,
    ingest,
    ingest_constitution_pdf,
)


def _make_pdf(path, pages_text: list[str]) -> None:
    doc = fitz.open()
    for text in pages_text:
        page = doc.new_page()
        if text:
            page.insert_text((72, 72), text, fontsize=11)
    doc.save(str(path))
    doc.close()


# -- basic extraction / schema -------------------------------------------------

def test_extraction_produces_expected_top_level_schema(tmp_path):
    pdf_path = tmp_path / "constitution.pdf"
    _make_pdf(pdf_path, ["PREAMBLE\nWhereas sovereignty belongs to Allah Almighty."])

    doc = ingest_constitution_pdf(pdf_path)

    assert set(doc.keys()) == {"doc_id", "source_file", "metadata", "pages", "raw_text"}


def test_metadata_has_fixed_document_type_and_title(tmp_path):
    pdf_path = tmp_path / "constitution.pdf"
    _make_pdf(pdf_path, ["1. The Republic and its territories."])

    doc = ingest_constitution_pdf(pdf_path)

    assert doc["metadata"]["document_type"] == DOCUMENT_TYPE == "constitution"
    assert doc["metadata"]["title"] == TITLE == "Constitution of Pakistan"


def test_source_file_records_the_input_pdf_path(tmp_path):
    pdf_path = tmp_path / "constitution.pdf"
    _make_pdf(pdf_path, ["Some article text."])

    doc = ingest_constitution_pdf(pdf_path)

    assert doc["source_file"] == str(pdf_path)


# -- page-level extraction / ordering -------------------------------------------

def test_pages_are_extracted_with_page_numbers_in_order(tmp_path):
    pdf_path = tmp_path / "constitution.pdf"
    _make_pdf(pdf_path, ["Page one text.", "Page two text.", "Page three text."])

    doc = ingest_constitution_pdf(pdf_path)

    assert [p["page_number"] for p in doc["pages"]] == [1, 2, 3]
    assert "Page one text." in doc["pages"][0]["text"]
    assert "Page two text." in doc["pages"][1]["text"]
    assert "Page three text." in doc["pages"][2]["text"]


def test_each_page_has_page_number_and_text_keys_only(tmp_path):
    pdf_path = tmp_path / "constitution.pdf"
    _make_pdf(pdf_path, ["Only page."])

    doc = ingest_constitution_pdf(pdf_path)

    assert set(doc["pages"][0].keys()) == {"page_number", "text"}


def test_page_text_is_preserved_exactly_as_extracted_no_cleaning(tmp_path):
    # This stage must not strip, normalize, or otherwise touch whatever
    # the PDF extractor itself returns -- that is the next stage's job
    # entirely. Compared directly against extract_text_layer()'s own
    # output (not against the raw PyMuPDF insert_text() input), since a
    # PDF text round-trip does not guarantee byte-identical whitespace
    # to begin with -- that discrepancy belongs to the extractor, not to
    # this module, which must simply not add any discrepancy of its own.
    from src.extraction.pdf_extractor import extract_text_layer

    noisy = "CONSTITUTION OF PAKISTAN\n\n\ni\n\n1[ANNEX\n________\n   extra   spaces  "
    pdf_path = tmp_path / "constitution.pdf"
    _make_pdf(pdf_path, [noisy])

    expected = extract_text_layer(pdf_path, max_pages=100_000).pages[0].text
    doc = ingest_constitution_pdf(pdf_path)

    assert doc["pages"][0]["text"] == expected


# -- raw_text construction -------------------------------------------------------

def test_raw_text_is_built_from_page_texts_in_original_order(tmp_path):
    pdf_path = tmp_path / "constitution.pdf"
    _make_pdf(pdf_path, ["Alpha section.", "Beta section.", "Gamma section."])

    doc = ingest_constitution_pdf(pdf_path)

    alpha_pos = doc["raw_text"].index("Alpha section.")
    beta_pos = doc["raw_text"].index("Beta section.")
    gamma_pos = doc["raw_text"].index("Gamma section.")
    assert alpha_pos < beta_pos < gamma_pos


def test_no_cleaning_applied_to_raw_text(tmp_path):
    noisy = "CONTENTS\n____\nPage 1 of 252\n   extra    spaces   here  "
    pdf_path = tmp_path / "constitution.pdf"
    _make_pdf(pdf_path, [noisy])

    doc = ingest_constitution_pdf(pdf_path)

    assert "CONTENTS" in doc["raw_text"]
    assert "____" in doc["raw_text"]
    assert "extra    spaces" in doc["raw_text"] or "extra" in doc["raw_text"]


# -- deterministic doc_id ---------------------------------------------------------

def test_doc_id_is_deterministic_for_the_same_path(tmp_path):
    pdf_path = tmp_path / "constitution.pdf"
    _make_pdf(pdf_path, ["Text."])

    first = doc_id_for_constitution(pdf_path)
    second = doc_id_for_constitution(pdf_path)
    assert first == second


def test_doc_id_differs_for_different_paths(tmp_path):
    pdf_path_a = tmp_path / "a" / "constitution.pdf"
    pdf_path_b = tmp_path / "b" / "constitution.pdf"
    pdf_path_a.parent.mkdir()
    pdf_path_b.parent.mkdir()
    _make_pdf(pdf_path_a, ["Text."])
    _make_pdf(pdf_path_b, ["Text."])

    assert doc_id_for_constitution(pdf_path_a) != doc_id_for_constitution(pdf_path_b)


def test_ingestion_doc_id_matches_the_standalone_function(tmp_path):
    pdf_path = tmp_path / "constitution.pdf"
    _make_pdf(pdf_path, ["Text."])

    doc = ingest_constitution_pdf(pdf_path)
    assert doc["doc_id"] == doc_id_for_constitution(pdf_path)


# -- empty text layer fails loudly ------------------------------------------------

def test_empty_text_layer_raises_clear_error(tmp_path):
    pdf_path = tmp_path / "blank.pdf"
    _make_pdf(pdf_path, ["", "", ""])

    with pytest.raises(ConstitutionIngestionError):
        ingest_constitution_pdf(pdf_path)


# -- end-to-end ingest() / file output --------------------------------------------

def test_ingest_writes_one_json_file_to_output_dir(tmp_path):
    pdf_path = tmp_path / "constitution.pdf"
    _make_pdf(pdf_path, ["1. The Republic and its territories."])
    output_dir = tmp_path / "out"

    summary = ingest(pdf_path, output_dir)

    assert summary == {"processed": 1, "failed": 0, "error": None}
    written = list(output_dir.glob("*.json"))
    assert len(written) == 1
    doc = json.loads(written[0].read_text(encoding="utf-8"))
    assert doc["metadata"]["document_type"] == "constitution"


def test_ingest_does_not_modify_the_original_pdf(tmp_path):
    pdf_path = tmp_path / "constitution.pdf"
    _make_pdf(pdf_path, ["1. The Republic and its territories."])
    original_bytes = pdf_path.read_bytes()
    output_dir = tmp_path / "out"

    ingest(pdf_path, output_dir)

    assert pdf_path.read_bytes() == original_bytes


def test_ingest_reports_failure_without_raising_for_empty_pdf(tmp_path):
    pdf_path = tmp_path / "blank.pdf"
    _make_pdf(pdf_path, [""])
    output_dir = tmp_path / "out"

    summary = ingest(pdf_path, output_dir)

    assert summary["processed"] == 0
    assert summary["failed"] == 1
    assert summary["error"]
    assert list(output_dir.glob("*.json")) == []


# -- pipeline isolation ------------------------------------------------------------

def test_constitution_ingestion_does_not_import_statute_or_case_law_code():
    import ast

    import src.rag_prep.constitution_ingestion as mod

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
        assert not name.startswith(forbidden_prefixes), f"constitution_ingestion.py must not import {name!r}"
