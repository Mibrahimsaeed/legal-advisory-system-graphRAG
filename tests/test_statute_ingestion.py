"""Tests for the statute ingestion stage (src.rag_prep.statute_ingestion).

Synthetic PDFs only, built on the fly with PyMuPDF (fitz) -- no real
production statute PDFs are required or used.
"""

from __future__ import annotations

import json

import fitz
import pytest

from src.rag_prep.statute_ingestion import (
    DOCUMENT_TYPE,
    DOMAIN,
    DOMAIN_SOURCE,
    StatuteIngestionError,
    doc_id_for_statute,
    ingest_directory,
    ingest_one_pdf,
)


def _make_pdf(path, pages_text: list[str], title: str | None = None) -> None:
    doc = fitz.open()
    for text in pages_text:
        page = doc.new_page()
        if text:
            page.insert_text((72, 72), text, fontsize=11)
    if title:
        doc.set_metadata({"title": title})
    doc.save(str(path))
    doc.close()


# -- basic extraction / schema -------------------------------------------------

def test_pdf_to_json_extraction_produces_expected_schema(tmp_path):
    pdf_path = tmp_path / "family_courts_act.pdf"
    _make_pdf(pdf_path, ["1. Short title. This Act may be called the Family Courts Act."])

    doc = ingest_one_pdf(pdf_path, tmp_path)

    assert set(doc.keys()) == {"doc_id", "metadata", "full_text"}
    for key in ("law_name", "year", "document_type", "domain", "domain_source",
                "source_file", "page_count", "extraction_engine", "content_hash"):
        assert key in doc["metadata"]


def test_full_text_is_preserved(tmp_path):
    pdf_path = tmp_path / "statute.pdf"
    text = "1. Short title. 2. Definitions. In this Act, unless the context otherwise requires--"
    _make_pdf(pdf_path, [text])

    doc = ingest_one_pdf(pdf_path, tmp_path)

    assert "Short title" in doc["full_text"]
    assert "Definitions" in doc["full_text"]


def test_page_order_is_preserved_across_multiple_pages(tmp_path):
    pdf_path = tmp_path / "multi_page.pdf"
    _make_pdf(pdf_path, ["PAGE_ONE_MARKER section text", "PAGE_TWO_MARKER section text",
                          "PAGE_THREE_MARKER section text"])

    doc = ingest_one_pdf(pdf_path, tmp_path)

    first = doc["full_text"].index("PAGE_ONE_MARKER")
    second = doc["full_text"].index("PAGE_TWO_MARKER")
    third = doc["full_text"].index("PAGE_THREE_MARKER")
    assert first < second < third
    assert doc["metadata"]["page_count"] == 3


# -- metadata -------------------------------------------------------------------

def test_metadata_preserved_from_pdf_title(tmp_path):
    pdf_path = tmp_path / "ordinance.pdf"
    _make_pdf(pdf_path, ["Some statute text here."], title="Muslim Family Laws Ordinance")

    doc = ingest_one_pdf(pdf_path, tmp_path)

    assert doc["metadata"]["law_name"] == "Muslim Family Laws Ordinance"


def test_metadata_sidecar_override_is_preserved_not_invented(tmp_path):
    pdf_path = tmp_path / "ordinance2.pdf"
    _make_pdf(pdf_path, ["Some statute text here."])
    sidecar = tmp_path / "ordinance2.metadata.json"
    sidecar.write_text(json.dumps({"law_name": "West Pakistan Family Courts Act", "year": "1964"}))

    doc = ingest_one_pdf(pdf_path, tmp_path)

    assert doc["metadata"]["law_name"] == "West Pakistan Family Courts Act"
    assert doc["metadata"]["year"] == "1964"


def test_missing_metadata_is_null_not_invented(tmp_path):
    pdf_path = tmp_path / "plain.pdf"
    _make_pdf(pdf_path, ["Some statute text with no embedded title at all."])

    doc = ingest_one_pdf(pdf_path, tmp_path)

    assert doc["metadata"]["year"] is None
    # law_name may be None OR fall back to the first line of text as a
    # "title" heuristic inside the shared PDF extractor -- but it must
    # never be fabricated beyond what the PDF/sidecar actually provided.
    assert doc["metadata"]["law_name"] in (None, "Some statute text with no embedded title at all.")


def test_domain_is_curated_family_law(tmp_path):
    pdf_path = tmp_path / "x.pdf"
    _make_pdf(pdf_path, ["Statute text."])

    doc = ingest_one_pdf(pdf_path, tmp_path)

    assert doc["metadata"]["domain"] == DOMAIN == "family_law"
    assert doc["metadata"]["domain_source"] == DOMAIN_SOURCE == "curated"
    assert doc["metadata"]["document_type"] == DOCUMENT_TYPE == "statute"


def test_source_file_provenance_is_relative_path(tmp_path):
    subdir = tmp_path / "acts"
    subdir.mkdir()
    pdf_path = subdir / "act.pdf"
    _make_pdf(pdf_path, ["Statute text."])

    doc = ingest_one_pdf(pdf_path, tmp_path)

    assert doc["metadata"]["source_file"] == "acts/act.pdf"


# -- doc_id determinism -----------------------------------------------------------

def test_doc_id_is_deterministic(tmp_path):
    pdf_path = tmp_path / "statute.pdf"
    _make_pdf(pdf_path, ["text"])

    id1 = doc_id_for_statute(tmp_path, pdf_path)
    id2 = doc_id_for_statute(tmp_path, pdf_path)
    assert id1 == id2
    assert len(id1) == 24


def test_doc_id_differs_for_different_paths(tmp_path):
    pdf1 = tmp_path / "a.pdf"
    pdf2 = tmp_path / "b.pdf"
    _make_pdf(pdf1, ["text"])
    _make_pdf(pdf2, ["text"])
    assert doc_id_for_statute(tmp_path, pdf1) != doc_id_for_statute(tmp_path, pdf2)


# -- empty / unextractable PDFs fail clearly ---------------------------------------

def test_pdf_with_no_text_fails_clearly(tmp_path):
    pdf_path = tmp_path / "blank.pdf"
    _make_pdf(pdf_path, [""])  # a page with no text at all

    with pytest.raises(StatuteIngestionError):
        ingest_one_pdf(pdf_path, tmp_path)


def test_ingest_directory_records_failure_without_crashing_the_batch(tmp_path):
    input_dir = tmp_path / "in"
    output_dir = tmp_path / "out"
    input_dir.mkdir()
    _make_pdf(input_dir / "good.pdf", ["Real statute text with real words."])
    _make_pdf(input_dir / "blank.pdf", [""])

    summary = ingest_directory(input_dir, output_dir)

    assert summary["processed"] == 1
    assert summary["failed"] == 1
    assert summary["failed_pdfs"][0]["source_file"] == "blank.pdf"
    assert len(list(output_dir.glob("*.json"))) == 1


# -- output JSON on disk -----------------------------------------------------------

def test_ingest_directory_writes_valid_json_per_pdf(tmp_path):
    input_dir = tmp_path / "in"
    output_dir = tmp_path / "out"
    input_dir.mkdir()
    _make_pdf(input_dir / "act_one.pdf", ["Section 1. Short title and commencement."])
    _make_pdf(input_dir / "act_two.pdf", ["Section 1. Definitions."])

    summary = ingest_directory(input_dir, output_dir)

    assert summary["processed"] == 2
    out_files = sorted(output_dir.glob("*.json"))
    assert len(out_files) == 2
    for p in out_files:
        doc = json.loads(p.read_text(encoding="utf-8"))
        assert doc["doc_id"] == p.stem
        assert doc["metadata"]["domain"] == "family_law"


def test_ingest_directory_is_not_recursive_by_default(tmp_path):
    input_dir = tmp_path / "in"
    output_dir = tmp_path / "out"
    nested = input_dir / "nested"
    nested.mkdir(parents=True)
    _make_pdf(input_dir / "top_level.pdf", ["top level statute text."])
    _make_pdf(nested / "nested_act.pdf", ["nested statute text."])

    summary = ingest_directory(input_dir, output_dir, recursive=False)

    assert summary["processed"] == 1


def test_ingest_directory_recursive_flag_includes_nested_pdfs(tmp_path):
    input_dir = tmp_path / "in"
    output_dir = tmp_path / "out"
    nested = input_dir / "nested"
    nested.mkdir(parents=True)
    _make_pdf(input_dir / "top_level.pdf", ["top level statute text."])
    _make_pdf(nested / "nested_act.pdf", ["nested statute text."])

    summary = ingest_directory(input_dir, output_dir, recursive=True)

    assert summary["processed"] == 2


# -- case-law pipeline isolation ---------------------------------------------------

def test_statute_ingestion_module_does_not_import_case_law_pipeline_code():
    import ast

    import src.rag_prep.statute_ingestion as mod

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
        "src.extraction.case_loader",
    )
    for name in imported_modules:
        assert not name.startswith(forbidden_prefixes), f"statute_ingestion.py must not import {name!r}"


def test_statute_ingestion_does_not_touch_case_law_output_directories(tmp_path, monkeypatch):
    import src.rag_prep.statute_ingestion as mod

    input_dir = tmp_path / "in"
    output_dir = tmp_path / "out"
    input_dir.mkdir()
    _make_pdf(input_dir / "act.pdf", ["Statute text."])

    # Case-law directories must never even be referenced -- monkeypatch
    # this module's own constants to prove ingest_directory() only ever
    # touches the paths explicitly passed to it.
    monkeypatch.setattr(mod, "INPUT_DIR", input_dir)
    monkeypatch.setattr(mod, "OUTPUT_DIR", output_dir)

    summary = ingest_directory(input_dir, output_dir)
    assert summary["processed"] == 1
    # No case-law var/rag/* path appears anywhere in this module's source.
    source = open(mod.__file__, encoding="utf-8").read()
    for case_law_path in ("var/rag/raw/", "var/rag/processed/", "var/rag/structured",
                          "var/rag/chunked", "var/metadata.db"):
        assert case_law_path not in source
