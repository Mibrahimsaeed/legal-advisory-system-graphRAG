"""Tests for the thin reference_extractor.py / statute_auditor.py import
wrappers -- confirms they expose the real, already-validated
implementation (in statute_chunker.py / statute_chunk_validate.py)
under the requested module names, without duplicating or diverging from
it. The underlying logic itself is tested exhaustively in
tests/test_statute_chunker.py; this file only checks the wrapper layer.
"""

from __future__ import annotations

from src.rag_prep import reference_extractor, statute_auditor
from src.rag_prep.statute_chunker import build_statute_chunks, extract_references as chunker_extract_references


def _doc(cleaned_text: str, doc_id: str = "doc1") -> dict:
    return {
        "doc_id": doc_id,
        "metadata": {"law_name": "Sample Act", "year": "1970", "document_type": "statute",
                      "domain": "family_law", "domain_source": "curated", "source_file": "x.pdf"},
        "cleaned_text": cleaned_text,
    }


def test_reference_extractor_is_the_same_function_as_the_chunker_uses():
    assert reference_extractor.extract_references is chunker_extract_references


def test_reference_extractor_wrapper_produces_identical_results():
    text = "Section 9(1)(b) and the Guardians and Wards Act, 1890."
    assert reference_extractor.extract_references(text) == chunker_extract_references(text)


def test_statute_auditor_wrapper_validates_a_real_chunk_set():
    doc = _doc("1. Short title. X.\n\n2. Definitions. In this Act--\n(a) \"X\" means Y.")
    parent = build_statute_chunks(doc)
    result = statute_auditor.validate_statute_chunk_set(parent)
    assert result.ok, result.errors


def test_statute_auditor_wrapper_resumability_check():
    doc = _doc("1. Short title. X.")
    parent = build_statute_chunks(doc)
    result = statute_auditor.validate_statute_chunk_set(parent)
    serialized = {"parent": parent.to_dict(), "validation": result.to_dict()}
    assert statute_auditor.is_valid_existing_output(serialized) is True
    serialized["parent"]["chunks"][0]["text"] = "tampered"
    assert statute_auditor.is_valid_existing_output(serialized) is False
