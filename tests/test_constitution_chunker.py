"""Tests for the Constitution parent-child chunking stage
(src.rag_prep.constitution_chunker).

Synthetic fixtures mirror the real structural patterns this module was
validated against (the real Constitution of Pakistan document in
var/rag/constitution_parsed/ -- 555 chunks, 328 Articles represented,
0 validation errors), but every fixture here is hand-constructed, built
directly from constitution_parser.py's own Node so this file never
needs the real parsed JSON.
"""

from __future__ import annotations

import json

import pytest

from src.rag_prep.constitution_chunker import (
    ARTICLE_TOKEN_THRESHOLD,
    CLAUSE_TOKEN_THRESHOLD,
    Chunk,
    ConstitutionChunkingError,
    build_chunks_for_document,
    chunk_directory,
    chunk_document,
    count_tokens,
    is_valid_existing_output,
    validate_chunks,
)
from src.rag_prep.constitution_parser import parse_constitution_structure


def _parsed_doc(cleaned_text: str, doc_id: str = "doc1") -> dict:
    root = parse_constitution_structure(cleaned_text)
    from src.rag_prep.constitution_parser import validate_constitution_structure

    result = validate_constitution_structure(root, cleaned_text)
    return {
        "doc_id": doc_id, "document_type": "constitution", "title": "Constitution of Pakistan",
        "cleaned_text": cleaned_text, "structure": root.to_dict(), "validation": result.to_dict(),
    }


def _words(n: int, prefix: str = "word") -> str:
    return " ".join(f"{prefix}{i}" for i in range(n))


_SMALL_ARTICLE_TEXT = (
    "PREAMBLE\np\n\nPART I\nIntroductory\n"
    "## 1. The Republic and its territories.\n"
    "1. (1) Pakistan shall be Federal Republic to be known as Pakistan.\n"
    "(2) The territories of Pakistan shall comprise the Provinces."
)


def _large_article_text(n_filler_clauses: int = 3) -> str:
    # ~650 tokens overall: enough to cross ARTICLE_TOKEN_THRESHOLD, with
    # one deliberately oversized clause (clause 2) split-worthy via its
    # own sub-clauses, and ordinary small clauses alongside it.
    big_subclause_text = _words(120)
    clauses = [f"(1) {_words(40)}"]
    clauses.append(
        f"(2) {_words(30)}\n(a) {big_subclause_text}\n(b) {big_subclause_text}\n"
        f"Provided that {_words(30)}"
    )
    for i in range(n_filler_clauses):
        clauses.append(f"({i + 3}) {_words(20)}")
    return (
        "PREAMBLE\np\n\nPART VII\nThe Judicature\nCHAPTER 1.--THE COURTS\n"
        "## 175A. Appointment of Judges.\n175A. " + "\n".join(clauses)
    )


# -- 1. Article <=600 tokens -> one Parent == Child ---------------------------------

def test_small_article_produces_exactly_one_parent_equals_child_chunk():
    doc = _parsed_doc(_SMALL_ARTICLE_TEXT)
    chunks = build_chunks_for_document(doc)
    article_chunks = [c for c in chunks if c.article_number == "1"]
    assert len(article_chunks) == 1
    c = article_chunks[0]
    assert c.parent_id == c.child_id == "art_1"
    assert c.is_parent is True
    assert c.token_count <= ARTICLE_TOKEN_THRESHOLD


# -- 2. Article >600 tokens -> one parent + clause children -------------------------

def test_large_article_produces_one_parent_plus_clause_children():
    text = _large_article_text()
    doc = _parsed_doc(text)
    chunks = build_chunks_for_document(doc)
    group = [c for c in chunks if c.article_number == "175A"]
    parents = [c for c in group if c.is_parent]
    assert len(parents) == 1
    parent = parents[0]
    assert parent.parent_id == "art_175a_parent"
    assert parent.child_id is None
    assert parent.token_count > ARTICLE_TOKEN_THRESHOLD
    clause_1 = next(c for c in group if c.child_id == "art_175a_c1")
    assert clause_1.parent_id == "art_175a_parent"
    assert clause_1.clause_number == "1"


# -- 3/4. oversized clause splitting / preserved intact if no boundary --------------

def test_oversized_clause_split_by_subclause_and_proviso():
    text = _large_article_text()
    doc = _parsed_doc(text)
    chunks = build_chunks_for_document(doc)
    group = [c for c in chunks if c.article_number == "175A"]
    sub_a = next(c for c in group if c.child_id == "art_175a_c2_a")
    sub_b = next(c for c in group if c.child_id == "art_175a_c2_b")
    proviso = next(c for c in group if c.child_id == "art_175a_c2_proviso")
    assert sub_a.clause_number == "2" and sub_b.clause_number == "2" and proviso.clause_number == "2"
    assert count_tokens(sub_a.canonical_text) <= CLAUSE_TOKEN_THRESHOLD


def test_oversized_clause_without_internal_boundary_preserved_whole():
    big_flat_clause = f"(1) {_words(500)}"
    text = "PREAMBLE\np\n\nPART I\nIntro\n9. " + big_flat_clause
    doc = _parsed_doc(text)
    chunks = build_chunks_for_document(doc)
    group = [c for c in chunks if c.article_number == "9"]
    clause_chunks = [c for c in group if c.clause_number == "1"]
    assert len(clause_chunks) == 1
    assert count_tokens(clause_chunks[0].canonical_text) > CLAUSE_TOKEN_THRESHOLD


# -- 5/6/7. multiple clauses / sub-clauses / proviso ---------------------------------

def test_article_with_multiple_clauses_each_become_their_own_child():
    text = _large_article_text(n_filler_clauses=4)
    doc = _parsed_doc(text)
    chunks = build_chunks_for_document(doc)
    group = [c for c in chunks if c.article_number == "175A"]
    clause_numbers = sorted({c.clause_number for c in group if c.clause_number})
    assert {"1", "2", "3", "4", "5", "6"}.issubset(set(clause_numbers))


def test_subclauses_preserved_with_parent_clause_number():
    text = _large_article_text()
    doc = _parsed_doc(text)
    chunks = build_chunks_for_document(doc)
    sub_a = next(c for c in chunks if c.child_id == "art_175a_c2_a")
    assert "Provided that" not in sub_a.canonical_text


def test_proviso_preserved_as_its_own_child():
    text = _large_article_text()
    doc = _parsed_doc(text)
    chunks = build_chunks_for_document(doc)
    proviso = next(c for c in chunks if c.child_id == "art_175a_c2_proviso")
    assert proviso.canonical_text.startswith("Provided that")


# -- 8/9. Article 2A / Article 175A --------------------------------------------------

def test_article_2a_chunked_with_lowercase_id():
    text = "PREAMBLE\np\n\nPART I\nIntro\n2. Islam.\n\n2A. The Objectives Resolution text here."
    doc = _parsed_doc(text)
    chunks = build_chunks_for_document(doc)
    c = next(c for c in chunks if c.article_number == "2A")
    assert c.parent_id == "art_2a"


def test_article_175a_real_shape_chunked_correctly():
    text = _large_article_text()
    doc = _parsed_doc(text)
    chunks = build_chunks_for_document(doc)
    assert any(c.article_number == "175A" and c.is_parent for c in chunks)


# -- 10. Schedule item chunking -------------------------------------------------------

def test_schedule_chunked_as_single_parent_equals_child_unit():
    text = _SMALL_ARTICLE_TEXT + "\n\nFIRST SCHEDULE\n[Article 8(3)]\nLaws exempted from Article 8."
    doc = _parsed_doc(text)
    chunks = build_chunks_for_document(doc)
    schedule_chunks = [c for c in chunks if c.parent_id == "schedule_first_schedule"]
    assert len(schedule_chunks) == 1
    assert schedule_chunks[0].child_id == "schedule_first_schedule"
    assert schedule_chunks[0].article_number is None


def test_no_schedule_item_boundary_invented():
    text = _SMALL_ARTICLE_TEXT + "\n\nFIRST SCHEDULE\n[Article 8(3)]\n1. First law.\n2. Second law."
    doc = _parsed_doc(text)
    chunks = build_chunks_for_document(doc)
    schedule_chunks = [c for c in chunks if "schedule" in c.parent_id]
    # The parser identifies no item-level boundary inside a Schedule --
    # this stage must not invent one either.
    assert len(schedule_chunks) == 1
    assert "First law." in schedule_chunks[0].canonical_text
    assert "Second law." in schedule_chunks[0].canonical_text


# -- 11. metadata correctness ---------------------------------------------------------

def test_metadata_fields_populated_correctly_for_a_clause_child():
    text = _large_article_text()
    doc = _parsed_doc(text)
    chunks = build_chunks_for_document(doc)
    c = next(c for c in chunks if c.child_id == "art_175a_c1")
    assert c.doc_id == "doc1"
    assert c.part_number == "VII"
    assert c.part_title == "The Judicature"
    assert c.chapter_number == "1"
    assert c.chapter_title == "THE COURTS"
    assert c.article_number == "175A"
    assert c.clause_number == "1"
    assert c.is_parent is False


def test_metadata_uses_null_for_fields_that_do_not_apply():
    text = _SMALL_ARTICLE_TEXT + "\n\nFIRST SCHEDULE\n[Article 8(3)]\nLaws exempted."
    doc = _parsed_doc(text)
    chunks = build_chunks_for_document(doc)
    schedule_chunk = next(c for c in chunks if c.parent_id == "schedule_first_schedule")
    assert schedule_chunk.article_number is None
    assert schedule_chunk.part_number is None
    assert schedule_chunk.clause_number is None


# -- 12. breadcrumb generation ---------------------------------------------------------

def test_breadcrumb_contains_full_structural_path():
    text = _large_article_text()
    doc = _parsed_doc(text)
    chunks = build_chunks_for_document(doc)
    c = next(c for c in chunks if c.child_id == "art_175a_c1")
    assert c.embedding_text.startswith(
        "[Constitution of Pakistan | Part VII: The Judicature | Chapter 1: THE COURTS | "
        "Article 175A: Appointment of Judges. | Clause (1)]"
    )


def test_breadcrumb_strips_footnote_sentinel_from_title_but_not_metadata():
    text = (
        "PREAMBLE\np\n\nPART I\nIntro\n"
        "## 1. The Republic and its territories.\n"
        "[^5.1]1. (1) Pakistan shall be Federal Republic to be known as Pakistan."
    )
    doc = _parsed_doc(text)
    chunks = build_chunks_for_document(doc)
    c = next(c for c in chunks if c.article_number == "1")
    assert "[^5.1]" not in c.embedding_text.split("\n\n", 1)[0]


def test_strip_footnote_refs_removes_only_the_sentinel_marker():
    # Regression test for the ImportError that broke this module's
    # import of src.rag_prep.constitution_cleaning.strip_footnote_refs.
    # Exercises the function directly (not just through the breadcrumb
    # integration above), against the real Article 175A heading shape,
    # which has a footnote sentinel immediately followed by a genuine
    # amendment bracket -- confirming only the "[^id]" wrapper is
    # dropped and the substantive bracketed legal text survives.
    from src.rag_prep.constitution_cleaning import strip_footnote_refs

    title = (
        "Appointment of Judges to [^91.8][the Federal Constitutional Court,] "
        "the Supreme Court, High Courts and the Federal Shariat Court."
    )
    cleaned = strip_footnote_refs(title)
    assert "[^91.8]" not in cleaned
    assert "[the Federal Constitutional Court,]" in cleaned
    assert "Appointment of Judges to [the Federal Constitutional Court,] the Supreme Court" in cleaned


def test_strip_footnote_refs_is_a_noop_without_any_sentinel():
    from src.rag_prep.constitution_cleaning import strip_footnote_refs

    text = "The Republic and its territories."
    assert strip_footnote_refs(text) == text


def test_strip_footnote_refs_handles_multiple_markers_in_one_string():
    from src.rag_prep.constitution_cleaning import strip_footnote_refs

    text = "[^5.3]Balochistan, the [^5.4]Khyber Pakhtunkhwa, the Punjab and [^5.5]Sindh"
    cleaned = strip_footnote_refs(text)
    assert "[^5.3]" not in cleaned and "[^5.4]" not in cleaned and "[^5.5]" not in cleaned
    assert cleaned == "Balochistan, the Khyber Pakhtunkhwa, the Punjab and Sindh"


# -- 13. canonical_text vs embedding_text ----------------------------------------------

def test_canonical_text_never_contains_the_breadcrumb():
    text = _large_article_text()
    doc = _parsed_doc(text)
    chunks = build_chunks_for_document(doc)
    for c in chunks:
        assert not c.canonical_text.startswith("[Constitution of Pakistan")


def test_embedding_text_is_breadcrumb_plus_unaltered_canonical_text():
    text = _large_article_text()
    doc = _parsed_doc(text)
    chunks = build_chunks_for_document(doc)
    for c in chunks:
        assert c.embedding_text.endswith(c.canonical_text)


# -- 14/15. source-offset preservation / no text loss ----------------------------------

def test_canonical_text_matches_cleaned_text_offsets_for_every_chunk():
    text = _large_article_text()
    doc = _parsed_doc(text)
    chunks = build_chunks_for_document(doc)
    for c in chunks:
        assert doc["cleaned_text"][c.source_start:c.source_end] == c.canonical_text


def test_large_article_children_fully_reconstruct_the_article_no_loss():
    text = _large_article_text()
    doc = _parsed_doc(text)
    chunks = build_chunks_for_document(doc)
    group = [c for c in chunks if c.article_number == "175A"]
    parent = next(c for c in group if c.is_parent)
    children = sorted((c for c in group if not c.is_parent), key=lambda c: c.source_start)
    cursor = parent.source_start
    for child in children:
        assert child.source_start == cursor
        cursor = child.source_end
    assert cursor == parent.source_end


# -- 16/17. deterministic IDs / deterministic ordering ----------------------------------

def test_chunking_is_deterministic_across_repeated_runs():
    doc = _parsed_doc(_large_article_text())
    first = [c.to_dict() for c in build_chunks_for_document(doc)]
    second = [c.to_dict() for c in build_chunks_for_document(doc)]
    assert first == second


def test_chunk_ids_never_use_uuid_or_random_values():
    doc = _parsed_doc(_large_article_text())
    chunks = build_chunks_for_document(doc)
    for c in chunks:
        assert c.parent_id.startswith("art_") or c.parent_id.startswith("schedule_") or c.parent_id in ("preamble", "annex")


# -- 18. omitted/repealed provisions remain represented ----------------------------------

def test_omitted_article_remains_represented_in_a_chunk():
    text = (
        "PREAMBLE\np\n\nPART I\nIntro\n246. Something.\n\n"
        "247. [Administration of Tribal Areas.] Omitted by the Constitution "
        "(Twenty-fifth Amdt.) Act, 2018.\n\n248. Next."
    )
    doc = _parsed_doc(text)
    chunks = build_chunks_for_document(doc)
    c = next(c for c in chunks if c.article_number == "247")
    assert "Omitted by the Constitution" in c.canonical_text


# -- 19. amendment/footnote info already in parsed input is preserved -------------------

def test_amendment_bracket_text_preserved_verbatim_in_canonical_text():
    text = "PREAMBLE\np\n\nPART I\nIntro\n3. [Omitted] by some Amendment Act.\n\n4. Next article."
    doc = _parsed_doc(text)
    chunks = build_chunks_for_document(doc)
    c = next(c for c in chunks if c.article_number == "3")
    assert "[Omitted]" in c.canonical_text


# -- 20. malformed/incomplete AST fails validation rather than silently dropping text ---

def test_chunking_refuses_an_ast_whose_own_validation_failed():
    doc = _parsed_doc(_SMALL_ARTICLE_TEXT)
    doc["validation"]["ok"] = False
    doc["validation"]["errors"] = ["synthetic failure for this test"]
    with pytest.raises(ConstitutionChunkingError):
        build_chunks_for_document(doc)


def test_chunking_refuses_an_ast_with_no_content_at_all():
    doc = {
        "doc_id": "empty", "cleaned_text": "",
        "structure": {"type": "root", "id": "root", "source_start": 0, "source_end": 0, "text": "", "children": []},
        "validation": {"ok": True, "errors": [], "warnings": [], "diagnostics": {}},
    }
    with pytest.raises(ConstitutionChunkingError):
        build_chunks_for_document(doc)


# -- validator --------------------------------------------------------------------------

def test_validate_chunks_passes_for_well_formed_output():
    doc = _parsed_doc(_large_article_text())
    chunks = build_chunks_for_document(doc)
    result = validate_chunks(chunks, doc)
    assert result.ok, result.errors


def test_validate_chunks_detects_offset_mismatch():
    from dataclasses import replace

    doc = _parsed_doc(_SMALL_ARTICLE_TEXT)
    chunks = build_chunks_for_document(doc)
    bad = [replace(chunks[0], canonical_text="this no longer matches the source")] + chunks[1:]
    result = validate_chunks(bad, doc)
    assert not result.ok
    assert any("does not match" in e for e in result.errors)


def test_validate_chunks_detects_missing_article():
    doc = _parsed_doc(_large_article_text())
    chunks = [c for c in build_chunks_for_document(doc) if c.article_number != "175A"]
    result = validate_chunks(chunks, doc)
    assert not result.ok
    assert any("not represented" in e for e in result.errors)


# -- document-level / resumability -------------------------------------------------------

def test_chunk_document_end_to_end():
    doc = _parsed_doc(_large_article_text())
    result = chunk_document(doc)
    assert result["validation"]["ok"], result["validation"]["errors"]
    assert len(result["chunks"]) > 1


def test_is_valid_existing_output_true_for_genuinely_valid_output():
    doc = _parsed_doc(_SMALL_ARTICLE_TEXT)
    result = chunk_document(doc)
    assert is_valid_existing_output(result) is True


def test_is_valid_existing_output_false_for_corrupted_output():
    doc = _parsed_doc(_SMALL_ARTICLE_TEXT)
    result = chunk_document(doc)
    result["chunks"][0]["canonical_text"] = "corrupted beyond recognition"
    assert is_valid_existing_output(result) is False


def test_is_valid_existing_output_false_for_missing_fields():
    assert is_valid_existing_output({}) is False
    assert is_valid_existing_output({"chunks": []}) is False


def test_chunk_directory_resumability_skips_valid_regenerates_invalid(tmp_path):
    input_dir = tmp_path / "in"
    output_dir = tmp_path / "out"
    input_dir.mkdir()
    doc = _parsed_doc(_SMALL_ARTICLE_TEXT, doc_id="doc1")
    (input_dir / "doc1.json").write_text(json.dumps(doc), encoding="utf-8")

    first_summary = chunk_directory(input_dir, output_dir)
    assert first_summary["processed"] == 1
    assert first_summary["failed"] == 0

    out_path = output_dir / "doc1.json"
    corrupted = json.loads(out_path.read_text(encoding="utf-8"))
    corrupted["chunks"][0]["canonical_text"] = "corrupted"
    out_path.write_text(json.dumps(corrupted), encoding="utf-8")

    second_summary = chunk_directory(input_dir, output_dir)
    assert second_summary["skipped_valid"] == 0
    assert second_summary["regenerated"] == 1
    assert second_summary["failed"] == 0


def test_chunk_directory_does_not_modify_parsed_input(tmp_path):
    input_dir = tmp_path / "in"
    output_dir = tmp_path / "out"
    input_dir.mkdir()
    doc = _parsed_doc(_SMALL_ARTICLE_TEXT)
    raw_path = input_dir / "doc1.json"
    raw_path.write_text(json.dumps(doc), encoding="utf-8")
    original_bytes = raw_path.read_bytes()

    chunk_directory(input_dir, output_dir)

    assert raw_path.read_bytes() == original_bytes


# -- pipeline isolation --------------------------------------------------------------------

def test_constitution_chunker_does_not_import_statute_or_case_law_code():
    import ast

    import src.rag_prep.constitution_chunker as mod

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
        assert not name.startswith(forbidden_prefixes), f"constitution_chunker.py must not import {name!r}"
