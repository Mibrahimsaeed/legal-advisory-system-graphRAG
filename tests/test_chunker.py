"""Tests for the case-law parent-child chunking engine
(src.rag_prep.chunker / chunk_types / chunk_validate).

Synthetic fixtures only -- no real corpus documents are loaded anywhere
in this file. Tests emphasize PROPERTIES (exact preservation, coverage,
ordering, provenance, determinism) over any single example's exact chunk
count, per the task brief's explicit test-design principle.
"""

from __future__ import annotations

import pytest

from src.rag_prep.chunk_types import ChunkingError, ParentCase
from src.rag_prep.chunker import chunk_document, denormalize_chunk, paragraph_offsets
from src.rag_prep.chunk_validate import is_valid_existing_output, validate_chunk_set


def _words(n: int, tag: str) -> str:
    return " ".join(f"{tag}{i}" for i in range(n))


def _doc(paras: list[str], spans: list[dict], doc_id: str = "doc1", metadata: dict | None = None) -> dict:
    full_text = "\n\n".join(paras)
    return {
        "doc_id": doc_id,
        "metadata": metadata or {"case_title": "Test v. Case", "citation": "2024 X 1"},
        "full_text": full_text,
        "structure": {"spans": spans},
    }


def _span(kind: str, start: int, end: int, text: str, label: str | None = None) -> dict:
    d = {"type": kind, "paragraph_start": start, "paragraph_end": end, "text": text}
    if label is not None:
        d["label"] = label
    return d


def _assert_exact_offsets(parent: ParentCase) -> None:
    for c in parent.chunks:
        assert parent.full_text[c.source_start:c.source_end] == c.text


# -- basic grouping / size targets -------------------------------------------

def test_paragraphs_group_toward_target_range():
    paras = [_words(180, "a"), _words(160, "b"), _words(210, "c"), _words(170, "d")]
    spans = [_span("semantic_section", 0, 3, "\n\n".join(paras), label="facts")]
    doc = _doc(paras, spans)
    parent = chunk_document(doc)

    assert len(parent.chunks) == 1
    assert parent.chunks[0].word_count == 180 + 160 + 210 + 170 == 720
    _assert_exact_offsets(parent)
    assert validate_chunk_set(parent, spans).ok


def test_adding_a_paragraph_that_would_exceed_soft_max_starts_a_new_chunk():
    # A+B+C+D = 720 (fine); E would push to 1150 > 1000 soft max -> new chunk.
    paras = [_words(180, "a"), _words(160, "b"), _words(210, "c"), _words(170, "d"), _words(430, "e")]
    spans = [_span("semantic_section", 0, 4, "\n\n".join(paras), label="court_reasoning")]
    doc = _doc(paras, spans)
    parent = chunk_document(doc)

    assert len(parent.chunks) == 2
    assert parent.chunks[0].paragraph_end == 3
    assert parent.chunks[1].paragraph_start == 4
    assert parent.chunks[0].word_count == 720
    _assert_exact_offsets(parent)
    assert validate_chunk_set(parent, spans).ok


def test_chunk_never_unnecessarily_exceeds_soft_maximum():
    paras = [_words(300, f"p{i}") for i in range(6)]  # 1800 words total
    spans = [_span("semantic_section", 0, 5, "\n\n".join(paras), label="arguments")]
    doc = _doc(paras, spans)
    parent = chunk_document(doc)

    for c in parent.chunks:
        # A chunk may only exceed SOFT_MAX if it is a single oversized
        # paragraph that cannot be split further.
        if c.word_count > 1000:
            assert c.paragraph_start == c.paragraph_end
            assert c.oversized_single_paragraph
    assert validate_chunk_set(parent, spans).ok


# -- paragraph atomicity ------------------------------------------------------

def test_a_normal_paragraph_is_never_split():
    paras = [_words(900, "x"), _words(50, "y")]
    spans = [_span("semantic_section", 0, 1, "\n\n".join(paras), label="findings")]
    doc = _doc(paras, spans)
    parent = chunk_document(doc)

    # Paragraph 0 (900 words) must appear whole in exactly one chunk.
    owning = [c for c in parent.chunks if c.paragraph_start <= 0 <= c.paragraph_end]
    assert len(owning) == 1
    assert _words(900, "x") in owning[0].text
    _assert_exact_offsets(parent)


# -- small sections stay coherent, not padded ---------------------------------

def test_small_section_is_not_artificially_inflated():
    paras = ["Petition accepted."]
    spans = [_span("final_order", 0, 0, paras[0], label="final_order")]
    doc = _doc(paras, spans)
    parent = chunk_document(doc)

    assert len(parent.chunks) == 1
    assert parent.chunks[0].text == "Petition accepted."
    assert parent.chunks[0].word_count == 2


def test_small_issues_section_not_merged_with_unrelated_reasoning():
    issues_text = "Whether the Family Court had jurisdiction to entertain the suit."
    reasoning = _words(600, "r")
    paras = [issues_text, reasoning]
    spans = [
        _span("semantic_section", 0, 0, issues_text, label="issues"),
        _span("semantic_section", 1, 1, reasoning, label="court_reasoning"),
    ]
    doc = _doc(paras, spans)
    parent = chunk_document(doc)

    sections = {c.section for c in parent.chunks}
    assert sections == {"issues", "court_reasoning"}
    issues_chunk = next(c for c in parent.chunks if c.section == "issues")
    assert issues_chunk.text == issues_text
    assert validate_chunk_set(parent, spans).ok


# -- semantic section boundaries are hard chunk boundaries ---------------------

def test_chunks_do_not_cross_semantic_section_boundaries():
    facts = _words(100, "f")
    issues = _words(100, "i")
    paras = [facts, issues]
    spans = [
        _span("semantic_section", 0, 0, facts, label="facts"),
        _span("semantic_section", 1, 1, issues, label="issues"),
    ]
    doc = _doc(paras, spans)
    parent = chunk_document(doc)

    assert len(parent.chunks) == 2
    assert parent.chunks[0].section == "facts"
    assert parent.chunks[1].section == "issues"
    # Neither chunk's paragraph range reaches into the other section.
    assert parent.chunks[0].paragraph_end < parent.chunks[1].paragraph_start


# -- long special sections are split, not blindly kept whole -------------------

def test_long_final_order_span_is_split_into_multiple_chunks():
    paras = [_words(400, f"o{i}") for i in range(6)]  # 2400 words
    spans = [_span("final_order", 0, 5, "\n\n".join(paras), label="final_order")]
    doc = _doc(paras, spans)
    parent = chunk_document(doc)

    assert len(parent.chunks) > 1
    assert all(c.section == "final_order" for c in parent.chunks)
    assert validate_chunk_set(parent, spans).ok


def test_short_quoted_material_stays_one_chunk():
    text = "Quoted order: the application is allowed."
    spans = [_span("quoted_material", 0, 0, text, label="quoted_material")]
    doc = _doc([text], spans)
    parent = chunk_document(doc)
    assert len(parent.chunks) == 1
    assert parent.chunks[0].section == "quoted_material"


# -- exact offsets / repeated text --------------------------------------------

def test_offsets_exactly_match_full_text_slice():
    paras = [_words(50, "a"), _words(700, "b"), _words(30, "c")]
    spans = [_span("semantic_section", 0, 2, "\n\n".join(paras), label="evidence")]
    doc = _doc(paras, spans)
    parent = chunk_document(doc)
    _assert_exact_offsets(parent)


def test_repeated_paragraph_text_gets_distinct_offsets_not_found_via_search():
    repeated = "The same text appears twice in this document."
    unique = "middle paragraph unique content here."
    paras = [repeated, unique, repeated]
    spans = [
        _span("semantic_section", 0, 0, repeated, label="facts"),
        _span("semantic_section", 1, 1, unique, label="issues"),
        _span("semantic_section", 2, 2, repeated, label="findings"),
    ]
    doc = _doc(paras, spans)
    parent = chunk_document(doc)

    first = next(c for c in parent.chunks if c.section == "facts")
    last = next(c for c in parent.chunks if c.section == "findings")
    assert first.text == last.text == repeated
    # If offsets were found via naive str.find(), both would incorrectly
    # resolve to the FIRST occurrence's position.
    assert first.source_start != last.source_start
    assert doc["full_text"][first.source_start:first.source_end] == repeated
    assert doc["full_text"][last.source_start:last.source_end] == repeated
    assert validate_chunk_set(parent, spans).ok


# -- ordering -------------------------------------------------------------------

def test_chunks_preserve_source_order():
    paras = [_words(60, f"p{i}") for i in range(10)]
    spans = [
        _span("semantic_section", 0, 4, "\n\n".join(paras[0:5]), label="facts"),
        _span("semantic_section", 5, 9, "\n\n".join(paras[5:10]), label="court_reasoning"),
    ]
    doc = _doc(paras, spans)
    parent = chunk_document(doc)

    starts = [c.source_start for c in parent.chunks]
    assert starts == sorted(starts)
    assert all(a < b for a, b in zip(starts, starts[1:]))


# -- deterministic IDs / prev-next ---------------------------------------------

def test_chunk_ids_and_ordering_are_deterministic_across_runs():
    paras = [_words(300, f"p{i}") for i in range(5)]
    spans = [_span("semantic_section", 0, 4, "\n\n".join(paras), label="arguments")]
    doc = _doc(paras, spans)

    first_run = chunk_document(doc)
    second_run = chunk_document(doc)

    assert [c.chunk_id for c in first_run.chunks] == [c.chunk_id for c in second_run.chunks]
    assert [c.source_start for c in first_run.chunks] == [c.source_start for c in second_run.chunks]


def test_prev_next_links_are_correct_and_never_cross_documents():
    paras = [_words(300, f"p{i}") for i in range(5)]
    spans = [_span("semantic_section", 0, 4, "\n\n".join(paras), label="arguments")]
    parent = chunk_document(_doc(paras, spans, doc_id="docA"))

    assert parent.chunks[0].prev_chunk_id is None
    assert parent.chunks[-1].next_chunk_id is None
    for i, c in enumerate(parent.chunks):
        if i > 0:
            assert c.prev_chunk_id == parent.chunks[i - 1].chunk_id
        if i < len(parent.chunks) - 1:
            assert c.next_chunk_id == parent.chunks[i + 1].chunk_id
        assert c.chunk_id.startswith("docA:")


# -- metadata inheritance -------------------------------------------------------

def test_denormalize_chunk_inherits_purposeful_metadata_only():
    metadata = {
        "case_title": "A v. B", "citation": "2024 X 1", "court": "Lahore",
        "court_location": "Lahore", "decision_date": "2024-01-01", "judges": ["J1"],
        "primary_domain": "family_law", "classification_status": "auto_accepted",
        "disposition": "dismissed", "statutes_cited": ["Act I"], "provisions_cited": ["S.5"],
        "source_relpath": "x", "content_hash": "y",
    }
    paras = ["Petition dismissed."]
    spans = [_span("final_order", 0, 0, paras[0], label="final_order")]
    doc = _doc(paras, spans, metadata=metadata)
    parent = chunk_document(doc)

    flat = denormalize_chunk(parent.chunks[0], parent)
    assert flat["case_title"] == "A v. B"
    assert flat["disposition"] == "dismissed"
    assert flat["provisions_cited"] == ["S.5"]
    # Internal fields (classification_status/source_relpath/content_hash)
    # are deliberately NOT inherited -- "purposeful", not a blanket copy.
    assert "classification_status" not in flat
    assert "source_relpath" not in flat
    # Primary ParentCase/ChildChunk representation does not duplicate
    # metadata onto the chunk itself.
    assert not hasattr(parent.chunks[0], "case_title")


# -- conditional overlap --------------------------------------------------------

def test_overlap_applied_for_long_eligible_narrative_section():
    paras = [_words(50, c) for c in "abcdefghijklmnopqrstuv"]  # 22 x 50 = 1100 words
    spans = [_span("semantic_section", 0, len(paras) - 1, "\n\n".join(paras), label="facts")]
    doc = _doc(paras, spans)
    parent = chunk_document(doc)

    assert len(parent.chunks) >= 2
    assert any(c.is_overlap for c in parent.chunks[1:])
    overlap_chunk = next(c for c in parent.chunks if c.is_overlap)
    assert 0 < overlap_chunk.overlap_paragraph_count < len(paras)
    assert validate_chunk_set(parent, spans).ok


def test_no_overlap_for_short_section():
    text = "Petition accepted."
    spans = [_span("final_order", 0, 0, text, label="final_order")]
    doc = _doc([text], spans)
    parent = chunk_document(doc)
    assert all(not c.is_overlap for c in parent.chunks)


def test_no_overlap_for_ineligible_section_even_when_long():
    # 'arguments' is not in OVERLAP_ELIGIBLE_SECTIONS.
    paras = [_words(50, c) for c in "abcdefghijklmnopqrstuv"]
    spans = [_span("semantic_section", 0, len(paras) - 1, "\n\n".join(paras), label="arguments")]
    doc = _doc(paras, spans)
    parent = chunk_document(doc)
    assert len(parent.chunks) >= 2
    assert all(not c.is_overlap for c in parent.chunks)


def test_no_overlap_across_different_semantic_sections():
    facts = [_words(50, c) for c in "abcde"]
    reasoning = [_words(50, c) for c in "fghij"]
    paras = facts + reasoning
    spans = [
        _span("semantic_section", 0, 4, "\n\n".join(facts), label="facts"),
        _span("semantic_section", 5, 9, "\n\n".join(reasoning), label="court_reasoning"),
    ]
    doc = _doc(paras, spans)
    parent = chunk_document(doc)
    # Each section alone is short enough to be one chunk -- no overlap
    # possible, and certainly none crossing the section boundary.
    assert all(not c.is_overlap for c in parent.chunks)
    boundary_chunk = next(c for c in parent.chunks if c.section == "court_reasoning")
    assert boundary_chunk.paragraph_start == 5  # never reaches back into 'facts'


# -- large paragraph behavior ----------------------------------------------------

def test_single_paragraph_exceeding_soft_max_is_preserved_intact_and_flagged():
    big = _words(1500, "z")
    paras = ["short intro.", big, "short closing."]
    spans = [_span("semantic_section", 0, 2, "\n\n".join(paras), label="court_reasoning")]
    doc = _doc(paras, spans)
    parent = chunk_document(doc)

    big_chunk = next(c for c in parent.chunks if c.paragraph_start == c.paragraph_end == 1)
    assert big_chunk.text == big
    assert big_chunk.oversized_single_paragraph is True
    assert big_chunk.word_count == 1500
    _assert_exact_offsets(parent)
    assert validate_chunk_set(parent, spans).ok


# -- context-only spans (case_caption / judgment_marker) -----------------------

def test_case_caption_and_judgment_marker_produce_no_chunks_but_coverage_still_passes():
    caption = "IN THE HIGH COURT ... PETITIONER vs RESPONDENT"
    marker = "JUDGMENT"
    body = "The petition is accordingly dismissed."
    paras = [caption, marker, body]
    spans = [
        _span("case_caption", 0, 0, caption),
        _span("judgment_marker", 1, 1, marker, label="JUDGMENT"),
        _span("final_order", 2, 2, body, label="final_order"),
    ]
    doc = _doc(paras, spans)
    parent = chunk_document(doc)

    assert len(parent.chunks) == 1
    assert parent.chunks[0].paragraph_start == 2
    # The full source text is still retained on the parent regardless.
    assert parent.full_text == "\n\n".join(paras)
    result = validate_chunk_set(parent, spans)
    assert result.ok, result.errors


# -- malformed input -> safe failure, never a guessed offset --------------------

def test_malformed_span_missing_field_raises_chunking_error():
    doc = _doc(["one paragraph."], [{"type": "semantic_section", "paragraph_start": 0}])
    with pytest.raises(ChunkingError):
        chunk_document(doc)


def test_span_with_out_of_range_paragraph_indices_raises_chunking_error():
    doc = _doc(["only one paragraph."], [_span("semantic_section", 0, 5, "x", label="facts")])
    with pytest.raises(ChunkingError):
        chunk_document(doc)


def test_span_with_inverted_range_raises_chunking_error():
    doc = _doc(["a.", "b."], [_span("semantic_section", 1, 0, "x", label="facts")])
    with pytest.raises(ChunkingError):
        chunk_document(doc)


def test_unrecognized_span_type_raises_chunking_error():
    doc = _doc(["some paragraph text."], [_span("unknown_span_type", 0, 0, "some paragraph text.")])
    with pytest.raises(ChunkingError, match="unknown_span_type"):
        chunk_document(doc)


# -- validator: coverage / duplication / bounds ----------------------------------

def test_validator_flags_missing_coverage():
    paras = ["para zero.", "para one."]
    spans = [_span("semantic_section", 0, 0, paras[0], label="facts")]  # paragraph 1 never covered
    doc = _doc(paras, spans)
    parent = chunk_document(doc)
    result = validate_chunk_set(parent, spans)
    assert not result.ok
    assert any("missing paragraph" in e for e in result.errors)


def test_validator_flags_unintended_duplication_not_marked_as_overlap():
    paras = ["alpha paragraph text.", "beta paragraph text."]
    spans = [_span("semantic_section", 0, 1, "\n\n".join(paras), label="facts")]
    doc = _doc(paras, spans)
    parent = chunk_document(doc)
    # Manually corrupt: duplicate the first chunk's range into a second
    # chunk WITHOUT marking it as intentional overlap.
    from dataclasses import replace
    bad_chunk = replace(
        parent.chunks[0], chunk_id="doc1:9999", chunk_index=99,
        prev_chunk_id=parent.chunks[-1].chunk_id, next_chunk_id=None,
    )
    parent.chunks[-1] = replace(parent.chunks[-1], next_chunk_id=bad_chunk.chunk_id)
    parent.chunks.append(bad_chunk)
    result = validate_chunk_set(parent, spans)
    assert not result.ok
    assert any("unintended duplicated" in e for e in result.errors)


def test_validator_flags_out_of_bounds_offsets():
    paras = ["only paragraph."]
    spans = [_span("semantic_section", 0, 0, paras[0], label="facts")]
    doc = _doc(paras, spans)
    parent = chunk_document(doc)
    from dataclasses import replace
    parent.chunks[0] = replace(parent.chunks[0], source_end=len(parent.full_text) + 50)
    result = validate_chunk_set(parent, spans)
    assert not result.ok
    assert any("out-of-bounds" in e for e in result.errors)


def test_validator_size_diagnostics_never_fail_validation_alone():
    # A deliberately tiny, otherwise-perfectly-valid chunk set.
    paras = ["tiny."]
    spans = [_span("semantic_section", 0, 0, paras[0], label="issues")]
    doc = _doc(paras, spans)
    parent = chunk_document(doc)
    result = validate_chunk_set(parent, spans)
    assert result.ok  # size alone (well under 250 words) must not fail validation
    assert "<250" in result.diagnostics["size_histogram"]


# -- resumability: corrupt/invalid existing output is never "valid" -------------

def test_is_valid_existing_output_true_for_a_genuinely_valid_chunk_set():
    paras = [_words(300, f"p{i}") for i in range(3)]
    spans = [_span("semantic_section", 0, 2, "\n\n".join(paras), label="facts")]
    doc = _doc(paras, spans)
    parent = chunk_document(doc)
    assert is_valid_existing_output(parent.to_dict()) is True


def test_is_valid_existing_output_false_for_empty_dict():
    assert is_valid_existing_output({}) is False


def test_is_valid_existing_output_false_for_tampered_text():
    paras = [_words(300, f"p{i}") for i in range(3)]
    spans = [_span("semantic_section", 0, 2, "\n\n".join(paras), label="facts")]
    doc = _doc(paras, spans)
    parent = chunk_document(doc)
    d = parent.to_dict()
    d["chunks"][0]["text"] = "this does not match full_text at all"
    assert is_valid_existing_output(d) is False


def test_is_valid_existing_output_false_for_missing_chunks_key():
    assert is_valid_existing_output({"doc_id": "x", "metadata": {}, "full_text": "a\n\nb"}) is False


# -- paragraph_offsets helper (used independently by the validator too) ----------

def test_paragraph_offsets_round_trip_matches_full_text():
    full_text = "para one.\n\npara two is longer here.\n\nshort."
    offsets = paragraph_offsets(full_text)
    paragraphs = full_text.split("\n\n")
    assert len(offsets) == len(paragraphs)
    for (start, end), para in zip(offsets, paragraphs):
        assert full_text[start:end] == para
