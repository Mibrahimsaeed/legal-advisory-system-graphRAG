"""Tests for the section-aware statute chunker
(src.rag_prep.statute_chunker / statute_chunk_validate).

Synthetic fixtures only -- no real cleaned statute JSON required.
"""

from __future__ import annotations

import json

import pytest

from src.rag_prep.statute_chunker import (
    StatuteChunkingError,
    StatuteParent,
    build_statute_chunks,
    chunk_directory,
    extract_references,
)
from src.rag_prep.statute_chunk_validate import is_valid_existing_output, validate_statute_chunk_set


def _doc(cleaned_text: str, doc_id: str = "doc1", **metadata_overrides) -> dict:
    metadata = {
        "law_name": "Sample Act", "year": "1970", "document_type": "statute",
        "domain": "family_law", "domain_source": "curated", "source_file": "x.pdf",
    }
    metadata.update(metadata_overrides)
    return {"doc_id": doc_id, "metadata": metadata, "cleaned_text": cleaned_text}


_SIMPLE_ACT = """THE SAMPLE FAMILY ACT, 1970

ACT No. XII OF 1970
[1st January, 1970]

Preamble. WHEREAS it is expedient to provide for family matters;
It is hereby enacted as follows:--

1. Short title and extent. (1) This Act may be called the Sample Family Act, 1970.
(2) It extends to the whole of Pakistan.

2. Definitions. In this Act, unless the context otherwise requires--
(a) "Court" means a Family Court;
(b) "minor" means a person under the age of eighteen years.

3. Jurisdiction. The Family Court shall have exclusive jurisdiction under Section 9(1)(b)
and in accordance with the Guardians and Wards Act, 1890."""


# -- preamble -----------------------------------------------------------------------

def test_preamble_is_one_initial_chunk():
    parent = build_statute_chunks(_doc(_SIMPLE_ACT))
    assert parent.chunks[0].chunk_type == "preamble"
    assert parent.chunks[0].chunk_index == 0
    assert "ACT No. XII OF 1970" in parent.chunks[0].text
    assert "WHEREAS it is expedient" in parent.chunks[0].text
    # Not duplicated into any other chunk.
    for c in parent.chunks[1:]:
        assert "WHEREAS it is expedient" not in c.text


def test_no_preamble_fabricated_when_none_present():
    text = "1. Short title. This Act may be called the Sample Act."
    parent = build_statute_chunks(_doc(text))
    assert parent.preamble_present is False
    assert parent.chunks[0].chunk_type != "preamble"


# -- one chunk per section -----------------------------------------------------------

def test_one_chunk_per_numbered_section():
    parent = build_statute_chunks(_doc(_SIMPLE_ACT))
    section_chunks = [c for c in parent.chunks if c.chunk_type in ("section", "definitions")]
    assert [c.section_id for c in section_chunks] == ["1", "2", "3"]


def test_alphanumeric_section_identifier_preserved():
    text = "1. Short title. This Act may be called X.\n\n17A. Special provision. This applies specially."
    parent = build_statute_chunks(_doc(text))
    ids = [c.section_id for c in parent.chunks if c.chunk_type == "section"]
    assert "17A" in ids


def test_hyphenated_alphanumeric_section_identifier_preserved():
    text = "1. Short title. X.\n\n25-A. Transfer of cases. Cases may be transferred."
    parent = build_statute_chunks(_doc(text))
    ids = [c.section_id for c in parent.chunks if c.chunk_type == "section"]
    assert "25-A" in ids


def test_section_never_split_by_word_count():
    long_body = " ".join(f"word{i}" for i in range(2000))
    text = f"1. Short title. X.\n\n2. Long section. {long_body}"
    parent = build_statute_chunks(_doc(text))
    section_2 = next(c for c in parent.chunks if c.section_id == "2")
    assert "word0" in section_2.text and "word1999" in section_2.text
    assert len([c for c in parent.chunks if c.chunk_type == "section"]) == 2


def test_subsections_clauses_provisos_stay_within_their_section():
    parent = build_statute_chunks(_doc(_SIMPLE_ACT))
    section_1 = next(c for c in parent.chunks if c.section_id == "1")
    assert "(1) This Act may be called" in section_1.text
    assert "(2) It extends to the whole of Pakistan." in section_1.text


def test_section_with_proviso_and_explanation_preserved():
    text = ('1. Short title. X.\n\n'
            '2. Grounds. A woman may obtain relief:\n'
            'Provided that the marriage has not been consummated.\n\n'
            'Explanation.-- Lian means a false accusation.')
    parent = build_statute_chunks(_doc(text))
    section_2 = next(c for c in parent.chunks if c.section_id == "2")
    assert "Provided that the marriage has not been consummated." in section_2.text
    assert "Explanation.-- Lian means a false accusation." in section_2.text


# -- definitions ----------------------------------------------------------------------

def test_definitions_section_is_single_dedicated_chunk():
    parent = build_statute_chunks(_doc(_SIMPLE_ACT))
    defs = [c for c in parent.chunks if c.chunk_type == "definitions"]
    assert len(defs) == 1
    assert defs[0].section_id == "2"
    assert '"Court" means a Family Court' in defs[0].text
    assert '"minor" means a person under the age of eighteen years' in defs[0].text


def test_definitions_not_assumed_at_section_2_when_titled_differently():
    text = ('1. Short title. X.\n\n'
            '2. Jurisdiction. The court shall have jurisdiction.\n\n'
            '3. Definitions. In this Act--\n(a) "X" means Y.')
    parent = build_statute_chunks(_doc(text))
    defs = [c for c in parent.chunks if c.chunk_type == "definitions"]
    assert len(defs) == 1
    assert defs[0].section_id == "3"


def test_section_mentioning_definitions_in_passing_is_not_misflagged():
    text = ('1. Short title. X.\n\n'
            '2. Power to amend definitions. The Government may amend any definition.')
    parent = build_statute_chunks(_doc(text))
    defs = [c for c in parent.chunks if c.chunk_type == "definitions"]
    assert len(defs) == 0


# -- schedules --------------------------------------------------------------------------

def test_multiple_schedules_with_parts_are_separate_chunks():
    text = ('1. Short title. X.\n\n'
            'SCHEDULE\n[see Section 5]\n1[Part I]\n1.\nDissolution of marriage.\n2.\nDower.\n'
            'PART II\nOffences under the Pakistan Penal Code, 1860.')
    parent = build_statute_chunks(_doc(text))
    schedules = [c for c in parent.chunks if c.chunk_type == "schedule"]
    assert [s.schedule_id for s in schedules] == ["Part I", "Part II"]
    assert "Dissolution of marriage." in schedules[0].text
    assert "Dower." in schedules[0].text
    assert "Offences under the Pakistan Penal Code, 1860." in schedules[1].text
    # Schedule-internal numbered items must NOT be treated as sections.
    assert all(s.section_id is None for s in schedules)
    section_ids = [c.section_id for c in parent.chunks if c.chunk_type == "section"]
    assert "1" not in section_ids or len(section_ids) == 1  # only the real Section 1


def test_schedule_without_parts_is_one_chunk():
    text = "1. Short title. X.\n\nSCHEDULE\n[ENACTMENTS REPEALED.] Rep. by the Repealing Act, 1938."
    parent = build_statute_chunks(_doc(text))
    schedules = [c for c in parent.chunks if c.chunk_type == "schedule"]
    assert len(schedules) == 1
    assert "ENACTMENTS REPEALED" in schedules[0].text


def test_schedule_content_not_merged_into_preceding_section():
    text = "1. Short title. X.\n\nSCHEDULE\nSome schedule text here."
    parent = build_statute_chunks(_doc(text))
    section_1 = next(c for c in parent.chunks if c.section_id == "1")
    assert "Some schedule text here." not in section_1.text


def test_inline_schedule_cross_reference_does_not_trigger_schedule_heading():
    # Lowercase/Title-case "Schedule" inside a sentence must never be
    # mistaken for the structural ALL-CAPS "SCHEDULE" heading.
    text = "1. Repeal. Rep. by the Code of Civil Procedure, 1908, s. 156 and Schedule V."
    parent = build_statute_chunks(_doc(text))
    assert not any(c.chunk_type == "schedule" for c in parent.chunks)


# -- chapter/part/source order ------------------------------------------------------------

def test_chunks_preserve_source_order():
    parent = build_statute_chunks(_doc(_SIMPLE_ACT))
    starts = [c.source_start for c in parent.chunks]
    assert starts == sorted(starts)
    assert all(a < b for a, b in zip(starts, starts[1:]))


# -- cross-reference extraction ------------------------------------------------------------

def test_internal_section_reference_with_subsection_and_clause():
    refs = extract_references("See Section 9(1)(b) for details.")
    assert refs["internal_sections"] == [
        {"section_id": "9", "subsection": "1", "clause": "b", "source_text": "Section 9(1)(b)"}
    ]


def test_internal_reference_list_with_and():
    refs = extract_references("As provided in Sections 5 and 7 of this Act.")
    ids = {r["section_id"] for r in refs["internal_sections"]}
    assert ids == {"5", "7"}
    assert all(r["source_text"] == "Sections 5 and 7" for r in refs["internal_sections"])


def test_internal_reference_alphanumeric_section():
    refs = extract_references("under Section 17A of the Act")
    assert refs["internal_sections"][0]["section_id"] == "17A"


def test_external_statute_reference_extracted():
    refs = extract_references("in accordance with the Guardians and Wards Act, 1890.")
    assert {"name": "the Guardians and Wards Act", "year": 1890,
            "source_text": "the Guardians and Wards Act, 1890"} in refs["external_statutes"] or \
        any(r["name"].endswith("Guardians and Wards Act") and r["year"] == 1890
            for r in refs["external_statutes"])


def test_external_statute_abbreviation_extracted():
    refs = extract_references("as per CPC 1908 procedure.")
    assert any(r["name"] == "Code of Civil Procedure" and r["year"] == 1908
               for r in refs["external_statutes"])


def test_false_references_not_extracted_from_dates_and_case_citations():
    text = "Decided on 15th July, 1961. See 2007 SCMR 49 for guidance. Subsection (1) applies."
    refs = extract_references(text)
    assert refs["internal_sections"] == []
    assert refs["external_statutes"] == []


def test_duplicate_references_within_a_chunk_are_deduplicated():
    text = "Section 5 applies. Later, Section 5 applies again in the same way."
    refs = extract_references(text)
    matching = [r for r in refs["internal_sections"] if r["section_id"] == "5"]
    assert len(matching) == 1


def test_references_attached_to_their_chunk():
    parent = build_statute_chunks(_doc(_SIMPLE_ACT))
    section_3 = next(c for c in parent.chunks if c.section_id == "3")
    assert any(r["section_id"] == "9" for r in section_3.references["internal_sections"])
    assert any("Guardians and Wards Act" in r["name"] for r in section_3.references["external_statutes"])


# -- exact source offsets / repeated text -------------------------------------------------

def test_exact_source_offsets_for_every_chunk():
    doc = _doc(_SIMPLE_ACT)
    parent = build_statute_chunks(doc)
    for c in parent.chunks:
        assert parent.cleaned_text[c.source_start:c.source_end] == c.text


def test_repeated_section_text_gets_distinct_offsets():
    text = ('1. Repeated. Identical body text appears here.\n\n'
            '2. Other. Something else.\n\n'
            '3. Repeated again. Identical body text appears here.')
    parent = build_statute_chunks(_doc(text))
    sections = [c for c in parent.chunks if c.chunk_type == "section"]
    assert sections[0].text != sections[2].text  # titles differ even if a phrase repeats
    assert sections[0].source_start != sections[2].source_start
    assert parent.cleaned_text[sections[0].source_start:sections[0].source_end] == sections[0].text
    assert parent.cleaned_text[sections[2].source_start:sections[2].source_end] == sections[2].text


# -- no gaps / overlaps / duplicated sections ----------------------------------------------

def test_no_gaps_no_overlaps_full_coverage():
    parent = build_statute_chunks(_doc(_SIMPLE_ACT))
    result = validate_statute_chunk_set(parent)
    assert result.ok, result.errors
    cursor = 0
    for c in sorted(parent.chunks, key=lambda c: c.chunk_index):
        assert c.source_start == cursor
        cursor = c.source_end
    assert cursor == len(parent.cleaned_text)


def test_validator_detects_duplicated_section_id():
    from dataclasses import replace
    parent = build_statute_chunks(_doc(_SIMPLE_ACT))
    bad = replace(parent.chunks[1], section_id=parent.chunks[2].section_id)
    parent.chunks[1] = bad
    result = validate_statute_chunk_set(parent)
    assert not result.ok
    assert any("more than one chunk" in e for e in result.errors)


# -- deterministic IDs ----------------------------------------------------------------------

def test_chunk_ids_deterministic_across_runs():
    doc = _doc(_SIMPLE_ACT)
    first = build_statute_chunks(doc)
    second = build_statute_chunks(doc)
    assert [c.chunk_id for c in first.chunks] == [c.chunk_id for c in second.chunks]


def test_chunk_indexes_sequential_from_zero():
    parent = build_statute_chunks(_doc(_SIMPLE_ACT))
    assert [c.chunk_index for c in parent.chunks] == list(range(len(parent.chunks)))


# -- malformed/ambiguous input fails clearly ------------------------------------------------

def test_no_section_header_at_all_raises_clear_error():
    with pytest.raises(StatuteChunkingError):
        build_statute_chunks(_doc("Just some prose with no section numbering whatsoever."))


def test_out_of_sequence_amendment_footnote_is_absorbed_not_treated_as_a_section():
    # A footnote numbered "2. ..." appearing AFTER section 3 has already
    # been confirmed is a number REGRESSION (2 < 3), the deterministic,
    # vocabulary-free signal that it's amendment-annotation noise
    # physically embedded in section 3's body, not a genuine second
    # "Section 2". It must be absorbed into section 3's text, not spawn
    # a bogus duplicate section_id -- validation passes cleanly.
    text = ("1. Short title. X.\n\n"
            "2. Real section. Y.\n\n"
            "3. Another. Z.\n\n"
            "2. The word \"District\" rep. by s. 4 of some amending Act.")
    parent = build_statute_chunks(_doc(text))
    result = validate_statute_chunk_set(parent)
    assert result.ok, result.errors
    section_ids = [c.section_id for c in parent.chunks if c.chunk_type == "section"]
    assert section_ids.count("2") == 1
    section_3 = next(c for c in parent.chunks if c.section_id == "3")
    assert 'The word "District" rep. by s. 4 of some amending Act.' in section_3.text


def test_genuine_repealed_section_in_correct_sequence_still_recognized():
    # "6. [Repealed]." in its correct forward position (5 -> 6) must
    # remain a real section -- the fix is purely about OUT-OF-SEQUENCE
    # position, never about repeal/amendment vocabulary.
    text = ("1. Short title. X.\n\n"
            "5. Rights to dower. Y.\n\n"
            "6. 1[Repealed].\n\n"
            "1Rep. by the Repealing and Amending Act, 1942 (XXV of 1942), s. 2 and 1st Sch.")
    parent = build_statute_chunks(_doc(text))
    result = validate_statute_chunk_set(parent)
    assert result.ok, result.errors
    section_ids = [c.section_id for c in parent.chunks if c.chunk_type == "section"]
    assert "6" in section_ids
    section_6 = next(c for c in parent.chunks if c.section_id == "6")
    assert "[Repealed]" in section_6.text


def test_normal_section_2_still_recognized_as_definitions():
    # Ordinary forward-sequence "2. Definitions." must be entirely
    # unaffected by the amendment-footnote-sequence fix.
    parent = build_statute_chunks(_doc(_SIMPLE_ACT))
    defs = next(c for c in parent.chunks if c.chunk_type == "definitions")
    assert defs.section_id == "2"


def test_real_affected_document_42769a60dc59c5473affbc32_now_passes():
    # The actual real document that exposed this ambiguity (sections
    # 1-47 in strict increasing order, then a footnote numbered "2."
    # embedded in section 47's body) now parses and validates cleanly,
    # with the footnote absorbed into section 47.
    from pathlib import Path

    path = Path("var/rag/cleaned_statutes/42769a60dc59c5473affbc32.json")
    if not path.exists():
        pytest.skip("real cleaned statute fixture not present in this environment")
    doc = json.loads(path.read_text(encoding="utf-8"))
    parent = build_statute_chunks(doc)
    result = validate_statute_chunk_set(parent)
    assert result.ok, result.errors
    section_ids = [c.section_id for c in parent.chunks if c.chunk_type == "section"]
    assert section_ids.count("2") == 1
    section_47 = next(c for c in parent.chunks if c.section_id == "47")
    assert "The word" in section_47.text and "District" in section_47.text and "rep. by s. 4" in section_47.text


# -- resumability ------------------------------------------------------------------------

def test_is_valid_existing_output_true_for_genuinely_valid_output():
    parent = build_statute_chunks(_doc(_SIMPLE_ACT))
    result = validate_statute_chunk_set(parent)
    serialized = {"parent": parent.to_dict(), "validation": result.to_dict()}
    assert is_valid_existing_output(serialized) is True


def test_is_valid_existing_output_false_for_corrupted_output():
    parent = build_statute_chunks(_doc(_SIMPLE_ACT))
    result = validate_statute_chunk_set(parent)
    serialized = {"parent": parent.to_dict(), "validation": result.to_dict()}
    serialized["parent"]["chunks"][0]["text"] = "this no longer matches the source at all"
    assert is_valid_existing_output(serialized) is False


def test_is_valid_existing_output_false_for_missing_fields():
    assert is_valid_existing_output({}) is False
    assert is_valid_existing_output({"parent": {}}) is False


def test_chunk_directory_resumability_skips_valid_regenerates_invalid(tmp_path):
    input_dir = tmp_path / "in"
    output_dir = tmp_path / "out"
    input_dir.mkdir()
    (input_dir / "doc1.json").write_text(json.dumps(_doc(_SIMPLE_ACT, doc_id="doc1")), encoding="utf-8")
    (input_dir / "doc2.json").write_text(
        json.dumps(_doc("1. Short title. X.\n\n2. Other. Y.", doc_id="doc2")), encoding="utf-8"
    )

    first_summary = chunk_directory(input_dir, output_dir)
    assert first_summary["processed"] == 2
    assert first_summary["failed"] == 0

    # Corrupt doc2's output on disk (simulating a previous interrupted/bad run).
    doc2_out = output_dir / "doc2.json"
    corrupted = json.loads(doc2_out.read_text(encoding="utf-8"))
    corrupted["parent"]["chunks"][0]["text"] = "corrupted"
    doc2_out.write_text(json.dumps(corrupted), encoding="utf-8")

    second_summary = chunk_directory(input_dir, output_dir)
    assert second_summary["skipped_valid"] == 1  # doc1 was untouched, still valid
    assert second_summary["regenerated"] == 1  # doc2 was invalid, regenerated
    assert second_summary["failed"] == 0

    doc2_fixed = json.loads(doc2_out.read_text(encoding="utf-8"))
    assert doc2_fixed["parent"]["chunks"][0]["text"] != "corrupted"


def test_chunk_directory_does_not_modify_cleaned_input(tmp_path):
    input_dir = tmp_path / "in"
    output_dir = tmp_path / "out"
    input_dir.mkdir()
    raw_path = input_dir / "doc1.json"
    raw_path.write_text(json.dumps(_doc(_SIMPLE_ACT)), encoding="utf-8")
    original_bytes = raw_path.read_bytes()

    chunk_directory(input_dir, output_dir)

    assert raw_path.read_bytes() == original_bytes


# -- case-law isolation ----------------------------------------------------------------------

def test_statute_chunker_does_not_import_case_law_chunking_or_cleaning_code():
    import ast

    import src.rag_prep.statute_chunker as mod

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
        "src.rag_prep.chunker", "src.rag_prep.chunk_types", "src.rag_prep.chunk_validate",
        "src.rag_prep.statute_cleaner",  # the existing case-law citation extractor
        "src.extraction.case_loader",
    )
    for name in imported:
        assert not name.startswith(forbidden_prefixes), f"statute_chunker.py must not import {name!r}"
