"""Tests for the Constitution structural parsing stage
(src.rag_prep.constitution_parser).

Synthetic fixtures mirror the real structural patterns this module was
validated against (the real Constitution of Pakistan document in
var/rag/cleaned_constitution/ -- 328/328 real Articles matched exactly,
0 missed, 0 false positives), but every fixture here is hand-
constructed -- no real cleaned-constitution JSON is required for these
tests.
"""

from __future__ import annotations

import json

import pytest

from src.rag_prep.constitution_parser import (
    ConstitutionParsingError,
    Node,
    is_valid_existing_output,
    parse_constitution_document,
    parse_constitution_structure,
    parse_directory,
    validate_constitution_structure,
)


def _find(node_dict: dict, **kwargs) -> dict | None:
    if all(node_dict.get(k) == v for k, v in kwargs.items()):
        return node_dict
    for c in node_dict.get("children", []):
        found = _find(c, **kwargs)
        if found:
            return found
    return None


_SIMPLE_CONSTITUTION = """PREAMBLE
WHEREAS sovereignty over the entire Universe belongs to Allah Almighty.

PART I
Introductory
1. (1) Pakistan shall be Federal Republic to be known as Pakistan.
(2) The territories of Pakistan shall comprise--
(a) the Provinces;
(b) the Federal Capital.

2. Islam shall be the State religion of Pakistan.

PART II
Fundamental Rights and Principles of Policy
7. In this Part, unless the context otherwise requires, the State means something.
CHAPTER 1.--FUNDAMENTAL RIGHTS

8. (1) Any law inconsistent with Fundamental Rights shall be void.
Provided that this clause shall not apply to any person employed by the Armed Forces.
Provided further that nothing in this Article shall affect any existing law.

FIRST SCHEDULE
[Article 8(3)]
Laws exempted from the operation of Article 8."""


def _doc(cleaned_text: str, **metadata_overrides) -> dict:
    metadata = {"document_type": "constitution", "title": "Constitution of Pakistan"}
    metadata.update(metadata_overrides)
    return {"doc_id": "doc1", "metadata": metadata, "cleaned_text": cleaned_text}


# -- Preamble detection -----------------------------------------------------------

def test_preamble_detected_as_first_top_level_node():
    root = parse_constitution_structure(_SIMPLE_CONSTITUTION)
    preamble = next(c for c in root.children if c.type == "preamble")
    assert "WHEREAS sovereignty over the entire Universe belongs" in preamble.text


# -- Part detection -----------------------------------------------------------------

def test_parts_detected_in_source_order():
    root = parse_constitution_structure(_SIMPLE_CONSTITUTION)
    parts = [c for c in root.children if c.type == "part"]
    assert [p.id for p in parts] == ["part_I", "part_II"]


# -- Chapter detection --------------------------------------------------------------

def test_chapter_detected_within_its_part():
    root = parse_constitution_structure(_SIMPLE_CONSTITUTION)
    part_ii = next(c for c in root.children if c.id == "part_II")
    chapter = next(c for c in part_ii.children if c.type == "chapter")
    assert chapter.id == "part_II_chapter1"
    assert chapter.heading == "FUNDAMENTAL RIGHTS"


def test_article_directly_under_part_before_first_chapter():
    # Confirmed real shape: Article 7 sits directly under PART II,
    # before CHAPTER 1 begins.
    root = parse_constitution_structure(_SIMPLE_CONSTITUTION)
    part_ii = next(c for c in root.children if c.id == "part_II")
    article_7 = next(c for c in part_ii.children if c.type == "article")
    assert article_7.article_number == "7"


# -- Article detection ----------------------------------------------------------------

def test_article_with_same_line_body_detected():
    root = parse_constitution_structure(_SIMPLE_CONSTITUTION)
    article_2 = _find(root.to_dict(), article_number="2")
    assert article_2 is not None
    assert "Islam shall be the State religion" in article_2["text"]


def test_article_2a_variant_detected():
    text = "PREAMBLE\np\n\nPART I\nIntro\n2. Islam.\n\n2A. The Objectives Resolution text here.\n\n3. Elimination."
    root = parse_constitution_structure(text)
    article_2a = _find(root.to_dict(), article_number="2A")
    assert article_2a is not None
    assert "Objectives Resolution" in article_2a["text"]


def test_article_175a_style_triple_digit_letter_variant_detected():
    text = ("PREAMBLE\np\n\nPART VII\nIntro\n"
            "174. Something.\n\n175A. Appointment of Judges to the Court.\n(1) There shall be a Commission.\n\n"
            "176. Something else.")
    root = parse_constitution_structure(text)
    article = _find(root.to_dict(), article_number="175A")
    assert article is not None
    assert "Appointment of Judges" in article["text"]


def test_bare_article_header_with_title_on_next_line_detected():
    # Confirmed real artifact: "<id>.\n(1) body..." with the title
    # pushed entirely onto the marginal-note line instead.
    text = "PREAMBLE\np\n\nPART I\nIntro\n## 116. Governor's assent to Bills.\n116.\n(1) When a Bill has been passed."
    root = parse_constitution_structure(text)
    article = _find(root.to_dict(), article_number="116")
    assert article is not None
    assert article["heading"] == "Governor's assent to Bills."
    assert "When a Bill has been passed." in article["text"]


def test_bare_year_at_end_of_citation_not_mistaken_for_an_article():
    # Confirmed real false-positive risk: a citation ending in a bare
    # year, followed by a CONTINUING clause "(2)" (not "(1)"), must
    # never be read as a new Article.
    text = ("PREAMBLE\np\n\nPART I\nIntro\n"
            "239. (1) A Bill to amend may originate in either House.\n"
            "That Bill was passed under the Constitution (Eighteenth Amendment) Act, 2010.\n"
            "(2) Before entering upon office, a Minister shall take oath.")
    root = parse_constitution_structure(text)
    numbers = []

    def _collect(n):
        if n.get("article_number"):
            numbers.append(n["article_number"])
        for c in n.get("children", []):
            _collect(c)

    _collect(root.to_dict())
    assert "2010" not in numbers
    assert numbers == ["239"]


def test_numeric_subclause_reference_not_mistaken_for_an_article():
    text = "PREAMBLE\np\n\nPART I\nIntro\n5. Something refers to clause (1) and (2) of Article 48."
    root = parse_constitution_structure(text)
    numbers = []

    def _collect(n):
        if n.get("article_number"):
            numbers.append(n["article_number"])
        for c in n.get("children", []):
            _collect(c)

    _collect(root.to_dict())
    assert numbers == ["5"]


# -- clauses / sub-clauses / provisos --------------------------------------------------

def test_numbered_clauses_detected_under_article():
    root = parse_constitution_structure(_SIMPLE_CONSTITUTION)
    article_1 = _find(root.to_dict(), article_number="1")
    clause_numbers = [c["number"] for c in article_1["children"] if c["type"] == "clause"]
    assert "2" in clause_numbers


def test_alphabetic_subclauses_detected_under_clause():
    root = parse_constitution_structure(_SIMPLE_CONSTITUTION)
    article_1 = _find(root.to_dict(), article_number="1")
    clause_2 = next(c for c in article_1["children"] if c.get("number") == "2")
    sub_letters = [s["number"] for s in clause_2["children"] if s["type"] == "subclause"]
    assert sub_letters == ["a", "b"]


def test_provisos_detected_as_children_of_their_clause():
    root = parse_constitution_structure(_SIMPLE_CONSTITUTION)
    article_8 = _find(root.to_dict(), article_number="8")
    clause_1 = next(c for c in article_8["children"] if c.get("number") == "1")
    provisos = [c for c in clause_1["children"] if c["type"] == "proviso"]
    assert len(provisos) == 2
    assert "shall not apply to any person employed" in provisos[0]["text"]
    assert "Provided further that" in provisos[1]["text"]


def test_ordinary_prose_containing_provided_is_not_misread_as_a_structural_boundary():
    text = ("PREAMBLE\np\n\nPART I\nIntro\n"
            "9. The arrangement provided that services continue is a policy matter, not a legal proviso.")
    root = parse_constitution_structure(text)
    article_9 = _find(root.to_dict(), article_number="9")
    provisos = [c for c in article_9["children"] if c["type"] == "proviso"]
    assert provisos == []


# -- schedules ------------------------------------------------------------------------

def test_schedule_is_a_separate_top_level_unit():
    root = parse_constitution_structure(_SIMPLE_CONSTITUTION)
    schedule = next(c for c in root.children if c.type == "schedule")
    assert schedule.id == "schedule_first_schedule"
    assert schedule.heading == "FIRST SCHEDULE"
    assert "Laws exempted from the operation of Article 8" in schedule.text


def test_schedule_internal_part_label_not_mistaken_for_a_top_level_part():
    # Confirmed real shape: a Schedule (e.g. the Fourth Schedule's
    # Legislative Lists) can use its OWN "PART I"/"PART II" labels.
    text = (_SIMPLE_CONSTITUTION + "\n\nPART I\nFederal Legislative List\n1. Defence.\n\n"
            "PART II\nConcurrent List\n1. Criminal law.")
    root = parse_constitution_structure(text)
    parts = [c for c in root.children if c.type == "part"]
    assert [p.id for p in parts] == ["part_I", "part_II"]


def test_schedule_not_assumed_when_absent():
    text = "PREAMBLE\np\n\nPART I\nIntro\n1. Something."
    root = parse_constitution_structure(text)
    assert not any(c.type == "schedule" for c in root.children)


# -- omitted/repealed Articles / asterisk preservation --------------------------------

def test_omitted_article_status_text_preserved_in_article_span():
    # Confirmed real shape: the omission note is on the SAME line as
    # the bare article number ("247. [Administration...] Omitted by
    # ...") -- not pushed to its own next line.
    text = ("PREAMBLE\np\n\nPART I\nIntro\n246. Something.\n\n"
            "247. [Administration of Tribal Areas.] Omitted by the Constitution "
            "(Twenty-fifth Amdt.) Act, 2018.\n\n248. Next.")
    root = parse_constitution_structure(text)
    article_247 = _find(root.to_dict(), article_number="247")
    assert article_247 is not None
    assert "Omitted by the Constitution" in article_247["text"]


def test_asterisk_omission_marker_preserved_verbatim():
    text = ("PREAMBLE\np\n\nPART I\nIntro\n"
            "51. (1) There shall be seats for members.\n"
            "(2) A person shall be entitled to vote if--\n"
            "4*          *           *           *           *\n"
            "(3) Nothing in this Article shall prevent anything.")
    root = parse_constitution_structure(text)
    article_51 = _find(root.to_dict(), article_number="51")
    assert "* " in article_51["text"] or "*" in article_51["text"]


# -- source-offset correctness ---------------------------------------------------------

def test_every_node_text_matches_cleaned_text_slice():
    root = parse_constitution_structure(_SIMPLE_CONSTITUTION)

    def _check(n: Node):
        assert _SIMPLE_CONSTITUTION[n.source_start:n.source_end] == n.text
        for c in n.children:
            _check(c)

    _check(root)


def test_validator_passes_for_well_formed_structure():
    root = parse_constitution_structure(_SIMPLE_CONSTITUTION)
    result = validate_constitution_structure(root, _SIMPLE_CONSTITUTION)
    assert result.ok, result.errors


def test_validator_detects_offset_mismatch():
    from dataclasses import replace

    root = parse_constitution_structure(_SIMPLE_CONSTITUTION)
    part_i = next(c for c in root.children if c.id == "part_I")
    bad_article = replace(part_i.children[1], text="this no longer matches the source")
    part_i.children[1] = bad_article
    result = validate_constitution_structure(root, _SIMPLE_CONSTITUTION)
    assert not result.ok
    assert any("does not match" in e for e in result.errors)


def test_validator_detects_gap_in_coverage():
    from dataclasses import replace

    root = parse_constitution_structure(_SIMPLE_CONSTITUTION)
    part_i = next(c for c in root.children if c.id == "part_I")
    shrunk = replace(part_i.children[1], source_end=part_i.children[1].source_end - 5)
    part_i.children[1] = shrunk
    result = validate_constitution_structure(root, _SIMPLE_CONSTITUTION)
    assert not result.ok
    assert any("do not fully cover" in e or "gap in coverage" in e for e in result.errors)


# -- nested hierarchy -------------------------------------------------------------------

def test_full_nested_hierarchy_part_chapter_article_clause_subclause():
    root = parse_constitution_structure(_SIMPLE_CONSTITUTION)
    part_i = next(c for c in root.children if c.id == "part_I")
    article_1 = next(c for c in part_i.children if c.article_number == "1")
    clause_2 = next(c for c in article_1.children if c.number == "2")
    subclause_a = next(c for c in clause_2.children if c.number == "a")
    assert subclause_a.type == "subclause"
    assert part_i.type == "part" and article_1.type == "article" and clause_2.type == "clause"


# -- unrecognized-text preservation ------------------------------------------------------

def test_unrecognized_text_retained_as_a_text_node_not_dropped():
    # No PREAMBLE anchor here, so the stray lead-in text before PART I
    # belongs to no other node's own span -- it must surface as its own
    # root-level "text" node, not disappear.
    text = "Some stray lead-in text with no recognizable structure at all.\n\nPART I\nIntro\n1. Something."
    root = parse_constitution_structure(text)
    assert "Some stray lead-in text" in root.to_dict()["text"]
    found = any(
        c["type"] == "text" and "Some stray lead-in text" in c["text"]
        for c in root.to_dict()["children"]
    )
    assert found


# -- deterministic output ----------------------------------------------------------------

def test_parsing_is_deterministic_across_repeated_runs():
    first = parse_constitution_structure(_SIMPLE_CONSTITUTION).to_dict()
    second = parse_constitution_structure(_SIMPLE_CONSTITUTION).to_dict()
    assert first == second


def test_document_parsing_is_deterministic_across_repeated_runs():
    doc = _doc(_SIMPLE_CONSTITUTION)
    first = parse_constitution_document(doc)
    second = parse_constitution_document(doc)
    assert first == second


# -- malformed/ambiguous input fails safely -----------------------------------------------

def test_no_structure_at_all_raises_clear_error():
    with pytest.raises(ConstitutionParsingError):
        parse_constitution_structure("Just some prose with no PART and no Article numbering whatsoever.")


# -- footnotes block exclusion ------------------------------------------------------------

def test_footnotes_block_excluded_from_structural_parsing():
    text = _SIMPLE_CONSTITUTION + "\n\n[FOOTNOTES]\n1Subs. by the Constitution (First Amdt.) Act, 1974.\n[/FOOTNOTES]"
    root = parse_constitution_structure(text)
    assert "[FOOTNOTES]" not in root.text
    assert root.source_end < len(text)


# -- resumability / document-level behaviour -----------------------------------------------

def test_is_valid_existing_output_true_for_genuinely_valid_output():
    doc = _doc(_SIMPLE_CONSTITUTION)
    parsed = parse_constitution_document(doc)
    assert is_valid_existing_output(parsed) is True


def test_is_valid_existing_output_false_for_corrupted_output():
    doc = _doc(_SIMPLE_CONSTITUTION)
    parsed = parse_constitution_document(doc)
    parsed["structure"]["children"][0]["text"] = "corrupted beyond recognition"
    assert is_valid_existing_output(parsed) is False


def test_is_valid_existing_output_false_for_missing_fields():
    assert is_valid_existing_output({}) is False
    assert is_valid_existing_output({"structure": {}}) is False


def test_parse_directory_resumability_skips_valid_regenerates_invalid(tmp_path):
    input_dir = tmp_path / "in"
    output_dir = tmp_path / "out"
    input_dir.mkdir()
    (input_dir / "doc1.json").write_text(json.dumps(_doc(_SIMPLE_CONSTITUTION, title="doc1")), encoding="utf-8")

    first_summary = parse_directory(input_dir, output_dir)
    assert first_summary["processed"] == 1
    assert first_summary["failed"] == 0

    out_path = output_dir / "doc1.json"
    corrupted = json.loads(out_path.read_text(encoding="utf-8"))
    corrupted["structure"]["children"][0]["text"] = "corrupted"
    out_path.write_text(json.dumps(corrupted), encoding="utf-8")

    second_summary = parse_directory(input_dir, output_dir)
    assert second_summary["skipped_valid"] == 0
    assert second_summary["regenerated"] == 1
    assert second_summary["failed"] == 0


def test_parse_directory_does_not_modify_cleaned_input(tmp_path):
    input_dir = tmp_path / "in"
    output_dir = tmp_path / "out"
    input_dir.mkdir()
    raw_path = input_dir / "doc1.json"
    raw_path.write_text(json.dumps(_doc(_SIMPLE_CONSTITUTION)), encoding="utf-8")
    original_bytes = raw_path.read_bytes()

    parse_directory(input_dir, output_dir)

    assert raw_path.read_bytes() == original_bytes


# -- pipeline isolation --------------------------------------------------------------------

def test_constitution_parser_does_not_import_statute_or_case_law_code():
    import ast

    import src.rag_prep.constitution_parser as mod

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
        assert not name.startswith(forbidden_prefixes), f"constitution_parser.py must not import {name!r}"
