"""Focused tests for the RAG case-law cleaning stage (src/rag_prep/).

Four groups, matching what was asked for:

1. Encoding correction (mojibake repair)
2. Line joining (wrap-artifact removal vs paragraph-boundary preservation)
3. Metadata preservation (existing values pass through unchanged; new
   fields are derived conservatively or left null/empty)
4. Legal-content preservation (cleaning must not drop or shorten
   substantive text)
"""

from __future__ import annotations

from src.rag_prep.case_cleaner import (
    REQUIRED_METADATA_FIELDS,
    build_processed_metadata,
    clean_full_text,
    derive_court_location,
    extract_disposition,
    fix_mojibake,
    join_wrapped_lines,
)
from src.rag_prep.statute_cleaner import extract_provisions_cited, extract_statutes_cited

# ---------------------------------------------------------------------------
# 1. Encoding correction
# ---------------------------------------------------------------------------


def test_mojibake_hyphen_is_repaired():
    # The dominant artifact in this corpus: U+2011 (non-breaking hyphen)
    # decoded as cp1252, three times in a row, as seen in real party lines.
    assert fix_mojibake("JOHORA JANA â€‘â€‘â€‘Petitioner") == "JOHORA JANA ‑‑‑Petitioner"


def test_mojibake_quote_mark_is_repaired():
    # The corpus's actual (non-standard) quote-mark mojibake, confirmed
    # against real text: "the childâ€Ÿs sense of identity" -> U+201F.
    assert fix_mojibake("childâ€Ÿs sense") == "child‟s sense"


def test_every_mojibake_variant_present_in_the_staged_corpus_is_repaired():
    """Surveyed directly against var/rag/raw/: exactly three 'â€<x>'
    variants occur in this corpus (35,901 / 24 / 3 occurrences). All three
    must round-trip cleanly -- this is not a hypothetical list."""

    assert fix_mojibake("â€‘") == "‑"  # non-breaking hyphen (dominant case)
    assert fix_mojibake("â€Ž") == "‎"  # left-to-right mark
    assert fix_mojibake("â€Ÿ") == "‟"  # double high-reversed-9 quote


def test_text_without_mojibake_is_unchanged():
    clean = "The petition is dismissed with costs."
    assert fix_mojibake(clean) == clean


def test_a_non_mojibake_ae_sequence_is_left_alone():
    """Only the exact 'â€<x>' artifact is touched -- an unrelated 'â' or '€'
    elsewhere in the text (e.g. a genuine euro sign) must survive as-is."""

    text = "Damages of â‚¬500 were awarded"  # â‚¬ is NOT the 'â€' artifact
    assert fix_mojibake(text) == text


def test_mojibake_fix_does_not_touch_surrounding_text():
    text = "before â€‘ after"
    fixed = fix_mojibake(text)
    assert fixed.startswith("before ")
    assert fixed.endswith(" after")


# ---------------------------------------------------------------------------
# 2. Line joining
# ---------------------------------------------------------------------------


def test_single_newline_within_a_paragraph_becomes_a_space():
    text = "ADDITIONAL DISTRICT JUDGE, KABIRWALA and 2\nothers----Respondents"
    assert join_wrapped_lines(text) == "ADDITIONAL DISTRICT JUDGE, KABIRWALA and 2 others----Respondents"


def test_blank_line_paragraph_boundary_is_preserved():
    text = "MUHAMMAD HANIF----Petitioner\n\nVersus\n\nADDITIONAL DISTRICT JUDGE"
    joined = join_wrapped_lines(text)
    assert joined == "MUHAMMAD HANIF----Petitioner\n\nVersus\n\nADDITIONAL DISTRICT JUDGE"


def test_a_sentence_wrapped_mid_phrase_is_rejoined():
    text = "Writ Petition No.1984 of 2008, decided on 29th\nJanuary, 2014."
    assert join_wrapped_lines(text) == "Writ Petition No.1984 of 2008, decided on 29th January, 2014."


def test_multiple_paragraphs_each_get_internally_joined():
    text = "First\nparagraph\nhere.\n\nSecond\nparagraph\nhere."
    assert join_wrapped_lines(text) == "First paragraph here.\n\nSecond paragraph here."


def test_clean_full_text_applies_mojibake_then_normalize_then_join_in_order():
    raw = "Mst. JOHORA JANA â€‘â€‘â€‘Petitioner\n\nWest\nPakistan\nFamily Courts Act (XXXV of 1964)"
    cleaned = clean_full_text(raw)
    assert "â€" not in cleaned  # mojibake fixed
    assert "West Pakistan Family Courts Act" in cleaned  # wrap-joined
    assert "\n\n" in cleaned  # paragraph boundary preserved


def test_clean_full_text_handles_empty_input():
    assert clean_full_text("") == ""
    assert clean_full_text(None) == ""


# ---------------------------------------------------------------------------
# 3. Metadata preservation
# ---------------------------------------------------------------------------


def _raw_metadata(**overrides):
    base = {
        "title": "MUHAMMAD AKRAM vs Mst. YASMIN",
        "citation": "1983 C L C 3098",
        "court": "Karachi",
        "decision_date": "1983-01-01",
        "judges": ["Saleem Akhtar, J"],
        "case_number": "1983K604",
        "primary_domain": "family_law",
        "classification_status": "auto_accepted",
        "source_relpath": "1983K604",
        "content_hash": "abc123",
    }
    base.update(overrides)
    return base


def test_all_required_metadata_fields_are_present():
    metadata = build_processed_metadata(_raw_metadata(), "Petition\n\nMQ/1/L Petition dismissed.", "doc1")
    assert set(metadata.keys()) == set(REQUIRED_METADATA_FIELDS)


def test_existing_scalar_metadata_passes_through_unchanged():
    raw = _raw_metadata()
    metadata = build_processed_metadata(raw, "text", "doc1")
    assert metadata["citation"] == raw["citation"]
    assert metadata["court"] == raw["court"]
    assert metadata["decision_date"] == raw["decision_date"]
    assert metadata["case_number"] == raw["case_number"]
    assert metadata["primary_domain"] == raw["primary_domain"]
    assert metadata["classification_status"] == raw["classification_status"]
    assert metadata["source_relpath"] == raw["source_relpath"]
    assert metadata["content_hash"] == raw["content_hash"]
    assert metadata["judges"] == raw["judges"]


def test_title_is_renamed_to_case_title_not_dropped():
    raw = _raw_metadata(title="MY CASE TITLE")
    metadata = build_processed_metadata(raw, "text", "doc1")
    assert metadata["case_title"] == "MY CASE TITLE"


def test_doc_id_comes_from_the_explicit_argument_not_guessed():
    metadata = build_processed_metadata(_raw_metadata(), "text", "the-real-doc-id")
    assert metadata["doc_id"] == "the-real-doc-id"


def test_missing_optional_metadata_is_null_not_invented():
    raw = _raw_metadata(citation=None, decision_date=None, judges=None)
    metadata = build_processed_metadata(raw, "text", "doc1")
    assert metadata["citation"] is None
    assert metadata["decision_date"] is None
    assert metadata["judges"] == []  # normalized to empty list, not None


def test_disposition_and_citations_are_derived_from_cleaned_text():
    raw = _raw_metadata()
    text = "West Pakistan Family Courts Act (XXXV of 1964)\n\nMQ/1/L Petition dismissed."
    metadata = build_processed_metadata(raw, text, "doc1")
    assert metadata["disposition"] == "dismissed"
    assert "West Pakistan Family Courts Act (XXXV of 1964)" in metadata["statutes_cited"]


def test_unavailable_derived_fields_are_null_or_empty_not_guessed():
    raw = _raw_metadata(court=None)
    metadata = build_processed_metadata(raw, "no clear outcome mentioned here at all", "doc1")
    assert metadata["court_location"] is None
    assert metadata["disposition"] is None
    assert metadata["statutes_cited"] == []
    assert metadata["provisions_cited"] == []


def test_court_location_prefers_the_bench_city():
    assert derive_court_location("Lahore (Multan Bench)") == "Multan"
    assert derive_court_location("Supreme Court of Pakistan") is None  # institution, not a location
    assert derive_court_location("Karuchil") is None  # a typo is not corrected/guessed
    assert derive_court_location(None) is None


# ---------------------------------------------------------------------------
# 4. Legal-content preservation
# ---------------------------------------------------------------------------


def _words(text: str) -> set[str]:
    import re

    return set(re.findall(r"[a-zA-Z]+", text.lower()))


def test_cleaning_preserves_every_alphabetic_word():
    """No word may be dropped, only whitespace/line-structure may change."""

    raw = (
        "The petitioner contends that the marriage was solemnized\n"
        "under duress and that the wife is entitled to khula as of\n"
        "right under the West Pakistan Family Courts Act, 1964.\n\n"
        "The learned trial Court dismissed the suit holding that no\n"
        "evidence of coercion was produced."
    )
    cleaned = clean_full_text(raw)
    assert _words(raw) == _words(cleaned)


def test_cleaning_does_not_shorten_a_real_judgment_excerpt():
    raw = (
        "1983 C L C 3098\n\n[Karachi]\n\nBefore Saleem Akhtar, J\n\n"
        "MUHAMMAD AKRAMâ€‘Petitioner\n\nversus\n\nMst. YASMIN AND ANOTHERâ€‘â€‘Respondents\n\n"
        "West Pakistan Family Courts Act (XXXV of 1964) â€‘\n\n"
        "â€‘â€‘â€‘ S. 7â€‘Khula`â€‘Wife,\nheld, entitled to Khula` as of right if she "
        "satisfies conscience of Court that\nit will otherwise mean forcing her "
        "into hateful union.\n\nThe petition is dismissed in\nlimine."
    )
    cleaned = clean_full_text(raw)

    words_before = _words(raw)
    words_after = _words(cleaned)
    # Exact set equality: cleaning must not lose or add any word.
    assert words_before == words_after

    # Key substantive legal terms explicitly survive cleaning.
    for term in ("petitioner", "respondents", "khula", "conscience", "dismissed", "limine"):
        assert term in cleaned.lower(), f"{term!r} missing after cleaning"


def test_disposition_outcomes_are_preserved_verbatim_in_cleaned_text():
    """The disposition sentence itself is never deleted or paraphrased --
    only reflowed. Check across several outcome types."""

    for fragment, expected_label in [
        ("The suit is dismissed with costs.", "dismissed"),
        ("The petition is allowed as prayed for.", "allowed"),
        ("The case is remanded for fresh decision.", "remanded"),
        ("The appeal is partly allowed.", "partly_allowed"),
    ]:
        cleaned = clean_full_text(fragment)
        assert fragment.rstrip(".") in cleaned or fragment in cleaned
        assert extract_disposition(cleaned) == expected_label


def test_facts_judges_and_precedent_citations_all_survive_cleaning():
    raw = (
        "Before Muhammad Afzal Lone, J\n\nThe facts are that the parties were "
        "married on 5-2-1958 in accordance with Sunni Hanafi Law. Reliance was "
        "placed on Khurshid Bibi v. Muhammad Amin P L D 1967 S C 97.\n\n"
        "Section 5 of the Family Courts Act, 1964 was considered.\n\n"
        "The appeal is dismissed."
    )
    cleaned = clean_full_text(raw)
    for phrase in (
        "Muhammad Afzal Lone",
        "married on 5-2-1958",
        "Sunni Hanafi Law",
        "Khurshid Bibi v. Muhammad Amin",
        "Section 5 of the Family Courts Act",
        "dismissed",
    ):
        assert phrase in cleaned, f"{phrase!r} missing after cleaning"


# ---------------------------------------------------------------------------
# Citation extraction sanity (supports the metadata tests above)
# ---------------------------------------------------------------------------


def test_statute_extraction_does_not_cross_a_paragraph_boundary():
    """Regression: the name-capture must not run backward across a blank
    line and swallow the previous paragraph's last word."""

    text = "Mst. JOHORA JANA Petitioner\n\nWest Pakistan Family Courts Act (XXXV of 1964)"
    statutes = extract_statutes_cited(text)
    assert statutes == ["West Pakistan Family Courts Act (XXXV of 1964)"]


def test_provisions_cited_extracts_sections_and_articles():
    text = "Under S. 5 and Art. 199 of the Constitution, the petition is maintainable."
    provisions = extract_provisions_cited(text)
    assert any(p.replace(" ", "") in ("S.5",) or p == "S. 5" for p in provisions)
    assert any("199" in p for p in provisions)


def test_extraction_functions_return_empty_not_none_when_nothing_found():
    assert extract_statutes_cited("") == []
    assert extract_provisions_cited("no citations in this sentence") == []
