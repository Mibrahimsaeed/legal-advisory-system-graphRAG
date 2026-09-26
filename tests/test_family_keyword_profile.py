"""The expanded Family Law keyword profile.

The profile was widened to close a recall gap: a family judgment written in
vocabulary the original fifteen terms did not cover ("divorce", "nafaqah",
"mahr", "guardianship", "visitation", the statute short-names) produced
almost no keyword evidence. These tests pin down that the new terminology
is detected, that the Criminal profile and the scoring logic are untouched,
and -- importantly -- that no single generic word decides anything.
"""

from __future__ import annotations

import pytest

from src.classification.keyword_signals import (
    compile_profiles,
    detect_keyword_signals,
)
from src.classification.taxonomy_registry import load_frozen_taxonomy
from src.common.config import get_settings

# The Criminal profile as it must remain: this change touches Family only.
CRIMINAL_TERMS = [
    "conviction", "acquittal", "bail", "sentence", "accused", "prosecution",
    "penal code", "narcotic", "fir", "investigating officer", "ocular account",
    "trial court", "offence", "charge", "complainant",
]


@pytest.fixture()
def profiles():
    settings = get_settings()
    return compile_profiles(settings.domain_signals.profiles, load_frozen_taxonomy())


@pytest.fixture()
def family_terms():
    return get_settings().domain_signals.profiles["family_law"]


def _family(text: str, profiles, min_matches: int = 2):
    return detect_keyword_signals("d", text, profiles, min_matches=min_matches)


def _matched(text: str, profiles) -> set[str]:
    signals = detect_keyword_signals("d", text, profiles, min_matches=1)
    return set(signals.scores["family_law"].matched_terms)


# ---------------------------------------------------------------------------
# 1. Every requested group is present and matches real phrasing
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "text, expected",
    [
        # marriage
        ("The marriage was solemnised by nikah.", {"marriage", "nikah"}),
        ("The nikah nama was produced.", {"nikah nama", "nikah"}),
        ("Matrimonial and marital obligations of the spouse.",
         {"matrimonial", "marital", "spouse"}),
        ("The husband denied that the wife was entitled.", {"husband", "wife"}),
        # dissolution
        ("A decree of divorce followed the dissolution of marriage.",
         {"divorce", "dissolution", "dissolution of marriage"}),
        ("She sought khula; he pronounced talaq.", {"khula", "talaq"}),
        ("The talaqnama and the mubarat deed were exhibited.",
         {"talaqnama", "mubarat"}),
        # financial
        ("Maintenance and nafaqah were claimed.", {"maintenance", "nafaqah"}),
        ("Recovery of dower, being her mahr.", {"dower", "mahr"}),
        ("The haq mehr remained unpaid.", {"haq mehr", "mehr"}),
        ("The meher was fixed at the time of nikah.", {"meher", "nikah"}),
        ("Return of dowry articles and bridal gifts.",
         {"dowry", "dowry articles", "bridal gifts"}),
        ("A suit for the wife's property and the wife's belongings.",
         {"wife", "wife's property", "wife's belongings"}),
        # children
        ("Custody and hizanat of the child.", {"custody", "hizanat"}),
        ("The hizaanat of the minor was disputed.", {"hizaanat", "minor"}),
        ("Visitation was allowed; guardianship was refused to the guardian.",
         {"visitation", "guardianship", "guardian"}),
        ("The minor child remained a ward of the court.",
         {"minor", "minor child", "ward"}),
        ("Decided with reference to the welfare of the minor.",
         {"welfare of the minor", "minor"}),
        # marital rights
        ("A suit for restitution of conjugal rights.",
         {"restitution of conjugal rights", "conjugal rights"}),
        ("A decree for jactitation of marriage.",
         {"jactitation of marriage", "marriage"}),
        # legislation
        ("Proceedings before the Judge Family Court under the Family Courts Act.",
         {"family court", "judge family court", "family courts act"}),
        ("Section 7 of the Muslim Family Laws Ordinance, 1961.",
         {"muslim family laws", "muslim family laws ordinance"}),
        ("An application under the MFLO.", {"mflo"}),
        ("A suit under the Dissolution of Muslim Marriages Act, 1939.",
         {"dissolution", "dissolution of muslim marriages act"}),
        ("Appointed under the Guardians and Wards Act, 1890.",
         {"guardians and wards", "guardians and wards act"}),
        ("An offence under the Child Marriage Restraint Act.",
         {"child marriage restraint act", "marriage"}),
        ("A claim under the Dowry and Bridal Gifts (Restriction) Act, 1976.",
         {"dowry", "bridal gifts", "dowry and bridal gifts (restriction) act"}),
    ],
)
def test_the_new_terminology_is_detected(text, expected, profiles):
    assert expected <= _matched(text, profiles)


def test_statute_names_with_punctuation_compile_and_match(profiles):
    """Parentheses are regex metacharacters -- they must be escaped, not active."""

    matched = _matched(
        "Relief was sought under the Dowry and Bridal Gifts (Restriction) Act.",
        profiles,
    )
    assert "dowry and bridal gifts (restriction) act" in matched


def test_apostrophes_in_terms_match(profiles):
    assert "wife's property" in _matched("A decree for the wife's property.", profiles)


def test_every_requested_group_is_represented(family_terms):
    """A structural check, so a group cannot silently go missing."""

    terms = set(family_terms)
    for group, member in [
        ("marriage", "nikah"),
        ("dissolution", "mubarat"),
        ("financial", "nafaqah"),
        ("children", "hizanat"),
        ("marital rights", "jactitation of marriage"),
        ("legislation", "child marriage restraint act"),
    ]:
        assert member in terms, f"the {group} group is missing {member!r}"


def test_the_profile_has_no_duplicate_terms(family_terms):
    assert len(family_terms) == len(set(family_terms))


# ---------------------------------------------------------------------------
# 2. Recall actually improved
# ---------------------------------------------------------------------------


FAMILY_ALTERNATIVE_VOCABULARY = (
    "The appellant husband challenges the decree of divorce. The respondent spouse "
    "claimed nafaqah and her mahr. Proceedings under the Muslim Family Laws Ordinance "
    "and the Family Courts Act concerned visitation and guardianship of the minor "
    "child, and restitution of conjugal rights was refused."
)


def test_a_judgment_in_alternative_vocabulary_now_produces_real_evidence(profiles):
    """The recall gap: this case matched only 3 of the original 15 terms."""

    signals = _family(FAMILY_ALTERNATIVE_VOCABULARY, profiles)
    family = signals.scores["family_law"]

    assert len(family.matched_terms) >= 12
    assert signals.top_domain == "family_law"
    assert family.score > 0


def test_classic_family_vocabulary_still_wins(profiles):
    """The original terms must keep working -- nothing was removed."""

    signals = _family(
        "Suit for recovery of dower, dowry articles and maintenance before the Judge "
        "Family Court. The wife seeks dissolution of marriage by khula; custody of the "
        "minor under the Guardians and Wards Act; the nikahnama was exhibited.",
        profiles,
    )
    assert signals.top_domain == "family_law"
    assert len(signals.scores["family_law"].matched_terms) >= 10


def test_none_of_the_original_fifteen_terms_was_dropped(family_terms):
    original = {
        "dower", "dowry", "maintenance", "khula", "dissolution of marriage",
        "custody", "guardian", "minor", "nikahnama", "talaq", "family court",
        "matrimonial", "conjugal rights", "guardians and wards",
        "muslim family laws",
    }
    assert original <= set(family_terms)


# ---------------------------------------------------------------------------
# 3. Generic words are signals, not verdicts
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("word", ["husband", "wife", "marriage", "minor", "ward"])
def test_a_single_generic_word_does_not_produce_a_family_score(word, profiles):
    """min_keyword_matches is what stops one incidental word deciding a domain."""

    signals = _family(f"The witness mentioned the {word} in passing.", profiles, 2)
    assert signals.scores["family_law"].score == 0.0
    assert signals.top_domain is None


def test_a_criminal_judgment_mentioning_a_wife_stays_criminal(profiles):
    """Generic family words appear in criminal judgments and must not flip them."""

    signals = _family(
        "The accused was convicted under the Penal Code of murdering his wife. The "
        "prosecution led the ocular account; the investigating officer produced the "
        "FIR and the trial court recorded the charge before sentence.",
        profiles,
    )
    assert signals.top_domain == "criminal_law"


def test_an_off_domain_judgment_still_matches_nothing(profiles):
    signals = _family(
        "This reference concerns the assessment of income tax and the limitation "
        "period prescribed by the Ordinance; the taxpayer contends it was time barred.",
        profiles,
    )
    assert signals.top_domain is None
    assert signals.scores["family_law"].score == 0.0
    assert signals.scores["criminal_law"].score == 0.0


def test_matching_is_still_whole_word(profiles):
    """'ward' must not fire on 'awkward', 'nikah' not on 'nikahnama' alone."""

    assert "ward" not in _matched("The awkward wording of the clause.", profiles)
    assert "minor" not in _matched("A minority shareholder dispute.", profiles)


# ---------------------------------------------------------------------------
# 4. Nothing outside the Family profile changed
# ---------------------------------------------------------------------------


def test_the_criminal_profile_is_untouched():
    assert get_settings().domain_signals.profiles["criminal_law"] == CRIMINAL_TERMS


def test_a_pure_criminal_judgment_scores_exactly_as_before(profiles):
    """The Criminal denominator is unchanged, so its score is unchanged."""

    signals = _family(
        "The accused was convicted under the Penal Code and sentenced by the trial "
        "court. The ocular account of the complainant is doubtful; the investigating "
        "officer joined no private witness and the prosecution failed to prove the "
        "charge.",
        profiles,
    )
    criminal = signals.scores["criminal_law"]
    assert len(criminal.matched_terms) == 8
    assert criminal.score == pytest.approx(8 / 15)


def test_the_scoring_formula_is_unchanged(profiles):
    """score == distinct matched / profile size, gated by min_matches."""

    settings = get_settings()
    size = len(settings.domain_signals.profiles["family_law"])
    signals = _family(FAMILY_ALTERNATIVE_VOCABULARY, profiles)
    family = signals.scores["family_law"]

    assert family.score == pytest.approx(len(family.matched_terms) / size)


def test_min_keyword_matches_setting_is_unchanged():
    assert get_settings().domain_signals.min_keyword_matches == 2


def test_the_evidence_payload_shape_is_unchanged(profiles):
    """Phase 3 stores this JSON; its shape must not drift."""

    evidence = _family(FAMILY_ALTERNATIVE_VOCABULARY, profiles).as_evidence()

    assert set(evidence) == {"top_domain", "margin", "total_matches", "domains"}
    assert set(evidence["domains"]) == {"family_law", "criminal_law"}
    for entry in evidence["domains"].values():
        assert set(entry) == {"score", "total_matches", "distinct_terms", "matched_terms"}


def test_the_profile_still_validates_against_the_frozen_taxonomy():
    """Expanding a profile must not introduce an unknown domain id."""

    compiled = compile_profiles(
        get_settings().domain_signals.profiles, load_frozen_taxonomy()
    )
    assert set(compiled) == {"family_law", "criminal_law"}
