"""Conservative statute/provision CITATION extraction from case-law text.

Not a statute-cleaning module -- it does not process statute, Act, or
constitutional *documents* (there are none staged in ``var/rag/raw/`` to
process; that folder holds only court judgments, verified before this
module was written). This extracts the ``statutes_cited`` /
``provisions_cited`` metadata fields FROM a case-law judgment's text -- i.e.
which Acts/sections/Articles that judgment refers to.

Run it on already-cleaned text (:func:`src.rag_prep.case_cleaner.clean_full_text`'s
output) -- line-joining must happen first, or a statute name wrapped
mid-phrase like "West\\nPakistan\\nFamily Courts Act" won't match.

Deliberately not a legal NER system. Two conservative goals, in this order:

* **Never invent.** Every pattern here requires an unambiguous anchor --
  the "(<roman-or-number> of <year>)" suffix for statutes, or a section/
  article keyword immediately before a number for provisions. Something
  outside these shapes is left unextracted rather than guessed at.
* **Prefer under-extraction to over-extraction.** Real Pakistani case-law
  citation formats are far more varied than any fixed regex set can cover
  (this was validated against samples from the actual corpus, not assumed).
  Callers should treat an empty list as "nothing reliably found", not as
  "this document cites nothing".
"""

from __future__ import annotations

import re

# "<Name...> Act/Ordinance/Code/Order/Rules/Regulation (<N> of <YYYY>)" --
# e.g. "West Pakistan Family Courts Act (XXXV of 1964)",
# "Muslim Family Laws Ordinance (VIII of 1961)". The name capture is a
# short, bounded run of Title-Case-ish words so it can't run away across an
# entire paragraph if the anchor keyword appears unexpectedly.
#
# Internal word-spacing is restricted to [ \t] (never \n): after
# clean_full_text()/join_wrapped_lines(), a newline only ever survives at a
# paragraph boundary, so allowing \s (which matches \n\n too) here let the
# name capture run backward across a paragraph break and swallow the last
# word of the *previous* paragraph (observed directly: "Petitioner\n\nWest
# Pakistan Family Courts Act (XXXV of 1964)" matched as one statute name
# including "Petitioner"). Restricting to [ \t] makes that structurally
# impossible rather than relying on the pattern happening not to reach that far.
_STATUTE_RE = re.compile(
    r"([A-Z][A-Za-z.,&'\-]*(?:[ \t]+(?:of|and|the)?[ \t]*[A-Za-z.,&'\-]+){0,8}?[ \t]+"
    r"(?:Act|Ordinance|Code|Order|Rules|Regulation))[ \t]*"
    r"\([ \t]*([IVXLCDM]+|\d{1,4})[ \t]+of[ \t]+(\d{4})[ \t]*\)"
)

# The Constitution is cited constantly but never carries an "(N of YYYY)"
# suffix -- just a bare year, e.g. "Constitution of Pakistan (1973)" or
# "Constitution of Pakistan, 1973". Handled as its own anchor.
_CONSTITUTION_RE = re.compile(r"\bConstitution of Pakistan\b[\s,]*\(?\s*(\d{4})\s*\)?")

# Section references: "S. 5", "Ss. 3 & 4", "Section 7", "section 488",
# "sec. 14-A". Suffix is letters only (subsection markers like "199A"),
# never hyphens -- an earlier version of this pattern let a trailing
# section-divider "---" get swallowed into the match.
_SECTION_RE = re.compile(
    r"\bS(?:s|ec(?:tion)?s?)?\.?\s*\d+[A-Za-z]{0,2}"
    r"(?:\s*(?:,|&|and)\s*\d+[A-Za-z]{0,2})*\b",
    re.IGNORECASE,
)

# Article references: "Art. 199", "Article 199", "Arts. 8 & 9".
_ARTICLE_RE = re.compile(r"\bArt(?:icle)?s?\.?\s*\d+[A-Za-z]{0,2}\b", re.IGNORECASE)

_MULTI_SPACE_RE = re.compile(r"\s{2,}")


def _normalize_span(span: str) -> str:
    return _MULTI_SPACE_RE.sub(" ", span).strip()


def extract_statutes_cited(cleaned_text: str) -> list[str]:
    """Statute names with an unambiguous "(N of YYYY)" or Constitution anchor.

    Returns distinct values, in first-appearance order. Empty list (never
    ``None``) when nothing matches the anchor shapes -- matching the field's
    documented type of a list.
    """

    if not cleaned_text:
        return []

    seen: dict[str, None] = {}
    for match in _STATUTE_RE.finditer(cleaned_text):
        name = _normalize_span(match.group(1))
        citation = f"{name} ({match.group(2)} of {match.group(3)})"
        seen.setdefault(citation, None)
    for match in _CONSTITUTION_RE.finditer(cleaned_text):
        citation = f"Constitution of Pakistan ({match.group(1)})"
        seen.setdefault(citation, None)

    return list(seen)


def extract_provisions_cited(cleaned_text: str) -> list[str]:
    """Section/Article references, distinct, in first-appearance order."""

    if not cleaned_text:
        return []

    seen: dict[str, None] = {}
    for match in _SECTION_RE.finditer(cleaned_text):
        seen.setdefault(_normalize_span(match.group(0)), None)
    for match in _ARTICLE_RE.finditer(cleaned_text):
        seen.setdefault(_normalize_span(match.group(0)), None)

    return list(seen)
