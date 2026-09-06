"""Phase 2: decide whether a case document is structurally substantive.

This is a *structural* filter, not a legal one. It answers one question --
"does this document contain a judgment at all?" -- and never "which legal
domain is it?". It runs before embedding so that cause lists, one-line
adjournments and broken scrapes never consume model time or distort the
clustering they would otherwise land in.

The filter is deliberately **high-precision**: a false drop silently
removes real law from the corpus, while a false keep merely leaves noise
for a later stage to catch. Three design consequences:

* **Single weak signals never drop a document.** A procedural phrase or an
  office-report marker only counts against a document that is also short;
  a long judgment that happens to say "adjourned to a date in office" is
  kept.
* **A substantive-length document with judgment markers is kept outright**
  (:func:`_looks_substantive`), short-circuiting the phrase-based rules
  entirely.
* **Ambiguity resolves to PASS.** Only the rules below can drop; anything
  they do not recognise continues to the next phase as ``pending``.

Every decision is deterministic, local to one document, and cheap: regex
and counting over already-extracted text. No model, no LLM, no I/O.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from src.common.logging_utils import get_logger

logger = get_logger(__name__)

# -- drop reasons (stored in document_representations.drop_reason) --------
REASON_INCOMPLETE_SCRAPE = "incomplete_scrape"
REASON_CAUSE_LIST = "cause_list"
REASON_OFFICE_REPORT = "office_report"
REASON_PROCEDURAL_ADJOURNMENT = "procedural_adjournment"
REASON_SHORT_DOCUMENT = "short_document"
REASON_INSUFFICIENT_SUBSTANTIVE_TEXT = "insufficient_substantive_text"

DROP_REASONS = (
    REASON_INCOMPLETE_SCRAPE,
    REASON_CAUSE_LIST,
    REASON_OFFICE_REPORT,
    REASON_PROCEDURAL_ADJOURNMENT,
    REASON_SHORT_DOCUMENT,
    REASON_INSUFFICIENT_SUBSTANTIVE_TEXT,
)

# -- thresholds (overridden from config; see DocumentSettings) ------------
DEFAULT_MIN_CHARACTERS = 1200
DEFAULT_MIN_WORDS = 250
DEFAULT_INCOMPLETE_SCRAPE_MAX_CHARACTERS = 200
DEFAULT_PROCEDURAL_MAX_CHARACTERS = 3000
DEFAULT_CAUSE_LIST_MIN_CASE_NUMBERS = 8
DEFAULT_CAUSE_LIST_MIN_LIST_RATIO = 0.30
DEFAULT_SUBSTANTIVE_MIN_MARKERS = 2

# Markers that a document is a real decision. Matched case-insensitively
# as whole words, so "ordered" does not count as "ORDER".
_JUDGMENT_MARKER_RE = re.compile(
    r"\b(order|orders|judgment|judgement|petition|petitioner|respondent|held|"
    r"impugned|bench|verdict|appellant|accused|decree|writ|appeal|revision|"
    r"conviction|acquittal|plaintiff|defendant)\b",
    re.IGNORECASE,
)

# Reasoning language: what a court writing a decision sounds like. Used to
# separate a genuine judgment from a list that merely mentions parties.
_REASONING_RE = re.compile(
    r"\b(learned counsel|we are of the view|it is held|held that|in view of|"
    r"therefore|accordingly|hereby|it is observed|perused|contends|contended|"
    r"submitted that|evidence|section \d+|article \d+|prosecution|"
    r"in the circumstances|for the foregoing|is allowed|is dismissed|"
    r"is disposed of|set aside)\b",
    re.IGNORECASE,
)

# Cause-list / roll-call structure: case numbers of the shapes Pakistani
# court lists use.
_CASE_NUMBER_RE = re.compile(
    r"\b(?:C\.?P|Cr\.?l?|W\.?P|R\.?F\.?A|C\.?M\.?A|I\.?C\.?A|F\.?A|S\.?A|J\.?C|"
    r"Civil|Criminal|Writ|Suit)\.?\s*(?:Appeal|Petition|Application|No)?\.?\s*"
    r"\d+[-/]?\d*\s*(?:of|/)\s*\d{4}\b",
    re.IGNORECASE,
)
# A numbered/bulleted line: "1." / "12)" / "(3)" at the start of a line.
_SERIAL_LINE_RE = re.compile(r"^\s*[\(\[]?\d{1,3}[\.\)\]]\s+\S")

_CAUSE_LIST_HEADING_RE = re.compile(
    r"\b(cause list|daily cause list|weekly cause list|case list|"
    r"list of cases|roster|court list|supplementary list)\b",
    re.IGNORECASE,
)

_OFFICE_REPORT_RE = re.compile(
    r"\b(office report|registry report|office objection|office note|"
    r"report of the registrar|administrative order|administrative notice|"
    r"filing report|status report|office memorandum|notice of motion)\b",
    re.IGNORECASE,
)

_PROCEDURAL_RE = re.compile(
    r"(adjourned (?:to|till|until|sine die)|adjournment|relisted|re-listed|"
    r"relist before|case called[, ]*(?:none|no one)? ?(?:present|appears)?|"
    r"none present|no one present|put up on|put up after|list on|"
    r"to come up on|at (?:the )?request of (?:the )?(?:learned )?counsel|"
    r"await(?:ing)? office|for office objection|deferred|"
    r"fresh notice be issued|notice for)",
    re.IGNORECASE,
)

# Scraper/portal failure text. Deliberately specific: "not found" alone
# appears in genuine judgments ("the accused was not found at the scene").
_SCRAPE_ERROR_RE = re.compile(
    r"\b(404 not found|page not found|error 404|access denied|"
    r"you are not authorized|please enable javascript|javascript is required|"
    r"session (?:has )?expired|login required|please log ?in to|"
    r"subscribe to (?:view|read)|under construction|"
    r"an error (?:has )?occurred|service unavailable|no record found)\b",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class StructuralDecision:
    """The verdict for one document, with the evidence behind it."""

    passed: bool
    drop_reason: str | None = None
    triggered_rule: str | None = None
    char_count: int = 0
    word_count: int = 0
    judgment_markers: int = 0
    reasoning_markers: int = 0
    case_numbers: int = 0
    list_line_ratio: float = 0.0
    notes: list[str] = field(default_factory=list)

    @property
    def dropped(self) -> bool:
        return not self.passed


def _count(pattern: re.Pattern[str], text: str) -> int:
    return len(pattern.findall(text))


def _list_line_ratio(text: str) -> float:
    """Share of non-empty lines that look like numbered list entries."""

    lines = [line for line in text.splitlines() if line.strip()]
    if not lines:
        return 0.0
    serial = sum(1 for line in lines if _SERIAL_LINE_RE.match(line))
    return serial / len(lines)


def _looks_substantive(
    char_count: int,
    judgment_markers: int,
    reasoning_markers: int,
    min_characters: int,
    min_markers: int,
) -> bool:
    """A long document that reads like a decision is kept, full stop.

    This is the guard against over-eager rejection: no phrase-based rule
    may drop a document that is comfortably long *and* carries several
    judgment markers *and* contains judicial reasoning language.
    """

    return (
        char_count >= 2 * min_characters
        and judgment_markers >= min_markers
        and reasoning_markers >= 1
    )


def validate_structure(
    text: str,
    title: str | None = None,
    min_characters: int = DEFAULT_MIN_CHARACTERS,
    min_words: int = DEFAULT_MIN_WORDS,
    incomplete_scrape_max_characters: int = DEFAULT_INCOMPLETE_SCRAPE_MAX_CHARACTERS,
    procedural_max_characters: int = DEFAULT_PROCEDURAL_MAX_CHARACTERS,
    cause_list_min_case_numbers: int = DEFAULT_CAUSE_LIST_MIN_CASE_NUMBERS,
    cause_list_min_list_ratio: float = DEFAULT_CAUSE_LIST_MIN_LIST_RATIO,
    substantive_min_markers: int = DEFAULT_SUBSTANTIVE_MIN_MARKERS,
) -> StructuralDecision:
    """Decide whether ``text`` is a structurally substantive judgment.

    ``title`` is consulted only for cause-list and office-report headings;
    it never rescues or condemns a document on its own.

    Rules are applied in order of confidence, most certain first:

    1. ``incomplete_scrape`` -- effectively no text, or portal error text.
    2. ``cause_list`` -- list structure with many case numbers and no
       reasoning. Checked before the substantive guard because a cause
       list can be long.
    3. *substantive guard* -- long + judgment markers + reasoning: PASS.
    4. ``office_report`` -- registry/administrative marker on a short doc.
    5. ``procedural_adjournment`` -- procedural phrase on a short doc.
    6. ``short_document`` -- below the configured length floors.

    Anything reaching the end passes: ambiguity is resolved in favour of
    keeping the document.
    """

    text = text or ""
    stripped = text.strip()
    char_count = len(stripped)
    word_count = len(stripped.split())
    haystack = f"{title or ''}\n{stripped}"

    judgment_markers = _count(_JUDGMENT_MARKER_RE, stripped)
    reasoning_markers = _count(_REASONING_RE, stripped)
    case_numbers = _count(_CASE_NUMBER_RE, haystack)
    list_ratio = _list_line_ratio(stripped)

    def _decide(passed: bool, reason: str | None = None, rule: str | None = None,
                note: str | None = None) -> StructuralDecision:
        return StructuralDecision(
            passed=passed,
            drop_reason=reason,
            triggered_rule=rule,
            char_count=char_count,
            word_count=word_count,
            judgment_markers=judgment_markers,
            reasoning_markers=reasoning_markers,
            case_numbers=case_numbers,
            list_line_ratio=round(list_ratio, 3),
            notes=[note] if note else [],
        )

    # 1. Nothing usable was extracted, or the page is a portal error.
    #    A *tiny* document that names a procedural event is a real (if
    #    trivial) court order, not a failed scrape -- the reason has to be
    #    right, because "the scraper is broken" and "this order says
    #    nothing" call for completely different follow-up.
    if char_count <= incomplete_scrape_max_characters:
        if _PROCEDURAL_RE.search(stripped):
            return _decide(
                False, REASON_PROCEDURAL_ADJOURNMENT, "procedural_phrase_and_short",
                f"procedural phrase in a {char_count}-character order",
            )
        if _OFFICE_REPORT_RE.search(haystack):
            return _decide(
                False, REASON_OFFICE_REPORT, "office_report_marker",
                f"administrative marker in a {char_count}-character document",
            )
        return _decide(
            False, REASON_INCOMPLETE_SCRAPE, "near_empty_text",
            f"only {char_count} characters extracted",
        )
    if _SCRAPE_ERROR_RE.search(stripped[:2000]) and reasoning_markers == 0:
        return _decide(
            False, REASON_INCOMPLETE_SCRAPE, "portal_error_text",
            "portal/scraper error text with no judicial reasoning",
        )

    # 2. Cause lists: many case numbers, list-shaped, no reasoning. A
    #    genuine judgment citing several cases still reasons about them,
    #    which is what separates the two.
    heading_says_list = bool(_CAUSE_LIST_HEADING_RE.search(haystack[:1000]))
    list_shaped = (
        case_numbers >= cause_list_min_case_numbers
        and (list_ratio >= cause_list_min_list_ratio or heading_says_list)
        and reasoning_markers <= 2
    )
    if list_shaped or (heading_says_list and reasoning_markers == 0):
        return _decide(
            False, REASON_CAUSE_LIST, "case_number_list",
            f"{case_numbers} case numbers, list ratio {list_ratio:.2f}, "
            f"{reasoning_markers} reasoning marker(s)",
        )

    # 3. Substantive guard -- a long, reasoned decision is never dropped
    #    by the phrase rules below.
    if _looks_substantive(
        char_count, judgment_markers, reasoning_markers, min_characters,
        substantive_min_markers,
    ):
        return _decide(True, note="substantive length and judgment markers")

    short = char_count < procedural_max_characters

    # 4. Office/registry/administrative records.
    if short and _OFFICE_REPORT_RE.search(haystack):
        return _decide(
            False, REASON_OFFICE_REPORT, "office_report_marker",
            "administrative marker on a short document",
        )

    # 5. Procedural events (adjournment, relisting, non-appearance).
    if short and _PROCEDURAL_RE.search(stripped):
        return _decide(
            False, REASON_PROCEDURAL_ADJOURNMENT, "procedural_phrase_and_short",
            "procedural phrase on a short document",
        )

    # 6. Length floors.
    if char_count < min_characters:
        return _decide(
            False, REASON_SHORT_DOCUMENT, "below_min_characters",
            f"{char_count} characters < {min_characters}",
        )
    if word_count < min_words:
        return _decide(
            False, REASON_SHORT_DOCUMENT, "below_min_words",
            f"{word_count} words < {min_words}",
        )

    # 7. Long enough, but reads like nothing in particular. Only dropped
    #    when there is no judicial language at all.
    if judgment_markers == 0 and reasoning_markers == 0:
        return _decide(
            False, REASON_INSUFFICIENT_SUBSTANTIVE_TEXT, "no_legal_markers",
            "no judgment or reasoning markers found",
        )

    return _decide(True)
