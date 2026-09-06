"""Normalize a case folder's ``metadata.json`` into typed case-law fields.

Real case-law exports disagree about key names ("court" vs "court_name",
"date" vs "decision_date" vs "judgment_date"), about types (``judges`` as
a list, or one comma-joined string, or absent), and about date formats
("2019-04-11", "11/04/2019", "11 April 2019", "2019"). This module is the
single place that guesswork lives, so :mod:`src.extraction.case_loader`
stays a pure HTML-to-text concern and every normalization assumption is
reviewable in one file.

Two rules throughout:

* **Never fail.** A key that is missing, null, or the wrong type yields
  ``None``/``[]`` plus a warning string -- never an exception, and never
  a silently wrong value.
* **Never lose the original.** The verbatim ``metadata.json`` is kept on
  :attr:`~src.extraction.doc_representation.DocumentRepresentation.metadata`,
  so anything this module declines to interpret is still recoverable.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Any

# Key aliases, most-specific first. Matching is case-insensitive and
# ignores separators, so "case_title", "caseTitle" and "Case Title" all
# hit the same entry.
_TITLE_KEYS = ("case_title", "case_name", "title", "name")
_COURT_KEYS = ("court", "court_name", "forum", "tribunal", "bench_court")
_DATE_KEYS = (
    "decision_date",
    "judgment_date",
    "date_decided",
    "decided_on",
    "date",
    "year",
)
_CITATION_KEYS = ("citation", "citations", "cite", "neutral_citation", "reported_as")
_JUDGE_KEYS = ("judges", "judge", "bench", "author", "authors", "coram", "justice")
_CASE_NUMBER_KEYS = (
    "case_number",
    "case_no",
    "cause_number",
    "appeal_number",
    "petition_number",
    "number",
)

# Judge strings are split on these only. Deliberately NOT on commas:
# "Mr. Justice Asif Saeed Khan Khosa, CJ" is one judge, and splitting it
# would invent a second judge called "CJ".
_JUDGE_SPLIT_RE = re.compile(r"\s*(?:;|&|\band\b)\s*", re.IGNORECASE)

# Date formats tried in order. Numeric day/month ordering is day-first
# (11/04/2019 -> 11 April 2019): South Asian court records follow the
# British convention, and a wrong guess here is silent, so the ambiguous
# case is resolved once, here, rather than per caller.
_DATE_FORMATS = (
    "%Y-%m-%d",
    "%d-%m-%Y",
    "%d/%m/%Y",
    "%d.%m.%Y",
    "%d %B %Y",
    "%d %b %Y",
    "%B %d, %Y",
    "%b %d, %Y",
    "%Y/%m/%d",
)
_YEAR_ONLY_RE = re.compile(r"^\s*(1[6-9]\d{2}|20\d{2})\s*$")

WARNING_UNPARSEABLE_DATE = "unparseable_date"
WARNING_UNEXPECTED_TYPE = "unexpected_metadata_type"


@dataclass(frozen=True)
class CaseMetadata:
    """The typed subset of ``metadata.json`` the pipeline understands."""

    title: str | None = None
    court: str | None = None
    decision_date: str | None = None  # ISO-8601 YYYY-MM-DD
    citation: str | None = None
    judges: list[str] = field(default_factory=list)
    case_number: str | None = None
    warnings: list[str] = field(default_factory=list)


def _normalize_key(key: str) -> str:
    return re.sub(r"[^a-z0-9]", "", key.lower())


def _lookup(metadata: dict[str, Any], keys: tuple[str, ...]) -> tuple[str, Any] | None:
    """Return the first ``(actual_key, value)`` whose key matches an alias."""

    normalized = {_normalize_key(k): k for k in metadata if isinstance(k, str)}
    for alias in keys:
        actual = normalized.get(_normalize_key(alias))
        if actual is not None and metadata[actual] not in (None, "", [], {}):
            return actual, metadata[actual]
    return None


def _clean_str(value: Any) -> str | None:
    """Collapse whitespace on a string-ish scalar; reject anything else."""

    if isinstance(value, str):
        cleaned = " ".join(value.split())
        return cleaned or None
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return str(value)
    return None


def parse_date(value: Any) -> str | None:
    """Best-effort ISO-8601 (``YYYY-MM-DD``) for a metadata date value.

    A bare year normalizes to January 1st of that year -- courts publish
    year-only citations often enough that dropping the field entirely
    loses more than the false precision costs, and the raw value is still
    on ``metadata`` for anything that needs to tell the two apart.
    Returns ``None`` when nothing recognizable is there.
    """

    if isinstance(value, (date, datetime)):
        return (value.date() if isinstance(value, datetime) else value).isoformat()

    if isinstance(value, int) and not isinstance(value, bool):
        value = str(value)

    if not isinstance(value, str):
        return None

    raw = value.strip()
    if not raw:
        return None

    year_only = _YEAR_ONLY_RE.match(raw)
    if year_only:
        return f"{year_only.group(1)}-01-01"

    # Trim a trailing time component ("2019-04-11T00:00:00Z") before
    # trying date-only formats.
    candidate = re.split(r"[T ]", raw, maxsplit=1)[0] if re.match(r"^\d{4}-\d{2}-\d{2}", raw) else raw

    for fmt in _DATE_FORMATS:
        try:
            return datetime.strptime(candidate, fmt).date().isoformat()
        except ValueError:
            continue
    return None


def parse_judges(value: Any) -> list[str]:
    """Normalize a judges/bench/author value into a list of names.

    Accepts a list of strings, a list of ``{"name": ...}`` objects, or a
    single string joined by ``;``/``&``/``and``. Anything else yields an
    empty list (the caller records a warning).
    """

    if isinstance(value, str):
        parts = _JUDGE_SPLIT_RE.split(value)
        return [name for name in (_clean_str(p) for p in parts) if name]

    if isinstance(value, list):
        names: list[str] = []
        for item in value:
            if isinstance(item, dict):
                item = item.get("name") or item.get("judge") or item.get("title")
            name = _clean_str(item)
            if name:
                names.append(name)
        return names

    return []


def parse_citation(value: Any) -> str | None:
    """Primary citation: the value itself, or the first entry of a list.

    Any additional citations stay on the verbatim ``metadata`` dict --
    this field is for identity/display, not for exhaustive citation
    capture (that belongs to the later citation-graph work).
    """

    if isinstance(value, list):
        for item in value:
            citation = _clean_str(item)
            if citation:
                return citation
        return None
    return _clean_str(value)


def extract_case_metadata(metadata: dict[str, Any]) -> CaseMetadata:
    """Normalize a parsed ``metadata.json`` mapping into :class:`CaseMetadata`.

    Never raises. Every field that is present but unusable adds a warning
    of the form ``"<warning_kind>:<actual_key>"`` so a scan over a large
    corpus can be aggregated by problem type.
    """

    if not isinstance(metadata, dict) or not metadata:
        return CaseMetadata()

    warnings: list[str] = []

    def _scalar(keys: tuple[str, ...]) -> str | None:
        found = _lookup(metadata, keys)
        if found is None:
            return None
        actual_key, raw = found
        value = _clean_str(raw)
        if value is None:
            warnings.append(f"{WARNING_UNEXPECTED_TYPE}:{actual_key}")
        return value

    title = _scalar(_TITLE_KEYS)
    court = _scalar(_COURT_KEYS)
    case_number = _scalar(_CASE_NUMBER_KEYS)

    decision_date = None
    found_date = _lookup(metadata, _DATE_KEYS)
    if found_date is not None:
        actual_key, raw = found_date
        decision_date = parse_date(raw)
        if decision_date is None:
            warnings.append(f"{WARNING_UNPARSEABLE_DATE}:{actual_key}")

    citation = None
    found_citation = _lookup(metadata, _CITATION_KEYS)
    if found_citation is not None:
        actual_key, raw = found_citation
        citation = parse_citation(raw)
        if citation is None:
            warnings.append(f"{WARNING_UNEXPECTED_TYPE}:{actual_key}")

    judges: list[str] = []
    found_judges = _lookup(metadata, _JUDGE_KEYS)
    if found_judges is not None:
        actual_key, raw = found_judges
        judges = parse_judges(raw)
        if not judges:
            warnings.append(f"{WARNING_UNEXPECTED_TYPE}:{actual_key}")

    return CaseMetadata(
        title=title,
        court=court,
        decision_date=decision_date,
        citation=citation,
        judges=judges,
        case_number=case_number,
        warnings=warnings,
    )
