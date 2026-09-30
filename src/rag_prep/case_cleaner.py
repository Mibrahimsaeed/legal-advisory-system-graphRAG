"""RAG text cleaning for staged Family Law CASE-LAW documents only.

Scope: every document under ``var/rag/raw/`` is a court judgment
(``document_representations.source_type = 'case_html'``, verified against
the live database before this module was written -- not assumed). This
module's cleaning rules and legal-content-preservation guarantees are
written for that shape of document specifically: a cause title, court/
bench, judges, parties, facts, arguments, statutes/provisions cited,
precedents, reasoning, and a final disposition. Statutes, Acts, rules and
constitutional texts are a different document shape with different
preservation requirements and are explicitly NOT handled here -- there is
currently nothing of that kind staged in ``var/rag/raw/`` to process by
mistake, and this module makes no attempt to generalize to that case.

Reads ``var/rag/raw/<doc_id>.json`` (built by ``var/rag/build_raw_staging.py``
from the classification pipeline's ``document_representations``) and
produces cleaned ``full_text`` plus enriched, case-level metadata.
Deliberately conservative: every transformation here is either a targeted,
reversible repair (mojibake) or a whitespace/formatting normalization (line
joining). Nothing is summarized, paraphrased, or deleted for being
"boilerplate" -- the legal-content-preservation tests in
``tests/test_rag_prep.py`` check this directly by comparing normalized word
counts before/after.

Three responsibilities, in the order ``clean_full_text`` applies them:

1. **Mojibake repair** (:func:`fix_mojibake`) -- a scoped, byte-level fix for
   the ``â€<x>`` pattern that appears throughout this corpus. It is UTF-8
   punctuation (smart quotes, en/em dash, the non-breaking hyphen U+2011)
   that was decoded as cp1252/Latin-1 upstream during scraping. The fix is
   the exact inverse: re-encode the 3-character mojibake span as cp1252,
   then decode those bytes as UTF-8. Scoped to only the ``â€.`` span (not
   the whole document), so it cannot touch or corrupt any text outside
   that exact pattern.
2. **Structural normalization** -- reuses
   :func:`src.extraction.case_loader.normalize_case_text` (NFKC, control-char
   stripping, CRLF handling, multi-space collapsing, run-of-3+-newlines ->
   exactly one blank line). The same function Phase 1/2 already uses, so
   cleaned RAG text and classification's own text agree on this baseline.
3. **Line joining** (:func:`join_wrapped_lines`) -- the source HTML wraps a
   single visual paragraph at a fixed character width using single ``\\n``
   breaks; the double ``\\n\\n`` is the *only* real paragraph boundary
   (confirmed against the corpus: every genuine structural break -- case
   title / court / bench / party block / heading -- is separated by a
   blank line, while sentences and even party names get broken by a single
   ``\\n`` mid-phrase). So: split on blank lines, join each paragraph's
   internal single newlines with a space, rejoin paragraphs with a blank
   line. This is the one step that cannot be pulled from the existing
   classification code, because classification's own text never gets
   reflowed this way -- it just embeds the wrapped text as-is.

This module also derives two of the requested metadata fields that *are*
reliably derivable from existing data without guessing:

* :func:`extract_disposition` -- the same tail-of-document, disposition-line
  pattern match validated earlier against this exact corpus (see the
  "how many dismissed vs accepted" read-only audit). ~98% match rate on
  the full 2,088-document set; returns ``None`` rather than a forced guess
  for the remainder (citation-list/footnote tails, or a handful of
  documents whose original scrape cuts off mid-word).
* :func:`derive_court_location` -- a closed safelist of Pakistani
  province/city/territory names. ``document_representations.court`` mixes
  bare locations ("Lahore"), bench-qualified locations ("Lahore (Multan
  Bench)"), institution names ("Supreme Court of Pakistan"), and a small
  amount of pre-existing upstream extraction noise ("emphasis added", a
  misplaced statute clause, "Karuchil" for Karachi). Rather than guess-
  correct typos or infer a location for an institution name, this returns
  a location only when a known name appears as a substring, and returns
  the *bench* city when present (the bench is where the case actually
  sat), else ``None``.
"""

from __future__ import annotations

import re

from src.extraction.case_loader import normalize_case_text
from src.rag_prep.statute_cleaner import extract_provisions_cited, extract_statutes_cited

# The exact metadata shape var/rag/processed/<doc_id>.json carries.
REQUIRED_METADATA_FIELDS = (
    "doc_id",
    "case_title",
    "citation",
    "court",
    "court_location",
    "decision_date",
    "judges",
    "case_number",
    "primary_domain",
    "classification_status",
    "disposition",
    "statutes_cited",
    "provisions_cited",
    "source_relpath",
    "content_hash",
)

# ---------------------------------------------------------------------------
# Mojibake repair
# ---------------------------------------------------------------------------

# Every mojibake instance observed in this corpus is exactly "â€" followed by
# one more character -- the 3-byte UTF-8 encoding of a codepoint in the
# General Punctuation block (U+2010-U+2026: hyphens, dashes, smart quotes,
# ellipsis) each decoded as one cp1252 character. Matching only this 2+1
# character span means a document that also contains unrelated, already-
# correct Unicode is never touched outside the exact broken spans.
_MOJIBAKE_RE = re.compile("â€.")


def fix_mojibake(text: str) -> str:
    """Repair the ``â€<x>`` double-encoding artifact, scoped and reversible.

    A span that doesn't round-trip cleanly through cp1252->utf-8 (i.e. isn't
    actually this artifact) is left untouched rather than guessed at.
    """

    if not text or "â€" not in text:
        return text

    def _repair(match: re.Match) -> str:
        chunk = match.group(0)
        try:
            return chunk.encode("cp1252").decode("utf-8")
        except (UnicodeEncodeError, UnicodeDecodeError):
            return chunk

    return _MOJIBAKE_RE.sub(_repair, text)


# ---------------------------------------------------------------------------
# Line joining
# ---------------------------------------------------------------------------


def join_wrapped_lines(text: str) -> str:
    """Join single-newline-wrapped lines within a paragraph into one line.

    Assumes ``text`` has already had runs of 3+ newlines collapsed to
    exactly two (i.e. has gone through :func:`normalize_case_text` first),
    so a blank line reliably means "paragraph boundary" and a lone ``\\n``
    reliably means "the source wrapped here, not a real break".
    """

    if not text:
        return text

    paragraphs = text.split("\n\n")
    joined = []
    for para in paragraphs:
        # Single newlines within the paragraph are wrap artifacts -> space.
        line = " ".join(line.strip() for line in para.split("\n"))
        line = re.sub(r" {2,}", " ", line).strip()
        if line:
            joined.append(line)
    return "\n\n".join(joined)


def clean_full_text(text: str) -> str:
    """Full conservative cleaning pass: mojibake -> normalize -> line-join."""

    if not text:
        return ""

    repaired = fix_mojibake(text)
    normalized = normalize_case_text(repaired)
    return join_wrapped_lines(normalized)


# ---------------------------------------------------------------------------
# Disposition extraction
# ---------------------------------------------------------------------------

# Priority-ordered; compound outcomes checked before plain ones so "partly
# dismissed" isn't miscounted as plain DISMISSED. Validated against this
# corpus's actual disposition-line convention: a reporter's editorial code
# followed by a one/two-word outcome, e.g. "MQ/133/P Petition\ndismissed."
_DISPOSITION_PATTERNS: list[tuple[str, re.Pattern]] = [
    ("partly_allowed", re.compile(r"\bpart(?:ly|ially)\b.{0,25}\b(allowed|accepted|granted)\b")),
    ("partly_dismissed", re.compile(r"\bpart(?:ly|ially)\b.{0,25}\bdismissed\b")),
    ("dismissed", re.compile(r"\b(dismiss(?:ed)?|rejected|refused|disallowed|dropped)\b")),
    ("allowed", re.compile(r"\b(allowed|accepted|granted|decreed)\b")),
    ("remanded", re.compile(r"\bremand(?:ed)?\b")),
    ("transferred", re.compile(r"\btransferred\b")),
    ("returned", re.compile(r"\breturned\b")),
    ("withdrawn", re.compile(r"\bwithdrawn\b")),
    ("abated", re.compile(r"\babated\b")),
    ("infructuous", re.compile(r"\binfructuous\b")),
    ("objection_sustained", re.compile(r"\bobjection\b.{0,15}\bsustained\b")),
    ("set_aside", re.compile(r"\border\b.{0,10}\bset aside\b")),
    ("disposed_accordingly", re.compile(r"\bdisposed\b|\baccordingly\b")),
    ("converted", re.compile(r"\bconverted\b")),
]

# A handful of documents in this corpus had their final word truncated in
# the *original* scrape (verified against the raw case.html directly, e.g.
# "...Petition\ndismisse<o:p></o:p>" with the source itself missing the
# trailing "d."). The intent is unambiguous from the stem, so these are
# recovered rather than left as None.
_DISPOSITION_STEMS: list[tuple[str, list[str]]] = [
    ("dismissed", ["dismisse", "dismiss"]),
    ("allowed", ["allowe", "accepte", "grante", "decree"]),
    ("remanded", ["remande"]),
    ("disposed_accordingly", ["accordingl", "dispose"]),
    ("transferred", ["transferre"]),
]

_TAIL_LINES = 6


def _tail_lines(text: str, n: int) -> str:
    lines = [line.strip() for line in text.strip().split("\n") if line.strip()]
    return " ".join(lines[-n:])


def extract_disposition(cleaned_text: str) -> str | None:
    """The case's final disposition, read from the last lines of the text.

    Returns a normalized label (``"dismissed"``, ``"allowed"``, ``"remanded"``,
    etc.) or ``None`` when no disposition can be reliably identified -- never
    a forced guess. See the module docstring for validation history.
    """

    if not cleaned_text:
        return None

    tail = _tail_lines(cleaned_text, _TAIL_LINES).lower()
    for label, pattern in _DISPOSITION_PATTERNS:
        if pattern.search(tail):
            return label

    last_word_matches = re.findall(r"[a-z]+", tail)
    last_word = last_word_matches[-1] if last_word_matches else ""
    for label, stems in _DISPOSITION_STEMS:
        if last_word in stems:
            return label

    return None


# ---------------------------------------------------------------------------
# Court location
# ---------------------------------------------------------------------------

# Longest-name-first so "Gilgit-Baltistan" isn't shadowed by a shorter
# unrelated substring, and AJ&K variants map to one canonical spelling.
_KNOWN_LOCATIONS: list[tuple[str, str]] = [
    ("gilgit-baltistan", "Gilgit-Baltistan"),
    ("balochistan", "Balochistan"),
    ("islamabad", "Islamabad"),
    ("peshawar", "Peshawar"),
    ("rawalpindi", "Rawalpindi"),
    ("bahawalpur", "Bahawalpur"),
    ("multan", "Multan"),
    ("hyderabad", "Hyderabad"),
    ("sukkur", "Sukkur"),
    ("larkana", "Larkana"),
    ("abbottabad", "Abbottabad"),
    ("abbotabad", "Abbottabad"),
    ("mingora", "Mingora"),
    ("mignora", "Mingora"),
    ("mangora", "Mingora"),
    ("bannu", "Bannu"),
    ("d.i. khan", "D.I. Khan"),
    ("d.i.khan", "D.I. Khan"),
    ("turbat", "Turbat"),
    ("lahore", "Lahore"),
    ("karachi", "Karachi"),
    ("quetta", "Quetta"),
    ("sindh", "Sindh"),
    ("punjab", "Punjab"),
    ("aj&k", "Azad Jammu and Kashmir"),
    ("ajk", "Azad Jammu and Kashmir"),
    ("azad kashmir", "Azad Jammu and Kashmir"),
    ("azad jammu", "Azad Jammu and Kashmir"),
]

_BENCH_RE = re.compile(r"\(([^)]+?)\s+Bench\)", re.IGNORECASE)


def derive_court_location(court: str | None) -> str | None:
    """A location name conservatively derived from ``court``, or ``None``.

    Deliberately does not correct spelling ("Karuchil", "Balolchistan") or
    infer a seat for a bare institution name ("Supreme Court of Pakistan")
    -- both would be guessing rather than extracting. If ``court`` names a
    bench, the bench's city is preferred (that's where the case actually
    sat), falling back to the main value otherwise.
    """

    if not court:
        return None

    bench_match = _BENCH_RE.search(court)
    if bench_match:
        bench_text = bench_match.group(1).lower()
        for needle, canonical in _KNOWN_LOCATIONS:
            if needle in bench_text:
                return canonical

    lowered = court.lower()
    for needle, canonical in _KNOWN_LOCATIONS:
        if needle in lowered:
            return canonical

    return None


# ---------------------------------------------------------------------------
# Metadata assembly
# ---------------------------------------------------------------------------


def build_processed_metadata(raw_metadata: dict, cleaned_text: str, doc_id: str) -> dict:
    """The ``var/rag/processed/<doc_id>.json`` metadata shape.

    Reuses every existing value from ``raw_metadata`` (the staged
    ``var/rag/raw/<doc_id>.json`` metadata) unchanged -- this function never
    overwrites a value the classification pipeline already determined, it
    only adds the fields that are newly derivable here
    (``court_location``, ``disposition``, ``statutes_cited``,
    ``provisions_cited``). A field with nothing reliably available is
    ``None`` (or ``[]`` for the two list fields), never a guess.
    """

    m = raw_metadata
    return {
        "doc_id": doc_id,
        "case_title": m.get("title"),
        "citation": m.get("citation"),
        "court": m.get("court"),
        "court_location": derive_court_location(m.get("court")),
        "decision_date": m.get("decision_date"),
        "judges": m.get("judges") or [],
        "case_number": m.get("case_number"),
        "primary_domain": m.get("primary_domain"),
        "classification_status": m.get("classification_status"),
        "disposition": extract_disposition(cleaned_text),
        "statutes_cited": extract_statutes_cited(cleaned_text),
        "provisions_cited": extract_provisions_cited(cleaned_text),
        "source_relpath": m.get("source_relpath"),
        "content_hash": m.get("content_hash"),
    }
