"""Text normalization for extracted (text-layer or OCR) page content.

Applied uniformly regardless of which extractor produced the text, so
downstream signature-building never has to know whether a page came from
:mod:`~src.extraction.pdf_extractor` or :mod:`~src.extraction.ocr_fallback`.

Responsibilities (per the Stage 1 spec):

* Removal of headers, footers, and watermarks -- detected as lines that
  repeat near-verbatim across a large fraction of the sampled pages.
* Correction of common OCR/PDF-extraction artifacts -- de-hyphenation of
  line-wrapped words, stray control characters, ligature/whitespace noise.
* Normalization of formatting -- Unicode NFKC normalization, whitespace
  collapsing.
* Deduplication of repeated content -- both the boilerplate-line removal
  above and collapsing of runs of identical blank lines.
"""

from __future__ import annotations

import re
import unicodedata
from collections import Counter
from dataclasses import replace

from src.extraction.pdf_extractor import PageText, page_quality

# Trailing hyphen at end-of-line followed by a lowercase continuation is
# treated as a line-wrap hyphenation artifact, e.g. "exam-\nple" -> "example".
# A hyphen followed by an uppercase letter or digit is left alone, since that
# is more likely a genuine compound/proper-noun hyphen ("Smith-\nJones" is
# ambiguous either way, but erring on the side of *not* merging avoids
# corrupting party/case names).
_DEHYPHENATE_RE = re.compile(r"(\w)-\n\s*([a-z])")

_CONTROL_CHAR_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
_MULTI_SPACE_RE = re.compile(r"[ \t]{2,}")
_MULTI_BLANK_LINE_RE = re.compile(r"\n{3,}")

_PAGE_NUMBER_LINE_RE = re.compile(
    r"^\s*(page\s+)?\d+(\s*/\s*\d+|\s+of\s+\d+)?\s*$", re.IGNORECASE
)


def normalize_text(text: str) -> str:
    """Normalize a single page's raw extracted text.

    Order matters: de-hyphenation must run before whitespace collapsing
    (it depends on the literal ``-\\n`` sequence), and NFKC normalization
    runs first so hyphen-like Unicode variants are canonicalized before
    the regex-based passes see them.
    """

    if not text:
        return ""

    normalized = unicodedata.normalize("NFKC", text)
    normalized = _CONTROL_CHAR_RE.sub("", normalized)
    normalized = _DEHYPHENATE_RE.sub(r"\1\2", normalized)
    normalized = _MULTI_SPACE_RE.sub(" ", normalized)
    normalized = "\n".join(line.rstrip() for line in normalized.split("\n"))
    normalized = _MULTI_BLANK_LINE_RE.sub("\n\n", normalized)
    return normalized.strip()


def _is_boilerplate_candidate(line: str) -> bool:
    """Lines short enough, or number-only enough, to plausibly be a
    header/footer/page-number rather than substantive body text."""

    stripped = line.strip()
    if not stripped:
        return False
    if _PAGE_NUMBER_LINE_RE.match(stripped):
        return True
    return len(stripped) <= 120


def strip_boilerplate_lines(
    pages: list[PageText], min_repeat_fraction: float = 0.5
) -> list[PageText]:
    """Drop lines that repeat verbatim across many pages (headers/footers).

    Only considered when there are at least 3 pages to compare -- with
    fewer samples, "repeats across half the pages" is too noisy a signal
    to safely strip content.
    """

    if len(pages) < 3:
        return pages

    line_page_counts: Counter[str] = Counter()
    for page in pages:
        seen_this_page: set[str] = set()
        for raw_line in page.text.split("\n"):
            line = raw_line.strip()
            if line and _is_boilerplate_candidate(line):
                seen_this_page.add(line)
        line_page_counts.update(seen_this_page)

    threshold = max(2, int(len(pages) * min_repeat_fraction))
    boilerplate = {
        line for line, count in line_page_counts.items() if count >= threshold
    }
    if not boilerplate:
        return pages

    cleaned_pages = []
    for page in pages:
        kept_lines = [
            raw_line
            for raw_line in page.text.split("\n")
            if raw_line.strip() not in boilerplate
        ]
        cleaned_text = "\n".join(kept_lines)
        char_count, alpha_ratio = page_quality(cleaned_text)
        cleaned_pages.append(
            replace(
                page,
                text=cleaned_text,
                char_count=char_count,
                alpha_ratio=alpha_ratio,
            )
        )
    return cleaned_pages


def strip_page_number_lines(pages: list[PageText]) -> list[PageText]:
    """Unconditionally drop pure page-number lines (e.g. "Page 3 of 10").

    Unlike :func:`strip_boilerplate_lines`, this doesn't require the line
    to repeat verbatim across pages -- a page-number-only line carries no
    document-signal value regardless of how many pages it appears on, and
    the actual digits necessarily differ page to page so it would never
    be caught by the repeat-based check.
    """

    cleaned_pages = []
    for page in pages:
        kept_lines = [
            raw_line
            for raw_line in page.text.split("\n")
            if not _PAGE_NUMBER_LINE_RE.match(raw_line.strip())
        ]
        cleaned_text = "\n".join(kept_lines)
        char_count, alpha_ratio = page_quality(cleaned_text)
        cleaned_pages.append(
            replace(page, text=cleaned_text, char_count=char_count, alpha_ratio=alpha_ratio)
        )
    return cleaned_pages


def clean_pages(
    pages: list[PageText], min_repeat_fraction: float = 0.5
) -> list[PageText]:
    """Full cleaning pass: normalize, drop page numbers, then strip boilerplate."""

    normalized = [replace(p, text=normalize_text(p.text)) for p in pages]
    for i, page in enumerate(normalized):
        char_count, alpha_ratio = page_quality(page.text)
        normalized[i] = replace(page, char_count=char_count, alpha_ratio=alpha_ratio)

    without_page_numbers = strip_page_number_lines(normalized)
    return strip_boilerplate_lines(
        without_page_numbers, min_repeat_fraction=min_repeat_fraction
    )


def build_body_preview(pages: list[PageText], char_limit: int = 20_000) -> str:
    """Concatenate cleaned page texts (in page order) up to ``char_limit`` chars.

    Stops at the last full page that fits rather than truncating mid-page,
    unless a single page alone exceeds the budget (in which case that page
    is hard-truncated so the preview is never silently empty).
    """

    ordered = sorted(pages, key=lambda p: p.page_number)
    chunks: list[str] = []
    used = 0

    for page in ordered:
        text = page.text.strip()
        if not text:
            continue

        remaining = char_limit - used
        if remaining <= 0:
            break

        if len(text) > remaining:
            if not chunks:
                chunks.append(text[:remaining])
                used = char_limit
            break

        chunks.append(text)
        used += len(text) + 2  # account for the "\n\n" join below

    return "\n\n".join(chunks).strip()