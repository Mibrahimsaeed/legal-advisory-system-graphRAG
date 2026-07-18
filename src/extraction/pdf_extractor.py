

from __future__ import annotations
import re

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from src.common.exceptions import ExtractionError
from src.common.logging_utils import get_logger

logger = get_logger(__name__)

try:  # pragma: no cover - import guard exercised implicitly by env
    import fitz  # PyMuPDF
except ImportError:  # pragma: no cover
    fitz = None  # type: ignore[assignment]

try:  # pragma: no cover
    import pdfplumber
except ImportError:  # pragma: no cover
    pdfplumber = None  # type: ignore[assignment]

try:  # pragma: no cover
    from pypdf import PdfReader
except ImportError:  # pragma: no cover
    PdfReader = None  # type: ignore[assignment]


DEFAULT_MAX_PAGES = 15

# A line is a heading *candidate* (plain-text heuristic, used when no font
# metadata is available) if it is short and looks structurally like a
# legal-document heading rather than prose.
_HEADING_MAX_LEN = 90
_NUMBERED_HEADING_RE = re.compile(
    r"^\s*("
    r"(article|section|chapter|part|schedule|annex|clause)\s+[ivxlcdm0-9]+"
    r"|\d+(\.\d+)*[.)]?\s+\S"
    r")",
    re.IGNORECASE,
)


@dataclass
class PageText:
    """One page's worth of extracted text plus quality metrics."""

    page_number: int  # 1-indexed
    text: str
    char_count: int
    alpha_ratio: float
    source: str = "text_layer"  # "text_layer" | "ocr"


@dataclass
class ExtractionResult:
    """Everything :func:`extract_text_layer` (or the OCR fallback) produces."""

    pages: list[PageText]
    title: str | None
    toc: list[str]
    engine: str  # "fitz" | "pdfplumber" | "pypdf" | "ocr"
    total_page_count: int | None = None
    warnings: list[str] = field(default_factory=list)


def page_quality(text: str) -> tuple[int, float]:
    """Return ``(char_count, alpha_ratio)`` for a page's raw text.

    ``alpha_ratio`` is the fraction of non-whitespace characters that are
    alphabetic. Garbled OCR-needed text (e.g. a scanned page that yielded
    a text layer of stray glyphs/ligature junk) tends to have a very low
    ratio; empty/near-empty pages have ``char_count`` near zero.
    """

    stripped = text.strip()
    char_count = len(stripped)
    if char_count == 0:
        return 0, 0.0

    non_ws = [c for c in stripped if not c.isspace()]
    if not non_ws:
        return char_count, 0.0

    alpha = sum(1 for c in non_ws if c.isalpha())
    return char_count, alpha / len(non_ws)


def heading_candidates_from_text(text: str) -> list[str]:
    """Plain-text heading heuristic used when no font-size metadata exists."""

    candidates: list[str] = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or len(line) > _HEADING_MAX_LEN:
            continue

        is_numbered = bool(_NUMBERED_HEADING_RE.match(line))
        is_all_caps = line.isupper() and any(c.isalpha() for c in line)
        is_title_case_short = (
            len(line.split()) <= 8
            and line[:1].isupper()
            and not line.endswith((".", ",", ";"))
        )

        if is_numbered or is_all_caps or is_title_case_short:
            candidates.append(line)

    return candidates


def _heading_candidates_from_spans(
    spans: list[tuple[str, float]], body_size: float
) -> list[str]:
    """Font-size-based heading heuristic for backends exposing span sizes.

    ``spans`` is a list of ``(text, font_size)`` pairs for one page.
    Anything meaningfully larger than the page's modal ("body") font size,
    and short enough to be a heading rather than a paragraph, is treated
    as a heading candidate.
    """

    candidates: list[str] = []
    threshold = body_size * 1.15 if body_size else 0.0
    for text, size in spans:
        line = text.strip()
        if not line or len(line) > _HEADING_MAX_LEN:
            continue
        if size > threshold:
            candidates.append(line)
    return candidates


_JUNK_TITLES = {"untitled", "untitled document", "unknown", ""}


def _clean_title(raw: str | None) -> str | None:
    """Discard placeholder PDF-producer metadata titles (e.g. "Untitled")."""

    if not raw:
        return None
    stripped = raw.strip()
    if stripped.lower() in _JUNK_TITLES:
        return None
    return stripped


def _modal_font_size(sizes: list[float]) -> float:
    if not sizes:
        return 0.0
    rounded = [round(s, 1) for s in sizes]
    return max(set(rounded), key=rounded.count)


def _group_words_into_lines(
    words: list[tuple[str, float, float, float]],
    tolerance: float = 3.0,
) -> list[tuple[str, float]]:
    """Group ``(text, size, top, x0)`` word tuples into visual lines.

    Font-size heading detection needs to compare *lines*, not individual
    words/spans -- otherwise a multi-word heading like "1. Definitions"
    gets reported as two separate "heading" fragments. Words are grouped
    by proximity in their vertical (``top``) position, then joined
    left-to-right by horizontal (``x0``) position.

    Returns a list of ``(line_text, max_font_size_in_line)`` tuples.
    """

    if not words:
        return []

    ordered = sorted(words, key=lambda w: (w[2], w[3]))
    lines: list[list[tuple[str, float, float, float]]] = []

    for word in ordered:
        _, _, top, _ = word
        if lines and abs(lines[-1][-1][2] - top) <= tolerance:
            lines[-1].append(word)
        else:
            lines.append([word])

    result = []
    for line_words in lines:
        line_words.sort(key=lambda w: w[3])
        text = " ".join(w[0] for w in line_words).strip()
        max_size = max(w[1] for w in line_words)
        if text:
            result.append((text, max_size))
    return result


# --------------------------------------------------------------------------
# Backend: PyMuPDF (fitz)
# --------------------------------------------------------------------------


def _extract_with_fitz(pdf_path: Path, max_pages: int) -> ExtractionResult:
    doc = fitz.open(str(pdf_path))
    try:
        total_pages = doc.page_count
        n = min(max_pages, total_pages)

        title = _clean_title((doc.metadata or {}).get("title"))

        toc_entries = doc.get_toc(simple=True) or []
        toc = [entry[1].strip() for entry in toc_entries if entry[1].strip()]

        pages: list[PageText] = []
        heading_fallback: list[str] = []

        for i in range(n):
            page = doc.load_page(i)
            text = page.get_text("text") or ""
            char_count, alpha_ratio = page_quality(text)
            pages.append(
                PageText(
                    page_number=i + 1,
                    text=text,
                    char_count=char_count,
                    alpha_ratio=alpha_ratio,
                    source="text_layer",
                )
            )

            if not toc:
                span_data = page.get_text("dict")
                words: list[tuple[str, float, float, float]] = []
                for block in span_data.get("blocks", []):
                    for line in block.get("lines", []):
                        line_spans = line.get("spans", [])
                        if not line_spans:
                            continue
                        line_text = "".join(s.get("text", "") for s in line_spans).strip()
                        if not line_text:
                            continue
                        max_size = max(s.get("size", 0.0) for s in line_spans)
                        top = line_spans[0].get("bbox", [0, 0, 0, 0])[1]
                        x0 = line_spans[0].get("bbox", [0, 0, 0, 0])[0]
                        words.append((line_text, max_size, top, x0))
                sizes = [w[1] for w in words]
                body_size = _modal_font_size(sizes)
                # Each entry here is already a full line, so feed
                # ``_group_words_into_lines`` a singleton-per-line grouping
                # by treating every distinct ``top`` as its own line.
                line_spans_grouped = [(text, size) for text, size, _, _ in words]
                heading_fallback.extend(
                    _heading_candidates_from_spans(line_spans_grouped, body_size)
                )

            if not title and i == 0:
                first_lines = [l.strip() for l in text.splitlines() if l.strip()]
                if first_lines:
                    title = first_lines[0]

        if not toc:
            toc = heading_fallback

        return ExtractionResult(
            pages=pages,
            title=title,
            toc=toc,
            engine="fitz",
            total_page_count=total_pages,
        )
    finally:
        doc.close()


# --------------------------------------------------------------------------
# Backend: pdfplumber
# --------------------------------------------------------------------------


def _extract_with_pdfplumber(pdf_path: Path, max_pages: int) -> ExtractionResult:
    with pdfplumber.open(str(pdf_path)) as pdf:
        total_pages = len(pdf.pages)
        n = min(max_pages, total_pages)

        title = None
        try:
            title = _clean_title((pdf.metadata or {}).get("Title"))
        except Exception:  # pragma: no cover - defensive, metadata is best-effort
            title = None

        pages: list[PageText] = []
        heading_fallback: list[str] = []

        for i in range(n):
            page = pdf.pages[i]
            text = page.extract_text() or ""
            char_count, alpha_ratio = page_quality(text)
            pages.append(
                PageText(
                    page_number=i + 1,
                    text=text,
                    char_count=char_count,
                    alpha_ratio=alpha_ratio,
                    source="text_layer",
                )
            )

            try:
                raw_words = page.extract_words(extra_attrs=["size"])
            except Exception:  # pragma: no cover - malformed page content stream
                raw_words = []

            word_tuples = [
                (w["text"], w.get("size", 0.0), w.get("top", 0.0), w.get("x0", 0.0))
                for w in raw_words
                if w.get("text")
            ]
            grouped_lines = _group_words_into_lines(word_tuples)
            sizes = [size for _, size in grouped_lines]
            body_size = _modal_font_size(sizes)
            heading_fallback.extend(
                _heading_candidates_from_spans(grouped_lines, body_size)
            )

            if not title and i == 0:
                first_lines = [l.strip() for l in text.splitlines() if l.strip()]
                if first_lines:
                    title = first_lines[0]

        return ExtractionResult(
            pages=pages,
            title=title,
            toc=heading_fallback,
            engine="pdfplumber",
            total_page_count=total_pages,
        )


# --------------------------------------------------------------------------
# Backend: pypdf (text-only fallback, no font metadata)
# --------------------------------------------------------------------------


def _flatten_pypdf_outline(outline: Any) -> list[str]:
    titles: list[str] = []
    for item in outline or []:
        if isinstance(item, list):
            titles.extend(_flatten_pypdf_outline(item))
        else:
            raw_title = getattr(item, "title", None)
            if raw_title:
                titles.append(str(raw_title).strip())
    return titles


def _extract_with_pypdf(pdf_path: Path, max_pages: int) -> ExtractionResult:
    reader = PdfReader(str(pdf_path))
    total_pages = len(reader.pages)
    n = min(max_pages, total_pages)

    title = None
    if reader.metadata is not None:
        title = getattr(reader.metadata, "title", None) or None

    toc: list[str] = []
    try:
        toc = _flatten_pypdf_outline(reader.outline)
    except Exception:  # pragma: no cover - some PDFs have malformed outlines
        toc = []

    pages: list[PageText] = []
    heading_fallback: list[str] = []

    for i in range(n):
        try:
            text = reader.pages[i].extract_text() or ""
        except Exception:  # pragma: no cover - corrupt page object
            text = ""

        char_count, alpha_ratio = page_quality(text)
        pages.append(
            PageText(
                page_number=i + 1,
                text=text,
                char_count=char_count,
                alpha_ratio=alpha_ratio,
                source="text_layer",
            )
        )

        if not toc:
            heading_fallback.extend(heading_candidates_from_text(text))

        if not title and i == 0:
            first_lines = [l.strip() for l in text.splitlines() if l.strip()]
            if first_lines:
                title = first_lines[0]

    if not toc:
        toc = heading_fallback

    return ExtractionResult(
        pages=pages,
        title=title,
        toc=toc,
        engine="pypdf",
        total_page_count=total_pages,
    )


# --------------------------------------------------------------------------
# Public API
# --------------------------------------------------------------------------


def available_backend() -> str | None:
    """Name of the backend :func:`extract_text_layer` would use, or ``None``."""

    if fitz is not None:
        return "fitz"
    if pdfplumber is not None:
        return "pdfplumber"
    if PdfReader is not None:
        return "pypdf"
    return None


def extract_text_layer(
    pdf_path: str | Path, max_pages: int = DEFAULT_MAX_PAGES
) -> ExtractionResult:
    """Extract title/TOC/first-N-pages text via the best available backend.

    Raises:
        ExtractionError: if no supported PDF library is importable, or if
            the file cannot be opened/parsed by the selected backend.
    """

    pdf_path = Path(pdf_path)
    backend = available_backend()

    if backend is None:
        raise ExtractionError(
            "No PDF text-extraction backend available "
            "(none of fitz/pdfplumber/pypdf could be imported)",
            phase="extract",
            doc_id=pdf_path.stem,
        )

    try:
        if backend == "fitz":
            result = _extract_with_fitz(pdf_path, max_pages)
        elif backend == "pdfplumber":
            result = _extract_with_pdfplumber(pdf_path, max_pages)
        else:
            result = _extract_with_pypdf(pdf_path, max_pages)
    except Exception as exc:
        raise ExtractionError(
            f"Text-layer extraction failed using backend={backend}",
            phase="extract",
            doc_id=pdf_path.stem,
            cause=exc,
        ) from exc

    logger.info(
        "Extracted text layer for %s via %s: %d page(s), title=%r, %d TOC entr(y/ies)",
        pdf_path.name,
        result.engine,
        len(result.pages),
        result.title,
        len(result.toc),
    )
    return result


def low_quality_page_numbers(
    pages: list[PageText],
    min_chars_per_page: float = 40.0,
    min_alpha_ratio: float = 0.6,
) -> list[int]:
    """1-indexed page numbers whose text layer looks empty or garbled."""

    flagged = []
    for page in pages:
        if page.char_count < min_chars_per_page or page.alpha_ratio < min_alpha_ratio:
            flagged.append(page.page_number)
    return flagged


def looks_low_quality(
    pages: list[PageText],
    min_chars_per_page: float = 40.0,
    min_alpha_ratio: float = 0.6,
    flagged_page_ratio: float = 0.5,
) -> bool:
    """True if enough pages look empty/garbled that OCR should be attempted.

    A document (rather than a single stray page) is considered a text-layer
    failure -- i.e. likely scanned -- when at least ``flagged_page_ratio``
    (default: half) of the sampled pages fail the per-page quality check,
    or when there are no pages at all.
    """

    if not pages:
        return True

    flagged = low_quality_page_numbers(pages, min_chars_per_page, min_alpha_ratio)
    return (len(flagged) / len(pages)) >= flagged_page_ratio