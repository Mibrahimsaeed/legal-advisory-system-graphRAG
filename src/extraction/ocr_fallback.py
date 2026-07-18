
from __future__ import annotations

from pathlib import Path

from PIL import Image

from src.common.exceptions import ExtractionError
from src.common.logging_utils import get_logger
from src.extraction.pdf_extractor import (
    ExtractionResult,
    PageText,
    heading_candidates_from_text,
    page_quality,
)

logger = get_logger(__name__)

try:  # pragma: no cover
    import fitz  # PyMuPDF
except ImportError:  # pragma: no cover
    fitz = None  # type: ignore[assignment]

try:  # pragma: no cover
    import pypdfium2 as pdfium
except ImportError:  # pragma: no cover
    pdfium = None  # type: ignore[assignment]

try:  # pragma: no cover
    import pytesseract
except ImportError:  # pragma: no cover
    pytesseract = None  # type: ignore[assignment]


DEFAULT_DPI = 300
DEFAULT_LANG = "eng"


def available() -> bool:
    """Whether OCR can run at all in this environment."""

    return pytesseract is not None and (fitz is not None or pdfium is not None)


def _render_with_fitz(pdf_path: Path, page_number: int, dpi: int) -> Image.Image:
    doc = fitz.open(str(pdf_path))
    try:
        page = doc.load_page(page_number - 1)
        zoom = dpi / 72.0
        pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom))
        return Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
    finally:
        doc.close()


def _render_with_pdfium(pdf_path: Path, page_number: int, dpi: int) -> Image.Image:
    pdf = pdfium.PdfDocument(str(pdf_path))
    try:
        page = pdf[page_number - 1]
        scale = dpi / 72.0
        bitmap = page.render(scale=scale)
        return bitmap.to_pil()
    finally:
        pdf.close()


def _render_page(pdf_path: Path, page_number: int, dpi: int) -> Image.Image:
    if fitz is not None:
        return _render_with_fitz(pdf_path, page_number, dpi)
    if pdfium is not None:
        return _render_with_pdfium(pdf_path, page_number, dpi)
    raise ExtractionError(
        "No PDF rendering backend available for OCR (need fitz or pypdfium2)",
        phase="ocr",
    )


def extract_via_ocr(
    pdf_path: str | Path,
    page_numbers: list[int],
    dpi: int = DEFAULT_DPI,
    lang: str = DEFAULT_LANG,
) -> ExtractionResult:
    """Run OCR over ``page_numbers`` (1-indexed) and return an ExtractionResult.

    Pages that fail to render or OCR are recorded with an empty
    :class:`~src.extraction.pdf_extractor.PageText` and a warning, rather
    than aborting the whole call -- a handful of unreadable pages
    shouldn't sink an otherwise-usable document signature.

    Raises:
        ExtractionError: if no OCR engine/renderer is available at all, or
            if the underlying Tesseract binary is missing.
    """

    if pytesseract is None:
        raise ExtractionError(
            "pytesseract is not installed; cannot run OCR fallback",
            phase="ocr",
        )
    if fitz is None and pdfium is None:
        raise ExtractionError(
            "No PDF rendering backend available for OCR (need fitz or pypdfium2)",
            phase="ocr",
        )

    pdf_path = Path(pdf_path)
    pages: list[PageText] = []
    warnings: list[str] = []
    title: str | None = None
    heading_fallback: list[str] = []

    for page_number in sorted(page_numbers):
        try:
            image = _render_page(pdf_path, page_number, dpi)
            text = pytesseract.image_to_string(image, lang=lang) or ""
        except Exception as exc:
            if "tesseract" in type(exc).__name__.lower() or "TesseractNotFound" in str(exc):
                raise ExtractionError(
                    "Tesseract OCR engine not found on PATH",
                    phase="ocr",
                    doc_id=pdf_path.stem,
                    cause=exc,
                ) from exc

            msg = f"OCR failed for page {page_number}: {type(exc).__name__}: {exc}"
            logger.warning(msg)
            warnings.append(msg)
            pages.append(
                PageText(
                    page_number=page_number,
                    text="",
                    char_count=0,
                    alpha_ratio=0.0,
                    source="ocr",
                )
            )
            continue

        char_count, alpha_ratio = page_quality(text)
        pages.append(
            PageText(
                page_number=page_number,
                text=text,
                char_count=char_count,
                alpha_ratio=alpha_ratio,
                source="ocr",
            )
        )
        heading_fallback.extend(heading_candidates_from_text(text))

        if title is None and page_number == min(page_numbers):
            first_lines = [l.strip() for l in text.splitlines() if l.strip()]
            if first_lines:
                title = first_lines[0]

    logger.info(
        "OCR fallback complete for %s: %d page(s) attempted, %d warning(s)",
        pdf_path.name,
        len(page_numbers),
        len(warnings),
    )

    return ExtractionResult(
        pages=pages,
        title=title,
        toc=heading_fallback,
        engine="ocr",
        total_page_count=None,
        warnings=warnings,
    )