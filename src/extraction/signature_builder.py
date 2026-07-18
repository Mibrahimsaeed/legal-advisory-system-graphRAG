"""Orchestrates one document through the Stage 1 signature pipeline.

    text-layer extraction -> (OCR fallback for low-quality pages only)
        -> cleaning/normalization -> signature assembly

This is the single entry point :mod:`orchestration.dags.feature_extraction_flow`
calls per document; everything else in :mod:`src.extraction` is a building
block this module composes. Kept deliberately exception-safe: any failure
anywhere in the chain is captured into the returned
:class:`~src.extraction.signature.DocumentSignature` with
``extraction_status="failed"`` rather than propagating, so one bad PDF in
a batch of 500 never aborts the other 499.
"""

from __future__ import annotations

from pathlib import Path

from src.common.exceptions import ExtractionError
from src.common.logging_utils import get_logger, log_context
from src.extraction.cleaner import build_body_preview, clean_pages
from src.extraction.ocr_fallback import extract_via_ocr
from src.extraction.pdf_extractor import (
    DEFAULT_MAX_PAGES,
    PageText,
    extract_text_layer,
    low_quality_page_numbers,
)
from src.extraction.signature import DocumentSignature, compute_signature_hash

logger = get_logger(__name__)

DEFAULT_MIN_CHARS_PER_PAGE = 40.0
DEFAULT_MIN_ALPHA_RATIO = 0.6
DEFAULT_OCR_DPI = 300
DEFAULT_OCR_LANG = "eng"
DEFAULT_BODY_PREVIEW_CHAR_LIMIT = 20_000
# A document is considered "scanned" once at least this fraction of its
# sampled pages needed OCR (as opposed to, say, one stray low-quality page
# in an otherwise-digital document).
SCANNED_PAGE_RATIO = 0.5


def _failed_signature(
    doc_id: str, source_uri: str, batch_id: str | None, error: str
) -> DocumentSignature:
    return DocumentSignature(
        doc_id=doc_id,
        source_uri=source_uri,
        signature_hash="",
        is_scanned=False,
        extraction_status="failed",
        extractor_used="none",
        title=None,
        toc=[],
        body_preview="",
        pages_used=0,
        char_count=0,
        quality_score=0.0,
        batch_id=batch_id,
        error=error,
    )


def _merge_ocr_pages(
    text_pages: list[PageText], ocr_pages: list[PageText]
) -> list[PageText]:
    """Replace a text-layer page with its OCR'd counterpart only if the OCR
    pass actually recovered more content -- guards against OCR noise
    overwriting a text layer that was merely borderline, not truly empty."""

    ocr_by_page = {p.page_number: p for p in ocr_pages}
    merged = []
    for page in text_pages:
        ocr_page = ocr_by_page.get(page.page_number)
        if ocr_page is not None and ocr_page.char_count > page.char_count:
            merged.append(ocr_page)
        else:
            merged.append(page)
    return merged


def build_signature(
    doc_id: str,
    source_uri: str,
    pdf_path: str | Path,
    *,
    max_pages: int = DEFAULT_MAX_PAGES,
    min_chars_per_page: float = DEFAULT_MIN_CHARS_PER_PAGE,
    min_alpha_ratio: float = DEFAULT_MIN_ALPHA_RATIO,
    ocr_dpi: int = DEFAULT_OCR_DPI,
    ocr_lang: str = DEFAULT_OCR_LANG,
    body_preview_char_limit: int = DEFAULT_BODY_PREVIEW_CHAR_LIMIT,
    batch_id: str | None = None,
) -> DocumentSignature:
    """Build a :class:`DocumentSignature` for a single pulled PDF.

    Never raises: any exception is converted into a ``failed`` signature
    so callers can persist a record either way and move on to the next
    document in the batch.
    """

    with log_context(doc_id=doc_id, phase="signature_build"):
        try:
            text_result = extract_text_layer(pdf_path, max_pages=max_pages)
        except ExtractionError as exc:
            logger.error("Text-layer extraction failed for %s: %s", doc_id, exc)
            return _failed_signature(doc_id, source_uri, batch_id, str(exc))
        except Exception as exc:  # pragma: no cover - unexpected/defensive
            logger.exception("Unexpected error extracting text layer for %s", doc_id)
            return _failed_signature(
                doc_id, source_uri, batch_id, f"{type(exc).__name__}: {exc}"
            )

        pages = text_result.pages
        title = text_result.title
        toc = text_result.toc
        engine_used = text_result.engine
        is_scanned = False
        notes: list[str] = []

        flagged = low_quality_page_numbers(pages, min_chars_per_page, min_alpha_ratio)
        if flagged:
            try:
                ocr_result = extract_via_ocr(
                    pdf_path, flagged, dpi=ocr_dpi, lang=ocr_lang
                )
            except ExtractionError as exc:
                # OCR being unavailable/failing shouldn't sink the document --
                # fall back to whatever text layer we already have and note it.
                logger.warning("OCR fallback unavailable for %s: %s", doc_id, exc)
                notes.append(f"ocr_unavailable: {exc}")
                ocr_result = None

            if ocr_result is not None:
                notes.extend(f"ocr_page_warning: {w}" for w in ocr_result.warnings)
                pages = _merge_ocr_pages(pages, ocr_result.pages)
                is_scanned = (len(flagged) / len(text_result.pages)) >= SCANNED_PAGE_RATIO
                engine_used = (
                    "ocr" if len(flagged) == len(text_result.pages) else "mixed"
                )
                if not title:
                    title = ocr_result.title
                if not toc:
                    toc = ocr_result.toc

        cleaned_pages = clean_pages(pages)
        body_preview = build_body_preview(
            cleaned_pages, char_limit=body_preview_char_limit
        )

        if not cleaned_pages or not body_preview:
            error = "; ".join(notes) if notes else "No extractable content after cleaning"
            logger.warning("Signature build produced no content for %s", doc_id)
            return _failed_signature(doc_id, source_uri, batch_id, error)

        still_flagged = low_quality_page_numbers(
            cleaned_pages, min_chars_per_page, min_alpha_ratio
        )
        if still_flagged and notes:
            status = "ok_partial_ocr"
        elif is_scanned:
            status = "ok_ocr"
        else:
            status = "ok"

        total_chars = sum(p.char_count for p in cleaned_pages)
        quality_score = sum(p.alpha_ratio for p in cleaned_pages) / len(cleaned_pages)
        signature_hash = compute_signature_hash(title, toc, body_preview)

        signature = DocumentSignature(
            doc_id=doc_id,
            source_uri=source_uri,
            signature_hash=signature_hash,
            is_scanned=is_scanned,
            extraction_status=status,
            extractor_used=engine_used,
            title=title,
            toc=toc,
            body_preview=body_preview,
            pages_used=len(cleaned_pages),
            char_count=total_chars,
            quality_score=round(quality_score, 4),
            batch_id=batch_id,
            error="; ".join(notes) if notes else None,
        )

        logger.info(
            "Signature built for %s: status=%s engine=%s is_scanned=%s pages=%d chars=%d",
            doc_id,
            signature.extraction_status,
            signature.extractor_used,
            signature.is_scanned,
            signature.pages_used,
            signature.char_count,
        )
        return signature