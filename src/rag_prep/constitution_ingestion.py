"""Constitution of Pakistan ingestion -- Stage 0 of a SEPARATE, independent
Constitution pipeline (the Constitution is not a statute or a case-law
judgment and never goes through either of those pipelines' modules).

    Constitution PDF
        -> extract_text_layer()          [existing, generic PDF utility -- not Constitution-specific]
        -> raw ingestion JSON            [doc_id, source_file, metadata, pages, raw_text]

This stage does exactly one thing: turn the Constitution PDF into the
authoritative raw-text JSON representation that the later, not-yet-built
Constitution cleaning stage will consume. It does not clean, normalize,
remove TOC/headers/footers/page numbers, repair hyphenation, touch
brackets, parse Articles/clauses, chunk, or classify anything -- see
"SCOPE" in the task this module implements.

Reused from the existing codebase, because this is a genuinely generic
utility with no document-type-specific behavior baked in:
    - src.extraction.pdf_extractor.extract_text_layer()  (multi-backend
      PDF TEXT-LAYER extraction: fitz/pdfplumber/pypdf -- never OCR; the
      Constitution PDF is electronic/text-based, and OCR is explicitly
      out of scope for this stage regardless)
    - src.common.exceptions.PipelineError
    - src.common.logging_utils.get_logger
    - hashlib (stdlib)

PIPELINE ISOLATION
-------------------
No import of src.ingestion/, src.rag_prep's case_cleaner.py/structurer*.py/
chunk*.py/statute_*.py, src.classification/, or src.extraction.case_loader.py.
Independent entry point, independent output directory. Running this never
touches the case-law or statute pipelines' output directories, var/metadata.db,
or any of their code paths, and it is never imported by them.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

from src.common.exceptions import ExtractionError, PipelineError
from src.common.logging_utils import get_logger
from src.extraction.pdf_extractor import extract_text_layer

logger = get_logger(__name__)

# -- Configuration: explicit module-level constants, not hardcoded inline
# elsewhere in this file's logic -- mirrors statute_ingestion.py's own
# "input PDF path must be explicitly configurable in this module"
# convention. There is exactly one Constitution of Pakistan document (no
# directory of many, unlike the statute corpus), so this is a single
# file path, not a directory to glob.
INPUT_PDF_PATH = Path("/Users/ibrahim/Desktop/extracted_pdfs_temp/6926e060076ed_467.pdf")
OUTPUT_DIR = Path("var/rag/constitution_ingested")

DOCUMENT_TYPE = "constitution"
# No reliable PDF-embedded title field exists for this document (its own
# metadata title is empty; the PDF-extractor's heuristic-derived title is
# just "THE", the first line of the cover page) -- a fixed, curated
# constant, not inferred/guessed from PDF content, same rationale as
# statute_ingestion.py's DOMAIN being a fixed, curated value.
TITLE = "Constitution of Pakistan"

# No "directory of PDFs" to be relative to (there is only one file), so
# the doc_id is derived directly from the PDF's own resolved absolute
# path -- still deterministic and reproducible for the same input path,
# same sha256-based scheme as doc_id_for_statute()/doc_id_for_case().
DOC_ID_HASH_LENGTH = 24


class ConstitutionIngestionError(PipelineError):
    """Raised when the Constitution PDF cannot be safely ingested -- e.g.
    no extractable text layer at all. Never raised to mask a
    successful-but-empty result; see ingest_constitution_pdf()'s explicit
    check."""

    retryable = False


@dataclass(frozen=True)
class ConstitutionPage:
    page_number: int
    text: str

    def to_dict(self) -> dict:
        return {"page_number": self.page_number, "text": self.text}


def doc_id_for_constitution(pdf_path: Path) -> str:
    """Stable doc_id derived from the PDF's own resolved absolute path --
    same sha256-truncated scheme as the statute/case-law pipelines'
    doc_id functions, reimplemented independently here so this module has
    no dependency on either of their code."""

    return hashlib.sha256(Path(pdf_path).resolve().as_posix().encode("utf-8")).hexdigest()[:DOC_ID_HASH_LENGTH]


def ingest_constitution_pdf(pdf_path: Path = INPUT_PDF_PATH) -> dict:
    """Extracts the Constitution PDF into the raw ingestion JSON shape
    (doc_id, source_file, metadata, pages, raw_text).

    Direct text-layer extraction ONLY -- extract_text_layer() never
    invokes OCR. Page order is preserved by construction (the extractor
    already returns pages in document order); each page's text is stored
    exactly as the extractor returned it, with no cleaning, filtering, or
    re-flowing of any kind. ``raw_text`` is built by joining the
    page texts, in that same original order, with a blank line between
    pages (consistent with statute_ingestion.py's own raw-text join,
    applied here for the same reason: the extractor already emits one
    page per call and page boundaries carry no legal significance of
    their own that would require a different join).

    Raises ConstitutionIngestionError if the PDF has no usable text layer
    at all -- never returns a document with an empty/near-empty raw_text.
    """

    pdf_path = Path(pdf_path)
    result = extract_text_layer(pdf_path, max_pages=100_000)

    pages = [ConstitutionPage(page_number=p.page_number, text=p.text) for p in result.pages]
    raw_text = "\n\n".join(p.text for p in pages)
    if not raw_text.strip():
        raise ConstitutionIngestionError(
            f"{pdf_path.name}: extracted text layer is empty",
            phase="constitution_ingestion",
            doc_id=pdf_path.stem,
        )

    doc_id = doc_id_for_constitution(pdf_path)

    return {
        "doc_id": doc_id,
        "source_file": str(pdf_path),
        "metadata": {
            "document_type": DOCUMENT_TYPE,
            "title": TITLE,
        },
        "pages": [p.to_dict() for p in pages],
        "raw_text": raw_text,
    }


def ingest(input_pdf: Path = INPUT_PDF_PATH, output_dir: Path = OUTPUT_DIR) -> dict:
    """Ingests the Constitution PDF into ``output_dir/<doc_id>.json``.
    Never modifies ``input_pdf``. Returns {"processed", "failed", "error"}
    -- a single-document equivalent of the statute pipeline's
    ingest_directory() summary shape, since there is only one Constitution
    document to process."""

    input_pdf = Path(input_pdf)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        doc = ingest_constitution_pdf(input_pdf)
    except (ConstitutionIngestionError, ExtractionError) as exc:
        logger.error("Constitution ingestion failed for %s: %s", input_pdf.name, exc)
        return {"processed": 0, "failed": 1, "error": str(exc)}

    out_path = output_dir / f"{doc['doc_id']}.json"
    tmp_path = out_path.with_suffix(".json.tmp")
    tmp_path.write_text(json.dumps(doc, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp_path.replace(out_path)

    return {"processed": 1, "failed": 0, "error": None}


def main() -> None:  # pragma: no cover -- thin CLI wrapper
    summary = ingest()
    print(f"Constitution ingestion: {summary['processed']} processed, {summary['failed']} failed")
    if summary["error"]:
        print(f"  FAILED: {summary['error']}")


if __name__ == "__main__":  # pragma: no cover
    main()
