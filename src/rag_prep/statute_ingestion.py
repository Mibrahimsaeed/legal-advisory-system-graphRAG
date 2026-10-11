"""Statute ingestion -- Stage 0 of a SEPARATE, independent statute
pipeline (statutes are not case law and never go through the case-law
ingestion/cleaning/structuring/chunking/classification/review modules).

    PDF file
        -> extract_text_layer()          [existing, generic PDF utility -- not case-law-specific]
        -> raw ingestion JSON            [doc_id, metadata, full_text]

This stage does exactly one thing: turn a statute PDF into the
authoritative raw-text JSON representation that later, not-yet-built
statute stages (cleaning, Act/Chapter/Section structuring, chunking) will
consume. It does not clean, normalize, summarize, or classify the text,
and it does not run any case-law-specific logic -- see the module
docstring section "PIPELINE ISOLATION" below.

Reused from the existing codebase, because these are genuinely generic
utilities with no case-law-specific behavior baked in:
    - src.extraction.pdf_extractor.extract_text_layer()  (multi-backend
      PDF text extraction: fitz/pdfplumber/pypdf, already document-type
      agnostic)
    - src.common.exceptions.ExtractionError              (generic PipelineError)
    - src.common.logging_utils.get_logger
    - hashlib (stdlib)

NOT reused, deliberately: anything under src.ingestion/, src.rag_prep's
case_cleaner.py/structurer*.py/chunk*.py, src.classification/, or
src.extraction.case_loader.py -- all of those are case-law-specific
(folder-per-case discovery, HTML parsing, case metadata schema,
disposition/classification logic) and importing any of them here would
violate the required pipeline separation.

PIPELINE ISOLATION
-------------------
This module has its own entry point (``main()`` / ``ingest_directory()``),
its own output directory, its own doc_id scheme, and its own metadata
schema. Running it never touches any of the existing case-law output
directories (raw/processed/structured/chunked) or the metadata database,
or any case-law code path. The existing `cases` pipeline is never invoked from
here, and this module is never imported by it.

OCR is explicitly NOT used here (per the task's own restriction) -- a
PDF whose text layer looks empty/garbled (see
src.extraction.pdf_extractor.looks_low_quality) fails loudly
(StatuteIngestionError) rather than silently producing an empty or
guessed document.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import yaml

from src.common.exceptions import ExtractionError, PipelineError
from src.common.logging_utils import get_logger
from src.extraction.pdf_extractor import extract_text_layer

logger = get_logger(__name__)

# -- Configuration: the input/output paths are explicit module-level
# constants, not hardcoded inline elsewhere in this file's logic. A
# caller (or a future CLI wrapper) overrides them by passing explicit
# arguments to ingest_directory()/ingest_one_pdf() rather than editing
# this module, but a sensible, clearly-named default lives here per the
# task's "input PDF folder path must be explicitly configurable in this
# module" instruction.
INPUT_DIR = Path("/Users/ibrahim/Desktop/extracted_pdfs_temp/statues")
OUTPUT_DIR = Path("var/rag/statutes_ingested")

# Statutes are full Acts, not short judgments -- the case-law pipeline's
# PDF-extraction default (DEFAULT_MAX_PAGES=15 in pdf_extractor.py, tuned
# for short documents) would silently truncate a multi-chapter Act.
# This is intentionally a very large ceiling, not "unlimited" (an
# unbounded value could hang on a pathological file), chosen so no
# realistic statute PDF is ever cut short.
MAX_STATUTE_PAGES = 100_000

DOC_ID_HASH_LENGTH = 24  # same convention as src.extraction.case_loader.doc_id_for_case

# The curated domain value lives in config/statute_domain.yaml, not as a
# Python literal here -- see that file's header comment for why (this
# repo enforces, project-wide, that domain ids are never hardcoded in
# Python; config/domains.yaml is the case-law classification taxonomy's
# own separate registry, intentionally not shared with statutes).
_DOMAIN_CONFIG_PATH = Path(__file__).resolve().parents[2] / "config" / "statute_domain.yaml"
_domain_config = yaml.safe_load(_DOMAIN_CONFIG_PATH.read_text(encoding="utf-8"))
DOMAIN: str = _domain_config["domain"]
DOMAIN_SOURCE: str = _domain_config["domain_source"]
DOCUMENT_TYPE = "statute"


class StatuteIngestionError(PipelineError):
    """Raised when a statute PDF cannot be safely ingested -- e.g. no
    extractable text layer. Never raised to mask a successful-but-empty
    result; see ingest_one_pdf()'s explicit quality check."""

    retryable = False


@dataclass(frozen=True)
class StatuteMetadataOverride:
    """Explicitly-provided, curated metadata for one statute PDF --
    never inferred or guessed from the filename or PDF content. Loaded
    from an optional ``<pdf_stem>.metadata.json`` sidecar file sitting
    next to the PDF (see _load_metadata_override); absent fields stay
    ``None`` rather than being invented."""

    law_name: str | None = None
    year: str | None = None


def doc_id_for_statute(root: Path, pdf_path: Path) -> str:
    """Stable doc_id derived from the PDF's path *relative to root* --
    same scheme as the case-law pipeline's doc_id_for_case() (sha256 of
    the relative POSIX path, truncated), reimplemented independently
    here rather than imported, so this module has no dependency on
    case-law code at all."""

    rel = pdf_path.resolve().relative_to(Path(root).resolve()).as_posix()
    return hashlib.sha256(rel.encode("utf-8")).hexdigest()[:DOC_ID_HASH_LENGTH]


def _load_metadata_override(pdf_path: Path) -> StatuteMetadataOverride:
    """Reads ``<pdf_path.stem>.metadata.json`` next to the PDF, if
    present -- the only mechanism by which law_name/year are ever
    populated with anything beyond the PDF's own embedded title. A
    missing or unreadable sidecar is not an error; it just means those
    fields stay null, per "if metadata is unavailable, use null"."""

    sidecar = pdf_path.with_suffix("").with_suffix(".metadata.json")
    if not sidecar.exists():
        return StatuteMetadataOverride()
    try:
        raw = json.loads(sidecar.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        logger.warning("Could not read metadata sidecar %s -- ignoring it", sidecar)
        return StatuteMetadataOverride()
    if not isinstance(raw, dict):
        return StatuteMetadataOverride()
    return StatuteMetadataOverride(
        law_name=raw.get("law_name") if isinstance(raw.get("law_name"), str) else None,
        year=raw.get("year") if isinstance(raw.get("year"), str) else None,
    )


def ingest_one_pdf(pdf_path: Path, input_root: Path) -> dict:
    """Extracts one statute PDF into the raw ingestion JSON shape.

    Raises StatuteIngestionError if the PDF has no usable text layer --
    never returns a document with an empty/near-empty full_text.
    """

    result = extract_text_layer(pdf_path, max_pages=MAX_STATUTE_PAGES)

    # Deliberately NOT using pdf_extractor.looks_low_quality() here: its
    # default thresholds (e.g. >=40 chars/page) are tuned for detecting
    # scanned case-law judgments that need OCR, and would misfire on a
    # genuinely short-but-real statute page (a title page, a short
    # schedule heading). This stage's only failure condition is the
    # literal one the task asks for -- no extractable text at all --
    # checked directly below via full_text itself.

    # Page order preserved by construction (result.pages is already in
    # document order); joined, never re-flowed/re-wrapped/cleaned.
    full_text = "\n\n".join(page.text for page in result.pages)
    if not full_text.strip():
        raise StatuteIngestionError(
            f"{pdf_path.name}: extracted text layer is empty",
            phase="statute_ingestion",
            doc_id=pdf_path.stem,
        )

    doc_id = doc_id_for_statute(input_root, pdf_path)
    override = _load_metadata_override(pdf_path)
    source_relpath = pdf_path.resolve().relative_to(Path(input_root).resolve()).as_posix()

    metadata = {
        "law_name": override.law_name or result.title,  # PDF's own embedded title, else sidecar, else null
        "year": override.year,  # no reliable PDF-embedded year field exists; sidecar-only
        "document_type": DOCUMENT_TYPE,
        "domain": DOMAIN,
        "domain_source": DOMAIN_SOURCE,
        "source_file": source_relpath,
        "page_count": result.total_page_count,
        "extraction_engine": result.engine,
        "content_hash": hashlib.sha256(full_text.encode("utf-8")).hexdigest(),
    }

    return {"doc_id": doc_id, "metadata": metadata, "full_text": full_text}


def ingest_directory(
    input_dir: Path = INPUT_DIR,
    output_dir: Path = OUTPUT_DIR,
    recursive: bool = False,
) -> dict:
    """Ingests every ``*.pdf`` directly under ``input_dir`` (or, if
    ``recursive=True``, under any subdirectory of it -- off by default;
    no existing project convention requires recursive statute discovery)
    into ``output_dir/<doc_id>.json``.

    The configured input directory is expected to be curated by the
    operator to contain only the intended Family Law statute PDFs --
    this function does not inspect file content to exclude, say, a
    Constitution PDF that was placed there by mistake; that is an input
    curation responsibility, not something this stage guesses about.

    Returns a small summary dict: {"processed", "failed", "failed_pdfs"}.
    Never raises on a single PDF's failure -- that PDF is recorded in
    "failed_pdfs" and ingestion continues with the rest.
    """

    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    pattern = "**/*.pdf" if recursive else "*.pdf"
    pdf_paths = sorted(input_dir.glob(pattern))

    output_dir.mkdir(parents=True, exist_ok=True)

    processed = 0
    failed_pdfs: list[dict] = []
    for pdf_path in pdf_paths:
        try:
            doc = ingest_one_pdf(pdf_path, input_dir)
        except (StatuteIngestionError, ExtractionError) as exc:
            logger.error("Statute ingestion failed for %s: %s", pdf_path.name, exc)
            failed_pdfs.append({"source_file": pdf_path.name, "error": str(exc)})
            continue

        out_path = output_dir / f"{doc['doc_id']}.json"
        tmp_path = out_path.with_suffix(".json.tmp")
        tmp_path.write_text(json.dumps(doc, ensure_ascii=False, indent=2), encoding="utf-8")
        tmp_path.replace(out_path)
        processed += 1

    return {"processed": processed, "failed": len(failed_pdfs), "failed_pdfs": failed_pdfs}


def main() -> None:  # pragma: no cover -- thin CLI wrapper
    summary = ingest_directory()
    print(f"Statute ingestion: {summary['processed']} processed, {summary['failed']} failed")
    for f in summary["failed_pdfs"]:
        print(f"  FAILED {f['source_file']}: {f['error']}")


if __name__ == "__main__":  # pragma: no cover
    main()
