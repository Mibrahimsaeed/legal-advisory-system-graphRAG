from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from src.common.db import connection_scope, init_schema
from src.extraction import pdf_extractor as pe
from src.extraction.cleaner import (
    build_body_preview,
    clean_pages,
    normalize_text,
    strip_boilerplate_lines,
    strip_page_number_lines,
)
from src.extraction.pdf_extractor import PageText
from src.extraction.signature import compute_signature_hash
from src.extraction.signature_builder import build_signature
from src.extraction.signature_store import (
    DEFAULT_SIGNATURE_SCHEMA_FILE,
    count_by_status,
    get_signature,
    get_signatures_for_batch,
    upsert_signatures,
)
from src.ingestion import manifest as manifest_db
from src.ingestion.scratch_manager import ScratchWorkspace
from orchestration.dags.feature_extraction_flow import run_stage1

MANIFEST_SCHEMA_FILE = "schemas/manifest_schema.sql"
SIGNATURE_SCHEMA_FILE = str(DEFAULT_SIGNATURE_SCHEMA_FILE)

OCR_AVAILABLE = shutil.which("tesseract") is not None

try:
    import pypdfium2 as _pdfium  # noqa: F401

    RENDER_AVAILABLE = True
except ImportError:
    RENDER_AVAILABLE = False

try:
    from reportlab.lib.pagesizes import letter
    from reportlab.lib.utils import ImageReader
    from reportlab.pdfgen import canvas

    REPORTLAB_AVAILABLE = True
except ImportError:
    REPORTLAB_AVAILABLE = False

pytestmark = pytest.mark.skipif(
    not REPORTLAB_AVAILABLE, reason="reportlab required to synthesize test PDFs"
)


# ---------------------------------------------------------------------------
# Fixtures: synthetic PDFs
# ---------------------------------------------------------------------------


@pytest.fixture()
def digital_pdf(tmp_path) -> Path:
    """A native text PDF: title page + two numbered sections."""

    path = tmp_path / "digital.pdf"
    c = canvas.Canvas(str(path), pagesize=letter)

    c.setFont("Helvetica-Bold", 18)
    c.drawString(72, 720, "MASTER SERVICES AGREEMENT")
    c.setFont("Helvetica", 11)
    c.drawString(72, 690, "This Master Services Agreement is entered into as of January 1, 2024,")
    c.drawString(72, 675, "by and between Acme Corp and Widget LLC.")
    c.showPage()

    c.setFont("Helvetica-Bold", 14)
    c.drawString(72, 720, "1. Definitions")
    c.setFont("Helvetica", 11)
    c.drawString(72, 690, "In this Agreement, the following terms shall have the meanings set")
    c.drawString(72, 675, "forth below, unless the context requires otherwise.")
    c.showPage()

    c.setFont("Helvetica-Bold", 14)
    c.drawString(72, 720, "2. Term and Termination")
    c.setFont("Helvetica", 11)
    c.drawString(72, 690, "This Agreement commences on the Effective Date and continues for")
    c.drawString(72, 675, "three (3) years unless earlier terminated under Section 5.")
    c.showPage()
    c.save()
    return path


@pytest.fixture()
def blank_pdf(tmp_path) -> Path:
    """A PDF with pages but no text layer at all (simulates a scan)."""

    path = tmp_path / "blank.pdf"
    c = canvas.Canvas(str(path), pagesize=letter)
    c.showPage()
    c.save()
    return path


@pytest.fixture()
def scanned_pdf(tmp_path) -> Path:
    """A PDF whose only content is a rendered image of text (no text layer)."""

    from PIL import Image, ImageDraw, ImageFont

    img = Image.new("RGB", (1000, 400), "white")
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 40
        )
        font_body = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 28
        )
    except Exception:
        font = font_body = ImageFont.load_default()

    draw.text((40, 40), "NOTICE OF APPEAL", font=font, fill="black")
    draw.text((40, 120), "This scanned exhibit is attached pursuant to Rule 4.", font=font_body, fill="black")

    png_path = tmp_path / "scanned_page.png"
    img.save(png_path)

    pdf_path = tmp_path / "scanned.pdf"
    c = canvas.Canvas(str(pdf_path), pagesize=letter)
    c.drawImage(ImageReader(str(png_path)), 0, letter[1] - 400 * 0.5, width=1000 * 0.5, height=400 * 0.5)
    c.showPage()
    c.save()
    return pdf_path


@pytest.fixture()
def db_path(tmp_path) -> Path:
    p = tmp_path / "metadata.db"
    init_schema(db_path=p, schema_file=MANIFEST_SCHEMA_FILE)
    init_schema(db_path=p, schema_file=SIGNATURE_SCHEMA_FILE)
    return p


# ---------------------------------------------------------------------------
# cleaner.py
# ---------------------------------------------------------------------------


def test_normalize_text_dehyphenates_and_collapses_whitespace():
    raw = "This is an exam-\nple of   line-wrapped   hyphen-\nation.\n\n\n\nTrailing.  "
    result = normalize_text(raw)
    assert "exam-\nple" not in result
    assert "example" in result
    assert "hyphenation." in result
    assert "\n\n\n" not in result


def test_normalize_text_strips_control_characters():
    raw = "Body text\x0cwith a form feed and \x07 a bell."
    result = normalize_text(raw)
    assert "\x0c" not in result
    assert "\x07" not in result


def test_strip_page_number_lines_removes_page_only_lines():
    pages = [
        PageText(1, "Body one.\nPage 1 of 3", 0, 0),
        PageText(2, "Body two.\n2", 0, 0),
    ]
    cleaned = strip_page_number_lines(pages)
    assert "Page 1 of 3" not in cleaned[0].text
    assert "Body one." in cleaned[0].text
    assert cleaned[1].text.strip() == "Body two."


def test_strip_boilerplate_lines_removes_repeated_headers():
    pages = [
        PageText(i, f"CONFIDENTIAL - ACME CORP\nBody text page {i}.", 0, 0)
        for i in range(1, 4)
    ]
    cleaned = strip_boilerplate_lines(pages, min_repeat_fraction=0.5)
    for i, page in enumerate(cleaned, start=1):
        assert "CONFIDENTIAL" not in page.text
        assert f"Body text page {i}." in page.text


def test_strip_boilerplate_lines_requires_at_least_three_pages():
    pages = [
        PageText(1, "REPEATED\nBody one.", 0, 0),
        PageText(2, "REPEATED\nBody two.", 0, 0),
    ]
    # With fewer than 3 pages, nothing should be stripped even though the
    # line repeats on every page -- too little signal to be confident.
    cleaned = strip_boilerplate_lines(pages, min_repeat_fraction=0.5)
    assert cleaned == pages


def test_build_body_preview_respects_char_limit():
    pages = [PageText(1, "a" * 50, 50, 1.0), PageText(2, "b" * 50, 50, 1.0)]
    preview = build_body_preview(pages, char_limit=60)
    assert len(preview) <= 60
    assert preview.startswith("a" * 50)


def test_clean_pages_full_pipeline():
    pages = [
        PageText(i, f"CONFIDENTIAL\nWidget para-\ngraph number {i}.\nPage {i} of 3", 0, 0)
        for i in range(1, 4)
    ]
    cleaned = clean_pages(pages)
    for page in cleaned:
        assert "CONFIDENTIAL" not in page.text
        assert "of 3" not in page.text
        assert "paragraph" in page.text


# ---------------------------------------------------------------------------
# pdf_extractor.py
# ---------------------------------------------------------------------------


def test_extract_text_layer_finds_title_and_toc(digital_pdf):
    result = pe.extract_text_layer(digital_pdf, max_pages=15)
    assert result.title == "MASTER SERVICES AGREEMENT"
    assert any("Definitions" in t for t in result.toc)
    assert any("Term and Termination" in t for t in result.toc)
    assert len(result.pages) == 3
    assert result.total_page_count == 3


def test_extract_text_layer_respects_max_pages(digital_pdf):
    result = pe.extract_text_layer(digital_pdf, max_pages=1)
    assert len(result.pages) == 1


def test_page_quality_flags_blank_pages_low_quality(blank_pdf):
    result = pe.extract_text_layer(blank_pdf, max_pages=15)
    assert pe.looks_low_quality(result.pages) is True
    assert pe.low_quality_page_numbers(result.pages) == [1]


def test_page_quality_does_not_flag_normal_text(digital_pdf):
    result = pe.extract_text_layer(digital_pdf, max_pages=15)
    assert pe.looks_low_quality(result.pages) is False
    assert pe.low_quality_page_numbers(result.pages) == []


def test_extract_text_layer_raises_extraction_error_for_garbage_file(tmp_path):
    from src.common.exceptions import ExtractionError

    bad = tmp_path / "not_a_pdf.pdf"
    bad.write_bytes(b"this is not a pdf")
    with pytest.raises(ExtractionError):
        pe.extract_text_layer(bad)


# ---------------------------------------------------------------------------
# ocr_fallback.py
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    not (OCR_AVAILABLE and RENDER_AVAILABLE),
    reason="tesseract binary and/or pypdfium2/fitz not available",
)
def test_ocr_fallback_recovers_text_from_scanned_page(scanned_pdf):
    from src.extraction import ocr_fallback as ocr

    text_result = pe.extract_text_layer(scanned_pdf, max_pages=15)
    flagged = pe.low_quality_page_numbers(text_result.pages)
    assert flagged, "fixture should have no text layer"

    ocr_result = ocr.extract_via_ocr(scanned_pdf, flagged)
    assert ocr_result.engine == "ocr"
    combined = " ".join(p.text for p in ocr_result.pages).upper()
    assert "NOTICE" in combined
    assert "APPEAL" in combined


# ---------------------------------------------------------------------------
# signature_builder.py
# ---------------------------------------------------------------------------


def test_build_signature_for_digital_document(digital_pdf):
    sig = build_signature("doc_1", "s3://landing/doc_1.pdf", digital_pdf)
    assert sig.extraction_status == "ok"
    assert sig.is_scanned is False
    assert sig.title == "MASTER SERVICES AGREEMENT"
    assert sig.pages_used == 3
    assert sig.signature_hash
    assert sig.error is None


def test_build_signature_for_corrupt_file_fails_gracefully(tmp_path):
    bad = tmp_path / "corrupt.pdf"
    bad.write_bytes(b"garbage")
    sig = build_signature("doc_bad", "s3://landing/doc_bad.pdf", bad)
    assert sig.extraction_status == "failed"
    assert sig.signature_hash == ""
    assert sig.error is not None


def test_signature_hash_is_deterministic_and_content_sensitive(digital_pdf, blank_pdf):
    sig_a = build_signature("doc_1", "s3://landing/doc_1.pdf", digital_pdf)
    sig_a2 = build_signature("doc_1", "s3://landing/doc_1.pdf", digital_pdf)
    assert sig_a.signature_hash == sig_a2.signature_hash

    assert compute_signature_hash("t", ["a"], "body") == compute_signature_hash(
        "t", ["a"], "body"
    )
    assert compute_signature_hash("t", ["a"], "body") != compute_signature_hash(
        "t", ["a"], "different body"
    )


@pytest.mark.skipif(
    not (OCR_AVAILABLE and RENDER_AVAILABLE),
    reason="tesseract binary and/or pypdfium2/fitz not available",
)
def test_build_signature_triggers_ocr_for_scanned_document(scanned_pdf):
    sig = build_signature("doc_scan", "s3://landing/doc_scan.pdf", scanned_pdf)
    assert sig.is_scanned is True
    assert sig.extraction_status in {"ok_ocr", "ok_partial_ocr"}
    assert sig.extractor_used in {"ocr", "mixed"}
    assert "NOTICE" in sig.body_preview.upper()


# ---------------------------------------------------------------------------
# signature_store.py
# ---------------------------------------------------------------------------


def test_signature_store_round_trip(db_path, digital_pdf):
    manifest_db.register_documents(
        [{"doc_id": "doc_1", "source_uri": "s3://landing/doc_1.pdf", "checksum": "x"}],
        db_path=db_path,
    )
    sig = build_signature(
        "doc_1", "s3://landing/doc_1.pdf", digital_pdf, batch_id="batch_a"
    )
    upsert_signatures([sig], db_path=db_path)

    fetched = get_signature("doc_1", db_path=db_path)
    assert fetched is not None
    assert fetched.signature_hash == sig.signature_hash
    assert fetched.toc == sig.toc

    by_batch = get_signatures_for_batch("batch_a", db_path=db_path)
    assert len(by_batch) == 1

    assert count_by_status(db_path=db_path) == {"ok": 1}


def test_signature_store_upsert_is_idempotent_update(db_path, digital_pdf):
    manifest_db.register_documents(
        [{"doc_id": "doc_1", "source_uri": "s3://landing/doc_1.pdf", "checksum": "x"}],
        db_path=db_path,
    )
    sig = build_signature("doc_1", "s3://landing/doc_1.pdf", digital_pdf)
    upsert_signatures([sig], db_path=db_path)

    from dataclasses import replace

    updated = replace(sig, extraction_status="ok_partial_ocr")
    upsert_signatures([updated], db_path=db_path)

    fetched = get_signature("doc_1", db_path=db_path)
    assert fetched.extraction_status == "ok_partial_ocr"

    with connection_scope(db_path) as conn:
        count = conn.execute(
            "SELECT COUNT(*) AS c FROM document_signatures"
        ).fetchone()["c"]
    assert count == 1


# ---------------------------------------------------------------------------
# orchestration/dags/feature_extraction_flow.py
# ---------------------------------------------------------------------------


def test_run_stage1_end_to_end(db_path, tmp_path, digital_pdf, blank_pdf, monkeypatch):
    from src.common.config import get_settings

    settings = get_settings()
    checkpoint_dir = tmp_path / "checkpoints"
    metrics_db_path = tmp_path / "metrics.db"
    scratch_root = tmp_path / "scratch"

    batch_id = "batch_stage1"
    manifest_db.register_documents(
        [
            {"doc_id": "doc_ok", "source_uri": "s3://landing/doc_ok.pdf", "checksum": "a"},
            {"doc_id": "doc_missing", "source_uri": "s3://landing/doc_missing.pdf", "checksum": "b"},
        ],
        db_path=db_path,
    )
    with connection_scope(db_path) as conn:
        for doc_id in ("doc_ok", "doc_missing"):
            conn.execute(
                "UPDATE document_manifest SET ingest_status='pulled', batch_id=? WHERE doc_id=?",
                (batch_id, doc_id),
            )
        conn.execute(
            """
            INSERT INTO batch_runs (batch_id, requested_size, actual_size, status, started_at)
            VALUES (?, 2, 2, 'pulled', datetime('now'))
            """,
            (batch_id,),
        )

    ws = ScratchWorkspace(batch_id=batch_id, root=scratch_root)
    ws._create()
    shutil.copy(digital_pdf, ws.path_for_doc("doc_ok", ".pdf"))
    # doc_missing: intentionally no file, to exercise the missing-file path

    result = run_stage1(
        batch_id,
        db_path=db_path,
        scratch_root=scratch_root,
        checkpoint_dir=checkpoint_dir,
        metrics_db_path=metrics_db_path,
    )

    assert result.succeeded_doc_ids == ["doc_ok"]
    assert result.failed_doc_ids == ["doc_missing"]
    assert not ws.batch_dir.exists(), "scratch should be purged after Stage 1"

    with connection_scope(db_path) as conn:
        statuses = {
            row["doc_id"]: row["ingest_status"]
            for row in conn.execute("SELECT doc_id, ingest_status FROM document_manifest")
        }
    assert statuses["doc_ok"] == "extracted"
    assert statuses["doc_missing"] == "extraction_failed"

    # Re-running with the same batch_id should hit the checkpoint and
    # return the already-persisted signatures without redoing extraction.
    result2 = run_stage1(
        batch_id,
        db_path=db_path,
        scratch_root=scratch_root,
        checkpoint_dir=checkpoint_dir,
        metrics_db_path=metrics_db_path,
    )
    assert {s.doc_id for s in result2.signatures} == {"doc_ok", "doc_missing"}