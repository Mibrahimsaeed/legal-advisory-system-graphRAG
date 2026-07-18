"""The Stage 1 output record: a lightweight per-document "signature".

Deliberately excludes full document text -- only ``doc_id``, ``source_uri``,
a content hash, scan/extraction status flags, and the bounded title/TOC/
body-preview fields described in the Phase 1 spec are ever persisted.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field


@dataclass
class DocumentSignature:
    """One row of the ``document_signatures`` table (see
    ``schemas/signature_schema.sql``)."""

    doc_id: str
    source_uri: str
    signature_hash: str
    is_scanned: bool
    extraction_status: str  # "ok" | "ok_ocr" | "ok_partial_ocr" | "failed"
    extractor_used: str  # "fitz" | "pdfplumber" | "pypdf" | "ocr" | "mixed" | "none"
    title: str | None = None
    toc: list[str] = field(default_factory=list)
    body_preview: str = ""
    pages_used: int = 0
    char_count: int = 0
    quality_score: float = 0.0
    batch_id: str | None = None
    error: str | None = None


def compute_signature_hash(
    title: str | None, toc: list[str], body_preview: str
) -> str:
    """Deterministic content hash over the signature's substantive fields.

    Used for change detection / dedup (e.g. re-ingested documents whose
    signature is byte-identical to one already on file don't need to be
    re-clustered downstream). Only the fields that feed downstream stages
    are hashed -- not ``doc_id``/``source_uri``, which are identity, not
    content.
    """

    payload = json.dumps(
        {"title": title or "", "toc": toc, "body_preview": body_preview},
        sort_keys=True,
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()