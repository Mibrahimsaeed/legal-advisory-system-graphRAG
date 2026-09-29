"""The corpus-neutral record the classification path consumes.

Stage 1.2 (embed -> pool -> UMAP -> HDBSCAN -> rank -> label) used to be
typed against :class:`~src.extraction.signature.DocumentSignature`, which
is a *PDF/book* record: it carries scan flags, an extractor name, a page
count and a table of contents because it is produced by pulling the first
N pages out of a PDF. Case law (``case_folder/case.html`` +
``metadata.json``) has no pages, no scan status and no extractor, so
requiring that record would mean fabricating a signature for every case.

This module holds the small structural contract the classification path
actually needs (:class:`EmbeddableDocument` -- four attributes) plus the
concrete record the case-law path produces
(:class:`DocumentRepresentation`).

``DocumentSignature`` still satisfies :class:`EmbeddableDocument` (it has
a ``headings`` alias for its ``toc``), so the legacy PDF path keeps
working unchanged -- but nothing on the classification path imports or
requires it any more.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

# The classification lifecycle of a case. Mirrors the CHECK constraint in
# schemas/document_representation_schema.sql -- keep the two in step.
CLASSIFICATION_STATUS_PENDING = "pending"
CLASSIFICATION_STATUS_AUTO_ACCEPTED = "auto_accepted"
CLASSIFICATION_STATUS_NEEDS_REVIEW = "needs_review"
CLASSIFICATION_STATUS_DROPPED_PROCEDURAL = "dropped_procedural"
CLASSIFICATION_STATUS_DROPPED_OFF_DOMAIN = "dropped_off_domain"

CLASSIFICATION_STATUSES = (
    CLASSIFICATION_STATUS_PENDING,
    CLASSIFICATION_STATUS_AUTO_ACCEPTED,
    CLASSIFICATION_STATUS_NEEDS_REVIEW,
    CLASSIFICATION_STATUS_DROPPED_PROCEDURAL,
    CLASSIFICATION_STATUS_DROPPED_OFF_DOMAIN,
)


@runtime_checkable
class EmbeddableDocument(Protocol):
    """Everything the classification path needs from a document record.

    Deliberately four attributes and no methods: any producer (case HTML
    today, PDF signatures still, whatever comes next) can satisfy it
    without inheriting anything.
    """

    doc_id: str
    title: str | None
    headings: list[str]
    body_preview: str


@dataclass
class DocumentRepresentation:
    """One row of the ``document_representations`` table (see
    ``schemas/document_representation_schema.sql``).

    Four groups of fields:

    * **identity + provenance** -- ``doc_id`` and enough path information
      to get back to the exact folder and file this came from.
    * **classification input** -- ``title``, ``headings``,
      ``body_preview``: the bounded text the embedding path reads.
    * **case-law facts** -- ``court``, ``decision_date``, ``citation``,
      ``judges``, ``case_number``, normalized out of ``metadata.json``
      by :mod:`src.extraction.case_metadata`. Nothing on the
      classification path reads these yet; they exist so filtering,
      dedup and the citation graph don't have to re-parse metadata later.
    * **outcome** -- ``status``, ``warnings``, ``error``: every degraded
      case is explicit rather than silently empty.

    ``full_text`` is the complete extracted text and is deliberately
    **not persisted** (see the schema file): it is re-derivable at any
    time from ``source_file``. Records read back out of SQLite therefore
    carry ``full_text=""`` and a populated ``body_preview``.
    """

    doc_id: str
    source_uri: str
    source_type: str  # "case_html" | ...

    # -- provenance ---------------------------------------------------
    source_file: str | None = None  # absolute path to the parsed file
    source_relpath: str | None = None  # path relative to the corpus root
    content_hash: str | None = None  # sha256 of the extracted text

    # -- classification input -----------------------------------------
    title: str | None = None
    headings: list[str] = field(default_factory=list)
    body_preview: str = ""
    char_count: int = 0

    # -- case-law facts (normalized from metadata.json) ---------------
    court: str | None = None
    decision_date: str | None = None  # ISO-8601 (YYYY-MM-DD) when parseable
    citation: str | None = None
    judges: list[str] = field(default_factory=list)
    case_number: str | None = None

    # -- legal metadata storage (Phase 1 provides storage only) -------
    # Populated by later phases; no extraction is implemented here.
    statute_citations: list[Any] = field(default_factory=list)
    court_metadata: dict[str, Any] = field(default_factory=dict)

    # -- current classification state ----------------------------------
    # A case exists as ``pending`` from the moment it is stored, before
    # any classification runs. The per-run history lives in
    # ``document_classifications``; these fields are the current answer.
    primary_domain: str | None = None
    secondary_domain: str | None = None
    domain_confidence: float | None = None
    classification_status: str = CLASSIFICATION_STATUS_PENDING
    drop_reason: str | None = None

    # -- outcome -------------------------------------------------------
    status: str = "ok"  # "ok" | "failed"
    warnings: list[str] = field(default_factory=list)
    error: str | None = None  # set only when status == "failed"

    # Verbatim metadata.json, kept for provenance/later phases. The
    # classification path does not read it.
    metadata: dict[str, Any] = field(default_factory=dict)
    batch_id: str | None = None

    # Full extracted text. Populated by the loader, never persisted --
    # see cleaned_text below and docs/data_retention_policy.md.
    full_text: str = ""

    # Cleaned case text from Phase 2 structural cleaning. Unlike
    # ``full_text`` this one IS persisted, but Phase 1 never populates it.
    cleaned_text: str | None = None
