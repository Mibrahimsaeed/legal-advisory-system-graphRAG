"""The compact, deterministic view of a case that Phase 3 reasons over.

One case folder -> one :class:`CaseRepresentation`. Three consumers share
it, which is the whole point of building it once:

* the embedder (it satisfies
  :class:`~src.extraction.doc_representation.EmbeddableDocument`, so
  :func:`src.embedding.doc_pooling.embed_documents` takes it unchanged),
* the keyword scanner (:mod:`src.classification.keyword_signals`),
* the LLM broad-domain assessor (:mod:`src.classification.domain_assessment`).

What goes in: the case's own words -- title, section headings, and the
substantive text Phase 2 cleaned. What stays out: court, bench, date,
citation and case number. Those are stored on the document row and remain
available to Phase 4 as separate signals; letting them into the text here
would push the embedding (and the clustering built on it) toward grouping
by forum rather than by subject matter.

Determinism is a requirement, not a nicety: the same document must yield
byte-identical text and the same ``representation_hash`` on every run, so
a re-run produces the same vector, the same cluster input and the same
keyword counts. Only the LLM step is allowed to vary, and it is isolated
in its own module.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any

from src.common.logging_utils import get_logger

logger = get_logger(__name__)

# How much substantive text the representation carries. Generous enough
# that a full judgment normally fits, bounded so one enormous document
# cannot dominate a batch's memory or an LLM prompt.
DEFAULT_MAX_TEXT_CHARS = 20_000
DEFAULT_MAX_HEADINGS = 20


@dataclass(frozen=True)
class CaseRepresentation:
    """One case, reduced to the content Phase 3 reasons about."""

    doc_id: str
    title: str | None = None
    headings: list[str] = field(default_factory=list)
    body_preview: str = ""  # the substantive text; named for EmbeddableDocument
    representation_hash: str = ""
    source_relpath: str | None = None
    text_source: str = "cleaned_text"  # "cleaned_text" | "body_preview"
    char_count: int = 0
    word_count: int = 0

    @property
    def signal_text(self) -> str:
        """Everything a keyword scanner should look at, in one string."""

        parts = [self.title or "", " ".join(self.headings), self.body_preview]
        return "\n".join(p for p in parts if p)

    def prompt_text(self, max_chars: int) -> str:
        """The bounded slice shown to the LLM."""

        return self.body_preview[:max_chars]


def _normalize_heading(heading: str) -> str:
    return " ".join((heading or "").split())


def build_case_representation(
    document: Any,
    max_text_chars: int = DEFAULT_MAX_TEXT_CHARS,
    max_headings: int = DEFAULT_MAX_HEADINGS,
) -> CaseRepresentation:
    """Build the Phase 3 representation for one stored document.

    The substantive text is ``cleaned_text`` when Phase 2 populated it,
    falling back to ``body_preview`` -- a document read back from SQLite
    before Phase 2 ran, or one whose cleaned text was not stored, still
    produces a usable representation rather than an empty one.

    ``representation_hash`` covers exactly the text that is embedded and
    scanned, so two runs agreeing on the hash agree on every deterministic
    signal downstream.
    """

    doc_id = getattr(document, "doc_id", "")
    cleaned = (getattr(document, "cleaned_text", None) or "").strip()
    preview = (getattr(document, "body_preview", "") or "").strip()

    if cleaned:
        text, text_source = cleaned, "cleaned_text"
    else:
        text, text_source = preview, "body_preview"

    text = text[:max_text_chars]
    title = " ".join((getattr(document, "title", None) or "").split()) or None
    headings = [
        _normalize_heading(h)
        for h in (getattr(document, "headings", None) or [])[:max_headings]
        if _normalize_heading(h)
    ]

    payload = " ".join([doc_id, title or "", "".join(headings), text])
    representation_hash = hashlib.sha256(payload.encode("utf-8")).hexdigest()

    return CaseRepresentation(
        doc_id=doc_id,
        title=title,
        headings=headings,
        body_preview=text,
        representation_hash=representation_hash,
        source_relpath=getattr(document, "source_relpath", None),
        text_source=text_source,
        char_count=len(text),
        word_count=len(text.split()),
    )


def build_case_representations(
    documents: list[Any],
    max_text_chars: int = DEFAULT_MAX_TEXT_CHARS,
    max_headings: int = DEFAULT_MAX_HEADINGS,
) -> list[CaseRepresentation]:
    """Build representations for a batch, skipping documents with no text.

    A representation with nothing to embed would become a meaningless
    near-zero vector and cluster with every other empty document, so those
    are dropped here and counted rather than passed on.
    """

    representations: list[CaseRepresentation] = []
    empty = 0
    for document in documents:
        representation = build_case_representation(
            document, max_text_chars=max_text_chars, max_headings=max_headings
        )
        if not representation.signal_text.strip():
            empty += 1
            continue
        representations.append(representation)

    if empty:
        logger.warning("Skipped %d document(s) with no representable text", empty)
    return representations
