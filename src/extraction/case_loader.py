"""Turn one case-law folder into a lightweight document representation.

    case_folder/
        case.html        -- the judgment/order markup
        metadata.json    -- court, date, citation, ... (optional)

is read into a :class:`~src.extraction.doc_representation.DocumentRepresentation`:
provenance, the classification text (title + headings + a char-capped
body preview), and the case-law facts normalized out of ``metadata.json``
by :mod:`src.extraction.case_metadata`. That is the whole Stage 1
equivalent for case law -- there is no PDF page sampling, no OCR decision
and no signature record involved, because none of those are meaningful
for HTML case law (see ``docs/architecture.md`` and the Phase 1 notes in
``src/extraction/doc_representation.py``).

Text extraction stops at "readable prose in document order": tags are
resolved, ``<script>``/``<style>`` content is dropped, entities are
decoded, and whitespace is normalized. Deeper cleaning (headnote/footer
segmentation, citation stripping, paragraph reflow) is *not* done here --
that belongs to the RAG pipeline, which will re-read ``case.html`` from
the provenance recorded on each record.

Parsing uses the standard library's :mod:`html.parser` rather than
BeautifulSoup/lxml so the active pipeline gains no new dependency, and
the small text-normalization pass below is deliberately *not* reused from
:mod:`src.extraction.cleaner` -- that module imports the PDF extractor
stack (page objects, quality scoring), and coupling the case-law path
back to the PDF path is precisely what Phase 1 removed.

Nothing here raises: an unreadable folder produces a ``status="failed"``
representation carrying an ``ERROR_*`` code, and every non-fatal problem
(no metadata, odd encoding, no headings, suspiciously short text) lands
in ``warnings`` as a ``WARNING_*`` code -- so one bad case in a 10k-case
scan never kills the run, and no degraded case passes silently.
"""

from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from html.parser import HTMLParser
from pathlib import Path
from typing import Any, Iterator

from src.common.logging_utils import get_logger
from src.extraction.case_metadata import extract_case_metadata
from src.extraction.doc_representation import DocumentRepresentation

logger = get_logger(__name__)

DEFAULT_CASE_HTML_FILENAME = "case.html"
DEFAULT_METADATA_FILENAME = "metadata.json"
DEFAULT_BODY_PREVIEW_CHAR_LIMIT = 20_000
DEFAULT_MAX_HEADINGS = 50

SOURCE_TYPE_CASE_HTML = "case_html"

# Tags whose text content is never document content: scripting/styling,
# plus <nav>, which by definition holds site navigation ("Home > Supreme
# Court"). Header/footer are deliberately NOT skipped -- court portals put
# the court name and cause title in them. Removing anything subtler than
# this (repeated boilerplate lines, page furniture, headnotes) is RAG
# cleaning and is out of scope here.
_SKIPPED_CONTENT_TAGS = frozenset({"script", "style", "noscript", "nav"})

# Tags that end a line of text when opened or closed.
_BLOCK_TAGS = frozenset(
    {
        "p", "div", "br", "hr", "li", "ul", "ol", "tr", "td", "th", "table",
        "section", "article", "header", "footer", "blockquote", "pre",
        "h1", "h2", "h3", "h4", "h5", "h6",
    }
)

_HEADING_TAGS = frozenset({"h1", "h2", "h3", "h4"})

_MAX_HEADING_CHARS = 200

# Below this many extracted characters a "successful" extraction is
# almost certainly a navigation stub or a paywall page rather than a
# judgment. It is recorded as a warning, not a failure: short orders do
# exist, so the record stays usable and the operator gets a signal.
#
# This is the fallback for direct callers; the pipeline passes
# ``document.min_characters`` from config (see case_ingest_flow), which is
# the tunable value of record.
DEFAULT_MIN_CHARACTERS = 1200

#: Deprecated alias kept so existing imports keep resolving.
MIN_MEANINGFUL_TEXT_CHARS = DEFAULT_MIN_CHARACTERS

_CONTROL_CHAR_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
_MULTI_SPACE_RE = re.compile(r"[ \t]{2,}")
_MULTI_BLANK_LINE_RE = re.compile(r"\n{3,}")

_META_CHARSET_RE = re.compile(
    rb"""<meta[^>]+charset\s*=\s*["']?\s*([a-zA-Z0-9_\-]+)""", re.IGNORECASE
)
_HTML_SNIFF_BYTES = 4096

# -- outcome vocabulary ---------------------------------------------------
# Fatal (status="failed"); the code is the first token of `error`.
ERROR_MISSING_HTML = "missing_case_html"
ERROR_UNREADABLE_HTML = "unreadable_case_html"
ERROR_PARSE_FAILED = "html_parse_failed"
ERROR_EMPTY_DOCUMENT = "empty_document"

# Non-fatal (status="ok"); appended to `warnings`.
WARNING_MISSING_METADATA = "missing_metadata_json"
WARNING_INVALID_METADATA = "invalid_metadata_json"
WARNING_FALLBACK_ENCODING = "fallback_encoding"
WARNING_NO_HEADINGS = "no_headings_found"
WARNING_SHORT_TEXT = "short_text"
WARNING_TITLE_FROM_FOLDER = "title_from_folder_name"


def decode_html_bytes(raw: bytes) -> tuple[str, str, list[str]]:
    """Decode case HTML to text, returning ``(html, encoding, warnings)``.

    Order: BOM, then a declared ``<meta charset=...>``, then UTF-8, then
    cp1252 as a last resort. Court portal exports are frequently cp1252
    or latin-1 despite claiming otherwise, and decoding those as UTF-8
    with ``errors="replace"`` silently corrupts every non-ASCII name in
    the judgment -- so the fallback is explicit and warned about rather
    than lossy-by-default.
    """

    warnings: list[str] = []

    if raw.startswith(b"\xef\xbb\xbf"):
        return raw.decode("utf-8-sig"), "utf-8-sig", warnings
    if raw.startswith((b"\xff\xfe", b"\xfe\xff")):
        return raw.decode("utf-16"), "utf-16", warnings

    declared = _META_CHARSET_RE.search(raw[:_HTML_SNIFF_BYTES])
    candidates: list[str] = []
    if declared:
        candidates.append(declared.group(1).decode("ascii", errors="ignore").lower())
    candidates.append("utf-8")

    for encoding in candidates:
        if not encoding:
            continue
        try:
            return raw.decode(encoding), encoding, warnings
        except (UnicodeDecodeError, LookupError):
            continue

    warnings.append(f"{WARNING_FALLBACK_ENCODING}:cp1252")
    return raw.decode("cp1252", errors="replace"), "cp1252", warnings


class _CaseHTMLParser(HTMLParser):
    """Collects plain text, ``<hN>`` headings and ``<title>`` from case HTML."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self._chunks: list[str] = []
        self._skip_depth = 0
        self._heading_tag: str | None = None
        self._heading_chunks: list[str] = []
        self._in_title = False
        self.headings: list[str] = []
        self.html_title: str | None = None

    # -- tag handling ----------------------------------------------------
    def _flush_heading(self) -> None:
        heading = " ".join("".join(self._heading_chunks).split())
        if heading:
            self.headings.append(heading[:_MAX_HEADING_CHARS])
        self._heading_tag = None
        self._heading_chunks = []

    def handle_starttag(self, tag: str, attrs) -> None:
        tag = tag.lower()
        if tag in _SKIPPED_CONTENT_TAGS:
            self._skip_depth += 1
            return
        if tag == "title":
            self._in_title = True
            return
        if tag in _BLOCK_TAGS:
            # A block start implies the end of an unclosed heading -- the
            # HTML5 rule, and the reason "<h1>Case Name<h1>" (a typo for
            # "</h1>", common in scraped court pages) still yields the
            # case name as a heading instead of swallowing the judgment.
            if self._heading_tag is not None:
                self._flush_heading()
            self._chunks.append("\n")
        if tag in _HEADING_TAGS and self._heading_tag is None:
            self._heading_tag = tag
            self._heading_chunks = []

    def handle_endtag(self, tag: str) -> None:
        tag = tag.lower()
        if tag in _SKIPPED_CONTENT_TAGS:
            self._skip_depth = max(0, self._skip_depth - 1)
            return
        if tag == "title":
            self._in_title = False
            return
        if tag in _BLOCK_TAGS:
            self._chunks.append("\n")
        if tag == self._heading_tag:
            self._flush_heading()

    def close(self) -> None:
        """Flush a heading left open at end of input (truncated markup)."""

        super().close()
        if self._heading_tag is not None:
            self._flush_heading()

    def handle_startendtag(self, tag: str, attrs) -> None:
        if tag.lower() in _BLOCK_TAGS:
            self._chunks.append("\n")

    def handle_data(self, data: str) -> None:
        if self._skip_depth:
            return
        if self._in_title:
            existing = self.html_title or ""
            self.html_title = (existing + data).strip() or None
            return
        self._chunks.append(data)
        if self._heading_tag is not None:
            self._heading_chunks.append(data)

    # -- output ----------------------------------------------------------
    @property
    def text(self) -> str:
        return "".join(self._chunks)


def normalize_case_text(text: str) -> str:
    """NFKC-normalize, drop control characters, collapse whitespace.

    The HTML-side equivalent of :func:`src.extraction.cleaner.normalize_text`
    minus everything that only makes sense per-PDF-page (de-hyphenation of
    line-wrapped words, page-number stripping, cross-page boilerplate
    detection).
    """

    if not text:
        return ""

    normalized = unicodedata.normalize("NFKC", text)
    normalized = _CONTROL_CHAR_RE.sub("", normalized)
    normalized = normalized.replace("\r\n", "\n").replace("\r", "\n")
    normalized = _MULTI_SPACE_RE.sub(" ", normalized)
    normalized = "\n".join(line.strip() for line in normalized.split("\n"))
    normalized = _MULTI_BLANK_LINE_RE.sub("\n\n", normalized)
    return normalized.strip()


def parse_case_html(
    html: str, max_headings: int = DEFAULT_MAX_HEADINGS
) -> tuple[str, list[str], str | None]:
    """Parse case HTML into ``(text, headings, html_title)``.

    ``headings`` is de-duplicated (preserving document order) and capped at
    ``max_headings`` -- a case with hundreds of numbered paragraph headers
    shouldn't dominate its own embedding.
    """

    parser = _CaseHTMLParser()
    parser.feed(html)
    parser.close()

    seen: set[str] = set()
    headings: list[str] = []
    for heading in parser.headings:
        if heading in seen:
            continue
        seen.add(heading)
        headings.append(heading)
        if len(headings) >= max_headings:
            break

    return normalize_case_text(parser.text), headings, parser.html_title


def doc_id_for_case(root: Path, case_folder: Path) -> str:
    """Stable doc_id derived from the folder path *relative to root*.

    Same scheme (and same reasoning about remounted drives) as
    :func:`src.ingestion.discovery._doc_id_for_path`, so a case keeps its
    doc_id across re-scans and across mount-point changes.
    """

    rel = case_folder.resolve().relative_to(Path(root).resolve()).as_posix()
    return hashlib.sha256(rel.encode("utf-8")).hexdigest()[:24]


def iter_case_folders(
    root: str | Path,
    case_html_filename: str = DEFAULT_CASE_HTML_FILENAME,
) -> Iterator[Path]:
    """Yield every folder under ``root`` that contains ``case.html``.

    Recursive and sorted, so a scan is deterministic and can be
    interrupted/resumed without reshuffling which cases were seen.
    """

    root = Path(root)
    for html_path in sorted(root.rglob(case_html_filename)):
        if html_path.is_file():
            yield html_path.parent


def _read_metadata(
    metadata_path: Path, doc_id: str
) -> tuple[dict[str, Any], list[str]]:
    """Read ``metadata.json`` if present. Never fatal.

    Returns ``(metadata, warnings)`` -- an absent file, a malformed file
    and a file whose top level is not an object are all reported as
    warnings, because the classification input comes from the HTML and a
    case without usable metadata is still a usable case.
    """

    if not metadata_path.is_file():
        return {}, [WARNING_MISSING_METADATA]

    try:
        loaded = json.loads(metadata_path.read_text(encoding="utf-8", errors="replace"))
    except (OSError, ValueError) as exc:
        logger.warning("Case %s: metadata.json unreadable: %s", doc_id, exc)
        return {}, [f"{WARNING_INVALID_METADATA}:{type(exc).__name__}"]

    if not isinstance(loaded, dict):
        logger.warning("Case %s: metadata.json is not a JSON object", doc_id)
        return {}, [f"{WARNING_INVALID_METADATA}:not_an_object"]

    return loaded, []


def load_case_folder(
    case_folder: str | Path,
    doc_id: str | None = None,
    root: str | Path | None = None,
    case_html_filename: str = DEFAULT_CASE_HTML_FILENAME,
    metadata_filename: str = DEFAULT_METADATA_FILENAME,
    body_preview_char_limit: int = DEFAULT_BODY_PREVIEW_CHAR_LIMIT,
    max_headings: int = DEFAULT_MAX_HEADINGS,
    min_characters: int = DEFAULT_MIN_CHARACTERS,
    batch_id: str | None = None,
) -> DocumentRepresentation:
    """Build a :class:`DocumentRepresentation` for one case folder.

    ``doc_id`` defaults to :func:`doc_id_for_case` against ``root`` (or
    the folder's own name if no ``root`` is given).

    Never raises. Every degraded outcome is explicit:

    * missing/unreadable ``case.html``, unparseable markup, or a document
      with no extractable text -> ``status="failed"`` with ``error``
      prefixed by one of the ``ERROR_*`` codes.
      :func:`src.extraction.representation_store.list_representations`
      excludes those from the classification corpus.
    * missing/malformed ``metadata.json``, a non-UTF-8 fallback decode,
      no headings, or suspiciously short text -> ``status="ok"`` with a
      ``WARNING_*`` code appended to ``warnings``.
    """

    # Resolved so provenance stays meaningful from any working directory:
    # a scan started with a relative --root would otherwise persist paths
    # that only resolve for the process that wrote them.
    case_folder = Path(case_folder).resolve()
    html_path = case_folder / case_html_filename

    if doc_id is None:
        doc_id = doc_id_for_case(Path(root), case_folder) if root else case_folder.name

    source_relpath: str | None = None
    if root is not None:
        try:
            source_relpath = (
                case_folder.resolve().relative_to(Path(root).resolve()).as_posix()
            )
        except ValueError:
            # case_folder is outside root -- keep the absolute paths and
            # carry on rather than failing an otherwise fine case.
            source_relpath = None

    def _representation(**overrides: Any) -> DocumentRepresentation:
        base: dict[str, Any] = dict(
            doc_id=doc_id,
            source_uri=str(case_folder),
            source_type=SOURCE_TYPE_CASE_HTML,
            source_file=str(html_path),
            source_relpath=source_relpath,
            batch_id=batch_id,
        )
        base.update(overrides)
        return DocumentRepresentation(**base)

    def _failed(code: str, detail: str = "") -> DocumentRepresentation:
        error = f"{code}: {detail}" if detail else code
        logger.warning("Case %s could not be loaded: %s", doc_id, error)
        return _representation(status="failed", error=error)

    # -- HTML: read -> decode -> parse ---------------------------------
    if not html_path.is_file():
        return _failed(ERROR_MISSING_HTML, f"{case_html_filename} not found")

    try:
        raw = html_path.read_bytes()
    except OSError as exc:
        return _failed(ERROR_UNREADABLE_HTML, f"{type(exc).__name__}: {exc}")

    warnings: list[str] = []
    try:
        html, _encoding, decode_warnings = decode_html_bytes(raw)
    except Exception as exc:  # noqa: BLE001 - no decoding path may kill a scan
        return _failed(ERROR_UNREADABLE_HTML, f"{type(exc).__name__}: {exc}")
    warnings.extend(decode_warnings)

    try:
        text, headings, html_title = parse_case_html(html, max_headings=max_headings)
    except Exception as exc:  # noqa: BLE001 - malformed markup must not kill a scan
        return _failed(ERROR_PARSE_FAILED, f"{type(exc).__name__}: {exc}")

    if not text.strip():
        return _failed(
            ERROR_EMPTY_DOCUMENT, f"no extractable text in {case_html_filename}"
        )

    # -- metadata.json: read -> normalize -------------------------------
    metadata, metadata_warnings = _read_metadata(case_folder / metadata_filename, doc_id)
    warnings.extend(metadata_warnings)
    case_metadata = extract_case_metadata(metadata)
    warnings.extend(case_metadata.warnings)

    # -- title: metadata -> first heading -> <title> -> folder name -----
    title = case_metadata.title or (headings[0] if headings else None) or html_title
    if not title:
        title = case_folder.name
        warnings.append(WARNING_TITLE_FROM_FOLDER)

    # Case HTML usually repeats the case title as its <h1>. Left in, that
    # string would be embedded twice -- once as the title input and again
    # inside the headings input -- silently giving it title_weight +
    # toc_weight instead of title_weight.
    headings = [h for h in headings if h.casefold() != title.casefold()]

    if not headings:
        warnings.append(WARNING_NO_HEADINGS)
    if len(text) < min_characters:
        warnings.append(WARNING_SHORT_TEXT)

    return _representation(
        title=title,
        headings=headings,
        body_preview=text[:body_preview_char_limit],
        # Length of the full extracted text, not of the (capped) preview --
        # so a later stratification/QA pass can still tell a two-page order
        # from a hundred-page judgment.
        char_count=len(text),
        content_hash=hashlib.sha256(text.encode("utf-8")).hexdigest(),
        court=case_metadata.court,
        decision_date=case_metadata.decision_date,
        citation=case_metadata.citation,
        judges=case_metadata.judges,
        case_number=case_metadata.case_number,
        status="ok",
        warnings=warnings,
        metadata=metadata,
        full_text=text,
    )
