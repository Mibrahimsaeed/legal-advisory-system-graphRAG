"""Stage 0a: local document discovery.

Walks a mounted filesystem root -- an internal drive, an external/USB
drive, or a network share -- and registers every matching file into
``document_manifest`` (``schemas/manifest_schema.sql``) so the rest of
the pipeline (claim -> pull -> extract) can pick it up.

This is the piece that makes "ingest straight off an external disk,
never copy it into the project" possible: ``source_uri`` for a
locally-discovered document is simply the file's own path on that disk.
Nothing here reads more than a file's ``stat()`` by default (see
``checksum_mode``), and nothing here -- or anywhere downstream, for
``storage.backend="local"`` -- ever copies the file's bytes into
``var/scratch`` or anywhere else in the project. See
``orchestration/dags/batch_ingest_flow.py`` for the "pull" step that
confirms this at claim time, and ``docs/data_retention_policy.md`` for
the full policy this is part of.

Typical use::

    from src.common.config import get_settings
    from src.ingestion.discovery import discover_local_documents

    settings = get_settings()
    discover_local_documents(
        root=settings.storage.source_root,
        checksum_mode=settings.storage.checksum_mode,
        extensions=settings.storage.file_extensions,
    )
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path

from src.common.db import DEFAULT_DB_PATH
from src.common.exceptions import ConfigurationError
from src.common.logging_utils import get_logger
from src.ingestion import manifest as manifest_db

logger = get_logger(__name__)

DEFAULT_EXTENSIONS: tuple[str, ...] = (".pdf",)
DEFAULT_BATCH_REGISTER_SIZE = 1000


def _doc_id_for_path(root: Path, path: Path) -> str:
    """A stable doc_id derived from the file's path *relative to root*.

    Deliberately not the absolute path: an external disk can get
    remounted at a different drive letter / mount point between runs
    (``/Volumes/LegalDocs`` today, ``/Volumes/LegalDocs 1`` tomorrow if
    macOS thinks it's a different volume, etc.) without its internal
    directory structure changing. Hashing the relative path keeps doc_id
    stable across that, so re-running discovery after a remount doesn't
    re-register every document under a brand-new doc_id.
    """

    rel = path.relative_to(root).as_posix()
    return hashlib.sha256(rel.encode("utf-8")).hexdigest()[:24]


def _fast_checksum(stat_result: os.stat_result) -> str:
    """Cheap change-detection signature: hash of size + mtime.

    Deliberately avoids reading file content -- for a corpus of many
    thousands of PDFs on a (possibly slow, e.g. USB 2.0) external disk,
    content-hashing every file on every scan would dwarf the cost of the
    pipeline's own bounded extraction. Detects "this file was replaced or
    edited since the last scan"; it is NOT a content-integrity checksum.
    """

    return hashlib.sha256(
        f"{stat_result.st_size}:{int(stat_result.st_mtime)}".encode("utf-8")
    ).hexdigest()


def _full_checksum(path: Path, chunk_size: int = 1024 * 1024) -> str:
    """Exact sha256 of file content. Slower; reads every byte."""

    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def discover_local_documents(
    root: str | Path | None,
    db_path: str | Path = DEFAULT_DB_PATH,
    extensions: tuple[str, ...] = DEFAULT_EXTENSIONS,
    checksum_mode: str = "fast",
    batch_register_size: int = DEFAULT_BATCH_REGISTER_SIZE,
) -> int:
    """Recursively scan ``root`` and register matching files into the manifest.

    Returns the number of *newly* registered documents (documents already
    present, by ``doc_id``, are left untouched -- see
    :func:`src.ingestion.manifest.register_documents`'s
    ``ON CONFLICT(doc_id) DO NOTHING``).

    Idempotent and safe to re-run: it's the intended way to pick up files
    added to the disk since the last scan. Files removed from disk since
    the last scan are *not* deleted from the manifest here -- Stage 1
    will simply fail cleanly with "source file not found" for those
    (see ``batch_ingest_flow._pull_local``), which is easier to audit
    than a discovery pass silently deleting manifest rows.

    Raises:
        ConfigurationError: if ``root`` is ``None`` (i.e.
            ``storage.source_root`` was never configured), or if it
            doesn't exist / isn't a directory -- e.g. the external disk
            isn't currently mounted.
    """

    if root is None:
        raise ConfigurationError(
            "discover_local_documents() requires a root path; set "
            "storage.source_root in config to the external disk's mount path"
        )

    root = Path(root)
    if not root.exists():
        raise ConfigurationError(
            f"Discovery root does not exist: {root} "
            "(is the external disk mounted?)"
        )
    if not root.is_dir():
        raise ConfigurationError(f"Discovery root is not a directory: {root}")

    normalized_ext = tuple(e.lower() for e in extensions)
    total_registered = 0
    scanned = 0
    skipped_unreadable = 0
    pending: list[dict] = []

    def _flush() -> None:
        nonlocal total_registered, pending
        if pending:
            total_registered += manifest_db.register_documents(pending, db_path=db_path)
            pending = []

    for path in root.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in normalized_ext:
            continue

        try:
            stat_result = path.stat()
        except OSError as exc:
            logger.warning("Skipping unreadable file %s: %s", path, exc)
            skipped_unreadable += 1
            continue

        doc_id = _doc_id_for_path(root, path)
        if checksum_mode == "full":
            checksum = _full_checksum(path)
            checksum_algo = "sha256"
        else:
            checksum = _fast_checksum(stat_result)
            checksum_algo = "sha256_fast_meta"

        pending.append(
            {
                "doc_id": doc_id,
                "source_uri": str(path),
                "checksum": checksum,
                "checksum_algo": checksum_algo,
                "byte_size": stat_result.st_size,
            }
        )
        scanned += 1
        if len(pending) >= batch_register_size:
            _flush()

    _flush()

    logger.info(
        "Discovery scanned %d matching file(s) under %s (%d unreadable "
        "skipped); %d newly registered",
        scanned, root, skipped_unreadable, total_registered,
    )
    return total_registered