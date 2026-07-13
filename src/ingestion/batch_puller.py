from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
from urllib.parse import urlparse

from src.common.db import DEFAULT_DB_PATH
from src.common.logging_utils import get_logger
from src.ingestion import manifest as manifest_db
from src.ingestion.manifest import ManifestEntry
from src.ingestion.scratch_manager import ScratchWorkspace

logger = get_logger(__name__)

CHUNK_SIZE = 1024 * 1024


@dataclass(frozen=True)
class PullResult:
    doc_id: str
    local_path: Optional[Path]
    success: bool
    error: Optional[str] = None


class StorageBackend:
    def download(self, source_uri: str, local_path: Path) -> None:
        raise NotImplementedError


class S3Backend(StorageBackend):
    def __init__(self, client=None):
        if client is not None:
            self._client = client
        else:
            import boto3
            self._client = boto3.client("s3")

    @staticmethod
    def _parse_s3_uri(uri: str) -> tuple[str, str]:
        parsed = urlparse(uri)

        if parsed.scheme != "s3":
            raise ValueError(f"Expected s3:// URI, got: {uri}")

        bucket = parsed.netloc
        key = parsed.path.lstrip("/")

        if not bucket or not key:
            raise ValueError(
                f"Malformed s3 URI (missing bucket or key): {uri}"
            )

        return bucket, key

    def download(self, source_uri: str, local_path: Path) -> None:
        bucket, key = self._parse_s3_uri(source_uri)

        local_path.parent.mkdir(parents=True, exist_ok=True)

        self._client.download_file(
            bucket,
            key,
            str(local_path),
        )


def _sha256_of(path: Path) -> str:
    hasher = hashlib.sha256()

    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(CHUNK_SIZE), b""):
            hasher.update(chunk)

    return hasher.hexdigest()


def _suffix_for(source_uri: str) -> str:
    suffix = Path(urlparse(source_uri).path).suffix
    return suffix or ".bin"


def pull_batch(
    entries: list[ManifestEntry],
    workspace: ScratchWorkspace,
    backend: Optional[StorageBackend] = None,
    db_path: str | Path = DEFAULT_DB_PATH,
) -> list[PullResult]:
    backend = backend or S3Backend()

    results: list[PullResult] = []

    pulled_ids: list[str] = []
    failed: list[tuple[str, str]] = []

    for entry in entries:
        local_path = workspace.path_for_doc(
            entry.doc_id,
            _suffix_for(entry.source_uri),
        )

        try:
            backend.download(
                entry.source_uri,
                local_path,
            )

            actual_checksum = _sha256_of(local_path)

            if actual_checksum != entry.checksum:
                msg = (
                    f"Checksum mismatch for {entry.doc_id}: "
                    f"expected {entry.checksum}, got {actual_checksum}"
                )

                logger.error(msg)

                local_path.unlink(missing_ok=True)

                failed.append((entry.doc_id, msg))

                results.append(
                    PullResult(
                        entry.doc_id,
                        None,
                        success=False,
                        error=msg,
                    )
                )

                continue

            pulled_ids.append(entry.doc_id)

            results.append(
                PullResult(
                    entry.doc_id,
                    local_path,
                    success=True,
                )
            )

            logger.info(
                "Pulled %s -> %s (checksum verified)",
                entry.doc_id,
                local_path,
            )

        except Exception as exc:
            msg = f"{type(exc).__name__}: {exc}"

            logger.exception(
                "Failed to pull %s from %s",
                entry.doc_id,
                entry.source_uri,
            )

            local_path.unlink(missing_ok=True)

            failed.append((entry.doc_id, msg))

            results.append(
                PullResult(
                    entry.doc_id,
                    None,
                    success=False,
                    error=msg,
                )
            )

    if pulled_ids:
        manifest_db.mark_status(
            pulled_ids,
            "pulled",
            db_path=db_path,
        )

    for doc_id, err in failed:
        manifest_db.mark_status(
            [doc_id],
            "failed",
            db_path=db_path,
            error=err,
            increment_attempts=True,
        )

    logger.info(
        "Batch pull complete: %d succeeded, %d failed (of %d total)",
        len(pulled_ids),
        len(failed),
        len(entries),
    )

    return results