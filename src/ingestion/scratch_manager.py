from __future__ import annotations

import shutil
import stat
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

from src.common.logging_utils import get_logger

logger = get_logger(__name__)

DEFAULT_SCRATCH_ROOT = Path("var/scratch")


class ScratchWorkspace:
    def __init__(self, batch_id: str, root: Path):
        self.batch_id = batch_id
        self.root = root
        self.batch_dir = root / batch_id
        self.raw_dir = self.batch_dir / "raw"
        self.tmp_dir = self.batch_dir / "tmp"

    def path_for_doc(self, doc_id: str, suffix: str = "") -> Path:
        safe_name = doc_id.replace("/", "_")
        return self.raw_dir / f"{safe_name}{suffix}"

    def _create(self) -> None:
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.tmp_dir.mkdir(parents=True, exist_ok=True)
        logger.info("Created scratch workspace at %s", self.batch_dir)

    def purge(self) -> None:
        if not self.batch_dir.exists():
            return

        def _on_rm_error(func, path, exc_info):
            try:
                Path(path).chmod(stat.S_IWRITE)
                func(path)
            except Exception:
                logger.exception(
                    "Failed to force-remove %s during scratch purge",
                    path,
                )
                raise

        shutil.rmtree(
            self.batch_dir,
            onerror=_on_rm_error,
        )

        logger.info(
            "Purged scratch workspace at %s",
            self.batch_dir,
        )

    def size_bytes(self) -> int:
        if not self.batch_dir.exists():
            return 0

        return sum(
            f.stat().st_size
            for f in self.batch_dir.rglob("*")
            if f.is_file()
        )


@contextmanager
def scratch_workspace(
    batch_id: str,
    root: str | Path = DEFAULT_SCRATCH_ROOT,
    purge_on_exit: bool = True,
) -> Iterator[ScratchWorkspace]:
    ws = ScratchWorkspace(
        batch_id=batch_id,
        root=Path(root),
    )

    ws._create()

    try:
        yield ws

    finally:
        if purge_on_exit:
            ws.purge()
        else:
            logger.warning(
                "Scratch workspace for batch %s NOT purged "
                "(purge_on_exit=False)",
                batch_id,
            )


def purge_stale_workspaces(
    root: str | Path = DEFAULT_SCRATCH_ROOT,
    known_active_batch_ids: set[str] | None = None,
) -> int:
    root = Path(root)

    if not root.exists():
        return 0

    known_active_batch_ids = known_active_batch_ids or set()

    removed = 0

    for child in root.iterdir():
        if not child.is_dir():
            continue

        if child.name in known_active_batch_ids:
            continue

        logger.warning(
            "Removing stale scratch directory: %s",
            child,
        )

        shutil.rmtree(
            child,
            ignore_errors=True,
        )

        removed += 1

    logger.info(
        "Stale scratch sweep removed %d director(y/ies)",
        removed,
    )

    return removed