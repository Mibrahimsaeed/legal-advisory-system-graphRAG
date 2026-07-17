"""Pre-flight validation of configuration, directories, database, and storage.

Intended to run once at process startup (CLI entrypoints, orchestration
DAGs) so misconfiguration fails fast with a clear, aggregated error rather
than surfacing as an obscure exception mid-pipeline.

Usage::

    from src.common.config import get_settings
    from src.common.validation import run_startup_checks

    run_startup_checks(get_settings())
"""

from __future__ import annotations

import importlib.util
import os
import sqlite3
from pathlib import Path

from src.common.config import Settings
from src.common.exceptions import ValidationError
from src.common.logging_utils import get_logger

logger = get_logger(__name__)


def _check_writable_dir(path: Path, issues: list[str], label: str) -> None:
    try:
        path.mkdir(parents=True, exist_ok=True)
        probe = path / ".write_check"
        probe.write_text("ok", encoding="utf-8")
        probe.unlink()
    except OSError as exc:
        issues.append(f"{label} directory {path} is not writable: {exc}")


def validate_config(settings: Settings) -> list[str]:
    """Sanity checks beyond what Pydantic field validation already covers."""

    issues: list[str] = []
    if settings.pipeline.batch_size <= 0:
        issues.append("pipeline.batch_size must be a positive integer")
    if settings.storage.backend == "s3" and not settings.storage.bucket and settings.env == "prod":
        issues.append("storage.bucket must be set when storage.backend='s3' in prod")
    return issues


def validate_directories(settings: Settings) -> list[str]:
    """Ensure scratch/checkpoint/local-storage directories exist and are writable."""

    issues: list[str] = []
    _check_writable_dir(settings.scratch.root, issues, "scratch.root")
    _check_writable_dir(settings.pipeline.checkpoint_dir, issues, "pipeline.checkpoint_dir")
    if settings.storage.backend == "local":
        _check_writable_dir(settings.storage.local_root, issues, "storage.local_root")

    db_parent = settings.database.path.parent
    if str(db_parent) not in ("", "."):
        _check_writable_dir(db_parent, issues, "database.path parent")

    metrics_parent = settings.metrics.db_path.parent
    if str(metrics_parent) not in ("", "."):
        _check_writable_dir(metrics_parent, issues, "metrics.db_path parent")

    return issues


def validate_database(settings: Settings) -> list[str]:
    """Confirm the manifest schema file exists and SQLite is usable."""

    issues: list[str] = []

    if not settings.database.schema_file.exists():
        issues.append(f"database.schema_file not found: {settings.database.schema_file}")

    try:
        conn = sqlite3.connect(str(settings.database.path))
        conn.execute("SELECT 1;")
        conn.close()
    except sqlite3.Error as exc:
        issues.append(f"Unable to open SQLite database at {settings.database.path}: {exc}")

    return issues


def validate_storage(settings: Settings) -> list[str]:
    """Confirm the configured storage backend is usable."""

    issues: list[str] = []

    if settings.storage.backend == "s3":
        if importlib.util.find_spec("boto3") is None:
            issues.append("storage.backend='s3' but the 'boto3' package is not installed")
    elif settings.storage.backend == "local":
        if not settings.storage.local_root.exists():
            issues.append(f"storage.local_root does not exist: {settings.storage.local_root}")
    else:  # pragma: no cover - unreachable given Settings validation
        issues.append(f"Unknown storage backend: {settings.storage.backend}")

    return issues


def run_startup_checks(settings: Settings) -> None:
    """Run all validation checks and raise if any fail.

    Raises:
        ValidationError: aggregating every failed check, so operators see
            the full picture instead of fixing issues one at a time.
    """

    issues: list[str] = []
    issues += validate_config(settings)
    issues += validate_directories(settings)
    issues += validate_database(settings)
    issues += validate_storage(settings)

    if issues:
        joined = "; ".join(issues)
        raise ValidationError(f"Startup validation failed ({len(issues)} issue(s)): {joined}")

    logger.info("Startup validation passed for env=%s", settings.env)


def main() -> int:  # pragma: no cover - thin CLI wrapper
    from src.common.config import get_settings

    settings = get_settings(env=os.environ.get("APP_ENV"))
    try:
        run_startup_checks(settings)
    except ValidationError as exc:
        logger.error(str(exc))
        return 1
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())