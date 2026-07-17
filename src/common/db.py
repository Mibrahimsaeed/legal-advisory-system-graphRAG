from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

from src.common.config import get_settings
from src.common.exceptions import DatabaseError

# Sourced from layered config (config/base.yaml -> config/{env}.yaml -> env
# vars); falls back to the config defaults if no config files are present.
DEFAULT_DB_PATH: Path = get_settings().database.path
DEFAULT_SCHEMA_FILE: Path = get_settings().database.schema_file


def _ensure_parent_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def get_connection(db_path: str | Path = DEFAULT_DB_PATH) -> sqlite3.Connection:
    db_path = Path(db_path)
    _ensure_parent_dir(db_path)
    try:
        conn = sqlite3.connect(str(db_path), timeout=30.0, isolation_level=None)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON;")
        conn.execute("PRAGMA journal_mode = WAL;")
    except sqlite3.Error as exc:
        raise DatabaseError(f"Failed to open SQLite database at {db_path}", cause=exc) from exc
    return conn


@contextmanager
def connection_scope(db_path: str | Path = DEFAULT_DB_PATH) -> Iterator[sqlite3.Connection]:
    conn = get_connection(db_path)
    conn.execute("BEGIN;")
    try:
        yield conn
        if conn.in_transaction:
            conn.execute("COMMIT;")
    except Exception:
        if conn.in_transaction:
            conn.execute("ROLLBACK;")
        raise
    finally:
        conn.close()


def init_schema(
    db_path: str | Path = DEFAULT_DB_PATH,
    schema_file: str | Path = DEFAULT_SCHEMA_FILE,
) -> None:
    schema_path = Path(schema_file)
    try:
        schema_sql = schema_path.read_text()
    except OSError as exc:
        raise DatabaseError(f"Failed to read schema file {schema_path}", cause=exc) from exc

    with connection_scope(db_path) as conn:
        try:
            conn.executescript(schema_sql)
        except sqlite3.Error as exc:
            raise DatabaseError(f"Failed to apply schema from {schema_path}", cause=exc) from exc