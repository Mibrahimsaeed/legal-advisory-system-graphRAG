"""Centralized logging for the pipeline.

Enhancements over a plain ``logging.getLogger``:

* Every record is tagged with a process-wide ``run_id``, plus the current
  ``batch_id`` / ``doc_id`` / ``phase`` when set via :func:`log_context`.
* Timestamps are ISO-8601 and each record carries ``elapsed_ms`` -- time
  since the current phase/context started (or process start, if none).
* Optional structured JSON output (``logging.json_format`` in config),
  suitable for shipping to a log aggregator.

Usage::

    from src.common.logging_utils import get_logger, log_context

    logger = get_logger(__name__)

    with log_context(batch_id="batch_123", phase="pull"):
        logger.info("Pulling batch")   # includes batch_id=batch_123 phase=pull
"""

from __future__ import annotations

import contextvars
import json
import logging
import os
import sys
import time
import uuid
from contextlib import contextmanager
from typing import Any, Iterator

_CONFIGURED = False

_PROCESS_START = time.monotonic()
_RUN_ID = os.environ.get("APP_RUN_ID") or uuid.uuid4().hex[:12]

_batch_id_var: contextvars.ContextVar[str | None] = contextvars.ContextVar("batch_id", default=None)
_doc_id_var: contextvars.ContextVar[str | None] = contextvars.ContextVar("doc_id", default=None)
_phase_var: contextvars.ContextVar[str | None] = contextvars.ContextVar("phase", default=None)
_phase_start_var: contextvars.ContextVar[float | None] = contextvars.ContextVar(
    "phase_start", default=None
)

_STANDARD_ATTRS = frozenset(
    {
        "name", "msg", "args", "levelname", "levelno", "pathname", "filename",
        "module", "exc_info", "exc_text", "stack_info", "lineno", "funcName",
        "created", "msecs", "relativeCreated", "thread", "threadName",
        "processName", "process", "message", "taskName",
    }
)


def current_run_id() -> str:
    """Return the run ID for this process (stable for its lifetime)."""

    return _RUN_ID


class _ContextFilter(logging.Filter):
    """Injects run/batch/doc/phase identifiers and elapsed time into records."""

    def filter(self, record: logging.LogRecord) -> bool:
        record.run_id = _RUN_ID
        record.batch_id = _batch_id_var.get()
        record.doc_id = _doc_id_var.get()
        record.phase = _phase_var.get()
        phase_start = _phase_start_var.get()
        reference = phase_start if phase_start is not None else _PROCESS_START
        record.elapsed_ms = round((time.monotonic() - reference) * 1000, 2)
        return True


class _PlainFormatter(logging.Formatter):
    default_fmt = (
        "%(asctime)s | %(levelname)-8s | run=%(run_id)s"
        "%(context_suffix)s | %(name)s | %(message)s | +%(elapsed_ms)sms"
    )

    def format(self, record: logging.LogRecord) -> str:
        context_bits = []
        for attr in ("phase", "batch_id", "doc_id"):
            value = getattr(record, attr, None)
            if value:
                context_bits.append(f"{attr}={value}")
        record.context_suffix = (" | " + " ".join(context_bits)) if context_bits else ""
        self._style._fmt = self.default_fmt  # noqa: SLF001
        return super().format(record)


class _JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "timestamp": self.formatTime(record, "%Y-%m-%dT%H:%M:%S%z"),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "run_id": getattr(record, "run_id", None),
            "batch_id": getattr(record, "batch_id", None),
            "doc_id": getattr(record, "doc_id", None),
            "phase": getattr(record, "phase", None),
            "elapsed_ms": getattr(record, "elapsed_ms", None),
        }
        extras = {
            k: v
            for k, v in record.__dict__.items()
            if k not in _STANDARD_ATTRS and k not in payload and not k.startswith("_")
        }
        if extras:
            payload["extra"] = extras
        if record.exc_info:
            payload["exc_info"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str)


def _configure_root(level: str | None = None, json_format: bool | None = None) -> None:
    global _CONFIGURED
    if _CONFIGURED:
        return

    resolved_level = getattr(
        logging, (level or os.environ.get("LOG_LEVEL", "INFO")).upper(), logging.INFO
    )
    use_json = json_format if json_format is not None else os.environ.get("LOG_JSON") == "1"

    handler = logging.StreamHandler(stream=sys.stdout)
    handler.addFilter(_ContextFilter())
    handler.setFormatter(_JsonFormatter() if use_json else _PlainFormatter())

    root = logging.getLogger()
    root.setLevel(resolved_level)
    root.addHandler(handler)
    _CONFIGURED = True


def configure_logging(level: str = "INFO", json_format: bool = False) -> None:
    """Explicitly (re)configure root logging, e.g. from :class:`Settings`.

    Safe to call at process startup; subsequent calls reset and reapply
    handlers so late configuration (e.g. after settings load) takes effect.
    """

    global _CONFIGURED
    _CONFIGURED = False
    root = logging.getLogger()
    for h in list(root.handlers):
        root.removeHandler(h)
    _configure_root(level=level, json_format=json_format)


def get_logger(name: str) -> logging.Logger:
    _configure_root()
    return logging.getLogger(name)


@contextmanager
def log_context(
    *,
    batch_id: str | None = None,
    doc_id: str | None = None,
    phase: str | None = None,
    reset_timer: bool = True,
) -> Iterator[None]:
    """Temporarily bind batch/doc/phase identifiers onto all log records.

    Nested contexts compose: an inner context only overrides the fields it
    was given, leaving unset fields inherited from the outer context.
    """

    tokens = []
    if batch_id is not None:
        tokens.append((_batch_id_var, _batch_id_var.set(batch_id)))
    if doc_id is not None:
        tokens.append((_doc_id_var, _doc_id_var.set(doc_id)))
    if phase is not None:
        tokens.append((_phase_var, _phase_var.set(phase)))
    if reset_timer:
        tokens.append((_phase_start_var, _phase_start_var.set(time.monotonic())))

    try:
        yield
    finally:
        for var, token in tokens:
            var.reset(token)