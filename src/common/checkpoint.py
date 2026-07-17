"""Phase-aware checkpoint / resume system.

Any multi-phase pipeline run (ingestion, extraction, embedding, graph
building, clustering, ...) can use a :class:`CheckpointManager` to:

* Record which phases have completed for a given ``run_key`` (typically a
  batch ID), so a crashed or killed process can be resumed without redoing
  finished work.
* Persist small pieces of state per phase (e.g. "which doc_ids succeeded")
  so downstream phases can pick up exactly where the previous one left off.
* Retry a phase's callable with exponential backoff (delegating to
  :mod:`src.common.retry`) before marking it failed.

State is persisted as a single JSON file per ``run_key`` under
``checkpoint_dir``, written atomically (temp file + ``os.replace``) so a
crash mid-write never corrupts the checkpoint.

Usage::

    from src.common.checkpoint import CheckpointManager

    manager = CheckpointManager(
        checkpoint_dir="var/checkpoints",
        run_key=batch_id,
        phases=["claim", "pull"],
    )

    claimed = manager.run_phase("claim", do_claim)
    if not manager.is_completed("pull"):
        manager.run_phase("pull", do_pull, claimed)
"""

from __future__ import annotations

import json
import os
import tempfile
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Callable, TypeVar

from src.common.exceptions import CheckpointError
from src.common.logging_utils import get_logger, log_context
from src.common.retry import RetryPolicy, call_with_retry

logger = get_logger(__name__)

T = TypeVar("T")


class PhaseStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class PhaseRecord:
    name: str
    status: PhaseStatus = PhaseStatus.PENDING
    attempts: int = 0
    last_error: str | None = None
    started_at: str | None = None
    finished_at: str | None = None
    state: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["status"] = self.status.value
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "PhaseRecord":
        d = dict(d)
        d["status"] = PhaseStatus(d.get("status", "pending"))
        return cls(**d)


class CheckpointManager:
    """Tracks phase completion/state for a single run, persisted to disk."""

    def __init__(
        self,
        checkpoint_dir: str | Path,
        run_key: str,
        phases: list[str] | None = None,
    ) -> None:
        self.checkpoint_dir = Path(checkpoint_dir)
        self.run_key = run_key
        self._phases: dict[str, PhaseRecord] = {
            name: PhaseRecord(name=name) for name in (phases or [])
        }
        self._load()

    # -- persistence ------------------------------------------------------

    @property
    def _path(self) -> Path:
        safe_key = self.run_key.replace("/", "_")
        return self.checkpoint_dir / f"{safe_key}.json"

    def _load(self) -> None:
        path = self._path
        if not path.exists():
            return
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise CheckpointError(
                f"Failed to load checkpoint file {path}", cause=exc
            ) from exc

        for name, record_dict in raw.get("phases", {}).items():
            self._phases[name] = PhaseRecord.from_dict(record_dict)

        logger.info(
            "Loaded checkpoint for run_key=%s (%d phase(s) recorded)",
            self.run_key,
            len(self._phases),
        )

    def _save(self) -> None:
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        payload = {
            "run_key": self.run_key,
            "updated_at": _now(),
            "phases": {name: rec.to_dict() for name, rec in self._phases.items()},
        }
        try:
            fd, tmp_name = tempfile.mkstemp(
                dir=str(self.checkpoint_dir), prefix=".tmp_ckpt_", suffix=".json"
            )
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                json.dump(payload, fh, indent=2, sort_keys=True)
            os.replace(tmp_name, self._path)
        except OSError as exc:
            raise CheckpointError(
                f"Failed to persist checkpoint file {self._path}", cause=exc
            ) from exc

    # -- phase bookkeeping --------------------------------------------------

    def _record(self, phase: str) -> PhaseRecord:
        if phase not in self._phases:
            self._phases[phase] = PhaseRecord(name=phase)
        return self._phases[phase]

    def status(self, phase: str) -> PhaseStatus:
        return self._record(phase).status

    def is_completed(self, phase: str) -> bool:
        return self.status(phase) == PhaseStatus.COMPLETED

    def get_state(self, phase: str) -> dict[str, Any]:
        """Return whatever state was stored when ``phase`` last completed."""

        return dict(self._record(phase).state)

    def set_state(self, phase: str, state: dict[str, Any]) -> None:
        """Persist arbitrary progress state for ``phase`` without changing status.

        Useful for checkpointing partial progress *within* a long-running
        phase (e.g. every N documents), so a retry can resume mid-phase.
        """

        self._record(phase).state.update(state)
        self._save()

    def start_phase(self, phase: str) -> None:
        record = self._record(phase)
        record.status = PhaseStatus.RUNNING
        record.started_at = _now()
        self._save()

    def complete_phase(self, phase: str, state: dict[str, Any] | None = None) -> None:
        record = self._record(phase)
        record.status = PhaseStatus.COMPLETED
        record.finished_at = _now()
        record.last_error = None
        if state is not None:
            record.state = state
        self._save()
        logger.info("Phase '%s' completed for run_key=%s", phase, self.run_key)

    def fail_phase(self, phase: str, error: str) -> None:
        record = self._record(phase)
        record.status = PhaseStatus.FAILED
        record.finished_at = _now()
        record.last_error = error
        self._save()
        logger.error("Phase '%s' failed for run_key=%s: %s", phase, self.run_key, error)

    def reset(self, phase: str | None = None) -> None:
        """Clear checkpoint state for one phase, or all phases if omitted."""

        if phase is None:
            self._phases = {name: PhaseRecord(name=name) for name in self._phases}
        else:
            self._phases[phase] = PhaseRecord(name=phase)
        self._save()

    # -- execution ----------------------------------------------------------

    def run_phase(
        self,
        phase: str,
        func: Callable[..., T],
        *args: Any,
        retry_policy: RetryPolicy | None = None,
        force: bool = False,
        **kwargs: Any,
    ) -> T | None:
        """Run ``func(*args, **kwargs)`` under checkpoint + retry protection.

        If the phase is already marked completed (and ``force`` is False),
        the callable is skipped and the previously stored state is returned
        instead. Otherwise the phase is marked running, executed with
        backoff retries via :func:`~src.common.retry.call_with_retry`, and
        marked completed/failed based on the outcome.

        Returns:
            The callable's return value on a fresh run, the stored state
            dict if the phase was already completed and skipped, or
            ``None`` if the phase had no stored state.
        """

        if self.is_completed(phase) and not force:
            logger.info(
                "Skipping already-completed phase '%s' for run_key=%s (resume)",
                phase,
                self.run_key,
            )
            return self.get_state(phase) or None  # type: ignore[return-value]

        with log_context(batch_id=self.run_key, phase=phase):
            self.start_phase(phase)
            self._record(phase).attempts += 1
            self._save()

            start = time.monotonic()
            try:
                result = call_with_retry(
                    func,
                    *args,
                    policy=retry_policy,
                    operation_name=phase,
                    **kwargs,
                )
            except Exception as exc:
                self.fail_phase(phase, f"{type(exc).__name__}: {exc}")
                raise

            elapsed = time.monotonic() - start
            state = result if isinstance(result, dict) else {"result_repr": repr(result)}
            self.complete_phase(phase, state=state)
            logger.info(
                "Phase '%s' finished in %.2fs for run_key=%s",
                phase,
                elapsed,
                self.run_key,
            )
            return result

    def summary(self) -> dict[str, Any]:
        """A compact overview of all tracked phases, useful for logging/CLI."""

        return {name: rec.status.value for name, rec in self._phases.items()}