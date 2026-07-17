"""Typed exception hierarchy for the Legal GraphRAG pipeline.

All pipeline-raised exceptions derive from :class:`PipelineError`, which
carries structured context (phase, doc_id, batch_id) so callers — loggers,
metrics recorders, checkpoint managers — can attach it without re-parsing
error strings.

Each exception class declares whether it is ``retryable`` by default; this
is used by :mod:`src.common.retry` and :mod:`src.common.checkpoint` to
decide whether a failure should be retried automatically or should fail
fast.
"""

from __future__ import annotations

from typing import Any


class PipelineError(Exception):
    """Base class for all errors raised by the pipeline.

    Attributes:
        message: Human readable description.
        phase: Pipeline phase in which the error occurred (e.g. "pull").
        doc_id: Document identifier associated with the error, if any.
        batch_id: Batch identifier associated with the error, if any.
        cause: The original exception that triggered this error, if any.
    """

    retryable: bool = False

    def __init__(
        self,
        message: str,
        *,
        phase: str | None = None,
        doc_id: str | None = None,
        batch_id: str | None = None,
        cause: BaseException | None = None,
        **context: Any,
    ) -> None:
        self.message = message
        self.phase = phase
        self.doc_id = doc_id
        self.batch_id = batch_id
        self.cause = cause
        self.context = context
        super().__init__(self._render())

    def _render(self) -> str:
        parts = [self.message]
        meta = {
            "phase": self.phase,
            "doc_id": self.doc_id,
            "batch_id": self.batch_id,
            **self.context,
        }
        rendered_meta = ", ".join(f"{k}={v}" for k, v in meta.items() if v is not None)
        if rendered_meta:
            parts.append(f"[{rendered_meta}]")
        if self.cause is not None:
            parts.append(f"caused by {type(self.cause).__name__}: {self.cause}")
        return " ".join(parts)

    def to_dict(self) -> dict[str, Any]:
        """Structured representation, useful for logging/metrics sinks."""

        return {
            "error_type": type(self).__name__,
            "message": self.message,
            "phase": self.phase,
            "doc_id": self.doc_id,
            "batch_id": self.batch_id,
            "retryable": self.retryable,
            "cause": f"{type(self.cause).__name__}: {self.cause}" if self.cause else None,
            **self.context,
        }


# --------------------------------------------------------------------------
# Configuration / validation
# --------------------------------------------------------------------------


class ConfigurationError(PipelineError):
    """Raised when configuration is missing, malformed, or invalid."""

    retryable = False


class ValidationError(PipelineError):
    """Raised when pre-flight environment validation fails."""

    retryable = False


# --------------------------------------------------------------------------
# Infrastructure
# --------------------------------------------------------------------------


class DatabaseError(PipelineError):
    """Raised for SQLite/manifest database failures."""

    retryable = True


class StorageError(PipelineError):
    """Raised for object-storage / filesystem backend failures."""

    retryable = True


class CheckpointError(PipelineError):
    """Raised when checkpoint state cannot be read, written, or resumed."""

    retryable = False


class RetryExhaustedError(PipelineError):
    """Raised when an operation failed after all retry attempts."""

    retryable = False


# --------------------------------------------------------------------------
# Pipeline stages
# --------------------------------------------------------------------------


class IngestionError(PipelineError):
    """Raised for failures during document ingestion (Stage 0)."""

    retryable = True


class DownloadError(IngestionError):
    """Raised when a document cannot be downloaded from its source."""

    retryable = True


class ChecksumMismatchError(IngestionError):
    """Raised when a downloaded document's checksum does not match the manifest."""

    retryable = False


class ExtractionError(PipelineError):
    """Raised for failures during text/PDF extraction and chunking."""

    retryable = True


class EmbeddingError(PipelineError):
    """Raised for failures during embedding generation."""

    retryable = True


class ClusteringError(PipelineError):
    """Raised for failures during domain clustering."""

    retryable = False


class GraphBuildError(PipelineError):
    """Raised for failures while building or updating the knowledge graph."""

    retryable = True


class IndexingError(PipelineError):
    """Raised for failures while building sparse/dense indexes."""

    retryable = True


class RetrievalError(PipelineError):
    """Raised for failures during query-time retrieval."""

    retryable = True