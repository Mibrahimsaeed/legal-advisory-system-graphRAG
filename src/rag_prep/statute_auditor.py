"""Statute chunk-set validation -- thin, stable import path over the
real implementation.

The actual audit logic lives in
:func:`src.rag_prep.statute_chunk_validate.validate_statute_chunk_set`
and :func:`src.rag_prep.statute_chunk_validate.is_valid_existing_output`
(built and validated there first, directly against the 9 real cleaned/
chunked statutes). This module exists only so callers that expect a
dedicated ``statute_auditor`` entry point have one, per the project's
"incremental addition" decision: avoid duplicating or forking the real,
validated validator into a second implementation.

See ``src.rag_prep.statute_chunk_validate``'s module docstring for what
is actually checked: schema validity, exact source-slice equality,
bounds, gapless/non-overlapping full-text coverage, deterministic
chunk IDs, unique section/schedule IDs, at most one definitions chunk,
and reference-schema validity. Nothing here silently repairs a failure
-- a bad result is always returned as ``ok=False`` with precise errors,
never auto-corrected.
"""

from __future__ import annotations

from src.rag_prep.statute_chunk_validate import (
    StatuteValidationResult,
    is_valid_existing_output,
    validate_statute_chunk_set,
)

__all__ = ["StatuteValidationResult", "is_valid_existing_output", "validate_statute_chunk_set"]
