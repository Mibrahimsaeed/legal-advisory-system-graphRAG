"""Internal/external statute cross-reference extraction -- thin, stable
import path over the real implementation.

The actual deterministic, regex-only extraction logic lives in
:func:`src.rag_prep.statute_chunker.extract_references` (built and
validated there first, directly against the 9 real cleaned statutes,
since a chunk's references are computed at chunk-construction time and
need to live right next to the chunking logic that calls them per
chunk). This module exists only so callers that expect a dedicated
``reference_extractor`` entry point have one, per the project's
"incremental addition" decision: avoid duplicating or forking the real,
validated logic into a second implementation.

No new behavior, no new patterns, no LLM. See
``src.rag_prep.statute_chunker``'s module docstring for the extraction
rules themselves (internal-section-reference and external-statute-
reference regexes, their real-document evidence, and what is
deliberately NOT extracted -- dates, bare numbers, case citations).
"""

from __future__ import annotations

from src.rag_prep.statute_chunker import extract_references

__all__ = ["extract_references"]
