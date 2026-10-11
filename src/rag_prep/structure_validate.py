"""Layer 4: no-content-loss validation. Mandatory, runs on every document.

Checks, against the paragraph-index coordinate system established in
:mod:`structure_types`:

1. **Coverage** -- every paragraph index ``0..len(paragraphs)-1`` is
   covered by exactly one span (no gap).
2. **No overlap** -- no paragraph index is covered by more than one span.
3. **No duplication** -- same as (2), stated the other direction.
4. **Ordering** -- spans, sorted by ``paragraph_start``, appear in strictly
   increasing, non-overlapping order (a structural guarantee that follows
   from 1+2, checked directly anyway as its own assertion).
5. **No rewriting** -- every span's recorded ``text`` is exactly
   ``"\\n\\n".join(paragraphs[start:end+1])``; nothing the LLM might have
   echoed is trusted as a substitute.
6. **Full reconstruction** -- concatenating every span's text in
   ``paragraph_start`` order, joined by ``"\\n\\n"``, reproduces
   ``full_text`` exactly.

A document that fails any of these does not get a partial/corrupted
"structured" result -- the caller (structurer.py) is expected to fall back
to the always-valid paragraph-group identity mapping for that document
rather than emit something this validator rejects.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from src.rag_prep.structure_types import Span


@dataclass
class ValidationResult:
    ok: bool
    errors: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {"ok": self.ok, "errors": self.errors}


def validate_spans(spans: list[Span], paragraphs: list[str], full_text: str) -> ValidationResult:
    errors: list[str] = []
    n = len(paragraphs)

    if not spans:
        return ValidationResult(ok=False, errors=["no spans produced"])

    # Bounds + per-span text correctness.
    for s in spans:
        if s.paragraph_start < 0 or s.paragraph_end >= n or s.paragraph_end < s.paragraph_start:
            errors.append(f"invalid range [{s.paragraph_start},{s.paragraph_end}] for n={n}")
            continue
        expected_text = "\n\n".join(paragraphs[s.paragraph_start : s.paragraph_end + 1])
        if s.text != expected_text:
            errors.append(
                f"span [{s.paragraph_start},{s.paragraph_end}] text does not match "
                f"the source paragraphs (possible rewrite/truncation)"
            )

    if errors:
        return ValidationResult(ok=False, errors=errors)

    # Coverage + overlap + ordering, via sorted spans.
    ordered = sorted(spans, key=lambda s: s.paragraph_start)
    covered: set[int] = set()
    cursor = -1
    for s in ordered:
        if s.paragraph_start <= cursor:
            errors.append(
                f"overlap or out-of-order span: [{s.paragraph_start},{s.paragraph_end}] "
                f"after cursor={cursor}"
            )
        span_range = set(range(s.paragraph_start, s.paragraph_end + 1))
        duplicated = span_range & covered
        if duplicated:
            errors.append(f"duplicated paragraph index(es): {sorted(duplicated)[:5]}")
        covered |= span_range
        cursor = s.paragraph_end

    missing = set(range(n)) - covered
    if missing:
        errors.append(f"missing paragraph index(es) -- content dropped: {sorted(missing)[:10]}")

    if errors:
        return ValidationResult(ok=False, errors=errors)

    # Full reconstruction, in paragraph order (not span-declaration order).
    reconstructed = "\n\n".join(paragraphs[s.paragraph_start] if s.paragraph_start == s.paragraph_end
                                 else "\n\n".join(paragraphs[s.paragraph_start:s.paragraph_end + 1])
                                 for s in ordered)
    # The line above re-derives text from paragraphs directly (not from
    # s.text) so this check is independent of whether span.text happened
    # to be built correctly -- a true source-of-truth reconstruction.
    if reconstructed != full_text:
        errors.append("full reconstruction does not match full_text byte-for-byte")
        return ValidationResult(ok=False, errors=errors)

    return ValidationResult(ok=True, errors=[])


def identity_fallback_spans(paragraphs: list[str]) -> list[Span]:
    """The always-valid fallback: one paragraph_group span per paragraph,
    in order. Used when semantic structuring fails validation -- this
    trivially passes validate_spans() by construction, since it's the
    identity mapping the reconstruction check is defined against."""

    return [
        Span(kind="paragraph_group", paragraph_start=i, paragraph_end=i, text=p, label="unclassified")
        for i, p in enumerate(paragraphs)
    ]
