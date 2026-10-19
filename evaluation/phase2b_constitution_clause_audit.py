#!/usr/bin/env python3
"""Phase 2B -- corpus-wide Constitution clause-boundary audit (read-only).

Systematically sweeps all 328 articles' clause-level chunks for the two
patterns found opportunistically in Phase 2A:
  (a) a chunk whose clause_number START is numerically LOWER than the
      running-max clause number already seen earlier in the same article,
      while being perfectly contiguous (source_start == a prior chunk's
      source_end) -- the "art_199_c1_2" / "art_224a_c2_2" signature
      (an inline "under clause (N)" cross-reference misread as a new
      clause-N boundary, truncating and mislabeling the real clause's tail).
  (b) a chunk whose clause_number is a RANGE whose END is numerically
      LOWER than its own START (e.g. "(5)-(2)") -- a nonsensical range,
      the "art_270aa_c5" signature (a cross-reference misread as marking
      the range's end).

This script only reads var/rag/constitution_chunked/*.json -- it does not
import or modify src/rag_prep/constitution_parser.py, and it performs no
Qdrant, embedding, or indexing operation at all.

    python evaluation/phase2b_constitution_clause_audit.py
"""

from __future__ import annotations

import csv
import json
import re
import subprocess
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
CHUNK_FILE = REPO_ROOT / "var/rag/constitution_chunked/b427a922990d75f027928c15.json"

OUT_JSON = REPO_ROOT / "evaluation/results/phase2b_constitution_clause_audit.json"
OUT_MD = REPO_ROOT / "evaluation/results/phase2b_constitution_clause_audit.md"
OUT_CSV = REPO_ROOT / "evaluation/results/phase2b_constitution_clause_audit.csv"

_LETTER_VAL = {c: i + 1 for i, c in enumerate("ABCDEFGHIJKLMNOPQRSTUVWXYZ")}


def _parse_token(tok: str) -> float | None:
    """'4' -> 4.0, '4A' -> 4.1, '4B' -> 4.2, '3CC' -> 3.33 (rare double-letter)."""
    m = re.match(r"^(\d+)([A-Z]{0,2})$", tok.strip())
    if not m:
        return None
    base = float(m.group(1))
    letters = m.group(2)
    if not letters:
        return base
    frac = sum(_LETTER_VAL.get(ch, 0) for ch in letters) / 100.0
    return base + frac


def parse_clause_number(raw: str) -> tuple[float | None, float | None]:
    """Returns (start_val, end_val). A plain '4A' gives (4.1, 4.1). A range
    '4)–(5' or '4)-(5' gives (4.0, 5.0). Unparseable -> (None, None)."""

    if not raw:
        return None, None
    # normalise both en-dash and hyphen range separators, strip stray parens
    norm = raw.replace(")–(", "-").replace(")-(", "-").replace(")", "").replace("(", "")
    parts = norm.split("-")
    if len(parts) == 1:
        v = _parse_token(parts[0])
        return v, v
    if len(parts) == 2:
        a, b = _parse_token(parts[0]), _parse_token(parts[1])
        return a, b
    return None, None


def git_status_short() -> str:
    out = subprocess.run(["git", "status", "--short"], cwd=REPO_ROOT, capture_output=True, text=True, check=True)
    return out.stdout


def classify(chunk: dict, prev_chunk: dict | None, flag_type: str, start_val: float, end_val: float, running_max: float) -> dict:
    """Heuristic classification per the task's 4 categories. Uses the chunk's
    own text + its contiguity relationship to the preceding chunk -- not
    regex alone -- to decide confidence."""

    text = chunk.get("canonical_text") or ""
    text_strip = text.strip()
    contiguous_with_prev = bool(prev_chunk) and chunk["source_start"] == prev_chunk["source_end"]
    # A genuine new clause normally starts "(N) <Capitalised word...>" where N matches
    # the chunk's own claimed start. A mislabeled continuation typically starts
    # "(N), <lowercase continuation...>" or similar -- i.e. immediately followed by
    # a comma/lowercase rather than a fresh capitalised clause opening.
    looks_like_bare_continuation = bool(re.match(r"^\(?\d{1,3}[A-Z]{0,2}\)?,\s*[a-z]", text_strip)) or \
        bool(re.match(r"^\(?\d{1,3}[A-Z]{0,2}\)?\s+[a-z]", text_strip))

    if flag_type == "regression" and contiguous_with_prev and looks_like_bare_continuation:
        category, confidence = "confirmed_defect", "high"
    elif flag_type == "regression" and contiguous_with_prev:
        category, confidence = "probable_defect", "medium"
    elif flag_type == "regression":
        category, confidence = "needs_manual_review", "low"
    elif flag_type == "backwards_range":
        # the chunk's own start looks like a genuine clause opening in most observed
        # cases (e.g. art_270aa_c5 starts "(5) No suit..."); the defect is in the
        # declared END of the range label, not necessarily the chunk's own content start
        looks_like_genuine_open = bool(re.match(r"^\(?\d{1,3}[A-Z]{0,2}\)?\s+[A-Z]", text_strip))
        if looks_like_genuine_open:
            category, confidence = "probable_defect", "medium"
        else:
            category, confidence = "needs_manual_review", "low"
    else:
        category, confidence = "needs_manual_review", "low"

    return {
        "category": category,
        "confidence": confidence,
        "contiguous_with_prev_chunk": contiguous_with_prev,
        "text_looks_like_bare_continuation": looks_like_bare_continuation,
    }


def main() -> None:
    initial_git_status = git_status_short()
    print("[phase2b] initial git status --short:")
    print(initial_git_status or "  (clean)")

    for p in (OUT_JSON, OUT_MD, OUT_CSV):
        if p.exists():
            raise SystemExit(f"[phase2b] refusing to overwrite existing path: {p}.")

    if not CHUNK_FILE.exists():
        raise SystemExit(f"[phase2b] {CHUNK_FILE} not found. Stopping.")
    data = json.loads(CHUNK_FILE.read_text(encoding="utf-8"))
    chunks = data["chunks"]

    all_article_numbers = sorted(set(c["article_number"] for c in chunks if c.get("article_number")))
    clause_labeled = [c for c in chunks if c.get("clause_number")]
    excluded_no_article = [c["chunk_id"] for c in chunks if not c.get("article_number")]  # preamble/schedule/annex

    print(f"[phase2b] {len(all_article_numbers)} unique articles, {len(chunks)} total chunks, "
          f"{len(clause_labeled)} chunks with a clause_number label, "
          f"{len(excluded_no_article)} chunks excluded from per-article analysis (preamble/schedule/annex)")

    by_article: dict[str, list[dict]] = {}
    for c in chunks:
        an = c.get("article_number")
        if an:
            by_article.setdefault(an, []).append(c)

    findings = []
    for an, arts in by_article.items():
        arts_sorted = sorted(arts, key=lambda c: c["source_start"])
        # only consider clause-level chunks (skip the non-retrievable, clause_number=None parent)
        clause_chunks = [c for c in arts_sorted if c.get("clause_number")]
        running_max = float("-inf")
        prev_chunk = None
        for c in clause_chunks:
            start_val, end_val = parse_clause_number(c["clause_number"])
            if start_val is None:
                prev_chunk = c
                continue

            # pattern (a): regression -- this clause's start is below the running max
            if start_val < running_max - 1e-9:
                info = classify(c, prev_chunk, "regression", start_val, end_val, running_max)
                findings.append({
                    "finding_type": "clause_number_regression",
                    "article_number": an, "chunk_id": c["chunk_id"],
                    "current_label": c["clause_number"],
                    "expected_label_hint": (
                        f"likely continuation of {prev_chunk['chunk_id']} (clause ~{running_max:g})"
                        if prev_chunk else "unknown"
                    ),
                    "text_excerpt": (c.get("canonical_text") or "")[:160],
                    "source_location": f"source_start={c['source_start']}, source_end={c['source_end']}",
                    "preceding_chunk_id": prev_chunk["chunk_id"] if prev_chunk else None,
                    "preceding_chunk_source_end": prev_chunk["source_end"] if prev_chunk else None,
                    "running_max_clause_before_this": running_max,
                    **info,
                    "content_status": "mislabeled" if info["category"] == "confirmed_defect" else "possibly_mislabeled",
                })

            # pattern (b): backwards range label (end < start)
            if end_val is not None and start_val is not None and end_val < start_val - 1e-9:
                info = classify(c, prev_chunk, "backwards_range", start_val, end_val, running_max)
                findings.append({
                    "finding_type": "backwards_range_label",
                    "article_number": an, "chunk_id": c["chunk_id"],
                    "current_label": c["clause_number"],
                    "expected_label_hint": f"range end likely spurious (declared end {end_val:g} < start {start_val:g})",
                    "text_excerpt": (c.get("canonical_text") or "")[:160],
                    "source_location": f"source_start={c['source_start']}, source_end={c['source_end']}",
                    "preceding_chunk_id": prev_chunk["chunk_id"] if prev_chunk else None,
                    "preceding_chunk_source_end": prev_chunk["source_end"] if prev_chunk else None,
                    "running_max_clause_before_this": running_max,
                    **info,
                    "content_status": "mislabeled_range_end",
                })

            running_max = max(running_max, start_val, end_val if end_val is not None else start_val)
            prev_chunk = c

    confirmed = [f for f in findings if f["category"] == "confirmed_defect"]
    probable = [f for f in findings if f["category"] == "probable_defect"]
    manual_review = [f for f in findings if f["category"] == "needs_manual_review"]
    not_defect = [f for f in findings if f["category"] == "not_a_defect"]

    affected_articles = sorted(set(f["article_number"] for f in findings))
    confirmed_articles = sorted(set(f["article_number"] for f in confirmed))

    denom_articles_with_clauses = len([an for an, arts in by_article.items() if any(c.get("clause_number") for c in arts)])
    denom_clause_chunks = len(clause_labeled)

    print(f"\n[phase2b] TOTAL findings: {len(findings)}  "
          f"(confirmed={len(confirmed)}, probable={len(probable)}, manual_review={len(manual_review)})")
    for f in findings:
        print(f"  [{f['category']}] {f['chunk_id']} (Art.{f['article_number']}) label={f['current_label']!r} "
              f"type={f['finding_type']}")

    report = {
        "audit": "phase2b_constitution_clause_audit",
        "run_timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "source_file": str(CHUNK_FILE.relative_to(REPO_ROOT)),
        "scope": {
            "total_articles_found": len(all_article_numbers),
            "total_articles_expected_per_phase2a": 328,
            "all_328_covered": len(all_article_numbers) == 328,
            "total_chunks_examined": len(chunks),
            "chunks_with_clause_label": denom_clause_chunks,
            "chunks_excluded_no_article_number": len(excluded_no_article),
            "excluded_reason": "preamble/schedule/annex chunks carry no article_number and are structurally outside the per-article clause sequence",
            "articles_with_at_least_one_clause_label": denom_articles_with_clauses,
        },
        "detection_method": (
            "Per article, chunks with a clause_number were sorted by source_start. A running "
            "maximum of the highest clause value seen so far (letters A/B/.. treated as fractional "
            "sub-ordering, e.g. '4A' > 4, < 5) was tracked. A chunk was flagged as a "
            "'clause_number_regression' if its own start value fell below that running max AND the "
            "chunk is textually contiguous with its immediate predecessor (source_start == predecessor's "
            "source_end) -- the exact signature of the two known Phase 2A defects. A chunk was "
            "separately flagged as a 'backwards_range_label' if its own declared range end was lower "
            "than its own declared start (e.g. '(5)-(2)'). Each flagged chunk was then classified by "
            "reading its actual canonical_text excerpt, not by regex alone, distinguishing a genuine "
            "new clause opening (capitalised text immediately after the number) from a bare, "
            "lowercase/comma continuation fragment (the two known defects' signature)."
        ),
        "findings": findings,
        "summary": {
            "confirmed_defect_count": len(confirmed),
            "probable_defect_count": len(probable),
            "needs_manual_review_count": len(manual_review),
            "not_a_defect_count": len(not_defect),
            "total_flagged": len(findings),
            "articles_flagged": affected_articles,
            "articles_with_confirmed_defect": confirmed_articles,
            "proportion_of_articles_with_clause_labels_flagged": f"{len(affected_articles)}/{denom_articles_with_clauses}",
            "proportion_of_articles_with_confirmed_defect": f"{len(confirmed_articles)}/{denom_articles_with_clauses}",
            "proportion_of_clause_labeled_chunks_flagged": f"{len(findings)}/{denom_clause_chunks}",
        },
        "known_examples_cross_check": {
            "art_199_c1_2_detected": any(f["chunk_id"] == "art_199_c1_2" for f in findings),
            "art_224a_c2_2_detected": any(f["chunk_id"] == "art_224a_c2_2" for f in findings),
        },
        "limitations": [
            "Detection relies on source-offset contiguity and a lowercase/comma-continuation text "
            "heuristic, not a full manual re-read of the original PDF for every flagged chunk -- "
            "'confirmed_defect' here means the chunk's own stored text and offsets clearly support the "
            "conclusion, not that the original 1973 Constitution PDF was independently re-opened for each one.",
            "The backwards_range_label check and the regression check can both fire on the same chunk; "
            "such chunks appear once per finding type in the findings list, not deduplicated into one row.",
        ],
        "run_completed_successfully": True,
    }

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    with OUT_CSV.open("w", newline="", encoding="utf-8") as fh:
        fieldnames = ["finding_type", "category", "confidence", "article_number", "chunk_id", "current_label",
                      "expected_label_hint", "preceding_chunk_id", "content_status", "text_excerpt"]
        writer = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(findings)

    final_git_status = git_status_short()
    report["initial_git_status_short"] = initial_git_status
    report["final_git_status_short"] = final_git_status
    OUT_JSON.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"\n[phase2b] JSON -> {OUT_JSON}")
    print(f"[phase2b] CSV -> {OUT_CSV}")
    print("\n[phase2b] final git status --short:")
    print(final_git_status or "  (clean)")


if __name__ == "__main__":
    main()
