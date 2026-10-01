#!/usr/bin/env python3
"""Read-only quality audit of the full chunked corpus (var/rag/chunked/)
against its structured source (var/rag/structured_gemini/).

Does not trust the stored `validation.ok` flag alone: for every case,
the ORIGINAL structural spans are re-read from structured_gemini/ and
fed through the real, unmodified validate_chunk_set() again, so a
disagreement between what was stored and what's independently
recomputed would surface here. Additionally checks parent-integrity
(doc_id/full_text/metadata unchanged from the structured source -- a
check validate_chunk_set() itself doesn't make, since it only checks
internal self-consistency) and chunk-ID determinism.

Read-only: never writes to structured_gemini/, chunked/, processed/,
raw/, or var/metadata.db. Only reads them and writes new files under
var/rag/audits/chunk_quality_final/.
"""

from __future__ import annotations

import csv
import json
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from src.rag_prep.chunk_types import OVERLAP_ELIGIBLE_SECTIONS, SOFT_MAX_WORDS, ParentCase  # noqa: E402
from src.rag_prep.chunk_validate import validate_chunk_set  # noqa: E402

STRUCTURED_DIR = REPO / "var" / "rag" / "structured_gemini"
CHUNKED_DIR = REPO / "var" / "rag" / "chunked"
OUT_DIR = REPO / "var" / "rag" / "audits" / "chunk_quality_final"

MANUAL_REVIEW_WORD_THRESHOLD = 3000  # well beyond the already-flagged >1000 soft max


def _load_json(path: Path):
    try:
        return json.loads(path.read_text(encoding="utf-8")), None
    except Exception as exc:  # noqa: BLE001
        return None, f"{type(exc).__name__}: {exc}"


def audit_one(doc_id: str, structured_path: Path, chunked_path: Path | None) -> dict:
    result: dict = {"doc_id": doc_id, "issues": [], "word_counts": [], "sections": [],
                     "overlap_chunks": [], "oversized_chunks": []}

    def flag(category: str, severity: str, message: str, evidence: str = ""):
        result["issues"].append({"doc_id": doc_id, "category": category, "severity": severity,
                                  "message": message, "evidence": evidence[:300]})

    structured_doc, err = _load_json(structured_path)
    if err:
        flag("completeness", "high", f"structured source unreadable: {err}")
        result["readiness"] = "not_ready"
        return result

    if chunked_path is None:
        flag("completeness", "high", "missing chunked output")
        result["readiness"] = "not_ready"
        return result

    chunked_doc, err = _load_json(chunked_path)
    if err:
        flag("completeness", "high", f"chunked output unreadable/corrupted: {err}")
        result["readiness"] = "not_ready"
        return result

    for key in ("parent", "validation"):
        if key not in chunked_doc:
            flag("schema", "high", f"chunked output missing top-level field '{key}'")
    parent_dict = chunked_doc.get("parent", {})
    for key in ("doc_id", "metadata", "full_text", "chunks"):
        if key not in parent_dict:
            flag("schema", "high", f"parent object missing field '{key}'")
    if result["issues"]:
        result["readiness"] = "not_ready"
        return result

    # -- parent integrity vs. the structured source (NOT checked by validate_chunk_set) --
    if parent_dict.get("doc_id") != structured_doc.get("doc_id"):
        flag("parent_integrity", "high", "parent.doc_id differs from structured source doc_id")
    if parent_dict.get("full_text") != structured_doc.get("full_text"):
        flag("parent_integrity", "high", "parent.full_text differs from structured source full_text")
    if parent_dict.get("metadata") != structured_doc.get("metadata"):
        flag("parent_integrity", "high", "parent.metadata differs from structured source metadata")

    try:
        parent = ParentCase.from_dict(parent_dict)
    except Exception as exc:
        flag("schema", "high", f"could not reconstruct ParentCase: {exc}")
        result["readiness"] = "not_ready"
        return result

    # -- independent re-validation using the ORIGINAL spans, not the stored flag --
    spans = structured_doc.get("structure", {}).get("spans", [])
    recomputed = validate_chunk_set(parent, spans)
    stored_ok = chunked_doc.get("validation", {}).get("ok")
    result["stored_validation_ok"] = stored_ok
    result["recomputed_validation_ok"] = recomputed.ok
    if stored_ok != recomputed.ok:
        flag("validator_agreement", "high",
             "stored validation.ok disagrees with independently recomputed validation",
             f"stored={stored_ok} recomputed={recomputed.ok} errors={recomputed.errors}")
    if not recomputed.ok:
        for e in recomputed.errors:
            flag("integrity", "high", e)

    # -- chunk ID determinism, uniqueness, sequential index --
    seen_ids: set[str] = set()
    for idx, c in enumerate(parent.chunks):
        expected_id = f"{doc_id}:{idx:04d}"
        if c.chunk_id != expected_id:
            flag("determinism", "high",
                 f"chunk_id {c.chunk_id!r} != expected deterministic id {expected_id!r}")
        if c.chunk_id in seen_ids:
            flag("determinism", "high", f"duplicate chunk_id {c.chunk_id!r}")
        seen_ids.add(c.chunk_id)
        if c.chunk_index != idx:
            flag("ordering", "high", f"chunk at position {idx} has non-sequential chunk_index {c.chunk_index}")

        word_count = len(c.text.split())  # independently recomputed, not read from c.word_count
        result["word_counts"].append(word_count)
        result["sections"].append(c.section)
        if c.is_overlap:
            result["overlap_chunks"].append({
                "chunk_id": c.chunk_id, "section": c.section, "span_kind": c.span_kind,
                "paragraph_start": c.paragraph_start, "paragraph_end": c.paragraph_end,
                "overlap_paragraph_count": c.overlap_paragraph_count, "word_count": word_count,
            })
            if c.section not in OVERLAP_ELIGIBLE_SECTIONS:
                flag("overlap", "high", f"overlap chunk {c.chunk_id} has section {c.section!r}, "
                                          f"not in OVERLAP_ELIGIBLE_SECTIONS")
            if c.overlap_paragraph_count <= 0:
                flag("overlap", "high", f"overlap chunk {c.chunk_id} has overlap_paragraph_count <= 0")
        if c.oversized_single_paragraph:
            result["oversized_chunks"].append({
                "chunk_id": c.chunk_id, "section": c.section, "span_kind": c.span_kind,
                "paragraph_start": c.paragraph_start, "word_count": word_count,
            })
            if c.paragraph_start != c.paragraph_end:
                flag("oversized", "high",
                     f"oversized chunk {c.chunk_id} spans more than one paragraph "
                     f"[{c.paragraph_start},{c.paragraph_end}]")
            if word_count <= SOFT_MAX_WORDS:
                flag("oversized", "medium",
                     f"chunk {c.chunk_id} flagged oversized_single_paragraph but word_count "
                     f"{word_count} <= SOFT_MAX_WORDS {SOFT_MAX_WORDS}")

    max_word_count = max(result["word_counts"]) if result["word_counts"] else 0
    has_unclassified_or_fallback = any(s in ("unclassified",) for s in result["sections"]) or \
        any(c.span_kind == "paragraph_group" for c in parent.chunks)
    small_chunk_fraction = (
        sum(1 for w in result["word_counts"] if w < 250) / len(result["word_counts"])
        if result["word_counts"] else 0.0
    )

    high_severity = any(i["severity"] == "high" for i in result["issues"])
    if high_severity:
        readiness = "not_ready"
    elif max_word_count > MANUAL_REVIEW_WORD_THRESHOLD:
        readiness = "manual_review_required"
    elif result["oversized_chunks"] or result["overlap_chunks"] or has_unclassified_or_fallback \
            or small_chunk_fraction > 0.5:
        readiness = "ready_with_caution"
    else:
        readiness = "ready"
    result["readiness"] = readiness
    result["max_word_count"] = max_word_count
    result["n_chunks"] = len(parent.chunks)
    result["structure_status"] = structured_doc.get("structure", {}).get("structure_status")

    return result


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    structured_ids = {
        p.stem for p in STRUCTURED_DIR.glob("*.json")
        if p.name not in {"progress.json", "full_run_report.json"}
    }
    chunked_paths = {
        p.stem: p for p in CHUNKED_DIR.glob("*.json") if p.name != "chunking_report.json"
    }
    chunked_ids = set(chunked_paths.keys())

    missing = sorted(structured_ids - chunked_ids)
    unexpected = sorted(chunked_ids - structured_ids)

    print(f"structured input cases: {len(structured_ids)}")
    print(f"chunked case outputs: {len(chunked_ids)}")
    print(f"missing: {len(missing)}")
    print(f"unexpected: {len(unexpected)}")

    per_doc_results: dict[str, dict] = {}
    all_issues: list[dict] = []
    readiness_counts: Counter = Counter()
    section_chunk_counts: Counter = Counter()
    section_case_counts: Counter = defaultdict(set)
    all_word_counts: list[int] = []
    all_overlap_chunks: list[dict] = []
    all_oversized_chunks: list[dict] = []
    stored_vs_recomputed_disagreements = 0
    total_chunks = 0

    all_doc_ids = sorted(structured_ids)
    for i, doc_id in enumerate(all_doc_ids, 1):
        if i % 400 == 0:
            print(f"  ...{i}/{len(all_doc_ids)}")
        res = audit_one(doc_id, STRUCTURED_DIR / f"{doc_id}.json", chunked_paths.get(doc_id))
        per_doc_results[doc_id] = res
        all_issues.extend(res["issues"])
        readiness_counts[res["readiness"]] += 1
        total_chunks += res.get("n_chunks", 0)
        all_word_counts.extend(res["word_counts"])
        for s in res["sections"]:
            section_chunk_counts[s] += 1
            section_case_counts[s].add(doc_id)
        for oc in res["overlap_chunks"]:
            all_overlap_chunks.append({**oc, "doc_id": doc_id})
        for oc in res["oversized_chunks"]:
            all_oversized_chunks.append({**oc, "doc_id": doc_id})
        if res.get("stored_validation_ok") != res.get("recomputed_validation_ok"):
            stored_vs_recomputed_disagreements += 1

    for doc_id in unexpected:
        all_issues.append({"doc_id": doc_id, "category": "completeness", "severity": "medium",
                            "message": "present in chunked/ but no corresponding structured source",
                            "evidence": ""})

    def pct(n, d):
        return round(100 * n / d, 2) if d else 0.0

    wc_sorted = sorted(all_word_counts)
    n_wc = len(wc_sorted)

    def percentile(p):
        if not n_wc:
            return None
        k = max(0, min(n_wc - 1, int(round(p / 100 * (n_wc - 1)))))
        return wc_sorted[k]

    size_buckets = {
        "<250": sum(1 for w in all_word_counts if w < 250),
        "250-499": sum(1 for w in all_word_counts if 250 <= w <= 499),
        "500-800": sum(1 for w in all_word_counts if 500 <= w <= 800),
        "801-1000": sum(1 for w in all_word_counts if 801 <= w <= 1000),
        ">1000": sum(1 for w in all_word_counts if w > 1000),
    }

    chunks_per_case = [r["n_chunks"] for r in per_doc_results.values() if "n_chunks" in r]

    cases_with_missing_content = sum(
        1 for r in per_doc_results.values()
        if any("missing paragraph" in i["message"] for i in r["issues"])
    )
    cases_with_unintended_duplication = sum(
        1 for r in per_doc_results.values()
        if any("unintended duplicated" in i["message"] for i in r["issues"])
    )
    cases_with_intentional_overlap = sum(1 for r in per_doc_results.values() if r["overlap_chunks"])

    summary = {
        "structured_cases": len(structured_ids),
        "chunked_cases": len(chunked_ids),
        "missing_count": len(missing),
        "missing_doc_ids": missing,
        "unexpected_count": len(unexpected),
        "unexpected_doc_ids": unexpected,
        "total_chunks_recounted": total_chunks,
        "readiness_distribution": dict(readiness_counts),
        "readiness_percentages": {k: pct(v, len(all_doc_ids)) for k, v in readiness_counts.items()},
        "integrity": {
            "exact_text_preservation_cases": sum(
                1 for r in per_doc_results.values() if r.get("recomputed_validation_ok") is True
            ),
            "offset_or_integrity_failures": sum(
                1 for r in per_doc_results.values() if r.get("recomputed_validation_ok") is False
            ),
            "stored_vs_recomputed_disagreements": stored_vs_recomputed_disagreements,
            "cases_with_missing_content": cases_with_missing_content,
            "cases_with_unintended_duplication": cases_with_unintended_duplication,
            "cases_with_intentional_overlap": cases_with_intentional_overlap,
        },
        "diagnostics": {
            "oversized_chunk_count": len(all_oversized_chunks),
            "overlap_chunk_count": len(all_overlap_chunks),
        },
        "size_distribution_words": {
            "min": wc_sorted[0] if n_wc else None,
            "median": statistics.median(wc_sorted) if n_wc else None,
            "mean": round(statistics.mean(wc_sorted), 2) if n_wc else None,
            "p90": percentile(90),
            "p95": percentile(95),
            "max": wc_sorted[-1] if n_wc else None,
            "buckets": size_buckets,
            "bucket_percentages": {k: pct(v, n_wc) for k, v in size_buckets.items()},
        },
        "chunks_per_case": {
            "min": min(chunks_per_case) if chunks_per_case else None,
            "median": statistics.median(chunks_per_case) if chunks_per_case else None,
            "max": max(chunks_per_case) if chunks_per_case else None,
        },
        "section_distribution": {
            "chunk_counts": dict(section_chunk_counts),
            "chunk_percentages": {k: pct(v, total_chunks) for k, v in section_chunk_counts.items()},
            "case_counts": {k: len(v) for k, v in section_case_counts.items()},
        },
        "total_issues_found": len(all_issues),
        "issues_by_severity": dict(Counter(i["severity"] for i in all_issues)),
        "issues_by_category": dict(Counter(i["category"] for i in all_issues)),
        "overlap_chunks_detail": all_overlap_chunks,
        "oversized_chunks_detail": all_oversized_chunks,
    }

    (OUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    per_doc_output = {
        doc_id: {k: v for k, v in r.items() if k not in ("word_counts", "sections")}
        for doc_id, r in per_doc_results.items()
    }
    (OUT_DIR / "per_doc_results.json").write_text(json.dumps(per_doc_output, indent=2), encoding="utf-8")

    with (OUT_DIR / "issues.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["doc_id", "severity", "category", "message", "evidence"])
        writer.writeheader()
        for issue in all_issues:
            writer.writerow(issue)

    _write_report(summary)

    print("\nAUDIT COMPLETE.")
    print(f"readiness: {dict(readiness_counts)}")
    print(f"integrity: {summary['integrity']}")


def _write_report(summary: dict) -> None:
    lines = []
    lines.append("# Full Chunk Quality Audit Report\n")
    lines.append("Read-only audit of `var/rag/chunked/` against `var/rag/structured_gemini/`. "
                  "No chunked/structured/processed/raw files, code, or database data were modified.\n")

    lines.append("## 1. Corpus completeness\n")
    lines.append(f"- Structured cases: **{summary['structured_cases']}**")
    lines.append(f"- Chunked cases: **{summary['chunked_cases']}**")
    lines.append(f"- Missing: **{summary['missing_count']}**")
    lines.append(f"- Unexpected: **{summary['unexpected_count']}**\n")

    lines.append("## Integrity (independently recomputed, not trusting stored validation.ok)\n")
    for k, v in summary["integrity"].items():
        lines.append(f"- `{k}`: {v}")
    lines.append("")

    lines.append("## Diagnostics\n")
    for k, v in summary["diagnostics"].items():
        lines.append(f"- `{k}`: {v}")
    lines.append("")

    lines.append("## Size distribution (words/chunk, independently recomputed from chunk text)\n")
    sd = summary["size_distribution_words"]
    lines.append(f"- min={sd['min']} median={sd['median']} mean={sd['mean']} "
                  f"p90={sd['p90']} p95={sd['p95']} max={sd['max']}")
    for k, v in sd["buckets"].items():
        lines.append(f"- `{k}`: {v} ({sd['bucket_percentages'][k]}%)")
    lines.append("")

    lines.append("## Section distribution\n")
    for k, v in summary["section_distribution"]["chunk_counts"].items():
        cases = summary["section_distribution"]["case_counts"].get(k, 0)
        pct_v = summary["section_distribution"]["chunk_percentages"].get(k, 0)
        lines.append(f"- `{k}`: {v} chunks ({pct_v}%), present in {cases} cases")
    lines.append("")

    lines.append("## Readiness\n")
    for k, v in summary["readiness_distribution"].items():
        lines.append(f"- `{k}`: {v} ({summary['readiness_percentages'].get(k, 0)}%)")
    lines.append("")

    lines.append("## Issues\n")
    lines.append(f"Total: **{summary['total_issues_found']}**, by severity: "
                  f"{summary['issues_by_severity']}, by category: {summary['issues_by_category']}\n")

    lines.append("## Overlap chunks (all, individually)\n")
    for oc in summary["overlap_chunks_detail"]:
        lines.append(f"- `{oc['doc_id']}` / `{oc['chunk_id']}` section={oc['section']} "
                      f"overlap_paragraph_count={oc['overlap_paragraph_count']} word_count={oc['word_count']}")
    lines.append("")

    lines.append("## Oversized chunks (all, individually)\n")
    for oc in summary["oversized_chunks_detail"]:
        lines.append(f"- `{oc['doc_id']}` / `{oc['chunk_id']}` section={oc['section']} "
                      f"paragraph={oc['paragraph_start']} word_count={oc['word_count']}")
    lines.append("")

    (OUT_DIR / "audit_report.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
