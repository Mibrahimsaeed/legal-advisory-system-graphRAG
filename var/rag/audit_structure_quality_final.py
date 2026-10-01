#!/usr/bin/env python3
"""Final read-only verification audit, post-regeneration of the one
corrupted file (0383244195c932e90c4d3520.json) found by the previous
audit (var/rag/audits/structure_quality/).

Deliberately reuses audit_structure_quality.py's audit_one() and
supporting logic verbatim via import -- this is a verification pass, not
a redesign, per the task brief's explicit "Do NOT perform another massive
heuristic redesign" / "Use the existing audit logic where appropriate".

Writes to a SEPARATE output directory (var/rag/audits/structure_quality_final/)
so the previous audit's files are never overwritten. Read-only: no
structured/processed JSON, source, database, or pipeline code is touched.
"""

from __future__ import annotations

import csv
import json
import sys
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

import audit_structure_quality as base  # noqa: E402

OUT_DIR = REPO / "var" / "rag" / "audits" / "structure_quality_final"
REGENERATED_DOC_ID = "0383244195c932e90c4d3520"


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    processed_ids = {p.stem for p in base.PROCESSED_DIR.glob("*.json")}
    structured_paths = {
        p.stem: p for p in base.STRUCTURED_DIR.glob("*.json")
        if p.name not in {"progress.json", "full_run_report.json"}
    }
    structured_ids = set(structured_paths.keys())

    missing = sorted(processed_ids - structured_ids)
    unexpected = sorted(structured_ids - processed_ids)

    # Duplicate doc_id check: filename == doc_id by construction (one file
    # per doc_id), so a "duplicate" would only show up as the field inside
    # the JSON disagreeing with another file's filename -- checked per-doc
    # inside audit_one() already (doc_id field vs filename mismatch).

    print(f"Expected (processed corpus): {len(processed_ids)}")
    print(f"Found (structured_gemini/): {len(structured_ids)}")
    print(f"Missing: {len(missing)}")
    print(f"Unexpected: {len(unexpected)}")

    per_doc_results: dict[str, dict] = {}
    all_issues: list[dict] = []
    readiness_counts: Counter = Counter()
    preservation_counts: Counter = Counter()
    status_counts: Counter = Counter()
    fallback_doc_count = 0
    used_llm_count = 0
    not_used_llm_count = 0

    all_doc_ids = sorted(processed_ids)
    for i, doc_id in enumerate(all_doc_ids, 1):
        if i % 400 == 0:
            print(f"  ...{i}/{len(all_doc_ids)}")
        res = base.audit_one(doc_id, base.PROCESSED_DIR / f"{doc_id}.json", structured_paths.get(doc_id))
        per_doc_results[doc_id] = res
        all_issues.extend(res["issues"])
        readiness_counts[res["readiness"]] += 1
        if "preservation" in res:
            preservation_counts[res["preservation"]] += 1
        if res.get("structure_status"):
            status_counts[res["structure_status"]] += 1
        if res.get("llm_fallback_reason"):
            fallback_doc_count += 1
        if res.get("used_llm") is True:
            used_llm_count += 1
        elif res.get("used_llm") is False:
            not_used_llm_count += 1

    for doc_id in unexpected:
        all_issues.append({
            "doc_id": doc_id, "category": "completeness", "severity": "medium",
            "check_type": "deterministic",
            "message": "present in structured_gemini/ but has no corresponding var/rag/processed/ source",
            "evidence": "",
        })

    n_audited = len(all_doc_ids)
    n_structured_present = len(structured_ids & processed_ids)

    exact_pct = round(100 * preservation_counts.get("exact_preservation", 0) / n_audited, 2) if n_audited else 0.0

    regenerated_case = per_doc_results.get(REGENERATED_DOC_ID, {})
    regenerated_path = structured_paths.get(REGENERATED_DOC_ID)
    regenerated_file_size = regenerated_path.stat().st_size if regenerated_path else None

    # Diff against the previous audit's summary, if present.
    prev_summary_path = REPO / "var" / "rag" / "audits" / "structure_quality" / "summary.json"
    prev_summary = json.loads(prev_summary_path.read_text(encoding="utf-8")) if prev_summary_path.exists() else None
    comparison = None
    if prev_summary:
        comparison = {
            "missing_count": {"previous": prev_summary.get("missing_count"), "current": len(missing)},
            "preservation_exact": {
                "previous": prev_summary.get("preservation_distribution", {}).get("exact_preservation"),
                "current": preservation_counts.get("exact_preservation", 0),
            },
            "preservation_unverifiable": {
                "previous": prev_summary.get("preservation_distribution", {}).get("unverifiable", 0),
                "current": preservation_counts.get("unverifiable", 0),
            },
            "readiness_not_ready": {
                "previous": prev_summary.get("readiness_distribution", {}).get("not_ready", 0),
                "current": readiness_counts.get("not_ready", 0),
            },
            "readiness_ready": {
                "previous": prev_summary.get("readiness_distribution", {}).get("ready", 0),
                "current": readiness_counts.get("ready", 0),
            },
        }

    summary = {
        "audit_type": "final_verification_post_regeneration",
        "previous_audit_dir": "var/rag/audits/structure_quality/",
        "expected_corpus_size": len(processed_ids),
        "found_in_structured_gemini": len(structured_ids),
        "missing_count": len(missing),
        "unexpected_count": len(unexpected),
        "documents_audited": n_audited,
        "documents_with_output_present": n_structured_present,
        "structure_status_distribution": dict(status_counts),
        "preservation_distribution": dict(preservation_counts),
        "preservation_exact_pct": exact_pct,
        "readiness_distribution": dict(readiness_counts),
        "readiness_percentages": {
            k: round(100 * v / n_audited, 2) for k, v in readiness_counts.items()
        },
        "docs_with_llm_fallback": fallback_doc_count,
        "docs_with_llm_fallback_pct_of_present": (
            round(100 * fallback_doc_count / n_structured_present, 2) if n_structured_present else 0.0
        ),
        "docs_used_llm": used_llm_count,
        "docs_no_llm_needed": not_used_llm_count,
        "total_issues_found": len(all_issues),
        "issues_by_severity": dict(Counter(i["severity"] for i in all_issues)),
        "issues_by_category": dict(Counter(i["category"] for i in all_issues)),
        "regenerated_case": {
            "doc_id": REGENERATED_DOC_ID,
            "file_size_bytes": regenerated_file_size,
            "structure_status": regenerated_case.get("structure_status"),
            "used_llm": regenerated_case.get("used_llm"),
            "llm_fallback_reason": regenerated_case.get("llm_fallback_reason"),
            "validation_ok_recomputed": regenerated_case.get("validation_ok_recomputed"),
            "validation_ok_stored": regenerated_case.get("validation_ok_stored"),
            "preservation": regenerated_case.get("preservation"),
            "readiness": regenerated_case.get("readiness"),
            "issue_count": len(regenerated_case.get("issues", [])),
        },
        "comparison_to_previous_audit": comparison,
    }

    (OUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (OUT_DIR / "per_doc_results.json").write_text(json.dumps(per_doc_results, indent=2), encoding="utf-8")

    with (OUT_DIR / "issues.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["doc_id", "severity", "category", "check_type", "message", "evidence"])
        writer.writeheader()
        for issue in all_issues:
            writer.writerow(issue)

    _write_report(summary, regenerated_case)

    print("\nFINAL VERIFICATION AUDIT COMPLETE.")
    print(f"  -> {OUT_DIR}")
    print(f"readiness: {dict(readiness_counts)}")
    print(f"regenerated case ({REGENERATED_DOC_ID}): {summary['regenerated_case']}")


def _write_report(summary: dict, regenerated_case: dict) -> None:
    lines = []
    lines.append("# Final Structure Quality Verification Audit\n")
    lines.append("Read-only verification pass after regenerating the one corrupted file "
                  "(`0383244195c932e90c4d3520.json`) found by the previous audit. "
                  "No structured/processed files, code, or database data were modified.\n")

    lines.append("## 1. Corpus completeness\n")
    lines.append(f"- Expected: **{summary['expected_corpus_size']}**")
    lines.append(f"- Found: **{summary['found_in_structured_gemini']}**")
    lines.append(f"- Missing: **{summary['missing_count']}**")
    lines.append(f"- Unexpected: **{summary['unexpected_count']}**\n")

    lines.append("## 2. Exact text preservation\n")
    for k, v in summary["preservation_distribution"].items():
        lines.append(f"- `{k}`: {v}")
    lines.append(f"- Exact preservation rate: **{summary['preservation_exact_pct']}%**\n")

    lines.append("## 3. Structural validity & semantic quality\n")
    lines.append(f"Total issues: **{summary['total_issues_found']}**")
    lines.append(f"By severity: {summary['issues_by_severity']}")
    lines.append(f"By category: {summary['issues_by_category']}\n")

    lines.append("## 4. Regenerated case\n")
    rc = summary["regenerated_case"]
    for k, v in rc.items():
        lines.append(f"- `{k}`: {v}")
    lines.append("")

    lines.append("## 5. Fallback verification\n")
    lines.append(f"- Docs with LLM fallback: **{summary['docs_with_llm_fallback']}** "
                  f"({summary['docs_with_llm_fallback_pct_of_present']}%)")
    lines.append(f"- Docs that used the LLM: **{summary['docs_used_llm']}**")
    lines.append(f"- Docs needing no LLM call: **{summary['docs_no_llm_needed']}**\n")

    lines.append("## 8. Comparison to previous audit\n")
    if summary["comparison_to_previous_audit"]:
        for k, v in summary["comparison_to_previous_audit"].items():
            lines.append(f"- `{k}`: previous={v['previous']}, current={v['current']}")
    else:
        lines.append("- Previous audit summary.json not found; no comparison available.")
    lines.append("")

    lines.append("## 9. Chunking readiness\n")
    for k, v in summary["readiness_distribution"].items():
        pct = summary["readiness_percentages"].get(k, 0)
        lines.append(f"- `{k}`: {v} ({pct}%)")
    lines.append("")

    (OUT_DIR / "audit_report.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
