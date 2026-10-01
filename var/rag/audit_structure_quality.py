#!/usr/bin/env python3
"""Read-only structural-quality audit of var/rag/structured_gemini/.

Does not modify any processed/structured JSON, source files, the database,
or any pipeline code. Reads only. Reuses the real Span/validate_spans
implementation from src.rag_prep so "independently recomputed" validation
means "re-run the actual validator against freshly reconstructed Span
objects from the stored JSON, against the PROCESSED source's full_text as
ground truth" -- not a reimplementation that could silently diverge from
what the pipeline itself considers valid.

Outputs (only under var/rag/audits/structure_quality/):
    summary.json            corpus-wide metrics and readiness counts
    per_doc_results.json    detailed checks for every doc_id
    issues.csv              actionable issue list
    manual_review_queue.csv prioritized + stratified-random review sample
    audit_report.md         readable narrative report
"""

from __future__ import annotations

import csv
import json
import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from src.rag_prep.structure_types import STATUS_FALLBACK  # noqa: E402
from src.rag_prep.structure_types import (  # noqa: E402
    SEMANTIC_LABELS,
    STATUS_INCOMPLETE_SOURCE,
    STATUS_NON_JUDGMENT_TEXT,
    STATUS_STRUCTURED,
    Span,
)
from src.rag_prep.structure_validate import validate_spans  # noqa: E402

PROCESSED_DIR = REPO / "var" / "rag" / "processed"
STRUCTURED_DIR = REPO / "var" / "rag" / "structured_gemini"
OUT_DIR = REPO / "var" / "rag" / "audits" / "structure_quality"

ALLOWED_STATUSES = {STATUS_STRUCTURED, STATUS_FALLBACK, STATUS_INCOMPLETE_SOURCE, STATUS_NON_JUDGMENT_TEXT}
ALLOWED_KINDS = {
    "case_caption", "headnotes", "judgment_marker", "semantic_section",
    "quoted_material", "paragraph_group", "final_order",
}

# The 12 doc_ids deliberately selected as representative/difficult cases
# during the original Qwen structuring pilot (see var/rag/build_structured_pilot.py)
# -- called out by name in the audit so their results are easy to find.
KNOWN_HARD_DOC_IDS = {
    "8dd0ca3d14364404d75b76fe": "headnotes+JUDGMENT, quoted lower-court order",
    "00ad7248bf77883e6b5975ed": "headnotes+ORDER, prose body",
    "a3ec28ba4900855d15a6e870": "no heading, continuous prose, 22K chars",
    "1d0a3afecc805e3d71cb54b9": "no headnotes, numbered-paragraph body",
    "0513f7c1139b729f4cccf83d": "no headnotes, ORDER, continuous prose",
    "04139076ff36792b38b72f8b": "multi-matter, 111K chars, high quote density",
    "177dbfa9f6115c8c4bb0cdcf": "longest in corpus, 193K chars",
    "f7afd3f61f59d8f9798b8742": "incomplete_source",
    "b3b83bb2ceb23d88cf08394a": "incomplete_source",
    "ae4a07835728a54c3dddb4ad": "non_judgment_text (academic article)",
    "2ee9704f426892ca99e2b92e": "shortest in corpus, 1,698 chars",
    "008d928c2926b2a5855bcc36": "headnotes+JUDGMENT+numbered body",
}

_COUNSEL_SUBMISSION_RE = re.compile(
    r"\b(learned\s+)?counsel\s+for\s+(the\s+)?\w+.{0,40}\b(contended|argued|submitted|urged|pleaded)\b",
    re.IGNORECASE,
)
_DISPOSITION_WORD_RE = re.compile(
    r"\b(allow|dismiss|grant|reject|refus|disallow|remand|transfer|withdraw|abate|"
    r"infructuous|disposed|accordingly|set aside|convert|decree)\w*\b",
    re.IGNORECASE,
)
_PROVISION_NORMALIZE_RE = re.compile(r"^(sections?|ss?\.)\s*", re.IGNORECASE)


def _normalize_provision(raw: str) -> str:
    s = raw.strip().lower()
    s = _PROVISION_NORMALIZE_RE.sub("", s)
    s = s.replace(" ", "").replace(".", "")
    return s


def _load_json(path: Path):
    try:
        return json.loads(path.read_text(encoding="utf-8")), None
    except Exception as exc:  # noqa: BLE001 -- audit must survive any malformed file
        return None, f"{type(exc).__name__}: {exc}"


def _spans_from_json(span_dicts: list[dict]) -> list[Span]:
    spans = []
    for d in span_dicts:
        spans.append(Span(
            kind=d.get("type"),
            paragraph_start=d.get("paragraph_start"),
            paragraph_end=d.get("paragraph_end"),
            text=d.get("text", ""),
            label=d.get("label"),
            confidence=d.get("confidence"),
            attribution=d.get("attribution"),
        ))
    return spans


def _classify_preservation(recomputed_ok: bool, errors: list[str]) -> str:
    if recomputed_ok:
        return "exact_preservation"
    joined = " | ".join(errors)
    if "missing paragraph index" in joined:
        return "possible_omission"
    if "duplicated paragraph index" in joined:
        return "possible_duplication"
    if "text does not match" in joined:
        return "possible_omission"  # span.text disagrees with the real slice
    if "overlap or out-of-order" in joined:
        return "ordering_error"
    if "full reconstruction does not match" in joined:
        return "possible_omission"
    if "invalid range" in joined:
        return "possible_omission"
    return "possible_omission"


def _identity_fallback_shape_ok(span_dicts: list[dict], n_paras: int) -> bool:
    if len(span_dicts) != n_paras:
        return False
    for i, d in enumerate(span_dicts):
        if d.get("type") != "paragraph_group" or d.get("label") != "unclassified":
            return False
        if d.get("paragraph_start") != i or d.get("paragraph_end") != i:
            return False
    return True


def audit_one(doc_id: str, processed_path: Path, structured_path: Path | None) -> dict:
    result: dict = {"doc_id": doc_id, "issues": []}

    def flag(category: str, severity: str, check_type: str, message: str, evidence: str = ""):
        result["issues"].append({
            "doc_id": doc_id, "category": category, "severity": severity,
            "check_type": check_type, "message": message, "evidence": evidence[:300],
        })

    processed_doc, proc_err = _load_json(processed_path)
    if proc_err:
        flag("completeness", "high", "deterministic", f"processed source unreadable: {proc_err}")
        result["preservation"] = "unverifiable"
        result["readiness"] = "not_ready"
        return result

    if structured_path is None:
        flag("completeness", "high", "deterministic", "missing from structured_gemini/ (not yet processed)")
        result["preservation"] = "unverifiable"
        result["readiness"] = "not_yet_structured"
        return result

    structured_doc, struct_err = _load_json(structured_path)
    if struct_err:
        flag("completeness", "high", "deterministic", f"structured output unreadable/corrupted: {struct_err}")
        result["preservation"] = "unverifiable"
        result["readiness"] = "not_ready"
        return result

    # -- completeness: required fields / doc_id match ------------------------
    for field in ("doc_id", "metadata", "full_text", "structure"):
        if field not in structured_doc:
            flag("completeness", "high", "deterministic", f"missing required top-level field '{field}'")
    if structured_doc.get("doc_id") != doc_id:
        flag("completeness", "high", "deterministic",
             f"doc_id field '{structured_doc.get('doc_id')}' != filename '{doc_id}'")

    structure = structured_doc.get("structure", {})
    status = structure.get("structure_status")
    result["structure_status"] = status
    if status not in ALLOWED_STATUSES:
        flag("structural_validity", "high", "deterministic", f"unrecognized structure_status '{status}'")

    # -- full_text / metadata identity vs. processed source -------------------
    proc_full_text = processed_doc.get("full_text", "")
    struct_full_text = structured_doc.get("full_text", "")
    if struct_full_text != proc_full_text:
        flag("preservation", "high", "deterministic",
             "structured full_text differs from processed source full_text (byte-for-byte)")
    if structured_doc.get("metadata") != processed_doc.get("metadata"):
        flag("metadata", "high", "deterministic",
             "structured metadata block differs from processed source metadata block")

    result["doc_length_chars"] = len(proc_full_text)
    paras = proc_full_text.split("\n\n")
    result["paragraph_count"] = len(paras)

    span_dicts = structure.get("spans", [])
    spans = _spans_from_json(span_dicts)

    # -- preservation: independently re-run the real validator ----------------
    recomputed = validate_spans(spans, paras, proc_full_text)
    stored_ok = structure.get("validation", {}).get("ok")
    result["validation_ok_recomputed"] = recomputed.ok
    result["validation_ok_stored"] = stored_ok
    if stored_ok != recomputed.ok:
        flag("structural_validity", "high", "deterministic",
             "stored validation.ok disagrees with independently recomputed validation",
             f"stored={stored_ok} recomputed={recomputed.ok} errors={recomputed.errors}")

    result["preservation"] = _classify_preservation(recomputed.ok, recomputed.errors)
    if result["preservation"] != "exact_preservation":
        flag("preservation", "high", "deterministic",
             f"preservation classified as {result['preservation']}",
             "; ".join(recomputed.errors)[:280])

    # -- structural validity: kinds/labels within the allowed vocabulary ------
    paragraph_group_count = 0
    for d in span_dicts:
        kind = d.get("type")
        label = d.get("label")
        if kind not in ALLOWED_KINDS:
            flag("structural_validity", "medium", "deterministic", f"unrecognized span kind '{kind}'")
        if kind == "paragraph_group":
            paragraph_group_count += 1
            if label != "unclassified":
                flag("structural_validity", "low", "deterministic",
                     f"paragraph_group span has unexpected label '{label}' (expected 'unclassified')")
        elif kind in ("semantic_section", "quoted_material"):
            if label not in SEMANTIC_LABELS:
                flag("structural_validity", "medium", "deterministic",
                     f"{kind} span has label '{label}' outside the allowed semantic vocabulary")
        elif kind == "judgment_marker":
            if label not in ("JUDGMENT", "ORDER"):
                flag("structural_validity", "low", "deterministic", f"judgment_marker span has label '{label}'")
        elif kind == "final_order":
            # By design (structurer.py::_final_order_spans), every final_order
            # SPAN's label is always the literal string "final_order" -- the
            # real per-sentence disposition words (dismissed/allowed/...) live
            # only in structure.final_orders, checked separately below via
            # _DISPOSITION_WORD_RE against the span's actual text.
            if label != "final_order":
                flag("structural_validity", "medium", "deterministic",
                     f"final_order span has unexpected label '{label}' (expected literal 'final_order')")
        if d.get("text", "") == "":
            flag("structural_validity", "medium", "deterministic", f"empty span text at kind={kind}")

    fallback_reason = structure.get("llm_fallback_reason")
    result["llm_fallback_reason"] = fallback_reason
    result["used_llm"] = structure.get("used_llm")
    result["paragraph_group_span_count"] = paragraph_group_count
    result["fallback_fraction_of_paragraphs"] = (
        round(paragraph_group_count / result["paragraph_count"], 4) if result["paragraph_count"] else 0.0
    )

    if status == STATUS_STRUCTURED:
        if fallback_reason and paragraph_group_count == 0:
            flag("structural_validity", "medium", "deterministic",
                 "llm_fallback_reason is set but no paragraph_group fallback spans were found")
        if not fallback_reason and paragraph_group_count > 0:
            flag("structural_validity", "medium", "deterministic",
                 "paragraph_group fallback spans present but llm_fallback_reason is null")
    elif status in (STATUS_INCOMPLETE_SOURCE, STATUS_NON_JUDGMENT_TEXT, STATUS_FALLBACK):
        if not _identity_fallback_shape_ok(span_dicts, result["paragraph_count"]):
            flag("structural_validity", "medium", "deterministic",
                 f"status={status} but spans are not the expected identity-fallback shape "
                 "(one paragraph_group span per paragraph)")

    # -- deterministic cross-check: disposition <=> short-circuit status ------
    disposition = processed_doc.get("metadata", {}).get("disposition")
    short_circuited = status in (STATUS_INCOMPLETE_SOURCE, STATUS_NON_JUDGMENT_TEXT)
    if (disposition is None) != short_circuited:
        flag("metadata", "high", "deterministic",
             f"disposition={disposition!r} but structure_status={status!r} "
             "(disposition=None should imply incomplete_source/non_judgment_text and vice versa)")

    # -- semantic-quality heuristics (explicitly heuristic, not proof) --------
    for d in span_dicts:
        text = d.get("text", "")
        label = d.get("label")
        kind = d.get("type")
        if kind == "semantic_section" and label and label != "arguments":
            if _COUNSEL_SUBMISSION_RE.search(text):
                flag("semantic_quality", "low", "heuristic",
                     f"span labeled '{label}' contains counsel-submission phrasing "
                     "(candidate: should possibly be 'arguments')",
                     text[:200])
        if kind == "quoted_material":
            if '"' not in text and "“" not in text and not re.search(r"\([a-z]\)|\(i+\)", text):
                flag("semantic_quality", "low", "heuristic",
                     "quoted_material span has no quotation marks or sub-item markers "
                     "(candidate: possible mislabel)", text[:200])
        if kind == "final_order":
            if not _DISPOSITION_WORD_RE.search(text):
                flag("semantic_quality", "low", "heuristic",
                     "final_order span text has no recognizable disposition wording "
                     "(candidate: may be a reporter tag-line fragment, not the operative sentence)",
                     text[:200])

    # -- metadata: provisions_cited internal variant duplicates ---------------
    provisions = processed_doc.get("metadata", {}).get("provisions_cited") or []
    seen_norm: dict[str, str] = {}
    for raw in provisions:
        norm = _normalize_provision(str(raw))
        if norm in seen_norm and seen_norm[norm] != raw:
            flag("metadata", "low", "heuristic",
                 f"provisions_cited contains variant forms of the same provision: "
                 f"'{seen_norm[norm]}' and '{raw}'")
        seen_norm.setdefault(norm, raw)

    # -- boundary / complex-case tags ------------------------------------------
    result["judgment_marker"] = structure.get("judgment_marker")
    result["headnotes_detected"] = structure.get("headnotes_detected")
    final_orders = structure.get("final_orders") or []
    result["final_order_count"] = len(final_orders)
    # Distinct OUTCOMES (labels), not distinct paragraph positions: almost
    # every judgment in this corpus emits two final_orders entries for the
    # same single outcome -- the body's operative sentence plus a separate
    # one-line reporter tag (e.g. "...dismissed." + "S.A.K./288/P Petition
    # dismissed."), which trivially differ in paragraph_start despite being
    # the same matter. A real multi-matter case (verified directly against
    # 04139076ff36792b38b72f8b) has multiple DIFFERENT labels, e.g.
    # "allowed"+"dismissed"+"disposed_accordingly" for distinct C.M.A.s.
    result["is_multi_matter_candidate"] = len({
        o.get("label") for o in final_orders if o.get("label") != "unclassified"
    }) > 1
    result["is_known_hard_case"] = doc_id in KNOWN_HARD_DOC_IDS

    # -- readiness classification (rule-based, see audit_report.md) -----------
    high_severity = any(i["severity"] == "high" for i in result["issues"])
    medium_severity = any(i["severity"] == "medium" for i in result["issues"])
    if high_severity or not recomputed.ok:
        readiness = "not_ready"
    elif status in (STATUS_FALLBACK,):
        readiness = "ready_with_caution"  # whole-doc identity fallback: text safe, no semantics
    elif status in (STATUS_INCOMPLETE_SOURCE, STATUS_NON_JUDGMENT_TEXT):
        readiness = "ready_with_caution"  # deliberately short-circuited, safe, coarse-grained
    elif medium_severity or result["fallback_fraction_of_paragraphs"] > 0.20:
        readiness = "manual_review_required"
    elif fallback_reason or result["is_multi_matter_candidate"]:
        readiness = "ready_with_caution"
    elif any(i["category"] == "semantic_quality" for i in result["issues"]):
        readiness = "ready_with_caution"
    else:
        readiness = "ready"
    result["readiness"] = readiness

    return result


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    processed_ids = {p.stem for p in PROCESSED_DIR.glob("*.json")}
    structured_paths = {
        p.stem: p for p in STRUCTURED_DIR.glob("*.json")
        if p.name not in {"progress.json", "full_run_report.json"}
    }
    structured_ids = set(structured_paths.keys())

    missing = sorted(processed_ids - structured_ids)
    unexpected = sorted(structured_ids - processed_ids)

    print(f"Expected (processed corpus): {len(processed_ids)}")
    print(f"Found (structured_gemini/): {len(structured_ids)}")
    print(f"Missing (not yet structured): {len(missing)}")
    print(f"Unexpected (structured but no processed source): {len(unexpected)}")

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
        if i % 200 == 0:
            print(f"  ...{i}/{len(all_doc_ids)}")
        processed_path = PROCESSED_DIR / f"{doc_id}.json"
        structured_path = structured_paths.get(doc_id)
        res = audit_one(doc_id, processed_path, structured_path)
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

    summary = {
        "expected_corpus_size": len(processed_ids),
        "found_in_structured_gemini": len(structured_ids),
        "missing_count": len(missing),
        "missing_doc_ids_sample": missing[:50],
        "unexpected_count": len(unexpected),
        "unexpected_doc_ids": unexpected,
        "documents_audited": n_audited,
        "documents_with_output_present": n_structured_present,
        "structure_status_distribution": dict(status_counts),
        "preservation_distribution": dict(preservation_counts),
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
        "known_hard_cases_present": {
            doc_id: per_doc_results.get(doc_id, {}).get("readiness", "missing")
            for doc_id in KNOWN_HARD_DOC_IDS
        },
        "note": (
            "This run audited the structured_gemini/ corpus as found on disk. "
            f"{len(missing)} of {len(processed_ids)} expected documents were not yet present "
            "at audit time (structuring run was not complete when this snapshot was taken)."
            if missing else "All expected documents were present at audit time."
        ),
    }

    (OUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (OUT_DIR / "per_doc_results.json").write_text(json.dumps(per_doc_results, indent=2), encoding="utf-8")

    with (OUT_DIR / "issues.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["doc_id", "severity", "category", "check_type", "message", "evidence"])
        writer.writeheader()
        for issue in all_issues:
            writer.writerow(issue)

    # -- manual review queue: priority-ranked flagged docs + stratified sample
    severity_rank = {"high": 0, "medium": 1, "low": 2}
    issue_docs = sorted(
        {i["doc_id"] for i in all_issues if i["category"] != "completeness" or i["doc_id"] in processed_ids},
        key=lambda d: min((severity_rank[i["severity"]] for i in all_issues if i["doc_id"] == d), default=9),
    )
    priority_queue = issue_docs[:300]

    rng = random.Random(42)
    clean_docs = [d for d in all_doc_ids if per_doc_results.get(d, {}).get("readiness") == "ready"]
    sample_size = min(40, len(clean_docs))
    random_sample = rng.sample(clean_docs, sample_size) if sample_size else []

    with (OUT_DIR / "manual_review_queue.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "doc_id", "queue_reason", "readiness", "structure_status",
            "paragraph_count", "fallback_fraction", "processed_path", "structured_path", "human_verdict",
        ])
        for doc_id in priority_queue:
            res = per_doc_results.get(doc_id, {})
            writer.writerow([
                doc_id, "flagged_issue", res.get("readiness"), res.get("structure_status"),
                res.get("paragraph_count"), res.get("fallback_fraction_of_paragraphs"),
                f"var/rag/processed/{doc_id}.json", f"var/rag/structured_gemini/{doc_id}.json", "",
            ])
        for doc_id in random_sample:
            res = per_doc_results.get(doc_id, {})
            writer.writerow([
                doc_id, "stratified_random_sample", res.get("readiness"), res.get("structure_status"),
                res.get("paragraph_count"), res.get("fallback_fraction_of_paragraphs"),
                f"var/rag/processed/{doc_id}.json", f"var/rag/structured_gemini/{doc_id}.json", "",
            ])

    _write_markdown_report(summary, per_doc_results, priority_queue, random_sample)

    print("\nAUDIT COMPLETE.")
    print(f"  summary.json, per_doc_results.json, issues.csv, manual_review_queue.csv, audit_report.md")
    print(f"  -> {OUT_DIR}")
    print(f"readiness: {dict(readiness_counts)}")


def _write_markdown_report(summary, per_doc_results, priority_queue, random_sample) -> None:
    lines = []
    lines.append("# Structure Quality Audit Report\n")
    lines.append("Read-only audit of `var/rag/structured_gemini/` against `var/rag/processed/`. "
                  "No structured/processed files, code, or database data were modified.\n")

    lines.append("## 1. Corpus completeness\n")
    lines.append(f"- Expected (processed corpus): **{summary['expected_corpus_size']}**")
    lines.append(f"- Found in structured_gemini/: **{summary['found_in_structured_gemini']}**")
    lines.append(f"- Missing (not yet structured): **{summary['missing_count']}**")
    lines.append(f"- Unexpected (no processed source): **{summary['unexpected_count']}**")
    lines.append(f"- {summary['note']}\n")
    lines.append("structure_status distribution:")
    for k, v in summary["structure_status_distribution"].items():
        lines.append(f"- `{k}`: {v}")
    lines.append("")

    lines.append("## 2. Legal text preservation\n")
    lines.append("Classification is derived by reconstructing each document's real `Span` objects from the "
                  "stored JSON and re-running `src.rag_prep.structure_validate.validate_spans` against the "
                  "processed source's `full_text` -- not by trusting the stored `validation.ok` flag.\n")
    for k, v in summary["preservation_distribution"].items():
        lines.append(f"- `{k}`: {v}")
    lines.append("")

    lines.append("## 3-4. Structural validity & semantic quality\n")
    lines.append(f"Total issues found: **{summary['total_issues_found']}**")
    lines.append(f"By severity: {summary['issues_by_severity']}")
    lines.append(f"By category: {summary['issues_by_category']}\n")
    lines.append("`heuristic` issues (semantic_quality category, and the provisions_cited variant check) are "
                  "candidate flags for human review, not confirmed defects. All other categories are "
                  "deterministic re-checks against the actual schema/validator.\n")

    lines.append("## 5. Known hard cases (from the original structuring pilot)\n")
    for doc_id, readiness in summary["known_hard_cases_present"].items():
        lines.append(f"- `{doc_id}` ({KNOWN_HARD_DOC_IDS[doc_id]}): **{readiness}**")
    lines.append("")

    lines.append("## 7. Fallback and LLM performance\n")
    lines.append(f"- Documents with an LLM fallback (partial or whole): "
                  f"**{summary['docs_with_llm_fallback']}** "
                  f"({summary['docs_with_llm_fallback_pct_of_present']}% of present docs)")
    lines.append(f"- Documents structured without any LLM call: **{summary['docs_no_llm_needed']}**")
    lines.append(f"- Documents that used the LLM: **{summary['docs_used_llm']}**")
    lines.append("- Per-document LLM call counts / retry counts are NOT stored in the structured JSON itself "
                  "(only `llm_fallback_reason` and `used_llm` are) -- that telemetry exists only in the "
                  "runtime `progress.json`/`full_run_report.json` files from each run session, which are "
                  "process logs, not corpus artifacts. This audit reports what is actually in the corpus.\n")

    lines.append("## 9. Readiness for chunking\n")
    lines.append("Rule-based categories (see `audit_structure_quality.py::audit_one` for the exact logic):\n")
    lines.append("- **not_ready**: a high-severity issue was found (unreadable file, missing required field, "
                  "full_text/metadata mismatch vs. source, or the independently recomputed validation failed).")
    lines.append("- **ready_with_caution**: whole-document identity fallback (`fallback_paragraph_groups`), "
                  "a deliberately short-circuited `incomplete_source`/`non_judgment_text` document, a partial "
                  "LLM fallback, a multi-matter candidate, or a heuristic semantic-quality flag -- text is "
                  "safe and chunkable, but semantic granularity is reduced or needs a second look.")
    lines.append("- **manual_review_required**: a medium-severity structural issue was found, or more than "
                  "20% of the document's paragraphs fell back to `paragraph_group`.")
    lines.append("- **ready**: none of the above.\n")
    for k, v in summary["readiness_distribution"].items():
        pct = summary["readiness_percentages"].get(k, 0)
        lines.append(f"- `{k}`: {v} ({pct}%)")
    lines.append("")

    lines.append("## 8. Manual review queue\n")
    lines.append(f"- Priority (flagged) docs queued: **{len(priority_queue)}**")
    lines.append(f"- Stratified random sample of apparently-clean docs: **{len(random_sample)}** "
                  "(seed=42, reproducible)")
    lines.append("- Full queue: `manual_review_queue.csv`\n")

    lines.append("## Limitations\n")
    lines.append("- Semantic-quality checks are regex-based heuristics over a small set of patterns "
                  "(counsel-submission phrasing, quotation-marker presence, disposition vocabulary). "
                  "They surface *candidates* for review; they do not prove a label is wrong, and they "
                  "will miss many real mislabels that don't match these specific patterns.")
    lines.append("- This audit does not re-run the LLM or re-score confidence; `confidence` values in the "
                  "stored spans are taken as-is from the structuring run.")
    lines.append("- If `missing_count` > 0 above, this audit reflects a snapshot of a still-in-progress "
                  "corpus, not the final 2,088-document result.\n")

    (OUT_DIR / "audit_report.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
