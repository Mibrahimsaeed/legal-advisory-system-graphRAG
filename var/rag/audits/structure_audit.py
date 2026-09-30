#!/usr/bin/env python3
"""READ-ONLY structure audit of var/rag/processed/. Writes only under
var/rag/audits/ (this directory). Never opens var/rag/raw/, var/metadata.db,
or any project config/schema file. Never writes to var/rag/processed/.
"""

from __future__ import annotations

import json
import re
import statistics as st
from collections import Counter, defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
PROCESSED = REPO / "var" / "rag" / "processed"
OUT = Path(__file__).resolve().parent

# ---------------------------------------------------------------------------
# Structural detectors
# ---------------------------------------------------------------------------

HEADNOTE_LETTER_RE = re.compile(r"^\([a-z]\)\s+\S")
NUMBERED_PARA_RE = re.compile(r"^\d{1,3}\.\s+\S")
ROMAN_SUBITEM_RE = re.compile(r"\((?:i|ii|iii|iv|v|vi|vii|viii|ix|x|xi|xii)\)", re.IGNORECASE)
LETTER_SUBITEM_RE = re.compile(r"\([a-z]\)")
CASE_CITATION_RE = re.compile(
    r"\b(P\s*L\s*D|S\s*C\s*M\s*R|C\s*L\s*C|Y\s*L\s*R|M\s*L\s*D|P\s*Cr\.?\s*L\s*J)\s+\d{4}\b",
    re.IGNORECASE,
)
VERSUS_RE = re.compile(r"\bversus\b|\bvs\.?\b", re.IGNORECASE)

# Candidate "standalone heading" vocabulary -- checked as an EXACT match
# (after stripping trailing punctuation) against a whole paragraph. This is
# the discovery list from the task prompt plus common variants; frequency
# results below are what actually determines which of these are real.
HEADING_CANDIDATES = [
    "JUDGMENT", "ORDER", "FINAL ORDER", "SHORT ORDER", "OPERATIVE ORDER",
    "FACTS", "BACKGROUND", "BRIEF FACTS", "FACTS OF THE CASE",
    "ARGUMENTS", "SUBMISSIONS", "CONTENTIONS",
    "REASONS", "REASONS FOR DECISION", "DISCUSSION", "ANALYSIS",
    "FINDINGS", "CONCLUSION", "DECISION", "HELD",
    "ISSUES", "POINTS FOR DETERMINATION", "QUESTION FOR DETERMINATION",
    "EVIDENCE", "PRELIMINARY OBJECTION", "PRELIMINARY OBJECTIONS",
    "OPINION", "OBSERVATIONS", "PRAYER", "RELIEF",
]


def paragraphs(text: str) -> list[str]:
    return [p.strip() for p in text.split("\n\n") if p.strip()]


def is_heading_shaped(p: str) -> bool:
    """Broad candidate filter for Audit 2/3's frequency table: short,
    not ending in a full stop, not obviously a citation/date/party line."""

    if len(p) > 70 or len(p) < 2:
        return False
    if p.endswith("."):
        return False
    if re.search(r"\d{4}", p) and ("decided on" in p.lower() or "no." in p.lower()):
        return False  # citation/petition-number lines
    if VERSUS_RE.search(p) and len(p) < 40:
        return False  # "Versus" itself, or short party separators
    letters = re.sub(r"[^A-Za-z]", "", p)
    if not letters:
        return False
    is_upper = letters.isupper()
    is_titleish = p == p.title() or (p[0].isupper() and p.count(" ") <= 6)
    return is_upper or is_titleish


def analyze_one(doc: dict) -> dict:
    text = doc["full_text"]
    meta = doc["metadata"]
    paras = paragraphs(text)
    n_paras = len(paras)

    headnote_paras = [p for p in paras if HEADNOTE_LETTER_RE.match(p)]
    numbered_paras = [p for p in paras if NUMBERED_PARA_RE.match(p)]
    heading_shaped = [p for p in paras if is_heading_shaped(p)]

    standalone_headings = []
    for p in paras:
        stripped = p.strip().rstrip(".:-").strip()
        for cand in HEADING_CANDIDATES:
            if stripped.upper() == cand:
                standalone_headings.append(cand)
                break

    has_judgment_word = bool(re.search(r"\bJUDGMENT\b", text))
    has_order_word = bool(re.search(r"\bORDER\b", text))
    has_final_order_word = bool(re.search(r"\bFINAL ORDER\b", text, re.IGNORECASE))
    judgment_standalone = "JUDGMENT" in standalone_headings
    order_standalone = "ORDER" in standalone_headings

    roman_subitems = len(ROMAN_SUBITEM_RE.findall(text))
    citation_count = len(CASE_CITATION_RE.findall(text))
    versus_count = len(VERSUS_RE.findall(text))

    numbered_frac = (len(numbered_paras) / n_paras) if n_paras else 0.0

    # Legal-content component detection -- reuses already-extracted
    # metadata (disposition/statutes_cited/provisions_cited) rather than
    # re-deriving, plus keyword/phrase heuristics for the rest.
    lower = text.lower()
    components = {
        "case_metadata": True,  # always present: citation/court/date are structured fields
        "parties": bool(versus_count),
        "procedural_history": bool(re.search(
            r"impugned (order|judgment)|against the order dated|filed an appeal|"
            r"filed a (writ petition|revision|suit)", lower)),
        "facts_background": bool(re.search(
            r"facts (of the case )?are|brief facts|facts leading to|facts, briefly", lower)),
        "legal_issues": bool(re.search(
            r"\bissue(s)? (for|is|was|arise)|question of law|whether\b.{0,60}\?", lower)),
        "arguments": bool(re.search(
            r"learned counsel|contended|argued|submission|submits? that", lower)),
        "applicable_law_statutes": bool(meta.get("statutes_cited")),
        "specific_provisions": bool(meta.get("provisions_cited")),
        "evidence": bool(re.search(r"\bevidence\b|\bexh\.|\bwitness|deposed", lower)),
        "court_reasoning": bool(re.search(
            r"i am of the view|in my (considered )?opinion|i have considered|"
            r"it (is|was) held that|reasons? .{0,20}(are|is) as follows", lower)),
        "findings_determination": bool(re.search(
            r"\bfind(ing)?s?\b.{0,20}(that|is|are)|i find that|conclu(de|sion)", lower)),
        "precedents_authorities": bool(citation_count or " rel." in lower or " ref." in lower),
        "lower_court_decisions": bool(re.search(
            r"trial court|family court|additional district judge|learned judge,? family|"
            r"guardian (court|judge)", lower)),
        "final_order": bool(meta.get("disposition")),
        "disposition_outcome": bool(meta.get("disposition")),
        "headnotes_propositions": bool(headnote_paras),
    }

    return {
        "doc_id": doc["doc_id"],
        "length": len(text),
        "n_paragraphs": n_paras,
        "n_headnote_paras": len(headnote_paras),
        "n_numbered_paras": len(numbered_paras),
        "numbered_frac": numbered_frac,
        "n_heading_shaped": len(heading_shaped),
        "heading_shaped_samples": heading_shaped[:6],
        "standalone_headings": standalone_headings,
        "has_judgment_word": has_judgment_word,
        "has_order_word": has_order_word,
        "has_final_order_word": has_final_order_word,
        "judgment_standalone": judgment_standalone,
        "order_standalone": order_standalone,
        "roman_subitems": roman_subitems,
        "citation_count": citation_count,
        "versus_count": versus_count,
        "components": components,
    }


def main() -> None:
    files = sorted(PROCESSED.glob("*.json"))
    print(f"Auditing {len(files)} processed documents (read-only)...")

    results = []
    for f in files:
        doc = json.loads(f.read_text(encoding="utf-8"))
        results.append(analyze_one(doc))

    # ---- Audit 1: corpus overview -------------------------------------
    lens = [r["length"] for r in results]
    overview = {
        "total_cases": len(results),
        "total_characters": sum(lens),
        "min_length": min(lens),
        "median_length": sorted(lens)[len(lens) // 2],
        "mean_length": round(st.mean(lens), 1),
        "max_length": max(lens),
        "cases_with_headnotes": sum(1 for r in results if r["n_headnote_paras"] > 0),
        "cases_mostly_numbered_paragraphs": sum(1 for r in results if r["numbered_frac"] >= 0.3),
        "cases_with_any_standalone_heading": sum(1 for r in results if r["standalone_headings"]),
        "cases_containing_JUDGMENT_word": sum(1 for r in results if r["has_judgment_word"]),
        "cases_containing_ORDER_word": sum(1 for r in results if r["has_order_word"]),
        "cases_containing_FINAL_ORDER_word": sum(1 for r in results if r["has_final_order_word"]),
        "cases_JUDGMENT_standalone_heading": sum(1 for r in results if r["judgment_standalone"]),
        "cases_ORDER_standalone_heading": sum(1 for r in results if r["order_standalone"]),
        "cases_with_roman_subitems": sum(1 for r in results if r["roman_subitems"] >= 2),
    }
    n = len(results)
    overview_pct = {k: (v, round(100 * v / n, 1)) for k, v in overview.items()
                     if k not in ("total_cases", "total_characters", "min_length",
                                  "median_length", "mean_length", "max_length")}

    # ---- Audit 2/3: heading frequency -----------------------------------
    heading_freq = Counter()
    for r in results:
        heading_freq.update(r["standalone_headings"])

    heading_shaped_freq = Counter()
    for r in results:
        for h in r["heading_shaped_samples"]:
            heading_shaped_freq[h.upper()] += 1

    # ---- Audit 4: structural signature clustering ------------------------
    def signature(r):
        return (
            "headnotes" if r["n_headnote_paras"] > 0 else "no_headnotes",
            "JUDGMENT_heading" if r["judgment_standalone"] else (
                "ORDER_heading" if r["order_standalone"] else "no_named_heading"),
            "numbered_body" if r["numbered_frac"] >= 0.3 else "prose_body",
            "has_sublist" if r["roman_subitems"] >= 2 else "no_sublist",
        )

    sig_counts = Counter(signature(r) for r in results)

    # ---- Audit 5: legal content component coverage -----------------------
    component_names = list(results[0]["components"].keys())
    component_coverage = {
        name: sum(1 for r in results if r["components"][name]) for name in component_names
    }

    # ---- Audit 6: content without headings (candidates for qualitative review)
    no_heading_long = [
        r["doc_id"] for r in results
        if not r["standalone_headings"] and r["numbered_frac"] >= 0.3 and r["length"] > 8000
    ]

    # ---- Audit 7: unusual/complex structure candidates --------------------
    many_versus = sorted(results, key=lambda r: -r["versus_count"])[:10]
    many_subitems = sorted(results, key=lambda r: -r["roman_subitems"])[:10]
    many_citations = sorted(results, key=lambda r: -r["citation_count"])[:10]
    longest = sorted(results, key=lambda r: -r["length"])[:10]
    shortest = sorted(results, key=lambda r: r["length"])[:10]
    most_headnotes = sorted(results, key=lambda r: -r["n_headnote_paras"])[:10]

    # ---- write outputs (var/rag/audits/ only) -----------------------------
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "per_doc_results.json").write_text(
        json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    summary = {
        "overview": overview,
        "overview_pct": overview_pct,
        "standalone_heading_frequency": heading_freq.most_common(40),
        "heading_shaped_candidate_frequency": heading_shaped_freq.most_common(40),
        "structural_signatures": [
            {"signature": list(sig), "count": c, "pct": round(100 * c / n, 1)}
            for sig, c in sig_counts.most_common()
        ],
        "component_coverage": {
            name: {"count": c, "pct": round(100 * c / n, 1)} for name, c in component_coverage.items()
        },
        "candidates_no_heading_long_numbered": no_heading_long[:20],
        "candidates_many_versus": [(r["doc_id"], r["versus_count"]) for r in many_versus],
        "candidates_many_subitems": [(r["doc_id"], r["roman_subitems"]) for r in many_subitems],
        "candidates_many_citations": [(r["doc_id"], r["citation_count"]) for r in many_citations],
        "candidates_longest": [(r["doc_id"], r["length"]) for r in longest],
        "candidates_shortest": [(r["doc_id"], r["length"]) for r in shortest],
        "candidates_most_headnotes": [(r["doc_id"], r["n_headnote_paras"]) for r in most_headnotes],
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    # ---- console report ----------------------------------------------
    print("\n=== AUDIT 1: CORPUS OVERVIEW ===")
    print(f"total cases: {overview['total_cases']}")
    print(f"total characters: {overview['total_characters']:,}")
    print(f"length: min={overview['min_length']} median={overview['median_length']} "
          f"mean={overview['mean_length']} max={overview['max_length']}")
    for k, (v, p) in overview_pct.items():
        print(f"  {k}: {v} ({p}%)")

    print("\n=== AUDIT 2: STANDALONE HEADING FREQUENCY (exact-match) ===")
    for h, c in heading_freq.most_common(25):
        print(f"  {h:30s} {c:5d}  ({100*c/n:.1f}%)")

    print("\n=== AUDIT 2b: HEADING-SHAPED CANDIDATE LINES (broader net) ===")
    for h, c in heading_shaped_freq.most_common(30):
        print(f"  {h[:50]:50s} {c:5d}")

    print("\n=== AUDIT 4: STRUCTURAL SIGNATURES ===")
    for s in summary["structural_signatures"][:15]:
        print(f"  {s['count']:5d} ({s['pct']:5.1f}%)  {s['signature']}")

    print("\n=== AUDIT 5: LEGAL CONTENT COMPONENT COVERAGE ===")
    for name, d in summary["component_coverage"].items():
        print(f"  {name:28s} {d['count']:5d} ({d['pct']:5.1f}%)")

    print(f"\n=== AUDIT 6 candidates: no standalone heading + numbered body + long "
          f"({len(no_heading_long)} found) ===")
    print(" ", no_heading_long[:10])

    print("\n=== AUDIT 7 candidates ===")
    print("most 'versus' occurrences (multi-party signal):", summary["candidates_many_versus"][:5])
    print("most roman-numeral sub-items (schedules):", summary["candidates_many_subitems"][:5])
    print("most case citations (precedent-heavy):", summary["candidates_many_citations"][:5])
    print("longest:", summary["candidates_longest"][:5])
    print("shortest:", summary["candidates_shortest"][:5])
    print("most headnote points:", summary["candidates_most_headnotes"][:5])

    print("\nwrote var/rag/audits/summary.json and per_doc_results.json")


if __name__ == "__main__":
    main()
