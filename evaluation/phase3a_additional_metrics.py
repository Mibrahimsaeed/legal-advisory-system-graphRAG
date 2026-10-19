#!/usr/bin/env python3
"""Phase 3A -- Recall@5 and MRR, computed from a judged CSV in the project's
existing judge_results.py format. Reuses calculate_metrics.py's own
recall_at_k() unmodified (called with k=5 instead of its default of 10,
which it already supports as a parameter) rather than duplicating its
logic. calculate_metrics.py itself already reports Precision@5, nDCG@10,
and Recall@10 and is not reproduced here.

MRR follows the same "never silently treat missing as irrelevant" rule as
every other metric in this project: scanning ranks in order, if an
unjudged rank is reached before any judged-relevant rank, the true first
relevant rank cannot be determined (an unjudged item at an earlier
position could turn out relevant), so MRR is reported unavailable for
that query/source rather than guessed.

    python evaluation/phase3a_additional_metrics.py --judged <path> --out <path>
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

_spec = importlib.util.spec_from_file_location("eval_calculate_metrics", REPO_ROOT / "evaluation" / "calculate_metrics.py")
_cm = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_cm)

SCORED_SOURCES = _cm.SCORED_SOURCES


def mrr_at_rank_list(relevances_by_rank: list[int | None]) -> tuple[float | None, str | None]:
    for i, r in enumerate(relevances_by_rank):
        if r is None:
            return None, f"first unjudged rank encountered at rank {i + 1} before any judged-relevant rank was found"
        if r >= 1:
            return 1.0 / (i + 1), None
    return 0.0, None  # every rank judged, none relevant -- a confirmed, valid 0, not a gap


def load_judged(path: Path) -> dict[tuple[str, str], list[dict]]:
    grouped: dict[tuple[str, str], list[dict]] = defaultdict(list)
    with path.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            st = row["source_type"]
            if st not in SCORED_SOURCES:
                continue
            rank = int(row["rank"])
            raw = (row["relevance"] or "").strip()
            relevance = int(raw) if raw else None
            grouped[(row["query_id"], st)].append({"rank": rank, "chunk_id": row.get("chunk_id"), "relevance": relevance})
    for k in grouped:
        grouped[k].sort(key=lambda r: r["rank"])
    return grouped


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--judged", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args(argv)

    judged = load_judged(a.judged)
    per_query: dict[str, dict] = {}
    for (qid, st), rows in sorted(judged.items()):
        relevances = [r["relevance"] for r in rows]
        chunk_ids = [r["chunk_id"] for r in rows]
        mrr, mrr_note = mrr_at_rank_list(relevances)
        recall5, recall5_note = _cm.recall_at_k(chunk_ids, None, k=5)  # unmodified reuse; always None (no qrels)
        per_query.setdefault(qid, {})[st] = {
            "mrr": mrr, "mrr_note": mrr_note,
            "recall_at_5": recall5, "recall_at_5_note": recall5_note,
        }

    macro = {}
    for st in SCORED_SOURCES:
        mrr_vals = [per_query[q][st]["mrr"] for q in per_query if st in per_query[q] and per_query[q][st]["mrr"] is not None]
        excluded = [q for q in per_query if st in per_query[q] and per_query[q][st]["mrr"] is None]
        macro[st] = {
            "mean_mrr": sum(mrr_vals) / len(mrr_vals) if mrr_vals else None,
            "mrr_n_queries": len(mrr_vals), "mrr_excluded_queries": excluded,
            "mean_recall_at_5": None, "recall_at_5_note": "unavailable for all queries -- no qrels reference set exists in this project",
        }

    result = {"per_query": per_query, "macro": macro}
    print(json.dumps(result, indent=2))
    if a.out:
        a.out.write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(f"\n[phase3a_additional_metrics] -> {a.out}")


if __name__ == "__main__":
    main()
