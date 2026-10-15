#!/usr/bin/env python3
"""Computes Precision@5, nDCG@10, and Recall@10 from a judged relevance CSV
(the output of evaluation/judge_results.py, with the `relevance` column
filled in by hand), separately for case law and statutes.

    python evaluation/calculate_metrics.py --judged evaluation/results/judging_<ts>.csv
    python evaluation/calculate_metrics.py --judged ... --qrels evaluation/results/qrels.json --out evaluation/results/metrics.json

Constitution rows (source_type == "constitution") are always excluded from
every metric, regardless of whether they're present in the judged CSV.

MISSING DATA IS NEVER SILENTLY TREATED AS IRRELEVANT
-----------------------------------------------------
* A retrieved result whose `relevance` cell is still blank is MISSING A
  LABEL. If any rank within a metric's window (top 5 for Precision@5, top
  10 for nDCG@10) has a missing label, that metric is reported as
  unavailable for that (query, source) pair, with the exact missing ranks
  named -- never computed by treating the blank as 0.
* Fewer than k results existing at all for a query (e.g. only 3 case-law
  hits were retrieved when 5 were requested) is a DIFFERENT, ordinary
  situation: Precision@5 still divides by 5 (ranks 4-5 simply contribute
  nothing to the numerator, exactly as instructed), and is NOT reported as
  unavailable on that basis alone.
* Recall@10 requires a COMPLETE set of known-relevant documents for that
  query+source, supplied via an optional --qrels JSON file
  ({"Q01": {"case": ["chunk_id", ...], "statute": [...]}, ...}). The
  documents judged relevant within the top 10 are NEVER substituted for
  that complete set. Without a qrels entry for a given (query, source),
  Recall@10 is reported as unavailable, not computed from the top 10 alone.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path

PRECISION_K = 5
NDCG_K = 10
RECALL_K = 10
SCORED_SOURCES = ("case", "statute")  # constitution is never scored


# =====================================================================================
# pure metric functions -- operate on plain Python data, no file I/O, fully unit-testable
# =====================================================================================

def precision_at_k(relevances_by_rank: list[int | None], k: int = PRECISION_K) -> tuple[float | None, str | None]:
    """relevances_by_rank[i] is the relevance label (0/1/2) for rank i+1, or
    None if that rank was retrieved but not yet judged. A rank beyond the
    list's length (fewer than k results retrieved) is treated as an empty
    slot contributing 0 to the numerator -- the denominator is always k.

    Returns (precision, note). note is None iff precision is not None.
    """

    top = relevances_by_rank[:k]
    missing = [i + 1 for i, r in enumerate(top) if r is None]
    if missing:
        return None, f"missing relevance label(s) at rank(s) {missing}"
    relevant = sum(1 for r in top if r is not None and r >= 1)
    return relevant / k, None


def ndcg_at_k(relevances_by_rank: list[int | None], k: int = NDCG_K) -> tuple[float | None, str | None]:
    """Graded relevance nDCG@k: gain = 2**relevance - 1, discount = log2(rank + 1)."""

    top = relevances_by_rank[:k]
    missing = [i + 1 for i, r in enumerate(top) if r is None]
    if missing:
        return None, f"missing relevance label(s) at rank(s) {missing}"
    if not top:
        return None, "no results retrieved"

    def _dcg(rels: list[int]) -> float:
        return sum((2 ** r - 1) / math.log2(i + 2) for i, r in enumerate(rels))

    dcg = _dcg(top)  # type: ignore[arg-type]
    idcg = _dcg(sorted(top, reverse=True))  # type: ignore[arg-type]
    if idcg == 0:
        return 0.0, None  # every judged label is 0 -- DCG is also 0, nDCG is well-defined as 0
    return dcg / idcg, None


def recall_at_k(
    ranked_chunk_ids: list[str], relevant_chunk_ids: set[str] | None, k: int = RECALL_K,
) -> tuple[float | None, str | None]:
    """relevant_chunk_ids is the COMPLETE known-relevant set for this query
    (from a qrels file), never inferred from the top-k themselves. None
    means no reference set was supplied at all for this query/source."""

    if relevant_chunk_ids is None:
        return None, "no complete relevance reference set available (no qrels entry for this query/source)"
    if not relevant_chunk_ids:
        return None, "relevance reference set is empty -- recall is undefined"
    top_k = set(ranked_chunk_ids[:k])
    found = top_k & relevant_chunk_ids
    return len(found) / len(relevant_chunk_ids), None


# =====================================================================================
# CSV / qrels loading
# =====================================================================================

def _parse_relevance(raw: str) -> int | None:
    raw = (raw or "").strip()
    if raw == "":
        return None
    if raw not in ("0", "1", "2"):
        raise ValueError(f"invalid relevance label {raw!r} -- must be blank, 0, 1, or 2")
    return int(raw)


def load_judged_csv(path: Path) -> dict[tuple[str, str], list[dict]]:
    """Returns {(query_id, source_type): [row, ...]} with rows sorted by
    rank ascending, for source types in SCORED_SOURCES only. Raises on any
    malformed relevance value -- never coerces it."""

    grouped: dict[tuple[str, str], list[dict]] = defaultdict(list)
    with path.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        required = {"query_id", "source_type", "rank", "chunk_id", "relevance"}
        missing_cols = required - set(reader.fieldnames or [])
        if missing_cols:
            raise ValueError(f"{path}: judged CSV is missing column(s): {sorted(missing_cols)}")
        for line_no, row in enumerate(reader, 2):  # header is line 1
            st = row["source_type"]
            if st not in SCORED_SOURCES:
                continue
            try:
                rank = int(row["rank"])
            except (TypeError, ValueError) as exc:
                raise ValueError(f"{path}:{line_no}: invalid rank {row.get('rank')!r}") from exc
            try:
                relevance = _parse_relevance(row["relevance"])
            except ValueError as exc:
                raise ValueError(f"{path}:{line_no}: {exc}") from exc
            grouped[(row["query_id"], st)].append({
                "rank": rank, "chunk_id": row.get("chunk_id"), "relevance": relevance,
            })
    for key in grouped:
        grouped[key].sort(key=lambda r: r["rank"])
    return grouped


def load_qrels(path: Path | None) -> dict[tuple[str, str], set[str]]:
    """{"Q01": {"case": ["chunk_id", ...], "statute": [...]}} -> {(query_id, source): {chunk_id, ...}}.
    Returns {} if path is None -- callers must then treat every (query, source) as having no reference set."""

    if path is None:
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    out: dict[tuple[str, str], set[str]] = {}
    for qid, by_source in data.items():
        for st, ids in by_source.items():
            out[(qid, st)] = set(ids)
    return out


# =====================================================================================
# aggregation
# =====================================================================================

def evaluate(judged: dict[tuple[str, str], list[dict]], qrels: dict[tuple[str, str], set[str]]) -> dict:
    per_query: dict[str, dict] = {}
    for (qid, st), rows in sorted(judged.items()):
        relevances = [r["relevance"] for r in rows]
        chunk_ids = [r["chunk_id"] for r in rows]

        p5, p5_note = precision_at_k(relevances, PRECISION_K)
        ndcg10, ndcg_note = ndcg_at_k(relevances, NDCG_K)
        rel_set = qrels.get((qid, st))
        recall10, recall_note = recall_at_k(chunk_ids, rel_set, RECALL_K)

        per_query.setdefault(qid, {})[st] = {
            "n_results": len(rows),
            "precision_at_5": p5, "precision_at_5_note": p5_note,
            "ndcg_at_10": ndcg10, "ndcg_at_10_note": ndcg_note,
            "recall_at_10": recall10, "recall_at_10_note": recall_note,
        }

    macro: dict[str, dict] = {}
    for st in SCORED_SOURCES:
        values = {"precision_at_5": [], "ndcg_at_10": [], "recall_at_10": []}
        excluded = {"precision_at_5": [], "ndcg_at_10": [], "recall_at_10": []}
        for qid, by_source in per_query.items():
            row = by_source.get(st)
            if row is None:
                continue
            for metric in values:
                v = row[metric]
                if v is None:
                    excluded[metric].append(qid)
                else:
                    values[metric].append(v)
        macro[st] = {
            f"mean_{m}": (sum(vs) / len(vs) if vs else None) for m, vs in values.items()
        }
        for m in values:
            macro[st][f"{m}_n_queries"] = len(values[m])
            macro[st][f"{m}_excluded_queries"] = excluded[m]
    return {"per_query": per_query, "macro": macro}


def print_report(result: dict) -> None:
    for st in SCORED_SOURCES:
        print(f"\n=== {st} ===")
        print(f"{'query_id':<10}{'n':>4}{'P@5':>10}{'nDCG@10':>10}{'Recall@10':>12}")
        for qid, by_source in result["per_query"].items():
            row = by_source.get(st)
            if row is None:
                continue
            p5 = f"{row['precision_at_5']:.3f}" if row["precision_at_5"] is not None else "n/a"
            ndcg = f"{row['ndcg_at_10']:.3f}" if row["ndcg_at_10"] is not None else "n/a"
            rec = f"{row['recall_at_10']:.3f}" if row["recall_at_10"] is not None else "n/a"
            print(f"{qid:<10}{row['n_results']:>4}{p5:>10}{ndcg:>10}{rec:>12}")
            for metric_key, note_key in (("precision_at_5", "precision_at_5_note"),
                                          ("ndcg_at_10", "ndcg_at_10_note"),
                                          ("recall_at_10", "recall_at_10_note")):
                if row[metric_key] is None and row[note_key]:
                    print(f"           {metric_key}: {row[note_key]}")
        m = result["macro"][st]
        macro_p5 = f"{m['mean_precision_at_5']:.3f}" if m["mean_precision_at_5"] is not None else "n/a"
        macro_ndcg = f"{m['mean_ndcg_at_10']:.3f}" if m["mean_ndcg_at_10"] is not None else "n/a"
        macro_recall = f"{m['mean_recall_at_10']:.3f}" if m["mean_recall_at_10"] is not None else "n/a"
        print(f"{'MACRO':<10}{'':>4}{macro_p5:>10}{macro_ndcg:>10}{macro_recall:>12}")
        print(f"  (P@5 over {m['precision_at_5_n_queries']} queries, "
              f"nDCG@10 over {m['ndcg_at_10_n_queries']}, Recall@10 over {m['recall_at_10_n_queries']})")
        if m["recall_at_10_excluded_queries"]:
            print(f"  Recall@10 unavailable for: {m['recall_at_10_excluded_queries']}")


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--judged", type=Path, required=True, help="judged relevance CSV (from judge_results.py)")
    ap.add_argument("--qrels", type=Path, default=None,
                    help="optional JSON file of complete known-relevant chunk_ids per query/source, for Recall@10")
    ap.add_argument("--out", type=Path, default=None, help="optional path to also write the full report as JSON")
    a = ap.parse_args(argv)

    judged = load_judged_csv(a.judged)
    qrels = load_qrels(a.qrels)
    result = evaluate(judged, qrels)
    print_report(result)

    if a.out:
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(f"\n[calculate_metrics] full report -> {a.out}")


if __name__ == "__main__":
    main()
