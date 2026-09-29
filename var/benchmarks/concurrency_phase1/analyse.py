#!/usr/bin/env python3
"""Concurrency Phase 1 analysis: wall-clock, latency, and prediction diff.

Read-only against JSONL produced by run_concurrency.py. Writes summary.json
here.
"""

from __future__ import annotations

import json
import statistics as st
from pathlib import Path

HERE = Path(__file__).resolve().parent
CONFIGS = (1, 2)


def load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        raise SystemExit(f"missing: {path} -- run the benchmark first")
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def percentile(values: list[float], p: float) -> float:
    if not values:
        return float("nan")
    s = sorted(values)
    k = (len(s) - 1) * p
    f, c = int(k), min(int(k) + 1, len(s) - 1)
    return s[f] if f == c else s[f] + (s[c] - s[f]) * (k - f)


def main() -> None:
    batch_wall = json.loads((HERE / "batch_wall_clock.json").read_text())
    wall_by_workers = {b["n_workers"]: b for b in batch_wall}

    runs = {n: {r["doc_id"]: r for r in load_jsonl(HERE / f"results_{n}worker.jsonl")} for n in CONFIGS}

    stats = {}
    for n in CONFIGS:
        rows = list(runs[n].values())
        times = [r["elapsed_seconds"] for r in rows]
        ok = [r for r in rows if r["success"]]
        wall = wall_by_workers.get(n, {}).get("batch_wall_seconds_this_run")
        stats[str(n)] = {
            "n_documents": len(rows),
            "successful": len(ok),
            "failed": len(rows) - len(ok),
            "wall_clock_seconds_full_batch": wall,
            "wall_clock_cases_per_minute": round(60 * len(rows) / wall, 2) if wall else None,
            "mean_call_latency_seconds": round(st.mean(times), 3) if times else None,
            "median_call_latency_seconds": round(st.median(times), 3) if times else None,
            "p90_call_latency_seconds": round(percentile(times, 0.90), 3) if times else None,
            "min_call_latency_seconds": round(min(times), 3) if times else None,
            "max_call_latency_seconds": round(max(times), 3) if times else None,
            "sum_of_individual_call_seconds": round(sum(times), 1) if times else None,
        }

    speedup = None
    if stats["1"]["wall_clock_seconds_full_batch"] and stats["2"]["wall_clock_seconds_full_batch"]:
        w1, w2 = stats["1"]["wall_clock_seconds_full_batch"], stats["2"]["wall_clock_seconds_full_batch"]
        speedup = {
            "wall_clock_1worker_seconds": w1,
            "wall_clock_2worker_seconds": w2,
            "pct_change_2_vs_1": round(100 * (w2 / w1 - 1), 1),
            "speedup_factor": round(w1 / w2, 3),
        }

    # Prediction diff -- must be computed on the SAME 100 documents present in both runs.
    shared = sorted(set(runs[1]) & set(runs[2]))
    diffs = []
    conf_diffs = 0
    both_ok = [d for d in shared if runs[1][d]["success"] and runs[2][d]["success"]]
    for d in both_ok:
        r1, r2 = runs[1][d], runs[2][d]
        if r1["predicted_domain"] != r2["predicted_domain"]:
            diffs.append({
                "doc_id": d, "domain_1worker": r1["predicted_domain"], "conf_1worker": r1["confidence"],
                "domain_2worker": r2["predicted_domain"], "conf_2worker": r2["confidence"],
            })
        if r1["confidence"] != r2["confidence"]:
            conf_diffs += 1
    failed_either = [d for d in shared if not (runs[1][d]["success"] and runs[2][d]["success"])]

    prediction_diff = {
        "compared_documents": len(both_ok),
        "identical_predictions": len(both_ok) - len(diffs),
        "changed_predictions": len(diffs),
        "changed_confidence_only": conf_diffs,
        "excluded_failure_in_either_run": len(failed_either),
        "failed_doc_ids": failed_either,
        "changed_cases": diffs,
    }

    ollama_check = json.loads((HERE / "ollama_server_check.json").read_text()) if (HERE / "ollama_server_check.json").exists() else {}

    projection = {}
    CORPUS_N = 6141
    for n in CONFIGS:
        cpm = stats[str(n)]["wall_clock_cases_per_minute"]
        if cpm:
            projection[str(n)] = {"projected_hours_for_6141_docs": round(CORPUS_N / cpm / 60, 1)}

    summary = {
        "ollama_backend_at_run_time": ollama_check,
        "per_config": stats,
        "speedup": speedup,
        "prediction_diff": prediction_diff,
        "corpus_projection_6141_docs": projection,
    }
    (HERE / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    print("=" * 72)
    print(f"OLLAMA BACKEND AT RUN TIME: num_parallel_slots = {ollama_check.get('num_parallel_slots')}")
    print("=" * 72)

    print("\nPER-CONFIG RESULTS")
    print(f"{'workers':>7} {'n':>4} {'ok':>4} {'wall_s':>8} {'/min':>6} {'mean_lat':>9} {'median':>7} {'p90':>7} {'min':>6} {'max':>7}")
    for n in CONFIGS:
        s = stats[str(n)]
        print(f"{n:>7} {s['n_documents']:>4} {s['successful']:>4} {s['wall_clock_seconds_full_batch']:>8} "
              f"{s['wall_clock_cases_per_minute']:>6} {s['mean_call_latency_seconds']:>9} "
              f"{s['median_call_latency_seconds']:>7} {s['p90_call_latency_seconds']:>7} "
              f"{s['min_call_latency_seconds']:>6} {s['max_call_latency_seconds']:>7}")

    if speedup:
        print(f"\nWALL-CLOCK SPEEDUP (2 workers vs 1):")
        print(f"  1-worker: {speedup['wall_clock_1worker_seconds']}s   2-worker: {speedup['wall_clock_2worker_seconds']}s")
        print(f"  change: {speedup['pct_change_2_vs_1']:+.1f}%   speedup factor: {speedup['speedup_factor']}x")

    print(f"\nCORPUS PROJECTION (6,141 documents):")
    for n in CONFIGS:
        p = projection.get(str(n))
        if p:
            print(f"  {n} worker(s): {p['projected_hours_for_6141_docs']} hours")

    print(f"\nPREDICTION STABILITY (n={prediction_diff['compared_documents']} compared, "
          f"{prediction_diff['excluded_failure_in_either_run']} excluded for a failure in either run)")
    print(f"  identical: {prediction_diff['identical_predictions']}   changed: {prediction_diff['changed_predictions']}   "
          f"confidence-only diffs: {prediction_diff['changed_confidence_only']}")
    for c in prediction_diff["changed_cases"]:
        print(f"    {c['doc_id']}: 1worker={c['domain_1worker']}({c['conf_1worker']})  "
              f"2worker={c['domain_2worker']}({c['conf_2worker']})")
    if prediction_diff["failed_doc_ids"]:
        print(f"  failed in either run: {prediction_diff['failed_doc_ids']}")

    print("\nwrote summary.json")


if __name__ == "__main__":
    main()
