#!/usr/bin/env python3
"""Phase 1B analysis: paired interleaved timing + human-label quality.

Read-only against JSONL produced by run_interleaved.py / run_human_labelled.py.
Writes timing_summary.json, quality_summary.json, summary.json here.
"""

from __future__ import annotations

import json
import statistics as st
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
LENGTHS = (3000, 1500, 1000)
BASELINE = 3000
DOMAINS = ("family_law", "criminal_law", "other_uncertain")


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
    if f == c:
        return s[f]
    return s[f] + (s[c] - s[f]) * (k - f)


# ---------------------------------------------------------------------------
# PART A: paired interleaved timing
# ---------------------------------------------------------------------------

def analyse_timing() -> dict:
    rows = load_jsonl(HERE / "interleaved_results.jsonl")
    by_len: dict[int, dict[str, dict]] = {n: {} for n in LENGTHS}
    for r in rows:
        by_len[r["prompt_chars"]][r["doc_id"]] = r

    block_stats = {}
    for n in LENGTHS:
        times = [r["elapsed_seconds"] for r in by_len[n].values()]
        ok = [r for r in by_len[n].values() if r["success"]]
        block_stats[str(n)] = {
            "n": len(times),
            "successful": len(ok),
            "failed": len(times) - len(ok),
            "mean": round(st.mean(times), 3) if times else None,
            "median": round(st.median(times), 3) if times else None,
            "p90": round(percentile(times, 0.90), 3) if times else None,
            "min": round(min(times), 3) if times else None,
            "max": round(max(times), 3) if times else None,
            "total_seconds": round(sum(times), 1) if times else None,
            "cases_per_minute": round(60 * len(times) / sum(times), 2) if times else None,
        }

    # Paired per-document differences -- the whole point of interleaving.
    shared = sorted(set(by_len[3000]) & set(by_len[1500]) & set(by_len[1000]))
    paired = {"n_complete_triples": len(shared)}
    for n in (1500, 1000):
        deltas = [by_len[n][d]["elapsed_seconds"] - by_len[3000][d]["elapsed_seconds"] for d in shared]
        pct = [100 * (by_len[n][d]["elapsed_seconds"] / by_len[3000][d]["elapsed_seconds"] - 1) for d in shared]
        paired[f"{n}_minus_3000"] = {
            "mean_delta_seconds": round(st.mean(deltas), 3),
            "median_delta_seconds": round(st.median(deltas), 3),
            "mean_pct_change": round(st.mean(pct), 1),
            "median_pct_change": round(st.median(pct), 1),
            "n_faster": sum(1 for d in deltas if d < 0),
            "n_slower": sum(1 for d in deltas if d > 0),
            "n_unchanged": sum(1 for d in deltas if d == 0),
        }

    # Call-order sanity check: does being called 1st/2nd/3rd for a doc bias timing?
    order_bias = defaultdict(list)
    for r in rows:
        order_bias[r["call_position_within_doc"]].append(r["elapsed_seconds"])
    order_check = {str(k): round(st.mean(v), 3) for k, v in sorted(order_bias.items())}

    return {"block_stats": block_stats, "paired": paired, "call_order_mean_seconds": order_check}


# ---------------------------------------------------------------------------
# PART B: human-label quality
# ---------------------------------------------------------------------------

def prf1(tp: int, fp: int, fn: int) -> tuple[float | None, float | None, float | None]:
    p = tp / (tp + fp) if (tp + fp) else None
    r = tp / (tp + fn) if (tp + fn) else None
    f1 = (2 * p * r / (p + r)) if (p is not None and r is not None and (p + r) > 0) else (0.0 if p is not None and r is not None else None)
    return p, r, f1


def analyse_human(human_labels: dict[str, str]) -> dict:
    rows = load_jsonl(HERE / "human_label_results.jsonl")
    by_len = {n: {r["doc_id"]: r for r in rows if r["prompt_chars"] == n} for n in LENGTHS}

    dist = Counter(human_labels.values())
    result = {
        "n_labelled_documents": len(human_labels),
        "class_distribution": dict(dist),
        "warning": "Family-law-heavy distribution; not a representative estimate of corpus-wide accuracy.",
        "per_prompt_length": {},
    }

    for n in LENGTHS:
        preds = by_len[n]
        failed = [d for d in human_labels if d in preds and not preds[d]["success"]]
        missing = [d for d in human_labels if d not in preds]
        valid = [d for d in human_labels if d in preds and preds[d]["success"]]

        y_true = [human_labels[d] for d in valid]
        y_pred = [preds[d]["predicted_domain"] for d in valid]

        correct = sum(1 for t, p in zip(y_true, y_pred) if t == p)
        accuracy = correct / len(valid) if valid else None

        per_class = {}
        f1s = []
        for cls in DOMAINS:
            tp = sum(1 for t, p in zip(y_true, y_pred) if t == cls and p == cls)
            fp = sum(1 for t, p in zip(y_true, y_pred) if t != cls and p == cls)
            fn = sum(1 for t, p in zip(y_true, y_pred) if t == cls and p != cls)
            support = sum(1 for t in y_true if t == cls)
            if support == 0 and fp == 0:
                per_class[cls] = {"precision": None, "recall": None, "f1": None, "support": 0}
                continue
            p, r, f1 = prf1(tp, fp, fn)
            per_class[cls] = {"precision": round(p, 3) if p is not None else None,
                               "recall": round(r, 3) if r is not None else None,
                               "f1": round(f1, 3) if f1 is not None else None,
                               "support": support}
            if f1 is not None:
                f1s.append(f1)
        macro_f1 = round(sum(f1s) / len(f1s), 3) if f1s else None

        confusion = defaultdict(int)
        for t, p in zip(y_true, y_pred):
            confusion[f"{t}->{p}"] += 1

        result["per_prompt_length"][str(n)] = {
            "n_valid_predictions": len(valid),
            "n_failed": len(failed),
            "n_missing": len(missing),
            "failed_doc_ids": failed,
            "accuracy": round(accuracy, 3) if accuracy is not None else None,
            "macro_f1": macro_f1,
            "per_class": per_class,
            "confusion_matrix": dict(confusion),
        }

    return result


def analyse_human_stability(human_labels: dict[str, str]) -> dict:
    rows = load_jsonl(HERE / "human_label_results.jsonl")
    by_len = {n: {r["doc_id"]: r for r in rows if r["prompt_chars"] == n} for n in LENGTHS}
    changed_cases = []
    for d in sorted(human_labels):
        preds = {n: by_len[n].get(d) for n in LENGTHS}
        if not all(preds.values()):
            continue
        domains = {n: preds[n]["predicted_domain"] if preds[n]["success"] else f"FAILED:{preds[n]['error']}" for n in LENGTHS}
        if len(set(domains.values())) > 1:
            changed_cases.append({
                "doc_id": d, "human_label": human_labels[d],
                "pred_3000": domains[3000], "conf_3000": preds[3000]["confidence"],
                "pred_1500": domains[1500], "conf_1500": preds[1500]["confidence"],
                "pred_1000": domains[1000], "conf_1000": preds[1000]["confidence"],
            })
    return {"changed_human_labelled_cases": changed_cases, "n_changed": len(changed_cases)}


# ---------------------------------------------------------------------------
# Stability across the 100-doc interleaved sample (mirrors Phase 1's compare)
# ---------------------------------------------------------------------------

def analyse_sample_stability() -> dict:
    rows = load_jsonl(HERE / "interleaved_results.jsonl")
    by_len = {n: {r["doc_id"]: r for r in rows if r["prompt_chars"] == n} for n in LENGTHS}

    def compare(a: int, b: int) -> dict:
        shared = sorted(set(by_len[a]) & set(by_len[b]))
        genuine_pairs = [d for d in shared if by_len[a][d]["success"] and by_len[b][d]["success"]]
        changed = [d for d in genuine_pairs if by_len[a][d]["predicted_domain"] != by_len[b][d]["predicted_domain"]]
        conf_changed = sum(1 for d in genuine_pairs if by_len[a][d]["confidence"] != by_len[b][d]["confidence"])
        other_flip = sum(1 for d in genuine_pairs
                          if (by_len[a][d]["predicted_domain"] == "other_uncertain")
                          != (by_len[b][d]["predicted_domain"] == "other_uncertain"))
        failed_either = [d for d in shared if not (by_len[a][d]["success"] and by_len[b][d]["success"])]
        return {
            "compared_documents": len(genuine_pairs),
            "changed_primary_domain": len(changed),
            "changed_pct": round(100 * len(changed) / len(genuine_pairs), 1) if genuine_pairs else None,
            "changed_confidence_values": conf_changed,
            "changed_other_uncertain_predictions": other_flip,
            "failures_excluded": len(failed_either),
            "changed_doc_ids": changed,
        }

    return {
        "3000_vs_1500": compare(3000, 1500),
        "3000_vs_1000": compare(3000, 1000),
        "1500_vs_1000": compare(1500, 1000),
    }


def analyse_taxonomy_failures() -> dict:
    rows = load_jsonl(HERE / "interleaved_results.jsonl") + \
        (load_jsonl(HERE / "human_label_results.jsonl") if (HERE / "human_label_results.jsonl").exists() else [])
    failures = [r for r in rows if not r["success"]]
    by_error = defaultdict(list)
    for r in failures:
        by_error[r["error"]].append({"doc_id": r["doc_id"], "prompt_chars": r["prompt_chars"], "source": "interleaved" if r in load_jsonl(HERE/"interleaved_results.jsonl") else "human"})
    return {"n_failures": len(failures), "by_error": {k: v for k, v in by_error.items()}}


def main() -> None:
    timing = analyse_timing()
    (HERE / "timing_summary.json").write_text(json.dumps(timing, indent=2), encoding="utf-8")

    stability = analyse_sample_stability()

    human_labels_doc = json.loads((HERE / "human_labels.json").read_text())["labels"]
    quality = analyse_human(human_labels_doc)
    human_stability = analyse_human_stability(human_labels_doc)
    quality["stability"] = human_stability
    (HERE / "quality_summary.json").write_text(json.dumps(quality, indent=2), encoding="utf-8")

    taxonomy_failures = analyse_taxonomy_failures()

    summary = {
        "timing": timing,
        "sample_stability_100doc": stability,
        "human_label_quality": quality,
        "taxonomy_output_failures": taxonomy_failures,
    }
    (HERE / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    # ---- console report ----
    print("=" * 70)
    print("PART A: PAIRED INTERLEAVED TIMING (100-doc sample)")
    print("=" * 70)
    print(f"{'prompt':>7} {'n':>4} {'ok':>4} {'mean':>7} {'median':>7} {'p90':>7} {'min':>6} {'max':>7} {'total':>8} {'/min':>6}")
    for n in LENGTHS:
        b = timing["block_stats"][str(n)]
        print(f"{n:>7} {b['n']:>4} {b['successful']:>4} {b['mean']:>7} {b['median']:>7} "
              f"{b['p90']:>7} {b['min']:>6} {b['max']:>7} {b['total_seconds']:>8} {b['cases_per_minute']:>6}")

    print(f"\nPAIRED (n={timing['paired']['n_complete_triples']} complete triples):")
    for n in (1500, 1000):
        p = timing["paired"][f"{n}_minus_3000"]
        print(f"  {n} vs 3000: mean Δ={p['mean_delta_seconds']:+.2f}s ({p['mean_pct_change']:+.1f}%)  "
              f"median Δ={p['median_delta_seconds']:+.2f}s ({p['median_pct_change']:+.1f}%)  "
              f"faster={p['n_faster']} slower={p['n_slower']} unchanged={p['n_unchanged']}")

    print(f"\nCall-order mean seconds (bias check): {timing['call_order_mean_seconds']}")

    print("\n" + "=" * 70)
    print("SAMPLE STABILITY (100-doc interleaved sample)")
    print("=" * 70)
    for k, c in stability.items():
        print(f"  {k}: {c['changed_primary_domain']}/{c['compared_documents']} changed "
              f"({c['changed_pct']}%), {c['changed_confidence_values']} conf changes, "
              f"{c['failures_excluded']} excluded (failure), other_uncertain flips={c['changed_other_uncertain_predictions']}")
        if c['changed_doc_ids']:
            print(f"    changed: {c['changed_doc_ids']}")

    print("\n" + "=" * 70)
    print(f"PART B: HUMAN-LABEL QUALITY (n={quality['n_labelled_documents']}, "
          f"dist={quality['class_distribution']})")
    print("=" * 70)
    print(f"  WARNING: {quality['warning']}")
    print(f"{'prompt':>7} {'valid':>6} {'failed':>7} {'acc':>6} {'macroF1':>8} {'famF1':>6} {'critF1':>7} {'otherF1':>8}")
    for n in LENGTHS:
        q = quality["per_prompt_length"][str(n)]
        fam = q["per_class"]["family_law"]["f1"]
        crim = q["per_class"]["criminal_law"]["f1"]
        oth = q["per_class"]["other_uncertain"]["f1"]
        print(f"{n:>7} {q['n_valid_predictions']:>6} {q['n_failed']:>7} "
              f"{q['accuracy']:>6} {q['macro_f1']:>8} {fam:>6} {crim:>7} {oth:>8}")
        if q["n_failed"]:
            print(f"    failed doc_ids: {q['failed_doc_ids']}")
        print(f"    confusion: {q['confusion_matrix']}")

    print(f"\nCHANGED HUMAN-LABELLED CASES (n={human_stability['n_changed']}):")
    for c in human_stability["changed_human_labelled_cases"]:
        print(f"  {c['doc_id']} human={c['human_label']}  "
              f"3000={c['pred_3000']}({c['conf_3000']})  1500={c['pred_1500']}({c['conf_1500']})  "
              f"1000={c['pred_1000']}({c['conf_1000']})")

    print("\n" + "=" * 70)
    print("TAXONOMY/OUTPUT-VALIDATION FAILURES")
    print("=" * 70)
    print(f"  total: {taxonomy_failures['n_failures']}")
    for err, cases in taxonomy_failures["by_error"].items():
        print(f"  {err}: {cases}")

    print("\nwrote timing_summary.json, quality_summary.json, summary.json")


if __name__ == "__main__":
    main()
