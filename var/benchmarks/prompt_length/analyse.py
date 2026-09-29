#!/usr/bin/env python3
"""Compare the three prompt-length runs. Read-only; writes summary.json only."""

from __future__ import annotations

import csv, json, sqlite3, statistics as st, sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
LENGTHS = (3000, 1500, 1000)
BASELINE = 3000


def load(n: int) -> dict:
    p = HERE / f"results_{n}.jsonl"
    out = {}
    for line in p.read_text(encoding="utf-8").splitlines():
        if line.strip():
            r = json.loads(line)
            out[r["doc_id"]] = r
    return out


def runtime(rows: dict) -> dict:
    ok = [r for r in rows.values() if r["success"]]
    times = [r["elapsed_seconds"] for r in rows.values()]   # all attempts count
    return {
        "total_documents": len(rows),
        "successful": len(ok),
        "failed": len(rows) - len(ok),
        "total_runtime_seconds": round(sum(times), 1),
        "avg_seconds_per_document": round(st.mean(times), 2) if times else None,
        "median_seconds_per_document": round(st.median(times), 2) if times else None,
        "min_seconds": round(min(times), 2) if times else None,
        "max_seconds": round(max(times), 2) if times else None,
        "avg_text_chars_sent": round(st.mean([r["text_chars_sent"] for r in rows.values()]), 1),
    }


def compare(a: dict, b: dict) -> dict:
    shared = sorted(set(a) & set(b))
    changed, conf_changed, other_delta = [], 0, 0
    for d in shared:
        ra, rb = a[d], b[d]
        if ra["predicted_domain"] != rb["predicted_domain"]:
            changed.append(d)
        if ra["confidence"] != rb["confidence"]:
            conf_changed += 1
        if (ra["predicted_domain"] == "other_uncertain") != (rb["predicted_domain"] == "other_uncertain"):
            other_delta += 1
    return {
        "compared_documents": len(shared),
        "changed_primary_domain": len(changed),
        "changed_pct": round(100 * len(changed) / len(shared), 1) if shared else None,
        "changed_confidence_values": conf_changed,
        "changed_other_uncertain_predictions": other_delta,
        "changed_doc_ids": changed,
    }


def main() -> None:
    runs = {}
    for n in LENGTHS:
        try:
            runs[n] = load(n)
        except FileNotFoundError:
            print(f"results_{n}.jsonl missing -- run the benchmark first")
            sys.exit(1)

    meta = json.loads((HERE / "sample_metadata.json").read_text()) if (HERE / "sample_metadata.json").exists() else {}
    manifest = json.loads((HERE / "sample_manifest.json").read_text())
    titles = {r["doc_id"]: r for r in csv.DictReader((HERE / "sample_100.csv").open(encoding="utf-8"))}

    summary = {
        "run_metadata": json.loads((HERE / "run_metadata.json").read_text()),
        "sample": {"size": manifest["sample_size"], "seed": manifest["sample_seed"],
                   "method": manifest["method"], "pool_size": manifest["pool_size"]},
        "runtime": {str(n): runtime(runs[n]) for n in LENGTHS},
        "stability": {
            "3000_vs_1500": compare(runs[3000], runs[1500]),
            "3000_vs_1000": compare(runs[3000], runs[1000]),
            "1500_vs_1000": compare(runs[1500], runs[1000]),
        },
    }

    # Every document whose label moved under ANY comparison, reported in full.
    moved = sorted({d for c in summary["stability"].values() for d in c["changed_doc_ids"]})
    summary["changed_documents"] = [{
        "doc_id": d,
        "title": (titles.get(d, {}).get("title") or "")[:90],
        "source_relpath": titles.get(d, {}).get("source_relpath"),
        "signal_text_chars": runs[3000][d]["signal_text_chars"],
        "pred_3000": runs[3000][d]["predicted_domain"], "conf_3000": runs[3000][d]["confidence"],
        "pred_1500": runs[1500][d]["predicted_domain"], "conf_1500": runs[1500][d]["confidence"],
        "pred_1000": runs[1000][d]["predicted_domain"], "conf_1000": runs[1000][d]["confidence"],
    } for d in moved]

    # Human labels: ledger + the completed blind-40 sheet. Reported only.
    con = sqlite3.connect(f"file:{REPO/'var/metadata.db'}?mode=ro", uri=True)
    con.row_factory = sqlite3.Row
    human = {}
    for r in con.execute("""SELECT doc_id, decision, decision_domain, machine_domain
                              FROM document_review_decisions"""):
        d = dict(r)
        lab = (d["machine_domain"] if d["decision"] == "human_accepted" else d["decision_domain"])
        if lab:
            human[d["doc_id"]] = lab
    con.close()
    blind = REPO / "var/review_queue/blind_validation_40.csv"
    if blind.exists():
        for r in csv.DictReader(blind.open(encoding="utf-8")):
            lab = (r.get("verified_domain") or "").strip()
            if lab:
                human[r["doc_id"]] = lab

    overlap = sorted(set(runs[3000]) & set(human))
    summary["human_validation"] = {
        "human_labels_available_corpus_wide": len(human),
        "overlap_with_benchmark_sample": len(overlap),
        "sufficient_for_metrics": len(overlap) >= 30,
        "note": ("Too few overlapping labels for accuracy / macro-F1 / confusion "
                 "matrix; the overlapping cases are listed individually instead. "
                 "No labels were invented and Qwen agreement is NOT reported as accuracy."
                 if len(overlap) < 30 else "sufficient"),
        "cases": [{"doc_id": d, "human_label": human[d],
                   "pred_3000": runs[3000][d]["predicted_domain"],
                   "pred_1500": runs[1500][d]["predicted_domain"],
                   "pred_1000": runs[1000][d]["predicted_domain"]} for d in overlap],
    }

    (HERE / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False),
                                       encoding="utf-8")

    print("RUNTIME")
    print(f"  {'prompt':>6} {'ok':>4} {'fail':>5} {'avg s':>7} {'med s':>7} {'min':>6} {'max':>7} {'total s':>9} {'chars sent':>11}")
    for n in LENGTHS:
        r = summary["runtime"][str(n)]
        print(f"  {n:>6} {r['successful']:>4} {r['failed']:>5} {r['avg_seconds_per_document']:>7} "
              f"{r['median_seconds_per_document']:>7} {r['min_seconds']:>6} {r['max_seconds']:>7} "
              f"{r['total_runtime_seconds']:>9} {r['avg_text_chars_sent']:>11}")
    base = summary["runtime"][str(BASELINE)]["avg_seconds_per_document"]
    print("\nSPEEDUP vs 3000")
    for n in (1500, 1000):
        a = summary["runtime"][str(n)]["avg_seconds_per_document"]
        print(f"  {n:>6}: {a} s/doc   {100*(1-a/base):+.1f}%   "
              f"projected 6141 docs = {a*6141/3600:.1f} h (vs {base*6141/3600:.1f} h)")
    print("\nSTABILITY")
    for k, c in summary["stability"].items():
        print(f"  {k:>13}: {c['changed_primary_domain']}/{c['compared_documents']} labels changed "
              f"({c['changed_pct']}%), {c['changed_confidence_values']} confidences changed, "
              f"{c['changed_other_uncertain_predictions']} other_uncertain flips")
    print(f"\nCHANGED DOCUMENTS ({len(summary['changed_documents'])})")
    for c in summary["changed_documents"]:
        print(f"  {c['doc_id']} [{c['signal_text_chars']:>6} chars] {c['source_relpath']:>10} "
              f"3000={c['pred_3000']}({c['conf_3000']}) 1500={c['pred_1500']}({c['conf_1500']}) "
              f"1000={c['pred_1000']}({c['conf_1000']})")
    h = summary["human_validation"]
    print(f"\nHUMAN LABELS: {h['overlap_with_benchmark_sample']} overlap of "
          f"{h['human_labels_available_corpus_wide']} available -- "
          f"sufficient={h['sufficient_for_metrics']}")
    for c in h["cases"]:
        print(f"  {c['doc_id']} human={c['human_label']} 3000={c['pred_3000']} "
              f"1500={c['pred_1500']} 1000={c['pred_1000']}")
    print("\nwrote summary.json")


if __name__ == "__main__":
    main()
