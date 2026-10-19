"""
Offline cross-encoder reranker experiment for case-law retrieval.

Reuses the FROZEN, UNMODIFIED hybrid top-40 rankings already saved by the 150-query
benchmark (evaluation/results/final_family_law_150_query_benchmark_20261019_212630/
retrieval_hybrid_case.jsonl) -- no retrieval is re-run, no Qdrant query is issued by this
script. The original candidate list, rank, and rrf_score are preserved exactly; this
script only ADDS a reranker_score and a reranked_rank computed offline over the SAME
40 candidates per query, using BAAI/bge-reranker-v2-m3 (sentence-transformers CrossEncoder).

The reranker is NEVER given relevance labels/qrels -- it sees only (query_text,
candidate_text) pairs, exactly as a production reranker would at inference time.
"""
import json, sqlite3, time, sys
from pathlib import Path
from collections import defaultdict

NEW_DIR = Path(__file__).resolve().parent
OLD_DIR = NEW_DIR.parents[0] / "final_family_law_150_query_benchmark_20261019_212630"
REPO_ROOT = NEW_DIR.parents[2]

MODEL_NAME = "BAAI/bge-reranker-v2-m3"
MAX_LENGTH = 512
DEVICE = "mps"
BATCH_SIZE = 16

con = sqlite3.connect(REPO_ROOT / "var/rag/docstore.sqlite")
cur = con.cursor()
_case_key_cache = {}
def get_case_key(chunk_id):
    if chunk_id not in _case_key_cache:
        cur.execute("select json_extract(json,'$.case_key') from chunks where chunk_id=?", (chunk_id,))
        row = cur.fetchone()
        _case_key_cache[chunk_id] = row[0] if row else None
    return _case_key_cache[chunk_id]

# ---------------------------------------------------------------------------
# Step 1: load the frozen hybrid top-40 case rankings (unmodified, read-only)
# ---------------------------------------------------------------------------
by_query = defaultdict(list)
for line in open(OLD_DIR / "retrieval_hybrid_case.jsonl"):
    r = json.loads(line)
    if r.get("source_type") != "case":
        continue
    by_query[r["query_id"]].append(r)

for qid in by_query:
    by_query[qid].sort(key=lambda r: r["rank"])

n_queries = len(by_query)
assert n_queries == 50, f"expected 50 CL queries in the frozen hybrid rankings, found {n_queries}"
for qid, rows in by_query.items():
    assert len(rows) == 40, f"{qid}: expected 40 candidates, found {len(rows)}"
    assert [r["rank"] for r in rows] == list(range(1, 41)), f"{qid}: ranks not contiguous 1..40"

print(f"[1/4] Loaded frozen hybrid rankings: {n_queries} queries x 40 candidates = {sum(len(v) for v in by_query.values())} rows")

# ---------------------------------------------------------------------------
# Step 2: write baseline_hybrid_rankings.jsonl (frozen input, copied verbatim + case_key)
# ---------------------------------------------------------------------------
n_empty_text = 0
baseline_rows = []
for qid, rows in by_query.items():
    for r in rows:
        text = r.get("canonical_text") or ""
        case_key = get_case_key(r["chunk_id"])
        if not text.strip():
            n_empty_text += 1
        baseline_rows.append({
            "query_id": qid, "query_text": r["query"], "candidate_id": r["point_id"],
            "chunk_id": r["chunk_id"], "case_id": case_key,
            "original_hybrid_rank": r["rank"], "original_hybrid_score": r["rrf_score"],
            "text_source": "canonical_text (frozen retrieval_hybrid_case.jsonl, unmodified)",
            "text_length": len(text), "text": text,
            "error_or_warning": "empty_text" if not text.strip() else None,
        })

with open(NEW_DIR / "baseline_hybrid_rankings.jsonl", "w") as f:
    for r in baseline_rows:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")

print(f"[2/4] Wrote baseline_hybrid_rankings.jsonl ({len(baseline_rows)} rows). Empty-text candidates: {n_empty_text}")

# ---------------------------------------------------------------------------
# Step 3: load the cross-encoder and score every candidate
# ---------------------------------------------------------------------------
from sentence_transformers import CrossEncoder
import torch

t_load_start = time.time()
model = CrossEncoder(MODEL_NAME, device=DEVICE, max_length=MAX_LENGTH)
model_dtype = str(next(model.model.parameters()).dtype)
t_load_end = time.time()
model_init_seconds = t_load_end - t_load_start
print(f"[3/4] Loaded {MODEL_NAME} on {DEVICE} in {model_init_seconds:.2f}s (dtype={model_dtype})")

per_query_timings = []
reranked_rows = []
n_truncated = 0

t_infer_start = time.time()
for qid, rows in by_query.items():
    pairs = []
    truncation_flags = []
    for r in rows:
        text = r.get("canonical_text") or ""
        tok_len = len(model.tokenizer.encode(r["query"], text, truncation=False))
        was_truncated = tok_len > MAX_LENGTH
        truncation_flags.append(was_truncated)
        if was_truncated:
            n_truncated += 1
        pairs.append((r["query"], text))

    t0 = time.time()
    scores = model.predict(pairs, batch_size=BATCH_SIZE, show_progress_bar=False)
    t1 = time.time()
    per_query_timings.append({"query_id": qid, "n_candidates": len(pairs), "seconds": t1 - t0})

    scored = []
    for r, score, trunc in zip(rows, scores, truncation_flags):
        case_key = get_case_key(r["chunk_id"])
        scored.append({
            "query_id": qid, "query_text": r["query"], "candidate_id": r["point_id"],
            "chunk_id": r["chunk_id"], "case_id": case_key,
            "original_hybrid_rank": r["rank"], "original_hybrid_score": r["rrf_score"],
            "reranker_score": float(score), "reranker_model": MODEL_NAME,
            "text_source": "canonical_text (frozen retrieval_hybrid_case.jsonl, unmodified)",
            "text_length": len(r.get("canonical_text") or ""), "was_truncated": trunc,
            "error_or_warning": "empty_text" if not (r.get("canonical_text") or "").strip() else None,
        })
    # deterministic sort: reranker_score desc, tie-break by original hybrid rank asc
    scored.sort(key=lambda x: (-x["reranker_score"], x["original_hybrid_rank"]))
    for i, s in enumerate(scored, 1):
        s["reranked_rank"] = i
    reranked_rows.extend(scored)

t_infer_end = time.time()
total_inference_seconds = t_infer_end - t_infer_start

with open(NEW_DIR / "reranked_rankings.jsonl", "w") as f:
    for r in reranked_rows:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")

print(f"[4/4] Scored and reranked {len(reranked_rows)} candidates across {n_queries} queries "
      f"in {total_inference_seconds:.2f}s total. Truncated candidates: {n_truncated}")

# ---------------------------------------------------------------------------
# Step 4: latency/resource report
# ---------------------------------------------------------------------------
seconds_per_query = [t["seconds"] for t in per_query_timings]
seconds_per_query.sort()
n = len(seconds_per_query)
median_per_query = seconds_per_query[n // 2] if n % 2 else (seconds_per_query[n//2 - 1] + seconds_per_query[n//2]) / 2
mean_per_query = sum(seconds_per_query) / n
total_candidates = sum(t["n_candidates"] for t in per_query_timings)

mps_mem = None
try:
    if torch.backends.mps.is_available():
        mps_mem = {
            "current_allocated_bytes": torch.mps.current_allocated_memory(),
            "driver_allocated_bytes": torch.mps.driver_allocated_memory(),
        }
except Exception as e:
    mps_mem = {"error": str(e)}

latency_report = {
    "model": MODEL_NAME, "device": DEVICE, "dtype": model_dtype, "max_length": MAX_LENGTH,
    "batch_size": BATCH_SIZE,
    "n_queries": n_queries, "n_candidates_total": total_candidates,
    "n_truncated_candidates": n_truncated, "n_empty_text_candidates": n_empty_text,
    "model_init_seconds": round(model_init_seconds, 3),
    "total_inference_seconds": round(total_inference_seconds, 3),
    "mean_seconds_per_query": round(mean_per_query, 4),
    "median_seconds_per_query": round(median_per_query, 4),
    "mean_seconds_per_candidate": round(total_inference_seconds / total_candidates, 5),
    "per_query_timings": per_query_timings,
    "mps_memory_bytes": mps_mem,
    "note": "Measured on this machine's MPS (Apple Silicon) backend; not comparable to CPU/CUDA figures without re-measurement on that hardware.",
}
json.dump(latency_report, open(NEW_DIR / "latency_report.json", "w"), indent=2)
print(json.dumps({k: v for k, v in latency_report.items() if k != "per_query_timings"}, indent=2))
