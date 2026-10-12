"""
Embed + index all three corpora (Constitution, statutes, case law) into one Qdrant collection.

    python src/rag_prep/embed_index.py check            # bge-m3 token lengths per corpus
    python src/rag_prep/embed_index.py embed            # dense bge-m3 vectors -> disk cache (resumable)
    python src/rag_prep/embed_index.py index            # docstore + Qdrant upsert (dense + BM25 sparse)
    python src/rag_prep/embed_index.py all              # embed + index
    python src/rag_prep/embed_index.py search "khula without returning haq mehr"

Design
  * One collection, `source_type` in {constitution, statute, case}; search per source, then merge/rerank.
  * Only `retrievable` chunks become points. Every chunk (parents, counsel, footnotes) goes to the
    SQLite docstore for parent / neighbour expansion at answer time.
  * Point id = uuid5(chunk_id)  -> re-runs overwrite, never duplicate.
  * Dense: BAAI/bge-m3 on `embedding_text` (CLS, L2-normalised, cosine). No query instruction needed.
  * Sparse: fastembed Qdrant/bm25 on `bm25_text`; Qdrant applies IDF (Modifier.IDF).
  * Dense vectors are cached by sha1(model | max_len | embedding_text): a re-chunk only re-embeds
    chunks whose text changed. Payload-only changes (metadata fixes) re-upsert without re-embedding.
  * `index` upserts only points whose content hash changed and deletes points whose chunk vanished.

Requirements
  pip install "qdrant-client>=1.10" fastembed sentence-transformers transformers torch numpy tqdm
  docker run -d --name qdrant -p 6333:6333 -p 6334:6334 \
      -v "$(pwd)/var/qdrant:/qdrant/storage" qdrant/qdrant
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sqlite3
import sys
import time
import uuid
import warnings
from pathlib import Path
from typing import Iterable, Iterator

import numpy as np

# -- config -------------------------------------------------------------------------------
SOURCES: dict[str, Path] = {
    "constitution": Path("var/rag/constitution_chunked"),
    "statute": Path("var/rag/statutes_chunked"),
    "case": Path("var/rag/caselaw_chunked"),
}
MODEL_NAME = "BAAI/bge-m3"
DENSE_DIM = 1024
MAX_SEQ_LEN = 1024                 # bge-m3 supports 8192; 1024 bounds memory and covers every chunk
BM25_MODEL = "Qdrant/bm25"
CACHE_DIR = Path("var/rag/embeddings/bge-m3")
DOCSTORE_PATH = Path("var/rag/docstore.sqlite")
META_PATH = Path("var/rag/index_meta.json")
QDRANT_URL = os.environ.get("QDRANT_URL", "http://localhost:6333")
COLLECTION = "pk_legal"
CASE_LIMIT: int | None = None      # --limit: only the first N case files (Constitution + statutes always full)
UPSERT_BATCH = 128
CACHE_FLUSH_EVERY = 1024
ID_NAMESPACE = uuid.UUID("6f1c2a7e-3b8d-5e4f-9a0b-1c2d3e4f5a6b")   # fixed: never change after first index

KEYWORD_INDEXES = [
    "source_type", "doc_id", "chunk_id", "unit_type", "base_unit_id", "ref_unit_ids", "ref_act_keys",
    "act_key", "section_number", "article_number", "case_key", "court", "disposition", "section",
    "cited_case_keys", "primary_domain",
]
INTEGER_INDEXES = ["year"]


# -- loading ------------------------------------------------------------------------------
def iter_docs(sources: dict[str, Path] = SOURCES) -> Iterator[tuple[str, Path, dict]]:
    for source_type, folder in sources.items():
        if not folder.exists():
            raise SystemExit(f"[error] missing chunk folder for {source_type}: {folder}")
        paths = sorted(folder.glob("*.json"))
        if source_type == "case" and CASE_LIMIT:
            paths = paths[:CASE_LIMIT]
        for path in paths:
            doc = json.loads(path.read_text(encoding="utf-8"))
            yield source_type, path, doc


def load_chunks(sources: dict[str, Path] = SOURCES) -> tuple[list[dict], list[dict]]:
    """Returns (all_chunks, retrievable_chunks). Each chunk gets `_source_type` and `_doc`."""
    all_chunks: list[dict] = []
    skipped: list[str] = []
    seen: dict[str, str] = {}
    for source_type, path, doc in iter_docs(sources):
        if not doc.get("validation", {}).get("ok", False):
            skipped.append(f"{source_type}:{path.name}")
            continue
        doc_meta = {k: doc.get(k) for k in ("doc_id", "act", "case", "metadata")}
        for c in doc["chunks"]:
            cid = c["chunk_id"]
            if cid in seen:
                raise SystemExit(f"[error] duplicate chunk_id {cid} in {path} and {seen[cid]}")
            seen[cid] = str(path)
            all_chunks.append({**c, "_source_type": source_type, "_doc": doc_meta})
    if skipped:
        print(f"[warn] skipped {len(skipped)} docs with validation.ok == False: {skipped[:10]}")
    retrievable = [c for c in all_chunks if c.get("retrievable")]
    return all_chunks, retrievable


# -- payload ------------------------------------------------------------------------------
_YEAR_RE = re.compile(r"\b(18|19|20)\d{2}\b")


def _dedupe(xs: Iterable) -> list:
    out, seen = [], set()
    for x in xs:
        if x and x not in seen:
            seen.add(x)
            out.append(x)
    return out


def build_payload(c: dict, unit_of: dict[str, str]) -> dict:
    st = c["_source_type"]
    p: dict = {
        "source_type": st,
        "chunk_id": c["chunk_id"],
        "doc_id": c["doc_id"],
        "parent_id": c.get("parent_id"),
        "unit_id": c.get("unit_id"),
        "base_unit_id": unit_of.get(c.get("parent_id")) or c.get("unit_id"),
        "unit_type": c.get("unit_type"),
        "context_header": c.get("context_header"),
        "lead_in": c.get("lead_in"),
        "canonical_text": c.get("canonical_text"),
        "token_count": c.get("token_count"),
    }
    if st == "constitution":
        p.update({k: c.get(k) for k in (
            "part_number", "part_title", "chapter_number", "chapter_title", "article_number",
            "article_title", "article_status", "clause_number", "sub_range", "schedule", "section_label")})
        p["ref_unit_ids"] = _dedupe(f"CONST:art:{a}" for a in c.get("references_articles") or [])
        p["ref_act_keys"] = []
        p["year"] = 1973
    elif st == "statute":
        p.update({k: c.get(k) for k in (
            "act_key", "act_name", "year", "part_number", "part_title", "chapter_number", "chapter_title",
            "section_number", "section_title", "section_status", "sub_range", "schedule", "schedule_unit")})
        p["subsections"] = c.get("subsections") or []
        p["defined_terms"] = c.get("defined_terms") or []
        refs = c.get("references") or {}
        ak = c.get("act_key")
        ext = refs.get("external_acts") or []
        p["ref_unit_ids"] = _dedupe(
            [f"{ak}:s:{s}" for s in refs.get("sections") or []]
            + [f"{e['act_key']}:s:{s}" for e in ext if e.get("act_key") for s in e.get("sections") or []])
        p["ref_act_keys"] = _dedupe(e.get("act_key") for e in ext if e.get("year"))
        p["primary_domain"] = (c.get("metadata") or {}).get("domain")
    else:  # case
        m = c.get("metadata") or {}
        p.update({k: m.get(k) for k in (
            "case_key", "citation", "court", "court_location", "case_title", "case_number",
            "decision_date", "judges", "disposition", "primary_domain")})
        p["case_key"] = c.get("case_key") or p["case_key"]
        p["citation"] = c.get("citation") or p["citation"]
        y = (p.get("decision_date") or "")[:4]
        if not y.isdigit():
            ym = _YEAR_RE.search(p.get("citation") or "")
            y = ym.group(0) if ym else ""
        p["year"] = int(y) if y.isdigit() else None
        p.update({k: c.get(k) for k in (
            "chunk_index", "part", "section", "labels", "headnote_letter", "headnote_topic",
            "opinion_author", "prev_chunk_id", "next_chunk_id")})
        refs = c.get("references") or {}
        p["cited_cases"] = refs.get("cases") or []
        p["cited_case_keys"] = _dedupe(x.get("case_key") for x in p["cited_cases"])
        p["statutes"] = refs.get("statutes") or []
        p["constitution_articles"] = refs.get("constitution_articles") or []
        p["ref_unit_ids"] = _dedupe(list(refs.get("unit_ids") or [])
                                    + [u for s in p["statutes"] for u in s.get("unit_ids") or []])
        p["ref_act_keys"] = _dedupe(s.get("act_key") for s in p["statutes"])
        p["flags"] = c.get("flags") or {}
    return p


def point_id(chunk_id: str) -> str:
    return str(uuid.uuid5(ID_NAMESPACE, chunk_id))


def cache_key(text: str) -> str:
    return hashlib.sha1(f"{MODEL_NAME}|{MAX_SEQ_LEN}|{text}".encode("utf-8")).hexdigest()


# -- dense embedding cache ----------------------------------------------------------------
class EmbeddingCache:
    """Append-only shards of (sha1 keys, float16 vectors). Safe to interrupt; resumes where it stopped."""

    def __init__(self, folder: Path = CACHE_DIR):
        self.folder = folder
        self.folder.mkdir(parents=True, exist_ok=True)
        self.index: dict[str, tuple[int, int]] = {}
        self.shards: list[np.ndarray] = []
        self._pending_keys: list[str] = []
        self._pending_set: set[str] = set()
        self._pending_vecs: list[np.ndarray] = []
        for f in sorted(self.folder.glob("shard_*.npz")):
            with np.load(f, allow_pickle=False) as z:
                keys, vecs = z["keys"], z["vecs"]
            if vecs.shape[1] != DENSE_DIM:
                raise SystemExit(f"[error] cache shard {f} has dim {vecs.shape[1]}, expected {DENSE_DIM}")
            si = len(self.shards)
            self.shards.append(vecs)
            for row, k in enumerate(keys.tolist()):
                self.index[k] = (si, row)

    def __contains__(self, key: str) -> bool:
        return key in self.index or key in self._pending_set

    def get(self, key: str) -> np.ndarray:
        if key in self._pending_set:
            self.flush()
        si, row = self.index[key]
        return self.shards[si][row].astype(np.float32)

    def add(self, keys: list[str], vecs: np.ndarray) -> None:
        self._pending_keys.extend(keys)
        self._pending_set.update(keys)
        self._pending_vecs.append(vecs.astype(np.float16))
        if len(self._pending_keys) >= CACHE_FLUSH_EVERY:
            self.flush()

    def flush(self) -> None:
        if not self._pending_keys:
            return
        vecs = np.concatenate(self._pending_vecs, axis=0)
        keys = np.array(self._pending_keys)
        name = f"shard_{time.strftime('%Y%m%d-%H%M%S')}_{len(self.shards):05d}.npz"
        tmp = self.folder / (name + ".tmp")
        with open(tmp, "wb") as fh:
            np.savez(fh, keys=keys, vecs=vecs)
        tmp.replace(self.folder / name)
        si = len(self.shards)
        self.shards.append(vecs)
        for row, k in enumerate(self._pending_keys):
            self.index[k] = (si, row)
        self._pending_keys, self._pending_vecs, self._pending_set = [], [], set()

    def __len__(self) -> int:
        return len(self.index) + len(self._pending_keys)


# -- models -------------------------------------------------------------------------------
def pick_device() -> str:
    import torch
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_dense(device: str | None = None, fp16: bool = True):
    from sentence_transformers import SentenceTransformer
    device = device or pick_device()
    model = SentenceTransformer(MODEL_NAME, device=device)
    model.max_seq_length = MAX_SEQ_LEN
    if fp16 and device in ("mps", "cuda"):
        model = model.half()
    print(f"[dense] {MODEL_NAME} on {device} ({'fp16' if fp16 and device != 'cpu' else 'fp32'}), "
          f"max_seq_length={MAX_SEQ_LEN}")
    return model


def encode_dense(model, texts: list[str], batch_size: int) -> np.ndarray:
    vecs = model.encode(texts, batch_size=batch_size, normalize_embeddings=True,
                        convert_to_numpy=True, show_progress_bar=False)
    vecs = np.asarray(vecs, dtype=np.float32)
    if not np.isfinite(vecs).all():
        raise RuntimeError("non-finite values in embeddings; re-run with --no-fp16")
    return vecs


def load_sparse(avg_len: float = 256.0):
    from fastembed.sparse.bm25 import Bm25
    return Bm25(BM25_MODEL, avg_len=avg_len)


def bm25_avg_len(model, texts: list[str]) -> float:
    """Average document length in BM25 tokens (after stopwords/stemming), as fastembed counts |d|."""
    try:
        from fastembed.common.utils import remove_non_alphanumeric
        lens = [len(model._stem(model.tokenizer.tokenize(remove_non_alphanumeric(t)))) for t in texts]
    except Exception:  # private API moved -> pre-stopword count is a close upper bound
        lens = [model.token_count(t) for t in texts]
    return float(np.mean(lens)) if lens else 256.0


def encode_sparse(model, texts: list[str]):
    return model.raw_embed(texts)


# -- commands -----------------------------------------------------------------------------
def cmd_check(args) -> None:
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    _, chunks = load_chunks()
    by_src: dict[str, list[tuple[int, str]]] = {}
    for c in chunks:
        n = len(tok(c["embedding_text"], add_special_tokens=True)["input_ids"])
        by_src.setdefault(c["_source_type"], []).append((n, c["chunk_id"]))
    print(f"{'source':<13}{'chunks':>8}{'mean':>7}{'p95':>6}{'max':>6}{'>512':>7}{'>' + str(MAX_SEQ_LEN):>7}")
    worst: list[tuple[int, str]] = []
    truncated = 0
    for src, rows in by_src.items():
        ns = np.array([r[0] for r in rows])
        over = int((ns > MAX_SEQ_LEN).sum())
        truncated += over
        print(f"{src:<13}{len(ns):>8}{ns.mean():>7.0f}{np.percentile(ns, 95):>6.0f}{ns.max():>6}"
              f"{int((ns > 512).sum()):>7}{over:>7}")
        worst.extend(rows)
    worst.sort(reverse=True)
    print("longest:", ", ".join(f"{cid} ({n})" for n, cid in worst[:5]))
    print("VERDICT:", "PASS (nothing truncated)" if truncated == 0
          else f"FAIL - {truncated} chunks exceed MAX_SEQ_LEN={MAX_SEQ_LEN}; raise it")


def cmd_embed(args) -> None:
    _, chunks = load_chunks()
    cache = EmbeddingCache()
    todo: dict[str, str] = {}
    for c in chunks:
        k = cache_key(c["embedding_text"])
        if k not in cache and k not in todo:
            todo[k] = c["embedding_text"]
    print(f"[embed] {len(chunks)} retrievable chunks, {len(cache)} cached, {len(todo)} to embed")
    if not todo:
        return
    model = load_dense(args.device, fp16=not args.no_fp16)
    items = sorted(todo.items(), key=lambda kv: -len(kv[1]))      # longest first: OOM shows up at once
    try:
        from tqdm import tqdm
        bar = tqdm(total=len(items), unit="chunk")
    except ImportError:
        bar = None
    t0 = time.time()
    try:
        for i in range(0, len(items), args.batch_size):
            batch = items[i:i + args.batch_size]
            vecs = encode_dense(model, [t for _, t in batch], args.batch_size)
            cache.add([k for k, _ in batch], vecs)
            if bar:
                bar.update(len(batch))
            elif (i // args.batch_size) % 50 == 0:
                print(f"  {i + len(batch)}/{len(items)}  {(i + len(batch)) / (time.time() - t0):.1f} chunks/s")
    finally:
        cache.flush()
        if bar:
            bar.close()
    print(f"[embed] done in {time.time() - t0:.0f}s; cache now {len(cache)} vectors")


def write_docstore(all_chunks: list[dict], path: Path | None = None) -> None:
    path = path or DOCSTORE_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".sqlite.tmp")
    if tmp.exists():
        tmp.unlink()
    con = sqlite3.connect(tmp)
    con.execute("""CREATE TABLE chunks (chunk_id TEXT PRIMARY KEY, point_id TEXT, source_type TEXT,
                   doc_id TEXT, parent_id TEXT, unit_id TEXT, unit_type TEXT, retrievable INTEGER,
                   prev_chunk_id TEXT, next_chunk_id TEXT, canonical_text TEXT, embedding_text TEXT,
                   json TEXT)""")
    rows = []
    for c in all_chunks:
        body = {k: v for k, v in c.items() if not k.startswith("_")}
        rows.append((c["chunk_id"], point_id(c["chunk_id"]), c["_source_type"], c["doc_id"],
                     c.get("parent_id"), c.get("unit_id"), c.get("unit_type"), int(bool(c.get("retrievable"))),
                     c.get("prev_chunk_id"), c.get("next_chunk_id"), c.get("canonical_text"),
                     c.get("embedding_text"), json.dumps(body, ensure_ascii=False)))
    con.executemany("INSERT INTO chunks VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)", rows)
    for col in ("point_id", "doc_id", "parent_id", "unit_id"):
        con.execute(f"CREATE INDEX idx_{col} ON chunks({col})")
    con.commit()
    con.close()
    tmp.replace(path)
    print(f"[docstore] {len(rows)} chunks -> {path}")


def connect(url: str = QDRANT_URL):
    from qdrant_client import QdrantClient
    return QdrantClient(url=url, timeout=120) if "://" in url else QdrantClient(path=url)


def ensure_collection(client, recreate: bool) -> None:
    from qdrant_client import models
    exists = client.collection_exists(COLLECTION)
    if exists and recreate:
        client.delete_collection(COLLECTION)
        exists = False
    if not exists:
        client.create_collection(
            COLLECTION,
            vectors_config={"dense": models.VectorParams(size=DENSE_DIM, distance=models.Distance.COSINE)},
            sparse_vectors_config={"bm25": models.SparseVectorParams(modifier=models.Modifier.IDF)},
        )
        print(f"[qdrant] created collection {COLLECTION}")
    else:
        info = client.get_collection(COLLECTION)
        size = info.config.params.vectors["dense"].size
        if size != DENSE_DIM:
            raise SystemExit(f"[error] {COLLECTION} has dense size {size}, expected {DENSE_DIM}; use --recreate")
    with warnings.catch_warnings():               # embedded local mode ignores payload indexes
        warnings.simplefilter("ignore")
        for f in KEYWORD_INDEXES:
            client.create_payload_index(COLLECTION, f, field_schema=models.PayloadSchemaType.KEYWORD)
        for f in INTEGER_INDEXES:
            client.create_payload_index(COLLECTION, f, field_schema=models.PayloadSchemaType.INTEGER)


def existing_hashes(client) -> dict[str, str]:
    out, offset = {}, None
    while True:
        pts, offset = client.scroll(COLLECTION, limit=2048, offset=offset,
                                    with_payload=["point_hash"], with_vectors=False)
        for p in pts:
            out[str(p.id)] = (p.payload or {}).get("point_hash")
        if offset is None:
            return out


def cmd_index(args) -> None:
    from qdrant_client import models
    all_chunks, chunks = load_chunks()
    write_docstore(all_chunks)

    cache = EmbeddingCache()
    missing = [c["chunk_id"] for c in chunks if cache_key(c["embedding_text"]) not in cache]
    if missing:
        raise SystemExit(f"[error] {len(missing)} chunks have no cached vector (e.g. {missing[:3]}); run `embed`")

    unit_of = {c["chunk_id"]: c.get("unit_id") for c in all_chunks}
    for c in all_chunks:                                    # cases: parent_id == doc_id -> case_key
        if c["_source_type"] == "case" and c.get("case_key"):
            unit_of.setdefault(c["doc_id"], c["case_key"])

    client = connect(args.qdrant)
    frozen = None
    if not args.recreate and client.collection_exists(COLLECTION) and META_PATH.exists():
        frozen = json.loads(META_PATH.read_text(encoding="utf-8")).get("bm25_avg_len")
    # avg_len is frozen after the first build so later edits stay incremental; --recreate recomputes it.
    avg_len = frozen or bm25_avg_len(load_sparse(), [c["bm25_text"] for c in chunks])
    sparse = load_sparse(avg_len=avg_len)
    print(f"[bm25] avg_len={avg_len:.1f}{' (frozen)' if frozen else ''}")
    ensure_collection(client, args.recreate)
    have = {} if args.recreate else existing_hashes(client)

    wanted: dict[str, tuple[dict, dict]] = {}
    for c in chunks:
        payload = build_payload(c, unit_of)
        h = hashlib.sha1(json.dumps([cache_key(c["embedding_text"]), c["bm25_text"], avg_len, payload],
                                    sort_keys=True, ensure_ascii=False, default=str).encode()).hexdigest()
        payload["point_hash"] = h
        wanted[point_id(c["chunk_id"])] = (c, payload)

    to_upsert = [pid for pid, (_, pl) in wanted.items() if have.get(pid) != pl["point_hash"]]
    stale = [pid for pid in have if pid not in wanted]
    print(f"[qdrant] {len(wanted)} points wanted, {len(have)} present, "
          f"{len(to_upsert)} to upsert, {len(stale)} stale to delete")

    for i in range(0, len(to_upsert), UPSERT_BATCH):
        ids = to_upsert[i:i + UPSERT_BATCH]
        batch = [wanted[pid] for pid in ids]
        sp = encode_sparse(sparse, [c["bm25_text"] for c, _ in batch])
        points = [
            models.PointStruct(
                id=pid,
                vector={"dense": cache.get(cache_key(c["embedding_text"])).tolist(),
                        "bm25": models.SparseVector(indices=s.indices.tolist(), values=s.values.tolist())},
                payload=pl)
            for pid, (c, pl), s in zip(ids, batch, sp)
        ]
        client.upsert(COLLECTION, points=points, wait=True)
        if (i // UPSERT_BATCH) % 20 == 0:
            print(f"  upserted {min(i + UPSERT_BATCH, len(to_upsert))}/{len(to_upsert)}")
    for i in range(0, len(stale), 1000):
        client.delete(COLLECTION, points_selector=models.PointIdsList(points=stale[i:i + 1000]), wait=True)

    counts = {st: client.count(COLLECTION, count_filter=models.Filter(must=[
        models.FieldCondition(key="source_type", match=models.MatchValue(value=st))]), exact=True).count
        for st in SOURCES}
    total = client.count(COLLECTION, exact=True).count
    expected = {st: sum(c["_source_type"] == st for c in chunks) for st in SOURCES}
    print(f"[qdrant] counts {counts} total={total}")
    ok = counts == expected and total == len(chunks)
    print("VERDICT:", "PASS" if ok else f"FAIL - expected {expected}")
    META_PATH.write_text(json.dumps({
        "collection": COLLECTION, "dense_model": MODEL_NAME, "dense_dim": DENSE_DIM, "max_seq_len": MAX_SEQ_LEN,
        "sparse_model": BM25_MODEL, "bm25_avg_len": avg_len, "id_namespace": str(ID_NAMESPACE),
        "counts": counts, "indexed_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }, indent=2), encoding="utf-8")


# -- search (smoke test; reranking comes next) --------------------------------------------
def hybrid_search(client, dense_model, sparse_model, query: str, per_source: dict[str, int],
                  candidates: int = 40, extra_filter=None) -> list:
    from qdrant_client import models
    qd = encode_dense(dense_model, [query], 1)[0].tolist()
    qs = next(iter(sparse_model.query_embed(query)))
    qs = models.SparseVector(indices=qs.indices.tolist(), values=qs.values.tolist())
    hits = []
    for st, k in per_source.items():
        must = [models.FieldCondition(key="source_type", match=models.MatchValue(value=st))]
        if extra_filter is not None:
            must.append(extra_filter)
        flt = models.Filter(must=must)
        res = client.query_points(
            COLLECTION,
            prefetch=[models.Prefetch(query=qd, using="dense", limit=candidates, filter=flt),
                      models.Prefetch(query=qs, using="bm25", limit=candidates, filter=flt)],
            query=models.FusionQuery(fusion=models.Fusion.RRF),
            limit=k, with_payload=True,
        ).points
        hits.extend(res)
    return hits


def cmd_search(args) -> None:
    client = connect(args.qdrant)
    dense = load_dense(args.device, fp16=not args.no_fp16)
    sparse = load_sparse()
    per = {"constitution": args.k_const, "statute": args.k_stat, "case": args.k_case}
    for h in hybrid_search(client, dense, sparse, args.query, per):
        p = h.payload
        print(f"{h.score:.4f}  {p['source_type']:<12} {p['context_header']}")
        print("        " + (p["canonical_text"] or "").replace("\n", " ")[:160])


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["check", "embed", "index", "all", "search"])
    ap.add_argument("query", nargs="?")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--device", default=None, help="mps | cuda | cpu (default: auto)")
    ap.add_argument("--no-fp16", action="store_true")
    ap.add_argument("--qdrant", default=QDRANT_URL, help="URL, or a folder path for embedded local mode")
    ap.add_argument("--recreate", action="store_true", help="drop and rebuild the collection")
    ap.add_argument("--limit", type=int, default=None,
                    help="test run: first N case files only; goes to collection pk_legal_sample unless --collection")
    ap.add_argument("--collection", default=None)
    ap.add_argument("--k-const", type=int, default=5)
    ap.add_argument("--k-stat", type=int, default=8)
    ap.add_argument("--k-case", type=int, default=10)
    args = ap.parse_args(argv)
    global COLLECTION, CASE_LIMIT, META_PATH, DOCSTORE_PATH
    CASE_LIMIT = args.limit
    COLLECTION = args.collection or (f"{COLLECTION}_sample" if args.limit else COLLECTION)
    if COLLECTION != "pk_legal":                 # keep sample artefacts apart from the real index
        META_PATH = META_PATH.with_name(f"index_meta_{COLLECTION}.json")
        DOCSTORE_PATH = DOCSTORE_PATH.with_name(f"docstore_{COLLECTION}.sqlite")
    print(f"[config] collection={COLLECTION}" + (f", case files limited to {CASE_LIMIT}" if CASE_LIMIT else ""))
    if args.cmd == "check":
        cmd_check(args)
    elif args.cmd == "embed":
        cmd_embed(args)
    elif args.cmd == "index":
        cmd_index(args)
    elif args.cmd == "all":
        cmd_embed(args)
        cmd_index(args)
    else:
        if not args.query:
            sys.exit("search needs a query")
        cmd_search(args)


if __name__ == "__main__":
    main()