"""Re-label ONLY the judgment-body paragraphs that the first structuring run left as
``unclassified`` because an LLM batch fell back -- without re-cleaning or
re-structuring anything else.

    var/rag/structured_gemini/<doc_id>.json   (existing structured files, read-only)
        -> label_body_paragraphs() on each run of fallback paragraphs
        -> var/rag/structured_gemini_relabeled/<doc_id>.json

Safe by construction:
  * paragraphs are rebuilt as ``full_text.split("\\n\\n")`` (verified to reproduce
    every existing span's text exactly); span text is always sliced from them;
  * caption, headnotes, judgment marker, final-order and already-labeled spans
    are copied unchanged;
  * only spans with type ``paragraph_group`` and label ``unclassified`` are sent to
    the LLM; whatever still fails stays ``paragraph_group``/``unclassified``;
  * every output is re-checked (spans tile the same paragraph range, text == slice).

Resumable: a document already present in the output directory is skipped.
Usage:  python -m src.rag_prep.relabel_unclassified [--in DIR] [--out DIR] [--limit N]
"""

from __future__ import annotations

import argparse
import json
import logging
import re
from pathlib import Path

from src.rag_prep.structurerLLM import RagStructuringLLMConfig, label_body_paragraphs

IN_DIR = Path("var/rag/structured_gemini")
OUT_DIR = Path("var/rag/structured_gemini_relabeled")

_LIST_MARKER_RE = re.compile(r"(?m)^\s*(?:\(?[ivx]+\)|\(?[a-z]\)|\d+[.)-])\s")


def _quotation_hints(paragraphs: list[str], idxs: list[int]) -> dict[int, int]:
    hints = {}
    for i in idxs:
        n = len(_LIST_MARKER_RE.findall(paragraphs[i]))
        if paragraphs[i].lstrip().startswith(("\"", "“")):
            n += 1
        if n:
            hints[i] = n
    return hints


def _span(paragraphs, start, end, label, confidence):
    text = "\n\n".join(paragraphs[start:end + 1])
    if label == "quoted_material":
        return {"type": "quoted_material", "paragraph_start": start, "paragraph_end": end, "text": text,
                "label": "quoted_material", "confidence": confidence, "attribution": "unknown"}
    return {"type": "semantic_section", "paragraph_start": start, "paragraph_end": end, "text": text,
            "label": label, "confidence": confidence}


def relabel_document(doc: dict, cfg: RagStructuringLLMConfig | None = None, llm_client=None) -> dict:
    paragraphs = doc["full_text"].split("\n\n")
    spans = doc["structure"]["spans"]
    out_spans: list[dict] = []
    diagnostics: list = []
    relabeled = still_unclassified = 0

    i = 0
    while i < len(spans):
        s = spans[i]
        if not (s["type"] == "paragraph_group" and s.get("label") in (None, "unclassified")):
            out_spans.append(s)
            i += 1
            continue
        j = i                                  # run of consecutive fallback spans
        while (j + 1 < len(spans) and spans[j + 1]["type"] == "paragraph_group"
               and spans[j + 1].get("label") in (None, "unclassified")
               and spans[j + 1]["paragraph_start"] == spans[j]["paragraph_end"] + 1):
            j += 1
        start, end = spans[i]["paragraph_start"], spans[j]["paragraph_end"]
        idxs = list(range(start, end + 1))
        labeled, failed = label_body_paragraphs(paragraphs, start, end, _quotation_hints(paragraphs, idxs),
                                                llm_client=llm_client, cfg=cfg, diagnostics=diagnostics)
        pieces = [_span(paragraphs, x["start"], x["end"], x["label"], x["confidence"]) for x in labeled]
        failed_idx = {k for a, b in failed for k in range(a, b + 1)}
        for k in sorted(failed_idx):
            pieces.append({"type": "paragraph_group", "paragraph_start": k, "paragraph_end": k,
                           "text": paragraphs[k], "label": "unclassified"})
        pieces.sort(key=lambda x: x["paragraph_start"])
        out_spans.extend(pieces)
        relabeled += len(idxs) - len(failed_idx)
        still_unclassified += len(failed_idx)
        i = j + 1

    _check(doc, paragraphs, out_spans)
    new = dict(doc)
    new["structure"] = {**doc["structure"], "spans": out_spans,
                        "llm_fallback_reason": (f"{still_unclassified} paragraph(s) still unclassified"
                                                if still_unclassified else None),
                        "relabel": {"relabeled_paragraphs": relabeled,
                                    "still_unclassified": still_unclassified,
                                    "diagnostics": diagnostics[:50]}}
    return new


def _check(doc, paragraphs, spans):
    old = doc["structure"]["spans"]
    lo, hi = old[0]["paragraph_start"], old[-1]["paragraph_end"]
    cursor = lo
    for s in sorted(spans, key=lambda x: x["paragraph_start"]):
        if s["paragraph_start"] != cursor:
            raise ValueError(f"span tiling broken at paragraph {cursor}")
        if s["text"] != "\n\n".join(paragraphs[s["paragraph_start"]:s["paragraph_end"] + 1]):
            raise ValueError(f"span text mismatch at paragraph {s['paragraph_start']}")
        cursor = s["paragraph_end"] + 1
    if cursor != hi + 1:
        raise ValueError("spans do not reach the end of the document")


def main() -> None:  # pragma: no cover
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", default=str(IN_DIR))
    ap.add_argument("--out", default=str(OUT_DIR))
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--backend", choices=["qwen", "gemini"], default="gemini")
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    client, cfg = None, None
    if a.backend == "gemini":
        from src.rag_prep.structurer_llm_gemini import GeminiStructuringClient, GeminiStructuringConfig
        gcfg = GeminiStructuringConfig(max_paragraphs_per_batch=12, max_chars_per_batch=5000)
        client = GeminiStructuringClient(cfg=gcfg)
        cfg = RagStructuringLLMConfig(max_paragraphs_per_batch=gcfg.max_paragraphs_per_batch,
                                      max_chars_per_batch=gcfg.max_chars_per_batch,
                                      max_chars_per_paragraph_shown=gcfg.max_chars_per_paragraph_shown)
    consecutive_dead = 0
    src, dst = Path(a.inp), Path(a.out)
    dst.mkdir(parents=True, exist_ok=True)
    done = skipped = 0
    totals = {"relabeled": 0, "still": 0}
    for path in sorted(src.glob("*.json")):
        out = dst / path.name
        if out.exists():
            skipped += 1
            continue
        doc = json.loads(path.read_text(encoding="utf-8"))
        if not any(s["type"] == "paragraph_group" and s.get("label") in (None, "unclassified")
                   for s in doc.get("structure", {}).get("spans", [])):
            out.write_text(json.dumps(doc, ensure_ascii=False, indent=2), encoding="utf-8")
            done += 1
            continue
        new = relabel_document(doc, cfg=cfg, llm_client=client)
        r = new["structure"]["relabel"]
        if any("circuit breaker" in str(x.get("error", "")) for x in r["diagnostics"]):
            # LLM unavailable / quota exhausted: do NOT write a fallback file (so the doc is retried
            # on the next run) and stop after 2 such documents in a row.
            consecutive_dead += 1
            print(f"{path.stem}: LLM unavailable -- not written. Last error: "
                  f"{next((x for x in reversed(r['diagnostics']) if 'attempt' in x), {}).get('error')}")
            if consecutive_dead >= 2:
                print("Stopping: LLM keeps failing (quota exhausted or service down). Re-run later; "
                      "finished documents are kept.")
                break
            continue
        consecutive_dead = 0
        totals["relabeled"] += r["relabeled_paragraphs"]
        totals["still"] += r["still_unclassified"]
        tmp = out.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(new, ensure_ascii=False, indent=2), encoding="utf-8")
        tmp.replace(out)
        done += 1
        print(f"{path.stem}: relabeled {r['relabeled_paragraphs']}, still unclassified {r['still_unclassified']}")
        if a.limit and done >= a.limit:
            break
    print(f"done {done}, skipped {skipped}, paragraphs relabeled {totals['relabeled']}, "
          f"still unclassified {totals['still']}")
    if client is not None and hasattr(client, "usage_summary"):
        print("LLM usage:", client.usage_summary())


if __name__ == "__main__":  # pragma: no cover
    main()