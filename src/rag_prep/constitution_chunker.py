"""Constitution of Pakistan parent-child chunking -- Stage 4 of the
independent Constitution pipeline (v2).

    parsed Constitution AST (var/rag/constitution_parsed/<doc_id>.json)
        -> build_chunks_for_document()   [deterministic, token-count-driven, no LLM]
        -> parent-child chunk JSON       (var/rag/constitution_chunked/<doc_id>.json)

Chunking rules (all sizes measured on the text that is actually embedded):

    Article  <= MAX_CHUNK_TOKENS     -> one chunk; parent == child; retrievable
    Article  >  MAX_CHUNK_TOKENS     -> one parent (whole Article, NOT embedded, used for
                                         context expansion) + retrievable children:
                                           * one child per Clause;
                                           * a Clause still too large is split at its own
                                             paragraph lines (sub-clauses, provisos,
                                             explanations, definitions) and every such child
                                             carries the clause's opening words ("lead-in");
                                           * a tiny Clause (< MIN_CHUNK_TOKENS) is merged into
                                             its neighbour.
    Preamble / Annex / Schedule      -> one parent + retrievable children split at their own
                                         paragraph lines; a Schedule also breaks at its own
                                         section labels ("### PART I", "PRESIDENT", ...) so an
                                         oath or a legislative-list Part never shares a chunk.

Text fields per chunk:
    canonical_text  exact slice of cleaned_text (display / citation; footnote refs kept)
    context_header  "[Constitution of Pakistan | Part II: ... | Article 25A: Right to education.]"
    embedding_text  context_header + lead-in (if any) + normalized body
    bm25_text       same content as embedding_text (titles and numbers help lexical match)
Normalization (embedding / BM25 only): footnote refs and amendment brackets removed,
"* * *" omission runs dropped, dashes and quotes unified, the "## N. Title" line dropped
(the header already carries it).

Never splits inside a line, never splits a Markdown table, never emits an empty chunk or
a heading-only chunk.

PIPELINE ISOLATION
-------------------
No import of src.ingestion/, src.classification/, src.rag_prep.*, or src.extraction.
Independent entry point, independent input/output directories. Never modifies the
parser's or cleaner's output.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

INPUT_DIR = Path("var/rag/constitution_parsed")
OUTPUT_DIR = Path("var/rag/constitution_chunked")

DOCUMENT_TITLE = "Constitution of Pakistan"
DOC_PREFIX = "CONST"

EMBED_TOKEN_LIMIT = 512        # hard limit of the embedding model (validated on embedding_text)
MAX_CHUNK_TOKENS = 400         # target ceiling for a chunk's body; leaves room for header + lead-in
MIN_CHUNK_TOKENS = 40          # smaller clauses / groups are merged with a neighbour
LEAD_IN_MAX_TOKENS = 60        # clause opening words prepended to split children

try:                                                     # deterministic token COUNTING only
    import tiktoken
    _ENCODING = tiktoken.get_encoding("cl100k_base")   # needs its vocab file (cached after first download)

    def count_tokens(text: str) -> int:
        return len(_ENCODING.encode(text))
except Exception:      # pragma: no cover - tiktoken missing or its vocab file can't be downloaded
    _ENCODING = None

    def count_tokens(text: str) -> int:
        # calibrated against cl100k_base on this corpus: never under-counts (max under-count ~0%)
        return int(len(re.findall(r"\w+|[^\w\s]", text)) * 1.30) + 1


class ConstitutionChunkingError(ValueError):
    """Raised when the input AST cannot be safely chunked."""


# -- text helpers ----------------------------------------------------------------
_REF_RE = re.compile(r"\[\^[^\]]+\]")
_ARTICLE_HEADING_LINE_RE = re.compile(r"^##[ \t]+\d+[A-Za-z]{0,3}\.[^\n]*\n*")
_STATUS_RE = re.compile(r"^\[\s*(Omitted|Repealed|Cease)", re.I)
_LEAD = r"(?:\[\^[^\]]+\]|\[)*"
_CLAUSE_LINE_RE = re.compile(rf"^{_LEAD}(?:\d+[A-Za-z]{{0,3}}\.[ \t]*{_LEAD})?\((\d+[A-Za-z]{{0,2}})\)")
_SUB_LINE_RE = re.compile(rf"^{_LEAD}\(([a-z]{{1,2}}|[ivxl]+)\)")
_SECTION_LABEL_RE = re.compile(r"^(?:###[ \t]+.*|[A-Z][A-Z .,'’()\-–—&]{2,60})$")   # "### PART I", "PRESIDENT"


def strip_refs(text: str) -> str:
    """Remove footnote refs and unwrap amendment brackets: '[^5.3][Balochistan]' -> 'Balochistan'."""
    prev = None
    while prev != text:
        prev = text
        text = re.sub(r"\[\^[^\]]+\]\[([^\[\]]*)\]", r"\1", text)
    return _REF_RE.sub("", text)


def normalize(text: str) -> str:
    text = strip_refs(text)
    text = re.sub(r"(?m)^#{1,3}[ \t]+", "", text)                 # Markdown heading markers
    text = re.sub(r"(?:\*[ \t]*){3,}", " ", text)                  # "* * *" omission runs
    text = re.sub(r"(?m)^\*(?=[A-Z])", "", text)                    # "*FIFTH SCHEDULE"
    text = re.sub(r"(?<=\w)[ \t]*\*(?=[ \t]*[.,;:)]|[ \t]*$)", "", text)   # lone omission star: "sex *."
    text = text.replace("–", "-").replace("—", "-").replace("―", "-")
    text = text.replace("“", '"').replace("”", '"').replace("‘", "'").replace("’", "'")
    text = re.sub(r"[ \t]{2,}", " ", text)
    text = re.sub(r"[ \t]+\n", "\n", text)
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def clean_title(title: str | None) -> str | None:
    if not title:
        return None
    t = normalize(re.sub(r"^#{1,3}[ \t]+", "", title)).strip(" []")
    return t or None


_ORDINALS = "FIRST|SECOND|THIRD|FOURTH|FIFTH|SIXTH|SEVENTH"


def chapter_title(title: str | None) -> str | None:
    """'THE MAJLIS-E-SHOORA (PARLIAMENT) Composition, Duration and ...' -> 'THE MAJLIS-E-SHOORA (PARLIAMENT)'
    (the mixed-case tail is a sub-heading printed under the chapter title)."""
    t = clean_title(title)
    if not t:
        return None
    m = re.match(r"^([A-Z0-9][A-Z0-9 ,.'()\-&]*?)(?:\s+[A-Z][a-z].*)?$", t)
    return (m.group(1).strip(" .,") if m else t) or t


def schedule_title(first_line: str) -> str | None:
    """'# [^189.1][FIRST SCHEDULE' -> 'First Schedule'; '# THIRD SCHEDULE Oaths of Office' ->
    'Third Schedule: Oaths of Office'."""
    t = clean_title(first_line) or ""
    m = re.match(rf"^\*?({_ORDINALS})\s+SCHEDULE\b\s*(.*)$", t)
    if not m:
        return t or None
    name = f"{m.group(1).title()} Schedule"
    return f"{name}: {m.group(2).strip()}" if m.group(2).strip() else name


_DEF_TERM_RE = re.compile(r"""^(?:\[\^[^\]]+\]|\[)*["“‘'`]{1,2}([^"”’']{1,60})["”’']{1,2}\s+(?:means|includes|shall|has)\b""")


def _truncate_tokens(text: str, limit: int) -> str:
    if count_tokens(text) <= limit:
        return text
    words, out = text.split(), []
    for w in words:
        if count_tokens(" ".join(out + [w])) > limit:
            break
        out.append(w)
    return " ".join(out) + " …"


# -- chunk record ----------------------------------------------------------------------
@dataclass(frozen=True)
class Chunk:
    doc_id: str
    chunk_id: str
    parent_id: str
    child_id: str
    unit_id: str
    unit_type: str                      # article | clause | clause_part | preamble | annex | schedule | schedule_part
    is_parent: bool
    retrievable: bool                   # True -> embed + BM25-index this chunk
    part_number: str | None
    part_title: str | None
    chapter_number: str | None
    chapter_title: str | None
    article_number: str | None
    article_title: str | None
    article_status: str | None          # in_force | omitted | repealed | ceased
    clause_number: str | None
    sub_range: str | None               # e.g. "(a)-(f)" for a split clause
    schedule: str | None
    section_label: str | None           # Schedule section, e.g. "PART I", "PRESIDENT"
    token_count: int                    # tokens of embedding_text
    source_start: int
    source_end: int
    canonical_text: str
    context_header: str
    lead_in: str | None
    embedding_text: str
    bm25_text: str
    footnote_ids: list = field(default_factory=list)
    references_articles: list = field(default_factory=list)

    def to_dict(self) -> dict:
        return dict(self.__dict__)

    @classmethod
    def from_dict(cls, d: dict) -> "Chunk":
        return cls(**d)


# -- header / references ---------------------------------------------------------------
def _header(ctx: dict, clause: str | None = None, sub_range: str | None = None,
            section_label: str | None = None) -> str:
    seg = [DOCUMENT_TITLE]
    if ctx.get("schedule"):
        seg.append(ctx["schedule"])
        if section_label:
            seg.append(section_label)
    elif ctx.get("unit") in ("preamble", "annex"):
        seg.append(ctx["unit_title"])
    else:
        if ctx.get("part_number"):
            seg.append(f"Part {ctx['part_number']}: {ctx['part_title']}" if ctx.get("part_title")
                       else f"Part {ctx['part_number']}")
        if ctx.get("chapter_number"):
            seg.append(f"Chapter {ctx['chapter_number']}: {ctx['chapter_title']}" if ctx.get("chapter_title")
                       else f"Chapter {ctx['chapter_number']}")
        if ctx.get("article_number"):
            t = ctx.get("article_title")
            seg.append(f"Article {ctx['article_number']}: {t}" if t else f"Article {ctx['article_number']}")
        if clause:
            seg.append(f"Clause ({clause})" + (f" {sub_range}" if sub_range else ""))
    return "[" + " | ".join(seg) + "]"


_ART_REF_RE = re.compile(
    r"\bArticles?\s+((?:\d+[A-Z]{0,3}(?:\s*\([^)]{1,6}\))*(?:\s*,\s*|\s+and\s+|\s+or\s+|\s+to\s+)?)+)")


def _article_refs(text: str, known: list[str], self_no: str | None) -> list[str]:
    order = {a: i for i, a in enumerate(known)}
    found: list[str] = []
    for m in _ART_REF_RE.finditer(text):
        span = re.sub(r"\([^)]*\)", "", m.group(1))
        nums = re.findall(r"\d+[A-Z]{0,3}|\bto\b", span)
        prev = None
        for i, n in enumerate(nums):
            if n == "to" and prev and i + 1 < len(nums) and nums[i + 1] in order and prev in order:
                found.extend(known[order[prev] + 1:order[nums[i + 1]]])
            elif n != "to":
                if n in order:
                    found.append(n)
                prev = n
    seen, out = set(), []
    for a in found:
        if a != self_no and a not in seen:
            seen.add(a)
            out.append(a)
    return out


# -- line segmentation ---------------------------------------------------------------
def _segments(text: str, base: int) -> list[tuple[int, int, str]]:
    """Paragraph lines of a span as (abs_start, abs_end, text). Consecutive table rows form one
    segment; a heading line ('## ...' / '# ...') is fused with the line after it so it can
    never become a chunk of its own."""
    segs: list[list] = []
    pos = 0
    pending_heading = None
    for raw in text.splitlines(keepends=True):
        start, end = base + pos, base + pos + len(raw)
        pos += len(raw)
        line = raw.strip()
        if not line:
            if segs and pending_heading is None:
                segs[-1][1] = end                         # keep blank lines inside spans (exact coverage)
            elif pending_heading is not None:
                pending_heading[1] = end
            continue
        if line.startswith("#"):
            if pending_heading is None:
                pending_heading = [start, end, raw]
            else:
                pending_heading[1], pending_heading[2] = end, pending_heading[2] + raw
            continue
        if pending_heading is not None:
            start = pending_heading[0]
            raw = pending_heading[2] + raw
            pending_heading = None
        if line.startswith("|") and segs and segs[-1][2].rstrip().split("\n")[-1].lstrip().startswith("|"):
            segs[-1][1], segs[-1][2] = end, segs[-1][2] + raw
            continue
        segs.append([start, end, raw])
    if pending_heading is not None:                      # trailing heading: attach to previous
        if segs:
            segs[-1][1], segs[-1][2] = pending_heading[1], segs[-1][2] + pending_heading[2]
        else:
            segs.append(pending_heading)
    return [(s, e, t) for s, e, t in segs]


def _body_tokens(text: str) -> int:
    return count_tokens(normalize(_ARTICLE_HEADING_LINE_RE.sub("", text)))


def _pack(segs: list[tuple[int, int, str]], max_tokens: int, breaks: set[int] | None = None) -> list[list]:
    """Greedy packing of whole segments into groups <= max_tokens; a segment index in
    `breaks` always starts a new group."""
    groups: list[list] = []
    cur: list = []
    cur_tok = 0
    for i, seg in enumerate(segs):
        t = _body_tokens(seg[2])
        if cur and (cur_tok + t > max_tokens or (breaks and i in breaks and cur_tok >= MIN_CHUNK_TOKENS)):
            groups.append(cur)
            cur, cur_tok = [], 0
        cur.append(seg)
        cur_tok += t
    if cur:
        groups.append(cur)
    # merge a tiny trailing group backwards when it fits
    if len(groups) > 1 and sum(_body_tokens(s[2]) for s in groups[-1]) < MIN_CHUNK_TOKENS:
        last = groups.pop()
        if sum(_body_tokens(s[2]) for s in groups[-1] + last) <= max_tokens and not (breaks and segs.index(last[0]) in breaks):
            groups[-1].extend(last)
        else:
            groups.append(last)
    return groups


# -- chunk builder -------------------------------------------------------------------------
class _Builder:
    def __init__(self, parsed_doc: dict):
        self.doc_id = parsed_doc["doc_id"]
        self.text = parsed_doc["cleaned_text"]
        self.root = parsed_doc["structure"]
        self.known_articles: list[str] = []
        self._collect(self.root)
        self.chunks: list[Chunk] = []
        self.ids: set[str] = set()

    def _collect(self, n: dict):
        if n.get("type") == "article":
            self.known_articles.append(n["article_number"])
        for c in n.get("children", []):
            self._collect(c)

    def _uid(self, base: str) -> str:
        cid, k = base, 2
        while cid in self.ids:
            cid, k = f"{base}_{k}", k + 1
        self.ids.add(cid)
        return cid

    def _make(self, ctx: dict, *, chunk_id: str, parent_id: str, unit_id: str, unit_type: str,
              is_parent: bool, retrievable: bool, start: int, end: int, clause: str | None = None,
              sub_range: str | None = None, lead_in: str | None = None, section_label: str | None = None) -> Chunk:
        canonical = self.text[start:end]
        header = _header(ctx, clause, sub_range, section_label)
        body = normalize(_ARTICLE_HEADING_LINE_RE.sub("", canonical.lstrip()))
        lead = normalize(lead_in) if lead_in else None
        emb = header + "\n\n" + (f"{lead} …\n" if lead else "") + body
        return Chunk(
            doc_id=self.doc_id, chunk_id=chunk_id, parent_id=parent_id, child_id=chunk_id, unit_id=unit_id,
            unit_type=unit_type, is_parent=is_parent, retrievable=retrievable,
            part_number=ctx.get("part_number"), part_title=ctx.get("part_title"),
            chapter_number=ctx.get("chapter_number"), chapter_title=ctx.get("chapter_title"),
            article_number=ctx.get("article_number"), article_title=ctx.get("article_title"),
            article_status=ctx.get("article_status"), clause_number=clause, sub_range=sub_range,
            schedule=ctx.get("schedule"), section_label=section_label,
            token_count=count_tokens(emb), source_start=start, source_end=end,
            canonical_text=canonical, context_header=header, lead_in=lead, embedding_text=emb, bm25_text=emb,
            footnote_ids=sorted(set(re.findall(r"\[\^([^\]]+)\]", canonical)), key=lambda s: [int(x) for x in s.split(".") if x.isdigit()]),
            references_articles=_article_refs(normalize(canonical), self.known_articles, ctx.get("article_number")),
        )

    # ---- articles ----
    def article(self, art: dict, ctx: dict):
        number = art["article_number"]
        title = clean_title(art.get("heading"))
        status = "in_force"
        m = _STATUS_RE.match(art.get("heading") or "")
        if m:
            status = {"omitted": "omitted", "repealed": "repealed", "cease": "ceased"}[m.group(1).lower()]
        ctx = {**ctx, "article_number": number, "article_title": title, "article_status": status}
        aid = f"art_{number.lower()}"
        unit = f"{DOC_PREFIX}:art:{number}"
        start, end = art["source_start"], self._trim_end(art["source_start"], art["source_end"])

        if _body_tokens(self.text[start:end]) <= MAX_CHUNK_TOKENS:
            self.chunks.append(self._make(ctx, chunk_id=self._uid(aid), parent_id=aid, unit_id=unit,
                                          unit_type="article", is_parent=True, retrievable=True, start=start, end=end))
            return

        self.chunks.append(self._make(ctx, chunk_id=self._uid(aid), parent_id=aid, unit_id=unit, unit_type="article",
                                      is_parent=True, retrievable=False, start=start, end=end))
        segs = _segments(self.text[start:end], start)

        # group lines into clause units
        units: list[dict] = []
        for seg in segs:
            first_line = seg[2].lstrip()
            first_line = _ARTICLE_HEADING_LINE_RE.sub("", first_line).lstrip()
            cm = _CLAUSE_LINE_RE.match(first_line)
            if cm or not units:
                units.append({"clause": cm.group(1) if cm else None, "segs": [seg]})
            else:
                units[-1]["segs"].append(seg)

        # merge tiny clause units forward (or into the previous one at the end)
        merged: list[dict] = []
        for u in units:
            tok = sum(_body_tokens(s[2]) for s in u["segs"])
            if merged:
                prev = merged[-1]
                prev_tok = sum(_body_tokens(s[2]) for s in prev["segs"])
                if (prev_tok < MIN_CHUNK_TOKENS or tok < MIN_CHUNK_TOKENS) and prev_tok + tok <= MAX_CHUNK_TOKENS:
                    prev["segs"].extend(u["segs"])
                    prev["clauses"].append(u["clause"])
                    continue
            merged.append({"clause": u["clause"], "clauses": [u["clause"]], "segs": list(u["segs"])})

        for u in merged:
            clause_ids = [c for c in u["clauses"] if c]
            clause_label = (clause_ids[0] if len(clause_ids) == 1 else
                            f"{clause_ids[0]})–({clause_ids[-1]}" if clause_ids else None)
            tok = sum(_body_tokens(s[2]) for s in u["segs"])
            if tok <= MAX_CHUNK_TOKENS:
                s0, s1 = u["segs"][0][0], u["segs"][-1][1]
                base = f"{aid}_c{clause_ids[0].lower()}" if clause_ids else f"{aid}_intro"
                self.chunks.append(self._make(
                    ctx, chunk_id=self._uid(base), parent_id=aid,
                    unit_id=f"{unit}:cl:{'-'.join(clause_ids)}" if clause_ids else f"{unit}:intro",
                    unit_type="clause", is_parent=False, retrievable=True, start=s0, end=s1, clause=clause_label))
                continue
            # oversized clause: split at its own paragraph lines, carry the clause lead-in
            lead_text = _ARTICLE_HEADING_LINE_RE.sub("", u["segs"][0][2].lstrip())
            lead_in = _truncate_tokens(normalize(lead_text), LEAD_IN_MAX_TOKENS)
            lead_tok = count_tokens(lead_in)
            for gi, grp in enumerate(_pack(u["segs"], MAX_CHUNK_TOKENS - lead_tok)):
                lines = [_ARTICLE_HEADING_LINE_RE.sub("", s[2].lstrip()).lstrip() for s in grp]
                terms = [normalize(m.group(1)).strip() for l in lines if (m := _DEF_TERM_RE.match(l))]
                subs = [m.group(1) for l in lines if (m := _SUB_LINE_RE.match(l))]
                if len(terms) >= 2:
                    sub_range = f'definitions "{terms[0]}" to "{terms[-1]}"'
                elif subs and _SUB_LINE_RE.match(lines[0]):
                    sub_range = f"({subs[0]})" if len(subs) == 1 else f"({subs[0]})–({subs[-1]})"
                else:
                    sub_range = None
                base = f"{aid}_c{clause_ids[0].lower()}_p{gi + 1}" if clause_ids else f"{aid}_p{gi + 1}"
                self.chunks.append(self._make(
                    ctx, chunk_id=self._uid(base), parent_id=aid,
                    unit_id=(f"{unit}:cl:{clause_ids[0]}" if clause_ids else unit) + f":part:{gi + 1}",
                    unit_type="clause_part", is_parent=False, retrievable=True, start=grp[0][0], end=grp[-1][1],
                    clause=clause_label, sub_range=sub_range, lead_in=None if gi == 0 else lead_in))

    # ---- preamble / annex / schedules ----
    def block(self, node: dict, kind: str):
        start, end = node["source_start"], self._trim_end(node["source_start"], node["source_end"])
        if kind == "schedule":
            ctx = {"schedule": schedule_title(self.text[start:end].split("\n", 1)[0])}
            pid = "sch_" + node["id"].removeprefix("schedule_").removesuffix("_schedule")
            unit = f"{DOC_PREFIX}:schedule:{pid.removeprefix('sch_')}"
        else:
            ctx = {"unit": kind, "unit_title": "Preamble" if kind == "preamble"
                   else "Annex: The Objectives Resolution"}
            pid, unit = kind, f"{DOC_PREFIX}:{kind}"

        small = _body_tokens(self.text[start:end]) <= MAX_CHUNK_TOKENS
        self.chunks.append(self._make(ctx, chunk_id=self._uid(pid), parent_id=pid, unit_id=unit, unit_type=kind,
                                      is_parent=True, retrievable=small, start=start, end=end))
        if small:
            return
        segs = _segments(self.text[start:end], start)
        breaks, labels, label, prev_was_label, base_label, run_start = set(), [], None, False, None, 0
        for i, s in enumerate(segs):
            lines = [l.strip() for l in s[2].strip().split("\n") if l.strip() and not l.strip().startswith("# ")]
            if kind == "schedule" and lines and lines[0].startswith("###"):       # "### PART I" (fused with entry 1)
                part = clean_title(lines[0])
                label = f"{base_label}, {part}" if base_label else part
                breaks.add(i)
                prev_was_label = False
                labels.append(label)
                continue
            cand = lines[-1] if lines else ""
            cand_n = normalize(cand).strip("[] ")
            is_label = (kind == "schedule" and bool(cand_n) and len(lines) == 1
                        and bool(_SECTION_LABEL_RE.match(cand_n)))
            if is_label:
                text_l = clean_title(cand) or ""
                if prev_was_label and label:
                    label = base_label = f"{label} {text_l}".strip()   # "THE FEDERAL CONSTITUTIONAL COURT AND THE SUPREME COURT"
                    k = len(labels) - 1
                    while k >= run_start:                              # relabel the whole run of label lines
                        labels[k] = label
                        k -= 1
                elif text_l.upper() not in ("AND", "OR"):
                    label = base_label = text_l
                    run_start = i
                    breaks.add(i)
            prev_was_label = is_label
            labels.append(label)
        for gi, grp in enumerate(_pack(segs, MAX_CHUNK_TOKENS, breaks)):
            sec = labels[segs.index(grp[-1])] if labels[segs.index(grp[0])] is None else labels[segs.index(grp[0])]
            lab_in = [labels[segs.index(x)] for x in grp if segs.index(x) in breaks]
            sec = lab_in[-1] if lab_in else sec
            self.chunks.append(self._make(
                ctx, chunk_id=self._uid(f"{pid}_p{gi + 1}"), parent_id=pid, unit_id=f"{unit}:part:{gi + 1}",
                unit_type=f"{kind}_part", is_parent=False, retrievable=True,
                start=grp[0][0], end=grp[-1][1], section_label=sec))

    def _trim_end(self, start: int, end: int) -> int:
        while end > start and self.text[end - 1] in " \t\r\n":
            end -= 1
        return end

    def build(self) -> list[Chunk]:
        for top in self.root.get("children", []):
            t = top["type"]
            if t == "part":
                pctx = {"part_number": top["id"].removeprefix("part_"), "part_title": clean_title(top.get("heading"))}
                for sub in top.get("children", []):
                    if sub["type"] == "chapter":
                        cctx = {**pctx, "chapter_number": sub["id"].rsplit("_chapter", 1)[-1],
                                "chapter_title": chapter_title(sub.get("heading"))}
                        for art in sub.get("children", []):
                            if art["type"] == "article":
                                self.article(art, cctx)
                    elif sub["type"] == "article":
                        self.article(sub, pctx)
            elif t in ("preamble", "annex", "schedule"):
                self.block(top, t)
        return self.chunks


def build_chunks_for_document(parsed_doc: dict) -> list[Chunk]:
    if not parsed_doc.get("validation", {}).get("ok"):
        raise ConstitutionChunkingError(
            "Refusing to chunk a parsed Constitution document whose own structural validation did not pass: "
            f"{parsed_doc.get('validation', {}).get('errors')}")
    chunks = _Builder(parsed_doc).build()
    if not chunks:
        raise ConstitutionChunkingError("No Article, Schedule, Preamble, or Annex content found to chunk.")
    return chunks


# -- validation -------------------------------------------------------------------------
@dataclass
class ChunkValidationResult:
    ok: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    diagnostics: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {"ok": self.ok, "errors": self.errors, "warnings": self.warnings, "diagnostics": self.diagnostics}


def validate_chunks(chunks: list[Chunk], parsed_doc: dict) -> ChunkValidationResult:
    errors: list[str] = []
    warnings: list[str] = []
    text = parsed_doc["cleaned_text"]

    ids = [c.chunk_id for c in chunks]
    dup = sorted({i for i in ids if ids.count(i) > 1})
    if dup:
        errors.append(f"duplicate chunk_id(s): {dup[:10]}")
    if any(not i for i in ids):
        errors.append("chunk without chunk_id")

    parents = {c.parent_id: c for c in chunks if c.is_parent}
    for c in chunks:
        if not (0 <= c.source_start < c.source_end <= len(text)):
            errors.append(f"{c.chunk_id}: bad offsets [{c.source_start},{c.source_end})")
            continue
        if text[c.source_start:c.source_end] != c.canonical_text:
            errors.append(f"{c.chunk_id}: canonical_text != cleaned_text[source_start:source_end]")
        body = normalize(_ARTICLE_HEADING_LINE_RE.sub("", c.canonical_text.lstrip()))
        if not body:
            errors.append(f"{c.chunk_id}: empty or heading-only chunk")
        if re.search(r"\n##[ \t]+\d", c.canonical_text):
            errors.append(f"{c.chunk_id}: contains another Article's heading (boundary leak)")
        if "#" in c.context_header or "[^" in c.context_header:
            errors.append(f"{c.chunk_id}: context_header contains markup: {c.context_header[:80]}")
        if c.retrievable and c.token_count > EMBED_TOKEN_LIMIT:
            errors.append(f"{c.chunk_id}: {c.token_count} tokens > embedding limit {EMBED_TOKEN_LIMIT}")
        if "[^" in c.embedding_text:
            errors.append(f"{c.chunk_id}: footnote refs left in embedding_text")
        if not c.is_parent and c.parent_id not in parents:
            errors.append(f"{c.chunk_id}: parent {c.parent_id} missing")

    # children of a non-retrievable parent must exactly tile the parent (ignoring whitespace)
    kids: dict[str, list[Chunk]] = {}
    for c in chunks:
        if not c.is_parent:
            kids.setdefault(c.parent_id, []).append(c)
    for pid, p in parents.items():
        ks = sorted(kids.get(pid, []), key=lambda c: c.source_start)
        if not p.retrievable and not ks:
            errors.append(f"{pid}: non-retrievable parent without children (content not indexed)")
        cursor = p.source_start
        for k in ks:
            if k.source_start < cursor:
                errors.append(f"{k.chunk_id}: overlaps previous sibling")
            elif text[cursor:k.source_start].strip():
                errors.append(f"{pid}: text lost before {k.chunk_id}: {text[cursor:k.source_start][:60]!r}")
            cursor = max(cursor, k.source_end)
        if ks and text[cursor:p.source_end].strip():
            errors.append(f"{pid}: text lost at end: {text[cursor:p.source_end][:60]!r}")

    def collect(n, out):
        if n.get("type") == "article":
            out.append(n["article_number"])
        for c in n.get("children", []):
            collect(c, out)
    real: list[str] = []
    collect(parsed_doc["structure"], real)
    represented = {c.article_number for c in chunks if c.article_number}
    missing = [a for a in real if a not in represented]
    if missing:
        errors.append(f"Articles not represented in any chunk: {missing[:20]}")
    parts = {c.part_number for c in chunks if c.part_number}

    retr = [c for c in chunks if c.retrievable]
    small = [c.chunk_id for c in retr if c.token_count - count_tokens(c.context_header) < 12]
    if small:
        warnings.append(f"{len(small)} retrievable chunks have < 12 body tokens: {small[:10]}")

    toks = sorted(c.token_count for c in retr)
    diagnostics = {
        "total_chunks": len(chunks),
        "parent_chunks": sum(c.is_parent for c in chunks),
        "child_chunks": sum(not c.is_parent for c in chunks),
        "retrievable_chunks": len(retr),
        "articles_represented": len(represented),
        "articles_in_ast": len(real),
        "parts_represented": len(parts),
        "omitted_articles": sorted({c.article_number for c in chunks if c.article_status not in (None, "in_force")}),
        "retrievable_token_min_median_max": [toks[0], toks[len(toks) // 2], toks[-1]] if toks else None,
        "embed_token_limit": EMBED_TOKEN_LIMIT,
    }
    return ChunkValidationResult(ok=not errors, errors=errors, warnings=warnings, diagnostics=diagnostics)


def is_valid_existing_output(doc_dict: dict) -> bool:
    try:
        chunks = [Chunk.from_dict(c) for c in doc_dict["chunks"]]
        parsed_doc = {"cleaned_text": doc_dict["cleaned_text"], "structure": doc_dict["structure"]}
        return validate_chunks(chunks, parsed_doc).ok
    except Exception:  # noqa: BLE001 -- a malformed or old-schema file is simply regenerated
        return False


def chunk_document(parsed_doc: dict) -> dict:
    chunks = build_chunks_for_document(parsed_doc)
    result = validate_chunks(chunks, parsed_doc)
    return {
        "doc_id": parsed_doc["doc_id"],
        "cleaned_text": parsed_doc["cleaned_text"],
        "structure": parsed_doc["structure"],
        "chunks": [c.to_dict() for c in chunks],
        "validation": result.to_dict(),
    }


def chunk_directory(input_dir: Path = INPUT_DIR, output_dir: Path = OUTPUT_DIR) -> dict:
    input_dir, output_dir = Path(input_dir), Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    processed = regenerated = skipped_valid = 0
    failed_docs: list[dict] = []

    for parsed_path in sorted(input_dir.glob("*.json")):
        doc_id = parsed_path.stem
        out_path = output_dir / f"{doc_id}.json"
        was_present = out_path.exists()
        if was_present:
            try:
                existing = json.loads(out_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                existing = None
            if existing is not None and is_valid_existing_output(existing):
                skipped_valid += 1
                continue
        try:
            parsed_doc = json.loads(parsed_path.read_text(encoding="utf-8"))
            chunked_doc = chunk_document(parsed_doc)
        except (OSError, json.JSONDecodeError, KeyError, ConstitutionChunkingError) as exc:
            failed_docs.append({"doc_id": doc_id, "error": str(exc)})
            continue
        if not chunked_doc["validation"]["ok"]:
            failed_docs.append({"doc_id": doc_id, "error": chunked_doc["validation"]["errors"]})
            continue
        tmp_path = out_path.with_suffix(".json.tmp")
        tmp_path.write_text(json.dumps(chunked_doc, ensure_ascii=False, indent=2), encoding="utf-8")
        tmp_path.replace(out_path)
        processed += 1
        regenerated += was_present

    return {"processed": processed, "skipped_valid": skipped_valid, "regenerated": regenerated,
            "failed": len(failed_docs), "failed_docs": failed_docs}


def main() -> None:  # pragma: no cover
    summary = chunk_directory()
    print(f"Constitution chunking: {summary['processed']} processed "
          f"({summary['regenerated']} regenerated), {summary['skipped_valid']} skipped (valid), "
          f"{summary['failed']} failed")
    for f in summary["failed_docs"]:
        print(f"  FAILED {f['doc_id']}: {f['error']}")
    if summary["failed"]:
        raise SystemExit(1)


if __name__ == "__main__":  # pragma: no cover
    main()