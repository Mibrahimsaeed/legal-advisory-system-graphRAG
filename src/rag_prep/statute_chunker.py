"""Section-aware statute chunking -- Stage 2 of the independent statute pipeline (v2).

    cleaned statute JSON (var/rag/cleaned_statutes/<doc_id>.json)     <- statute_cleaning.py v2
        -> parse_statute()            [preamble / chapters / sections / schedules / footnotes]
        -> build_chunks()             [parent-child, token-driven, deterministic, no LLM]
        -> chunked statute JSON       (var/rag/statutes_chunked/<doc_id>.json)

WHAT A CHUNK IS
---------------
The SECTION is the legal unit. Every chunk carries the full legal context in its header:

    [Guardians and Wards Act, 1890 | Chapter III: Duties, Rights and Liabilities of Guardians
     | Guardian of Property | Section 29: Limitation of powers of guardian of property ...]

    Section  <= MAX_CHUNK_TOKENS  -> ONE chunk (parent == child), retrievable.
    Section  >  MAX_CHUNK_TOKENS  -> one parent (whole section; NOT embedded; used to expand
                                     context for the LLM) + retrievable children made of WHOLE
                                     sub-sections packed into balanced groups (a sub-section is
                                     never cut, and its provisos / explanations stay with it).
                                     A single sub-section that is itself too long is split at its
                                     clause lines and every piece carries its opening words
                                     ("lead-in"). Definition sections are split at definition
                                     items and every piece carries the definitions lead-in
                                     ("In this Act, unless ... context,--").
    Preamble                      -> one retrievable chunk (long title, Act No., date, WHEREAS).
    Schedule                      -> one chunk per Part / Form (split the same way if too long).
    [FOOTNOTES] block             -> one NON-retrievable chunk (kept for coverage). Each chunk
                                     instead carries the amendment notes of the footnotes it
                                     cites in `amendments` (metadata, not embedded).

Repealed / omitted sections stay as their own (small) chunks with section_status set and
"[omitted]" / "[repealed]" in the header, so "is section X still in force?" is answerable.

TEXT FIELDS
-----------
    canonical_text  exact slice of cleaned_text (display / citation; footnote refs kept)
    embedding_text  context_header + lead-in (if any) + normalized body
    bm25_text       same content as embedding_text
Normalization (embedding/BM25 only): footnote refs removed, amendment brackets unwrapped,
"* * *" omission runs dropped, dashes/quotes unified, the "N. Title.--" prefix dropped (the
header carries it). Nothing is paraphrased; canonical_text is never modified.

GRAPH FIELDS (Neo4j)
--------------------
    unit_id          stable, human-readable key: "family-courts-act-1964:s:9:ss:1-1b"
    act_key          "family-courts-act-1964" (same key is produced for citations of this Act
                     in OTHER statutes, so CITES edges join without a lookup table)
    references       {"sections": [...], "external_acts": [{name, year, act_key, sections}],
                      "schedules": [...]}
    defined_terms    terms defined in the chunk (Term)-[:DEFINED_IN]->(Section)
    amendments       [{footnote_id, action, instrument{name, year, act_key}, text}]
    document-level   "act" (number, date, long title), "outline", "glossary"

PIPELINE ISOLATION
------------------
No import of the case-law chunker / validator, src.ingestion, src.classification, or the
Constitution pipeline. Self-contained validation (no statute_chunk_validate.py needed).
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

INPUT_DIR = Path("var/rag/cleaned_statutes")
OUTPUT_DIR = Path("var/rag/statutes_chunked")

EMBED_TOKEN_LIMIT = 512        # hard limit of the embedding model (checked on embedding_text)
MAX_CHUNK_TOKENS = 400         # ceiling for a chunk body; leaves room for header + lead-in
MIN_CHUNK_TOKENS = 40          # smaller pieces are merged with a neighbour
LEAD_IN_MAX_TOKENS = 60

INHERITABLE_METADATA_FIELDS = ("document_type", "domain", "domain_source", "source_file")

try:                                                     # deterministic token COUNTING only
    import tiktoken
    _ENCODING = tiktoken.get_encoding("cl100k_base")

    def count_tokens(text: str) -> int:
        return len(_ENCODING.encode(text))
except Exception:      # pragma: no cover - tiktoken missing or its vocab file can't be downloaded
    _ENCODING = None

    def count_tokens(text: str) -> int:
        # calibrated against cl100k_base on legal text: never under-counts
        return int(len(re.findall(r"\w+|[^\w\s]", text)) * 1.30) + 1


class StatuteChunkingError(ValueError):
    """Raised when the structure cannot be determined safely (e.g. no section found)."""


# =====================================================================================
# text helpers
# =====================================================================================
_REF = r"\[\^\d+\.\d+\]"
_REF_RE = re.compile(_REF)
_LEAD = rf"(?:{_REF}|\[|[ \t])*"                               # refs / amendment brackets / spaces

_SMALL_WORDS = {"a", "an", "and", "as", "at", "by", "for", "in", "of", "on", "or", "the", "to",
                "with", "under", "from", "into", "upon", "etc"}


def strip_refs(text: str) -> str:
    """'[^5.3][Pakistan]' -> 'Pakistan' (innermost first, so nested amendments unwrap);
    remaining bare refs are removed."""
    prev = None
    while prev != text:
        prev = text
        text = re.sub(rf"{_REF}\[([^\[\]]*)\]", r"\1", text)
    return _REF_RE.sub("", text)


def _drop_unbalanced_brackets(text: str) -> str:
    """An amendment bracket can open in one section and close in the next ('[^10.1][17A. ...
    17B. ...]'); after slicing, the orphan '[' / ']' would be noise in the embedding."""
    out, stack = list(text), []
    for i, ch in enumerate(text):
        if ch == "[":
            stack.append(i)
        elif ch == "]":
            if stack:
                stack.pop()
            else:
                out[i] = ""
    for i in stack:
        out[i] = ""
    return "".join(out)


def normalize(text: str) -> str:
    text = strip_refs(text)
    text = re.sub(r"\[[ \t\n]*\*(?:[ \t\n]*\*)*[ \t\n]*\]", " ", text)   # "[* * *]" omissions
    text = re.sub(r"\*(?:[ \t\n]*\*)+", " ", text)                     # "* * *" runs (never eats the line end)
    text = re.sub(r"(?<=\w)[ \t]*\*(?=[ \t]*[.,;:)\]]|[ \t]*$)", "", text, flags=re.M)
    text = re.sub(r"(?m)^[ \t]*\*[ \t]*$", "", text)
    text = re.sub(r"(?<=[ \t])\*(?=[ \t])", "", text)                     # lone omission star mid-line
    text = re.sub(r"(?m)^[ \t]*\d{1,2}[ \t]*$", "", text)                 # unlinked footnote digit alone on a line
    text = _drop_unbalanced_brackets(text)
    text = re.sub(r"\[\s*\]", "", text)
    text = text.replace("⸺", "—").replace("―", "—").replace("–", "-").replace("—", "-")
    text = re.sub(r"_{2,}", "-", text)
    text = text.replace("“", '"').replace("”", '"').replace("‘", "'").replace("’", "'")
    text = re.sub(r"[ \t]{2,}", " ", text)
    text = re.sub(r"[ \t]+\n|\n[ \t]+", "\n", text)
    text = re.sub(r"\n{2,}", "\n", text)
    return text.strip()


def smart_title(text: str) -> str:
    """'APPOINTMENT AND DECLARATION OF GUARDIANS' -> 'Appointment and Declaration of Guardians'."""
    words = []
    for i, w in enumerate(text.split()):
        lw = w.lower()
        core = lw.strip("()[],.")
        if i and core in _SMALL_WORDS:
            words.append(lw)
        else:
            words.append(re.sub(r"[a-z]", lambda m: m.group(0).upper(), lw, count=1))
    return " ".join(words)


def _truncate_tokens(text: str, limit: int) -> str:
    if count_tokens(text) <= limit:
        return text
    out: list[str] = []
    for w in text.split():
        if count_tokens(" ".join(out + [w])) > limit:
            break
        out.append(w)
    return " ".join(out) + " …"


def act_key(name: str, year: int | None) -> str:
    n = re.sub(r"^\s*the\s+", "", name.strip(), flags=re.I)
    n = re.sub(r"^(?:West\s+Pakistan|W\.\s?P\.)\s+", "", n)            # "West Pakistan Family Courts Act" == FCA
    slug = re.sub(r"[^a-z0-9]+", "-", n.lower()).strip("-")
    return f"{slug}-{year}" if year else slug


def _fn_sort_key(fid: str) -> list[int]:
    return [int(x) for x in fid.split(".") if x.isdigit()]


# =====================================================================================
# structure patterns (validated on the cleaned statutes: FCA 1964, G&W 1890, Divorce 1869,
# MFLO 1961, Dowry 1976, CMRA 1929)
# =====================================================================================
_FOOTNOTES_RE = re.compile(r"^\[FOOTNOTES\][ \t]*$", re.M)

# "7. Institution of suits.―", "[^6.5][4A. Power ...", "25-A. Transfer", "3. [^2.3][Omitted.]"
_SECTION_RE = re.compile(
    rf"^[ \t]*(?P<lead>(?:{_REF}\[|\[)*)(?P<num>\d{{1,3}}(?:-?[A-Z]{{1,2}})?)\.[ \t]*(?P<rest>[A-Z\[“\"‘(].*)$")

# heading lines between sections
_CHAPTER_RE = re.compile(r"^CHAPTER[ \t]+(?P<num>[IVXLC]+|\d+)[A-Z]?\b[.\s—–\-:]*(?P<title>.*)$", re.I)
_ROMAN_HEAD_RE = re.compile(r"^(?P<num>[IVXL]+)\s*\.\s*[—–\-]+\s*(?P<title>[A-Z][A-Z ,'&()\-]+)$")
_PART_HEAD_RE = re.compile(r"^PART[ \t]+(?P<num>[IVXLC]+|\d+)\b[.\s—–\-:]*(?P<title>.*)$")

# schedules
_SCHEDULE_HEAD_RE = re.compile(
    rf"^{_LEAD}(?:THE[ \t]+)?(?:(?:FIRST|SECOND|THIRD|FOURTH|FIFTH|SIXTH)[ \t]+)?SCHEDULE\b.*$")
_SCHED_PART_RE = re.compile(r"\bPART[ \t]+([IVXLC]+)\b", re.I)
_FORM_RE = re.compile(r"^[\[ \t]*No\.?[ \t]*[—–\-]*[ \t]*(?P<num>\d+)[ \t]*\.?[ \t]*[—–\-]*[ \t]*(?P<title>[A-Z][A-Z’'\-]+\b.*)$")

# lines inside sections
_SUBSEC_RE = re.compile(rf"^{_LEAD}\((?P<lab>\d+[A-Za-z]{{0,2}}|l[A-Z])\)")
_CLAUSE_RE = re.compile(rf"^{_LEAD}\((?P<lab>[a-z]{{1,2}}|[ivxl]+)\)")
_DEF_TERMS_RE = re.compile(r"[\"“‘']([^\"”’'\n]{1,70}?)[.]?[\"”’']")
_DEF_VERB_RE = re.compile(r"\b(?:means|includes|implies|shall,?\s+(?:mean|include|have)|has\s+the\s+meaning)\b")
_DEFINITIONS_TITLE_RE = re.compile(r"^(?:definitions?|interpretation(?:[- ]clause)?)\b", re.I)

# "[Repeal.] Rep. by ...", "[Pending suits.] Omitted by ...", "[^2.3][Omitted.]", "[Repeal]. Omitted"
_STATUS_TITLED_RE = re.compile(
    rf"^{_LEAD}\[(?P<title>[^\]\[]{{2,200}}?)\.?\]\.?[ \t]*(?P<word>Omitted|Repealed|Rep)\b\.?", re.I)
_STATUS_BARE_RE = re.compile(rf"^{_LEAD}(?P<word>Omitted|Repealed|Rep)\b\.?", re.I)

# title terminator: ". " + (dash | "(" | "[" | Capital | end) , or a dash
_TITLE_END_RE = re.compile(r"\.(?=[ \t]*(?:[—―⸺–]|_{2,}|:-|\(|\[|\d{1,2}\[|[A-Z“\"]|$))|[ \t]*[—―⸺](?=\S|[ \t])|:-")


# =====================================================================================
# references
# =====================================================================================
_ACT_KEYWORDS = r"(?:Act|Ordinance|Code|Order|Regulation|Rules)"
_CAP_WORD = r"(?:[A-Z][\w'’\-]*|[A-Z]{1,3}\.(?:[A-Z]\.)*|\([A-Z][\w .'’\-]{0,30}\))"   # no sentence-crossing "Courts."
_EXTERNAL_ACT_RE = re.compile(
    rf"(?P<name>{_CAP_WORD}(?:[ \t]+(?:(?:and|of|the|for|on|in|to)[ \t]+)*{_CAP_WORD})*?[ \t]+{_ACT_KEYWORDS})"
    rf"(?:[ \t]*,?[ \t]*(?P<year>1[6-9]\d\d|20\d\d)\b)?"
    rf"(?:[ \t]*\((?:[^()]{{0,30}}?)\b(?:of|OF)[ \t]+(?P<year2>1[6-9]\d\d|20\d\d)\.?\))?")
_SPECIAL_ACTS = [  # instruments whose name does not END with Act/Ordinance/Code/...
    (re.compile(r"Qanun-e-Shahadat(?:[ \t]+Order)?,?[ \t]*(?P<year>\d{4})?"), "Qanun-e-Shahadat Order"),
    (re.compile(r"Code[ \t]+of[ \t]+Civil[ \t]+Procedure(?:[ \t]*,?[ \t]*(?P<year>1[89]\d\d))?"), "Code of Civil Procedure"),
    (re.compile(r"Code[ \t]+of[ \t]+Criminal[ \t]+Procedure(?:[ \t]*,?[ \t]*(?P<year>1[89]\d\d))?"), "Code of Criminal Procedure"),
]
_DEFAULT_YEARS = {"Code of Civil Procedure": 1908, "Code of Criminal Procedure": 1898, "Pakistan Penal Code": 1860,
                  "Qanun-e-Shahadat Order": 1984}
_ABBREVIATIONS = {"CPC": ("Code of Civil Procedure", 1908), "CRPC": ("Code of Criminal Procedure", 1898),
                  "PPC": ("Pakistan Penal Code", 1860)}
_ABBR_RE = re.compile(r"\b(CPC|CrPC|PPC)\b")
_LEADING_NOISE = {"provisions", "provision", "amendment", "amendments", "of", "the", "save", "subject", "to", "notwithstanding", "nothing",
                  "in", "under", "where", "if", "when", "every", "any", "and", "or", "by", "for", "said"}
_NOT_A_NAME = {"this", "that", "the", "said", "such", "any", "an", "other", "every", "no", "each", "all",
               "an act", "the act", "this act", "said act", "that act"}

_SEC_LIST = r"\d{1,3}(?:-?[A-Z]{1,2})?(?:[ \t]*\([0-9a-zA-Z]{1,4}\))*"
_SECTION_REF_RE = re.compile(
    rf"\b(?i:sections?)[ \t]+(?P<list>{_SEC_LIST}(?:[ \t]*(?:,|and|or|to)[ \t]*(?:and[ \t]+)?{_SEC_LIST})*)")
_EXTERNAL_TAIL_RE = re.compile(
    r"^[ \t]*(?:of|under)[ \t]+(?:the[ \t]+|that[ \t]+|said[ \t]+)*(?:"
    r"[A-Z][\w'’\-]*|Code|Act|Ordinance|said|that)")
_SCHEDULE_REF_RE = re.compile(
    r"\b(?P<part>Part[ \t]+[IVX1l]+)[ \t]+of[ \t]+(?:the[ \t]+)?Schedule\b|\bthe[ \t]+Schedule[ \t]+to[ \t]+this[ \t]+(?:Act|Ordinance)\b")
_EXTERNAL_BEFORE_RE = re.compile(r"(?:Act|Ordinance|Code|Procedure|Order|Shahadat),?[ \t]*(?:\d{4})?[ \t]*(?:\([^()]{0,40}\))?,?[ \t]*(?:except|and|read[ \t]+with)?[ \t]*$")


def _clean_act_name(name: str) -> str | None:
    name = re.sub(r"\s+", " ", name).strip(" ,.")
    words = name.split()
    while len(words) > 1 and words[0].lower() in _LEADING_NOISE:
        words.pop(0)
    name = " ".join(words)
    # drop leading sentence words glued by the regex: keep from the last word that starts a title run
    if name.lower() in _NOT_A_NAME or name.split()[0].lower() in {"this", "that", "said", "such", "any"}:
        return None
    if len(name.split()) < 2 or not any(len(w.strip(".()")) >= 3 for w in name.split()[:-1]):
        return None                                           # "P. Act" (from "W.P. Act XXXV")
    if name.startswith("Code of") and not re.match(r"Code of (?:Civil|Criminal) Procedure$", name):
        return None
    return name


def extract_external_acts(text: str, self_key: str | None = None) -> list[dict]:
    plain = normalize(text)
    found: dict[str, dict] = {}

    def add(name: str, year: int | None, start: int, src: str):
        key = act_key(name, year)
        if self_key and (key == self_key or (year is None and self_key.startswith(act_key(name, None) + "-")
                                             and re.fullmatch(r"\d{4}", self_key[len(act_key(name, None)) + 1:]))):
            return                                            # the Act citing itself
        entry = found.setdefault(key, {"name": name, "year": year, "act_key": key, "sections": [],
                                       "source_text": src})
        # "section 115 of the Code of Civil Procedure" -> attach section numbers to the Act
        before = plain[max(0, start - 80):start]
        m = re.search(rf"\b(?:[Ss]ections?|[Ss]s?\.)[ \t]+({_SEC_LIST}(?:[ \t]*(?:,|and|to)[ \t]*{_SEC_LIST})*)"
                      rf"[ \t]+of[ \t]+(?:the[ \t]+)?$", before)
        if m:
            toks = re.findall(r"\d{1,3}(?:-?[A-Z]{1,2})?|\bto\b", re.sub(r"\([^)]*\)", "", m.group(1)))
            secs: list[str] = []
            for i, t in enumerate(toks):
                if t == "to" and secs and i + 1 < len(toks) and secs[-1].isdigit() and toks[i + 1].isdigit():
                    secs += [str(n) for n in range(int(secs[-1]) + 1, int(toks[i + 1]))]
                elif t != "to":
                    secs.append(t)
            for s in secs:
                if s not in entry["sections"]:
                    entry["sections"].append(s)

    for m in _EXTERNAL_ACT_RE.finditer(plain):
        name = _clean_act_name(m.group("name"))
        if not name or re.match(r"Code of (?:Civil|Criminal) Procedure", name):
            continue
        year = m.group("year") or m.group("year2")
        year = int(year) if year else _DEFAULT_YEARS.get(name)
        if year is None and not re.search(r"\b(?:Act|Ordinance|Code)$", name):
            continue                                          # "Holy Orders", "the said Rules"
        add(name, year, m.start() + m.group(0).find(name.split()[0]), m.group(0).strip())
    for rx, name in _SPECIAL_ACTS:
        for m in rx.finditer(plain):
            y = m.group("year")
            add(name, int(y) if y else _DEFAULT_YEARS.get(name), m.start(), m.group(0))
    for m in _ABBR_RE.finditer(plain):
        name, year = _ABBREVIATIONS[m.group(1).upper()]
        add(name, year, m.start(), m.group(0))
    # an entry without a year duplicates a dated one with the same name -> merge
    out = list(found.values())
    dated = {e["name"].lower(): e for e in out if e["year"]}
    merged = []
    for e in out:
        if not e["year"] and e["name"].lower() in dated:
            d = dated[e["name"].lower()]
            d["sections"] += [s for s in e["sections"] if s not in d["sections"]]
            continue
        merged.append(e)
    return merged


def extract_section_refs(text: str, known: list[str], self_no: str | None) -> list[str]:
    plain = normalize(text)
    order = {s: i for i, s in enumerate(known)}
    out: list[str] = []
    for m in _SECTION_REF_RE.finditer(plain):
        tail = plain[m.end():m.end() + 60]
        if _EXTERNAL_TAIL_RE.match(tail) and not re.match(r"^[ \t]*of[ \t]+this\b", tail):
            continue                                          # "section 25 of the General Clauses Act"
        if _EXTERNAL_BEFORE_RE.search(plain[max(0, m.start() - 60):m.start()]):
            continue                                          # "the Code of Civil Procedure, 1908, except sections 10 and 11"
        items = re.findall(r"\d{1,3}(?:-?[A-Z]{1,2})?|\bto\b", re.sub(r"\([^)]*\)", "", m.group("list")))
        prev = None
        for i, it in enumerate(items):
            if it == "to":
                nxt = items[i + 1] if i + 1 < len(items) else None
                if prev in order and nxt in order:
                    out.extend(known[order[prev] + 1:order[nxt]])
                continue
            norm = it.replace("-", "")
            cand = it if it in order else (norm if norm in order else None)
            if cand is None:   # "25A" written for "25-A"
                cand = next((k for k in known if k.replace("-", "") == norm), None)
            if cand:
                out.append(cand)
            prev = cand
    seen, res = set(), []
    for s in out:
        if s != self_no and s not in seen:
            seen.add(s)
            res.append(s)
    return res


def extract_schedule_refs(text: str) -> list[str]:
    res = []
    for m in _SCHEDULE_REF_RE.finditer(normalize(text)):
        part = m.group("part")
        lab = "Schedule" + (", Part " + part.split()[-1].upper().replace("L", "I").replace("1", "I") if part else "")
        if lab not in res:
            res.append(lab)
    return res


def defined_terms_in(line: str) -> list[str]:
    s = normalize(line)
    v = _DEF_VERB_RE.search(s)
    if not v or v.start() > 160:
        return []
    terms, seen = [], set()
    for t in _DEF_TERMS_RE.findall(s[:v.start()]):
        t = t.strip(" .,")
        if t and t.lower() not in seen:
            seen.add(t.lower())
            terms.append(t)
    return terms


_ACTION_RE = re.compile(r"\b(Subs|Ins|Added|Add|Omitted|Omit|Rep|Repealed|Renumbered|Amended|Re-numbered)\b\.?", re.I)
_ACTION_MAP = {"subs": "substituted", "ins": "inserted", "added": "added", "add": "added",
               "omitted": "omitted", "omit": "omitted", "rep": "repealed", "repealed": "repealed",
               "renumbered": "renumbered", "re-numbered": "renumbered", "amended": "amended"}


def describe_amendment(fid: str, text: str) -> dict:
    m = _ACTION_RE.search(text[:80])
    action = _ACTION_MAP.get(m.group(1).lower(), "note") if m else "note"
    acts = extract_external_acts(text)
    instrument = None
    if acts and action != "note":
        a = acts[0]
        instrument = {"name": a["name"], "year": a["year"], "act_key": a["act_key"]}
    else:
        ao = re.search(r"\bA\.\s?O\.,?\s*(\d{4})", text)
        if ao and action != "note":
            instrument = {"name": "A.O.", "year": int(ao.group(1)), "act_key": f"a-o-{ao.group(1)}"}
    return {"footnote_id": fid, "action": action, "instrument": instrument, "text": text}


# =====================================================================================
# parsing
# =====================================================================================
@dataclass
class Line:
    start: int
    end: int          # exclusive, without the newline
    text: str


def _lines(text: str, start: int, end: int) -> list[Line]:
    out, pos = [], start
    for raw in text[start:end].split("\n"):
        out.append(Line(pos, pos + len(raw), raw))
        pos += len(raw) + 1
    return out


def _heading_kind(line: str) -> str | None:
    s = strip_refs(line).strip().strip("[]").strip()
    if not s or len(s) > 90:
        return None
    if _CHAPTER_RE.match(s) or _ROMAN_HEAD_RE.match(s):
        return "chapter"
    if _PART_HEAD_RE.match(s):
        return "part"
    if s[-1] in ".,;:—―⸺–-" or not s[0].isupper():
        return None
    if not any(c.isalpha() for c in s):
        return None
    words = s.split()
    if s.upper() == s and len(words) <= 12:
        return "caps"
    if len(words) <= 8 and "," not in s and all(w[0].isupper() or w.lower() in _SMALL_WORDS for w in words):
        return "sub"
    return None


def _section_key(num: str) -> tuple[int, str]:
    m = re.match(r"(\d+)-?([A-Z]*)", num)
    return int(m.group(1)), m.group(2)


def _split_title(rest: str) -> tuple[str | None, str, int]:
    """rest = text after 'N.' on the header line -> (title, status, offset of body in rest)."""
    m = _STATUS_TITLED_RE.match(rest)
    if m:
        st = "omitted" if m.group("word").lower().startswith("omit") else "repealed"
        return smart_title_keep(m.group("title")), st, m.start("word")
    m = _STATUS_BARE_RE.match(rest)
    if m:
        st = "omitted" if m.group("word").lower().startswith("omit") else "repealed"
        return None, st, 0
    m = _TITLE_END_RE.search(rest)
    if not m:
        return None, "in_force", 0
    title = normalize(rest[:m.start()]).strip(" .")
    body_at = m.end()
    if title.endswith(" etc") or title.endswith(",etc"):
        title += "."
    words = title.split()
    if (not title or len(words) > 30 or re.search(r"\bshall\b", title) or not rest[:1].strip()
            or not re.match(r"^[\[“\"‘(]*[A-Z]", normalize(rest[:m.start()]) or "x")):
        return None, "in_force", 0
    while body_at < len(rest) and rest[body_at] in " \t—―⸺–-_:":
        body_at += 1
    return title, "in_force", body_at


def smart_title_keep(t: str) -> str:
    t = normalize(t).strip(" .[]")
    return t


@dataclass
class Section:
    number: str
    title: str | None
    status: str
    start: int
    end: int
    header_line: Line
    body_offset: int                  # absolute offset where the body starts on the header line
    chapter_number: str | None = None
    chapter_title: str | None = None
    part_number: str | None = None
    part_title: str | None = None
    subheading: str | None = None
    is_definitions: bool = False


@dataclass
class ScheduleUnit:
    schedule: str
    schedule_key: str
    label: str | None                 # "Part I", "Form No. 4: Petition for ..."
    unit_key: str                     # "part:I", "form:4", "" (whole schedule)
    status: str
    start: int
    end: int


def _trim(text: str, start: int, end: int) -> tuple[int, int]:
    while start < end and text[start] in " \t\r\n":
        start += 1
    while end > start and text[end - 1] in " \t\r\n":
        end -= 1
    return start, end


def parse_statute(cleaned_text: str) -> dict:
    text = cleaned_text
    fm = _FOOTNOTES_RE.search(text)
    content_end = fm.start() if fm else len(text)
    lines = _lines(text, 0, content_end)

    # --- schedule region: first SCHEDULE heading after at least one section header
    first_sec_idx = next((i for i, l in enumerate(lines) if _SECTION_RE.match(l.text)), None)
    if first_sec_idx is None:
        raise StatuteChunkingError("No section header ('1. Short title ...') found in the document body.")
    sched_idx = next((i for i in range(first_sec_idx + 1, len(lines))
                      if _SCHEDULE_HEAD_RE.match(lines[i].text)), None)
    body_lines = lines[:sched_idx] if sched_idx is not None else lines

    # --- section headers: strictly increasing numbers (amendment notes / list items that look
    #     like "2. ..." deep inside a later section can never become a boundary)
    accepted: list[tuple[int, re.Match]] = []
    rejected: list[str] = []
    last = (0, "")
    for i, l in enumerate(body_lines):
        m = _SECTION_RE.match(l.text)
        if not m:
            continue
        key = _section_key(m.group("num"))
        if key > last and key[0] - last[0] <= 30:
            accepted.append((i, m))
            last = key
        else:
            rejected.append(l.text[:80])

    # --- heading runs directly above each section header
    heading_runs: dict[int, list[tuple[Line, str]]] = {}
    for i, _m in accepted:
        run, j = [], i - 1
        prev_sec_line = max([k for k, _ in accepted if k < i], default=-1)
        while j > prev_sec_line:
            kind = _heading_kind(body_lines[j].text)
            if not kind:
                break
            run.insert(0, (body_lines[j], kind))
            j -= 1
        if run:
            heading_runs[i] = run

    # --- build sections
    sections: list[Section] = []
    headings: list[dict] = []
    ctx = {"chapter_number": None, "chapter_title": None, "part_number": None, "part_title": None,
           "subheading": None}
    first_heading_start = None
    for n, (i, m) in enumerate(accepted):
        run = heading_runs.get(i)
        if run:
            if first_heading_start is None:
                first_heading_start = run[0][0].start
            _apply_heading_run(run, ctx)
            headings.append({"start": run[0][0].start, "end": run[-1][0].end,
                             "chapter_number": ctx["chapter_number"], "chapter_title": ctx["chapter_title"],
                             "part_number": ctx["part_number"], "part_title": ctx["part_title"],
                             "subheading": ctx["subheading"]})
        hl = body_lines[i]
        rest_off = hl.start + m.start("rest")
        title, status, body_rel = _split_title(m.group("rest"))
        # end: next section's heading run (or header), else body end
        if n + 1 < len(accepted):
            ni = accepted[n + 1][0]
            nrun = heading_runs.get(ni)
            end = nrun[0][0].start if nrun else body_lines[ni].start
        else:
            end = body_lines[-1].end if body_lines else hl.end
        s0, s1 = _trim(text, hl.start, end)
        sections.append(Section(
            number=m.group("num"), title=title, status=status, start=s0, end=s1, header_line=hl,
            body_offset=rest_off + body_rel, chapter_number=ctx["chapter_number"],
            chapter_title=ctx["chapter_title"], part_number=ctx["part_number"], part_title=ctx["part_title"],
            subheading=ctx["subheading"],
            is_definitions=bool(title and _DEFINITIONS_TITLE_RE.match(title))))

    pre_end = sections[0].start
    if first_heading_start is not None and first_heading_start < pre_end:
        pre_end = first_heading_start
    preamble = _trim(text, 0, pre_end)

    # --- schedules
    schedules: list[ScheduleUnit] = []
    if sched_idx is not None:
        schedules = _parse_schedules(text, lines[sched_idx:], content_end)

    footnotes = _trim(text, fm.start(), len(text)) if fm else None
    return {"preamble": preamble if preamble[1] > preamble[0] else None, "sections": sections,
            "headings": headings, "schedules": schedules, "footnotes": footnotes,
            "rejected_section_like_lines": rejected}


def _apply_heading_run(run: list[tuple[Line, str]], ctx: dict) -> None:
    texts = [(strip_refs(l.text).strip().strip("[]").strip(), k) for l, k in run]
    if any(k in ("chapter", "part") for _, k in texts):
        title_parts: list[str] = []
        sub_parts: list[str] = []
        mode = None
        for t, k in texts:
            if k == "part":
                pm = _PART_HEAD_RE.match(t)
                ctx.update(part_number=pm.group("num"), part_title=smart_title(pm.group("title")) or None,
                           chapter_number=None, chapter_title=None, subheading=None)
                mode = "part"
            elif k == "chapter":
                cm = _CHAPTER_RE.match(t) or _ROMAN_HEAD_RE.match(t)
                ctx["chapter_number"] = cm.group("num")
                title_parts = [cm.group("title").strip()] if cm.group("title").strip() else []
                sub_parts = []
                mode = "chapter"
            elif k == "caps" and not sub_parts:
                if mode == "part" and ctx.get("part_title") is None:
                    ctx["part_title"] = smart_title(t)
                else:
                    title_parts.append(t)
            else:
                sub_parts.append(t)
        if mode == "chapter" or title_parts:
            ctx["chapter_title"] = smart_title(" ".join(title_parts)) if title_parts else None
        ctx["subheading"] = smart_title(" ".join(sub_parts)) if sub_parts else None
    else:
        joined = " ".join(t for t, _ in texts)
        ctx["subheading"] = smart_title(joined) if joined.upper() == joined else joined


def _schedule_name(line: str) -> tuple[str, str]:
    s = normalize(line)
    m = re.match(r"^((?:THE\s+)?(?:(?:FIRST|SECOND|THIRD|FOURTH|FIFTH|SIXTH)\s+)?SCHEDULE(?:\s+OF\s+[A-Z ]+)?)", s)
    name = smart_title(m.group(1)) if m else "Schedule"
    name = re.sub(r"^The\s+", "", name)
    key = re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")
    return name, key


def _parse_schedules(text: str, lines: list[Line], content_end: int) -> list[ScheduleUnit]:
    heads = [i for i, l in enumerate(lines) if _SCHEDULE_HEAD_RE.match(l.text)]
    units: list[ScheduleUnit] = []
    for hi, h in enumerate(heads):
        stop = heads[hi + 1] if hi + 1 < len(heads) else len(lines)
        block = lines[h:stop]
        name, key = _schedule_name(block[0].text)
        if len(heads) > 1 and sum(1 for x in heads if _schedule_name(lines[x].text)[1] == key) > 1:
            key = f"{key}-{hi + 1}"
        status = "in_force"
        head_plain = normalize(block[0].text)
        if re.search(r"\bRep\.\s*by\b|\bRepealed\b", head_plain):
            status = "repealed"
        elif re.search(r"\bOmitted\b", head_plain):
            status = "omitted"
        # sub-unit boundaries: Forms ("No. 4.—PETITION ...") or Parts (short line with PART n)
        cuts: list[tuple[int, str, str]] = []        # (line index in block, label, unit_key)
        for j, l in enumerate(block):
            fmatch = _FORM_RE.match(strip_refs(l.text))
            if fmatch:
                title_lines = [fmatch.group("title").strip()]
                k = j + 1
                while (k < len(block) and title_lines[-1] and not title_lines[-1].rstrip().endswith(".")
                       and normalize(block[k].text).upper() == normalize(block[k].text)
                       and re.search(r"[A-Z]{3}", block[k].text)):
                    title_lines.append(normalize(block[k].text))
                    k += 1
                title = smart_title(normalize(" ".join(title_lines)).strip(" ."))
                title = re.sub(r"\s*\(See [Ss]ections? [^)]*\)\s*$", "", title)
                cuts.append((j, f"Form No. {fmatch.group('num')}: {title}" if title else
                             f"Form No. {fmatch.group('num')}", f"form:{fmatch.group('num')}"))
                continue
            plain = normalize(l.text)
            pm = _SCHED_PART_RE.search(plain)
            if pm and len(plain) <= 60 and not re.search(r"\bof\s+the\s+Schedule\b", plain, re.I):
                roman = pm.group(1).upper()
                cuts.append((j, f"Part {roman}", f"part:{roman}"))
        if not cuts:
            s0, s1 = _trim(text, block[0].start, block[-1].end)
            units.append(ScheduleUnit(name, key, None, "", status, s0, s1))
            continue
        cuts[0] = (0, cuts[0][1], cuts[0][2])          # the heading line belongs to the first unit
        for ci, (j, label, ukey) in enumerate(cuts):
            j_end = cuts[ci + 1][0] if ci + 1 < len(cuts) else len(block)
            s0, s1 = _trim(text, block[j].start, block[j_end - 1].end)
            units.append(ScheduleUnit(name, key, label, ukey, status, s0, s1))
    return units


# =====================================================================================
# chunk record
# =====================================================================================
@dataclass(frozen=True)
class StatuteChunk:
    doc_id: str
    chunk_id: str
    parent_id: str
    unit_id: str
    unit_type: str          # preamble | section | section_part | definitions_part | schedule | schedule_part | footnotes
    is_parent: bool
    retrievable: bool       # True -> embed + BM25-index
    act_key: str
    act_name: str
    year: int | None
    part_number: str | None
    part_title: str | None
    chapter_number: str | None
    chapter_title: str | None
    subheading: str | None
    section_number: str | None
    section_title: str | None
    section_status: str | None      # in_force | omitted | repealed
    subsections: list
    sub_range: str | None
    schedule: str | None
    schedule_unit: str | None
    token_count: int
    source_start: int
    source_end: int
    canonical_text: str
    context_header: str
    lead_in: str | None
    embedding_text: str
    bm25_text: str
    footnote_ids: list = field(default_factory=list)
    amendments: list = field(default_factory=list)
    references: dict = field(default_factory=dict)
    defined_terms: list = field(default_factory=list)
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return dict(self.__dict__)

    @classmethod
    def from_dict(cls, d: dict) -> "StatuteChunk":
        return cls(**d)


# =====================================================================================
# builder
# =====================================================================================
@dataclass
class _Unit:
    start: int
    end: int
    label: str | None           # sub-section / clause label
    terms: list
    tokens: int


class _Builder:
    def __init__(self, doc: dict):
        self.doc = doc
        self.doc_id = doc["doc_id"]
        self.text = doc["cleaned_text"]
        md = doc.get("metadata", {})
        self.year = md.get("year")
        self.act_name = self._display_name(md)
        self.act_key = act_key(self.act_name.rsplit(",", 1)[0] if self.year else self.act_name, self.year)
        self.inherit = {k: md.get(k) for k in INHERITABLE_METADATA_FIELDS}
        self.footnotes = {f["id"]: f for f in doc.get("footnotes", [])}
        self.structure = parse_statute(self.text)
        self.known_sections = [s.number for s in self.structure["sections"]]
        self.chunks: list[StatuteChunk] = []
        self.ids: set[str] = set()

    def _display_name(self, md: dict) -> str:
        raw = md.get("law_title") or md.get("law_name") or "Statute"
        raw = re.sub(r",?\s*\d{4}\s*$", "", raw.strip())
        name = smart_title(raw) if raw.upper() == raw else raw
        name = re.sub(r"^The\s+", "", name)
        name = re.sub(r"\(([a-z])", lambda m: "(" + m.group(1).upper(), name)   # "(restriction)" -> "(Restriction)"
        return f"{name}, {md['year']}" if md.get("year") else name

    # ---- helpers ----
    def _uid(self, base: str) -> str:
        cid, k = base, 2
        while cid in self.ids:
            cid, k = f"{base}_{k}", k + 1
        self.ids.add(cid)
        return cid

    def _header(self, ctx: dict, extra: str | None = None) -> str:
        seg = [self.act_name]
        if ctx.get("kind") == "preamble":
            seg.append("Preamble")
        elif ctx.get("kind") == "footnotes":
            seg.append("Footnotes (amendment history)")
        elif ctx.get("schedule"):
            seg.append(ctx["schedule"] + (f" [{ctx['status']}]" if ctx.get("status") not in (None, "in_force") else ""))
            if ctx.get("schedule_unit"):
                seg.append(ctx["schedule_unit"])
        else:
            if ctx.get("part_number"):
                seg.append(f"Part {ctx['part_number']}" + (f": {ctx['part_title']}" if ctx.get("part_title") else ""))
            if ctx.get("chapter_number"):
                seg.append(f"Chapter {ctx['chapter_number']}" +
                           (f": {ctx['chapter_title']}" if ctx.get("chapter_title") else ""))
            if ctx.get("subheading"):
                seg.append(ctx["subheading"])
            sec = f"Section {ctx['section_number']}" + (f": {ctx['section_title']}" if ctx.get("section_title") else "")
            if ctx.get("section_status") not in (None, "in_force"):
                sec += f" [{ctx['section_status']}]"
            seg.append(sec)
        if extra:
            seg.append(extra)
        return "[" + " | ".join(seg) + "]"

    def _body_text(self, start: int, end: int, ctx: dict) -> str:
        """canonical slice -> body for embedding (drops the 'N. Title.--' prefix of a section)."""
        if ctx.get("body_offset") is not None and start <= ctx["body_offset"] <= end and start == ctx.get("section_start"):
            body = self.text[ctx["body_offset"]:end]       # orphan ']' of a dropped '[^n][' is removed by normalize
        else:
            body = self.text[start:end]
        return normalize(body)

    def _tokens(self, start: int, end: int, ctx: dict) -> int:
        return count_tokens(self._body_text(start, end, ctx))

    def _make(self, ctx: dict, *, chunk_id: str, parent_id: str, unit_id: str, unit_type: str,
              is_parent: bool, retrievable: bool, start: int, end: int, subsections: list | None = None,
              sub_range: str | None = None, lead_in: str | None = None) -> StatuteChunk:
        canonical = self.text[start:end]
        header = self._header(ctx, sub_range)
        body = self._body_text(start, end, ctx)
        if not body and ctx.get("section_title"):
            body = ctx["section_title"]
        lead = normalize(lead_in) if lead_in else None
        emb = header + "\n" + (f"{lead}{'' if lead.endswith('…') else ' …'}\n" if lead else "") + body
        fids = sorted(set(re.findall(r"\[\^(\d+\.\d+)\]", canonical)), key=_fn_sort_key)
        is_fn_block = unit_type == "footnotes"
        amendments = [] if is_fn_block else [
            describe_amendment(f, self.footnotes[f]["text"]) for f in fids if f in self.footnotes]
        refs = {"sections": [], "external_acts": [], "schedules": []} if is_fn_block else {
            "sections": extract_section_refs(canonical, self.known_sections, ctx.get("section_number")),
            "external_acts": extract_external_acts(canonical, self.act_key),
            "schedules": extract_schedule_refs(canonical) if not ctx.get("schedule") else [],
        }
        terms = []
        if ctx.get("is_definitions") and not is_fn_block:
            for l in canonical.split("\n"):
                terms += [t for t in defined_terms_in(l) if t not in terms]
        return StatuteChunk(
            doc_id=self.doc_id, chunk_id=chunk_id, parent_id=parent_id, unit_id=unit_id, unit_type=unit_type,
            is_parent=is_parent, retrievable=retrievable, act_key=self.act_key, act_name=self.act_name,
            year=self.year, part_number=ctx.get("part_number"), part_title=ctx.get("part_title"),
            chapter_number=ctx.get("chapter_number"), chapter_title=ctx.get("chapter_title"),
            subheading=ctx.get("subheading"), section_number=ctx.get("section_number"),
            section_title=ctx.get("section_title"), section_status=ctx.get("section_status"),
            subsections=subsections or [], sub_range=sub_range, schedule=ctx.get("schedule"),
            schedule_unit=ctx.get("schedule_unit"), token_count=count_tokens(emb),
            source_start=start, source_end=end, canonical_text=canonical, context_header=header,
            lead_in=lead, embedding_text=emb, bm25_text=emb, footnote_ids=fids, amendments=amendments,
            references=refs, defined_terms=terms, metadata=dict(self.inherit))

    # ---- splitting ----
    def _units(self, start: int, end: int, ctx: dict, level: str) -> list[_Unit]:
        """Split [start,end) into units at sub-section lines (level='sub'), clause lines
        (level='clause') or plain lines (level='line'). The first unit keeps everything before
        the first marker (e.g. the section header line)."""
        units: list[_Unit] = []
        for l in _lines(self.text, start, end):
            if not l.text.strip():
                if units:
                    units[-1].end = l.end
                continue
            lab = None
            if level == "sub":
                m = _SUBSEC_RE.match(l.text)
                lab = m.group("lab").replace("l", "1", 1) if m and m.group("lab").startswith("l") else (m.group("lab") if m else None)
                if ctx.get("is_definitions") and not lab and _CLAUSE_RE.match(l.text) and defined_terms_in(l.text):
                    lab = _CLAUSE_RE.match(l.text).group("lab")          # "(a) “child” means ..."
            elif level == "para":
                lab = "para" if re.match(rf"^{_LEAD}(?:\d{{1,2}}\.[ \t]|No\.|Provided\b|Explanation\b|\(Signed\)|Form of|Wherefore|Your petitioner)", l.text) else None
            elif level == "clause":
                m = _CLAUSE_RE.match(l.text) or _SUBSEC_RE.match(l.text)
                lab = m.group("lab") if m else None
            else:
                lab = "line"
            if lab is None and units:
                units[-1].end = l.end
                continue
            units.append(_Unit(l.start, l.end, None if lab in ("line", "para") else lab, [], 0))
        if not units:
            return []
        units[0].start = start
        # first line of a section: "7. Title.― (1) Every suit ..." -> label (1)
        if level == "sub" and units[0].label is None and ctx.get("body_offset") is not None and units[0].start == ctx.get("section_start"):
            m = re.match(rf"^{_LEAD}\((\d+[A-Za-z]{{0,2}})\)", self.text[ctx["body_offset"]:units[0].end])
            if m:
                units[0].label = m.group(1)
        for u in units:
            u.end = _trim(self.text, u.start, u.end)[1]
            u.tokens = self._tokens(u.start, u.end, ctx)
            if ctx.get("is_definitions"):
                u.terms = [t for l in self.text[u.start:u.end].split("\n") for t in defined_terms_in(l)]
        return units

    @staticmethod
    def _partition(tokens: list[int], max_tok: int) -> list[list[int]]:
        """Contiguous partition into the fewest groups with every group <= max_tok, balanced
        (minimises the largest group) -- avoids a 390-token chunk followed by a 15-token one."""
        n = len(tokens)
        if n == 0:
            return []
        # minimal number of groups (greedy is optimal for the count)
        k, cur = 1, 0
        for t in tokens:
            if cur and cur + t > max_tok:
                k, cur = k + 1, 0
            cur += t
        pre = [0]
        for t in tokens:
            pre.append(pre[-1] + t)
        INF = float("inf")
        best = [[INF] * (n + 1) for _ in range(k + 1)]
        cut = [[0] * (n + 1) for _ in range(k + 1)]
        best[0][0] = 0
        for g in range(1, k + 1):
            for i in range(1, n + 1):
                for j in range(g - 1, i):
                    if best[g - 1][j] == INF:
                        continue
                    v = max(best[g - 1][j], pre[i] - pre[j])
                    if v < best[g][i]:
                        best[g][i], cut[g][i] = v, j
        groups, i = [], n
        for g in range(k, 0, -1):
            j = cut[g][i]
            groups.insert(0, list(range(j, i)))
            i = j
        return [g for g in groups if g]

    def _merge_tiny(self, units: list[_Unit]) -> list[list[_Unit]]:
        """Each item is a 'block' of units that must stay together; tiny units join a neighbour."""
        blocks: list[list[_Unit]] = []
        for u in units:
            if blocks and (u.tokens < MIN_CHUNK_TOKENS or sum(x.tokens for x in blocks[-1]) < MIN_CHUNK_TOKENS) \
                    and sum(x.tokens for x in blocks[-1]) + u.tokens <= MAX_CHUNK_TOKENS:
                blocks[-1].append(u)
            else:
                blocks.append([u])
        return blocks

    @staticmethod
    def _range_label(labels: list[str], noun: str) -> str | None:
        labels = [l for l in labels if l]
        if not labels:
            return None
        if len(labels) == 1:
            return f"{noun} ({labels[0]})"
        return f"{noun}s ({labels[0]})–({labels[-1]})"

    # ---- sections ----
    def section(self, s: Section):
        ctx = {"section_number": s.number, "section_title": s.title, "section_status": s.status,
               "chapter_number": s.chapter_number, "chapter_title": s.chapter_title,
               "part_number": s.part_number, "part_title": s.part_title, "subheading": s.subheading,
               "is_definitions": s.is_definitions, "body_offset": s.body_offset, "section_start": s.start}
        sid = f"{self.doc_id}:s{s.number}"
        unit = f"{self.act_key}:s:{s.number}"
        total = self._tokens(s.start, s.end, ctx)
        if total <= MAX_CHUNK_TOKENS:
            labs = [u.label for u in self._units(s.start, s.end, ctx, "sub") if u.label]
            self.chunks.append(self._make(ctx, chunk_id=self._uid(sid), parent_id=sid, unit_id=unit,
                                          unit_type="section", is_parent=True, retrievable=True,
                                          start=s.start, end=s.end, subsections=labs))
            return
        self.chunks.append(self._make(ctx, chunk_id=self._uid(sid), parent_id=sid, unit_id=unit,
                                      unit_type="section", is_parent=True, retrievable=False,
                                      start=s.start, end=s.end,
                                      subsections=[u.label for u in self._units(s.start, s.end, ctx, "sub") if u.label]))
        units = self._units(s.start, s.end, ctx, "sub")
        if len(units) <= 1:
            units = self._units(s.start, s.end, ctx, "clause")
            if len(units) <= 1:
                units = self._units(s.start, s.end, ctx, "line")
        noun = "Sub-section" if any(_SUBSEC_RE.match(self.text[u.start:u.end]) or (u is units[0] and u.label and u.label[0].isdigit()) for u in units) else "Clause"

        # definitions: the opening words become the lead-in of every later piece
        defs_lead = None
        if s.is_definitions:
            first = self._body_text(units[0].start, units[0].end, ctx).split("\n")[0]
            defs_lead = _truncate_tokens(first, LEAD_IN_MAX_TOKENS)
        lead_tok = count_tokens(defs_lead) if defs_lead else 0

        pieces: list[tuple[int, int, list[str], list[str], str | None]] = []   # start,end,labels,terms,lead_in
        normal: list[_Unit] = []

        def flush_normal():
            if not normal:
                return
            blocks = self._merge_tiny(normal)
            groups = self._partition([sum(u.tokens for u in b) for b in blocks], MAX_CHUNK_TOKENS - lead_tok)
            for g in groups:
                us = [u for bi in g for u in blocks[bi]]
                pieces.append((us[0].start, us[-1].end, [u.label for u in us if u.label],
                               [t for u in us for t in u.terms], defs_lead))
            normal.clear()

        for u in units:
            if u.tokens <= MAX_CHUNK_TOKENS - lead_tok:
                normal.append(u)
                continue
            flush_normal()
            # oversized sub-section: split at its clause lines (then plain lines), carry its opening
            subs = self._units(u.start, u.end, {**ctx, "is_definitions": False}, "clause")
            if len(subs) <= 1 or max(x.tokens for x in subs) > MAX_CHUNK_TOKENS - LEAD_IN_MAX_TOKENS:
                subs = self._units(u.start, u.end, {**ctx, "is_definitions": False}, "line")
            opening = _truncate_tokens(self._body_text(u.start, subs[0].end, ctx).split("\n")[0], LEAD_IN_MAX_TOKENS)
            groups = self._partition([x.tokens for x in subs], MAX_CHUNK_TOKENS - count_tokens(opening))
            for gi, g in enumerate(groups):
                us = [subs[i] for i in g]
                clause_labs = [x.label for x in us if x.label and not x.label[0].isdigit()]
                pieces.append((us[0].start, us[-1].end, [u.label] if u.label else [], clause_labs,
                               None if gi == 0 else opening))
        flush_normal()

        for pi, (p0, p1, labs, terms_or_clauses, lead) in enumerate(pieces):
            if s.is_definitions and terms_or_clauses and all(isinstance(t, str) for t in terms_or_clauses) \
                    and any(defined_terms_in(l) for l in self.text[p0:p1].split("\n")):
                terms = [t for l in self.text[p0:p1].split("\n") for t in defined_terms_in(l)]
                sub_range = (f'Definitions "{terms[0]}"' if len(terms) == 1 else
                             f'Definitions "{terms[0]}" to "{terms[-1]}"') if terms else None
                utype = "definitions_part"
            else:
                sub_range = self._range_label(labs, noun)
                clause_labs = [c for c in terms_or_clauses if c and not s.is_definitions]
                if clause_labs:
                    cr = f"clause ({clause_labs[0]})" if len(clause_labs) == 1 else f"clauses ({clause_labs[0]})–({clause_labs[-1]})"
                    sub_range = f"{sub_range}, {cr}" if sub_range else cr.capitalize()
                utype = "section_part"
            same = [i for i, p in enumerate(pieces) if p[2] == labs]
            part = f":part:{same.index(pi) + 1}" if labs and len(same) > 1 else ""
            suffix = "-".join(labs) if labs else f"p{pi + 1}"
            self.chunks.append(self._make(
                ctx, chunk_id=self._uid(f"{sid}_{'ss' + suffix if labs else suffix}{part.replace(':part:', '_p')}"),
                parent_id=sid,
                unit_id=f"{unit}:{'ss:' + suffix if labs else 'part:' + str(pi + 1)}{part}",
                unit_type=utype, is_parent=False, retrievable=True, start=p0, end=p1,
                subsections=labs, sub_range=sub_range, lead_in=lead if p0 != s.start else None))

    # ---- preamble / schedules / footnotes ----
    def block(self, start: int, end: int, ctx: dict, base_id: str, unit_id: str, unit_type: str,
              retrievable: bool = True):
        if self._tokens(start, end, ctx) <= MAX_CHUNK_TOKENS or not retrievable:
            self.chunks.append(self._make(ctx, chunk_id=self._uid(base_id), parent_id=base_id, unit_id=unit_id,
                                          unit_type=unit_type, is_parent=True, retrievable=retrievable,
                                          start=start, end=end))
            return
        self.chunks.append(self._make(ctx, chunk_id=self._uid(base_id), parent_id=base_id, unit_id=unit_id,
                                      unit_type=unit_type, is_parent=True, retrievable=False, start=start, end=end))
        units = self._units(start, end, ctx, "para")
        if len(units) <= 1 or max(u.tokens for u in units) > MAX_CHUNK_TOKENS:
            units = self._units(start, end, ctx, "line")
        groups = self._partition([u.tokens for u in units], MAX_CHUNK_TOKENS)
        for gi, g in enumerate(groups):
            us = [units[i] for i in g]
            self.chunks.append(self._make(
                ctx, chunk_id=self._uid(f"{base_id}_p{gi + 1}"), parent_id=base_id,
                unit_id=f"{unit_id}:part:{gi + 1}", unit_type=f"{unit_type}_part", is_parent=False,
                retrievable=True, start=us[0].start, end=us[-1].end,
                sub_range=f"part {gi + 1} of {len(groups)}"))

    def build(self) -> list[StatuteChunk]:
        st = self.structure
        if st["preamble"]:
            self.block(*st["preamble"], {"kind": "preamble"}, f"{self.doc_id}:preamble",
                       f"{self.act_key}:preamble", "preamble")
        for s in st["sections"]:
            self.section(s)
        for u in st["schedules"]:
            ctx = {"schedule": u.schedule, "schedule_unit": u.label, "status": u.status}
            base = f"{self.doc_id}:{u.schedule_key}" + (f":{u.unit_key.replace(':', '')}" if u.unit_key else "")
            unit = f"{self.act_key}:{u.schedule_key}" + (f":{u.unit_key}" if u.unit_key else "")
            self.block(u.start, u.end, ctx, base, unit, "schedule")
        if st["footnotes"]:
            self.block(*st["footnotes"], {"kind": "footnotes"}, f"{self.doc_id}:footnotes",
                       f"{self.act_key}:footnotes", "footnotes", retrievable=False)
        return self.chunks

    # ---- document-level info for the graph ----
    def act_info(self) -> dict:
        pre = self.text[slice(*self.structure["preamble"])] if self.structure["preamble"] else ""
        plain = normalize(pre)
        num = re.search(r"\b(?:W\.\s?P\.\s+)?(?:Act|Ordinance)\s+(?:No\.?\s*)?([IVXLC]+|\d+)\s+of\s+(\d{4})", plain, re.I)
        date = re.search(r"\[(\d{1,2}(?:st|nd|rd|th|ST|ND|RD|TH)\s+[A-Za-z]+,?\s*\d{4})\]", plain)
        long_title = re.search(r"\b(An?\s+(?:Act|Ordinance)\s+to\s+[^.\n]+(?:\n(?-i:[a-z])[^.\n]+)*)", plain, re.I)
        return {"act_key": self.act_key, "name": self.act_name, "year": self.year,
                "act_number": f"{num.group(1)} of {num.group(2)}" if num else None,
                "date": date.group(1) if date else None,
                "long_title": re.sub(r"\s+", " ", long_title.group(1)).strip() if long_title else None}

    def outline(self) -> list[dict]:
        out = []
        for c in self.chunks:
            if c.is_parent and c.unit_type != "footnotes":
                out.append({"unit_id": c.unit_id, "chunk_id": c.chunk_id, "unit_type": c.unit_type,
                            "chapter": c.chapter_number, "chapter_title": c.chapter_title,
                            "section": c.section_number, "title": c.section_title or c.schedule_unit or c.schedule,
                            "status": c.section_status})
        return out

    def glossary(self) -> list[dict]:
        out, seen = [], set()
        for c in self.chunks:
            if c.is_parent and c.defined_terms:
                for t in c.defined_terms:
                    if t.lower() not in seen:
                        seen.add(t.lower())
                        out.append({"term": t, "section": c.section_number, "unit_id": c.unit_id,
                                    "chunk_id": c.chunk_id})
        return out


# =====================================================================================
# validation
# =====================================================================================
@dataclass
class ChunkValidationResult:
    ok: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    diagnostics: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {"ok": self.ok, "errors": self.errors, "warnings": self.warnings, "diagnostics": self.diagnostics}


def validate_chunks(chunks: list[StatuteChunk], doc: dict, structure: dict | None = None) -> ChunkValidationResult:
    errors: list[str] = []
    warnings: list[str] = []
    text = doc["cleaned_text"]
    structure = structure or parse_statute(text)

    ids = [c.chunk_id for c in chunks]
    dup = sorted({i for i in ids if ids.count(i) > 1})
    if dup:
        errors.append(f"duplicate chunk_id(s): {dup[:10]}")
    units = [c.unit_id for c in chunks]
    dupu = sorted({u for u in units if units.count(u) > 1})
    if dupu:
        errors.append(f"duplicate unit_id(s): {dupu[:10]}")

    parents = {c.parent_id: c for c in chunks if c.is_parent}
    known = {s.number for s in structure["sections"]}
    for c in chunks:
        if not (0 <= c.source_start < c.source_end <= len(text)):
            errors.append(f"{c.chunk_id}: bad offsets [{c.source_start},{c.source_end})")
            continue
        if text[c.source_start:c.source_end] != c.canonical_text:
            errors.append(f"{c.chunk_id}: canonical_text != cleaned_text[source_start:source_end]")
        if not c.embedding_text[len(c.context_header):].strip():
            errors.append(f"{c.chunk_id}: empty body")
        if "[^" in c.context_header or "[^" in c.embedding_text:
            errors.append(f"{c.chunk_id}: footnote refs left in context_header/embedding_text")
        if c.retrievable and c.token_count > EMBED_TOKEN_LIMIT:
            errors.append(f"{c.chunk_id}: {c.token_count} tokens > embedding limit {EMBED_TOKEN_LIMIT}")
        if not c.is_parent and c.parent_id not in parents:
            errors.append(f"{c.chunk_id}: parent {c.parent_id} missing")
        if c.unit_type == "footnotes" and c.retrievable:
            errors.append(f"{c.chunk_id}: footnotes block must not be retrievable")
        if "[FOOTNOTES]" in c.canonical_text and c.unit_type != "footnotes":
            errors.append(f"{c.chunk_id}: footnotes block leaked into a content chunk")
        if c.unit_type in ("section", "section_part", "definitions_part"):
            for line in c.canonical_text.split("\n")[1:]:
                m = _SECTION_RE.match(line)
                if m and m.group("num") in known and m.group("num") != c.section_number:
                    errors.append(f"{c.chunk_id}: contains the header of section {m.group('num')} (boundary leak)")
        for f in c.footnote_ids:
            if c.unit_type != "footnotes" and f not in {x['id'] for x in doc.get('footnotes', [])}:
                warnings.append(f"{c.chunk_id}: cites footnote {f} that is not in the footnotes list")

    # children tile their parent exactly (ignoring whitespace)
    kids: dict[str, list[StatuteChunk]] = {}
    for c in chunks:
        if not c.is_parent:
            kids.setdefault(c.parent_id, []).append(c)
    for pid, p in parents.items():
        ks = sorted(kids.get(pid, []), key=lambda c: c.source_start)
        if not p.retrievable and p.unit_type != "footnotes" and not ks:
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

    # whole-document coverage: every non-space character is in a parent chunk or a heading line
    covered = bytearray(len(text))
    for p in parents.values():
        covered[p.source_start:p.source_end] = b"\x01" * (p.source_end - p.source_start)
    for h in structure["headings"]:
        covered[h["start"]:h["end"]] = b"\x01" * (h["end"] - h["start"])
    lost = [i for i, ch in enumerate(text) if not covered[i] and not ch.isspace()]
    if lost:
        errors.append(f"{len(lost)} non-space characters not covered by any chunk, e.g. "
                      f"{text[lost[0]:lost[0] + 60]!r}")
    # every retrievable byte of legal text is indexed: retrievable chunks cover all parents except footnotes
    retr_cov = bytearray(len(text))
    for c in chunks:
        if c.retrievable:
            retr_cov[c.source_start:c.source_end] = b"\x01" * (c.source_end - c.source_start)
    for p in parents.values():
        if p.unit_type == "footnotes":
            continue
        miss = [i for i in range(p.source_start, p.source_end) if not retr_cov[i] and not text[i].isspace()]
        if miss:
            errors.append(f"{p.chunk_id}: {len(miss)} characters not in any retrievable chunk")

    # section sequence sanity
    nums = [s.number for s in structure["sections"]]
    bases = [_section_key(n)[0] for n in nums]
    gaps = [f"{bases[i - 1]}->{bases[i]}" for i in range(1, len(bases)) if bases[i] - bases[i - 1] > 1]
    if gaps:
        warnings.append(f"section number gaps (check for a swallowed section): {gaps}")
    if structure["rejected_section_like_lines"]:
        warnings.append(f"section-like lines not used as boundaries: {structure['rejected_section_like_lines'][:5]}")
    untitled = [s.number for s in structure["sections"] if not s.title and s.status == "in_force"]
    if untitled:
        warnings.append(f"sections without a marginal title (header uses the number only): {untitled}")
    retr = [c for c in chunks if c.retrievable]
    small = [c.chunk_id for c in retr if c.token_count - count_tokens(c.context_header) < 12
             and c.section_status in (None, "in_force")]
    if small:
        warnings.append(f"{len(small)} retrievable chunks have < 12 body tokens: {small[:10]}")
    unref = [f["id"] for f in doc.get("footnotes", []) if not any(f["id"] in c.footnote_ids for c in chunks if c.unit_type != "footnotes")]
    if unref:
        warnings.append(f"footnotes not cited by any content chunk: {unref}")

    toks = sorted(c.token_count for c in retr)
    diagnostics = {
        "total_chunks": len(chunks),
        "parent_chunks": sum(c.is_parent for c in chunks),
        "child_chunks": sum(not c.is_parent for c in chunks),
        "retrievable_chunks": len(retr),
        "sections": len(nums),
        "sections_split": sorted({c.section_number for c in chunks if c.unit_type in ("section_part", "definitions_part")},
                                 key=_section_key),
        "omitted_or_repealed_sections": [s.number for s in structure["sections"] if s.status != "in_force"],
        "chapters": list(dict.fromkeys(c.chapter_number for c in chunks if c.chapter_number)),
        "schedule_units": [f"{u.schedule}" + (f" / {u.label}" if u.label else "") for u in structure["schedules"]],
        "retrievable_token_min_median_max": [toks[0], toks[len(toks) // 2], toks[-1]] if toks else None,
        "embed_token_limit": EMBED_TOKEN_LIMIT,
        "token_counter": "tiktoken cl100k_base" if _ENCODING else "fallback estimator",
    }
    return ChunkValidationResult(ok=not errors, errors=errors, warnings=warnings, diagnostics=diagnostics)


# =====================================================================================
# entry points
# =====================================================================================
def chunk_document(doc: dict) -> dict:
    if "cleaned_text" not in doc:
        raise StatuteChunkingError("input is not a cleaned statute (no cleaned_text)")
    b = _Builder(doc)
    chunks = b.build()
    if not chunks:
        raise StatuteChunkingError("no content found to chunk")
    result = validate_chunks(chunks, doc, b.structure)
    return {
        "doc_id": doc["doc_id"],
        "act": b.act_info(),
        "metadata": doc.get("metadata", {}),
        "cleaned_text": doc["cleaned_text"],
        "footnotes": doc.get("footnotes", []),
        "outline": b.outline(),
        "glossary": b.glossary(),
        "chunks": [c.to_dict() for c in chunks],
        "validation": result.to_dict(),
    }


def is_valid_existing_output(out: dict) -> bool:
    try:
        chunks = [StatuteChunk.from_dict(c) for c in out["chunks"]]
        doc = {"cleaned_text": out["cleaned_text"], "footnotes": out.get("footnotes", [])}
        return validate_chunks(chunks, doc).ok
    except Exception:  # noqa: BLE001 -- malformed or old-schema output is regenerated
        return False


def chunk_directory(input_dir: Path = INPUT_DIR, output_dir: Path = OUTPUT_DIR) -> dict:
    """Chunks every <doc_id>.json in input_dir (skips the cleaner's _rejected/ folder). An
    existing output is skipped only if it re-validates. Never modifies input_dir."""
    input_dir, output_dir = Path(input_dir), Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    processed = regenerated = skipped_valid = 0
    failed_docs: list[dict] = []
    for path in sorted(input_dir.glob("*.json")):
        out_path = output_dir / path.name
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
            doc = json.loads(path.read_text(encoding="utf-8"))
            out = chunk_document(doc)
        except (OSError, json.JSONDecodeError, KeyError, StatuteChunkingError) as exc:
            failed_docs.append({"doc_id": path.stem, "error": str(exc)})
            continue
        if not out["validation"]["ok"]:
            failed_docs.append({"doc_id": path.stem, "error": out["validation"]["errors"]})
            continue
        tmp = out_path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
        tmp.replace(out_path)
        processed += 1
        regenerated += was_present
    return {"processed": processed, "skipped_valid": skipped_valid, "regenerated": regenerated,
            "failed": len(failed_docs), "failed_docs": failed_docs}


def main() -> None:  # pragma: no cover
    s = chunk_directory()
    print(f"Statute chunking: {s['processed']} processed ({s['regenerated']} regenerated), "
          f"{s['skipped_valid']} skipped (valid), {s['failed']} failed")
    for f in s["failed_docs"]:
        print(f"  FAILED {f['doc_id']}: {f['error']}")
    if s["failed"]:
        raise SystemExit(1)


if __name__ == "__main__":  # pragma: no cover
    main()