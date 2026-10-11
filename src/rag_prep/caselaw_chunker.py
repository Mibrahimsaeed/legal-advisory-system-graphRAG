"""Structure-aware parent-child chunking of case-law judgments for hybrid RAG (v2).

    structured case JSON (var/rag/structured_gemini_relabeled/<doc_id>.json)
        -> segment_front_matter()   [caption / headnotes / cited cases / counsel -- deterministic]
        -> repair_metadata()        [court, title, case number, dates, statutes, final orders]
        -> _Builder.build()         [headnote + judgment chunks, token-driven, no LLM]
        -> chunked case JSON        (var/rag/caselaw_chunked/<doc_id>.json)

Replaces chunker.py (+ chunk_types.py). Same output layout as statute_chunker.py v2, so one
indexer serves the Constitution, the statutes and the case law.

WHAT A CHUNK IS
---------------
    caption       citation, court, bench, parties, case number, date (+ counsel/hearing date
                  in the embedding text). Retrievable: answers "find Tufail v. Samiya".
    headnote      ONE chunk per headnote (a), (b), ... = topic line + summary + its cited-case
                  list. Headnotes are the reporter's legal summary -- the most valuable text
                  in a reported case. Cited-case lists go to bm25_text and references, NOT to
                  the embedding (a list of case names is noise for a dense model).
                  A headnote above MAX_CHUNK_TOKENS is split at its "---" catch-phrase
                  separators; every part carries "Statute --- Provision" as lead-in.
    judgment      whole paragraphs of the judgment grouped by their structuring label
                  (facts / arguments / court_reasoning / ...). A label change starts a new
                  chunk; a tiny paragraph ("3. Heard.") joins its neighbour; a quotation stays
                  with the sentence that introduces it ("... reads as under:--"). A paragraph
                  above MAX_CHUNK_TOKENS is split at sentence boundaries (abbreviation-safe:
                  "respondent No. 1", "S. 5", "Mst.", "C.P.C." never end a sentence) and each
                  continuation carries the previous sentence as lead-in.
    counsel       counsel / date-of-hearing lines (non-retrievable; values go to metadata).
    footnotes     editorial footnotes printed after the reporter line (non-retrievable).
    judgment marker lines ("JUDGMENT", "ORDER") are the only text not in a chunk.

    parent_id is the doc_id: the whole judgment (full_text, kept in the output) is the parent
    used for context expansion; doc-level "sections" lists every label run with its chunk ids.

TEXT FIELDS
-----------
    canonical_text  exact full_text[source_start:source_end] (display / citation)
    embedding_text  context_header + lead-in (if any) + normalized body
    bm25_text       embedding_text (+ cited-case lists for headnotes)
Normalization (embedding/BM25 only): Unicode dashes unified, "---" catch-phrase separators ->
" — ", "P L D"/"S C M R" -> "PLD"/"SCMR", "????" / U+FFFD junk dropped, reporter editor codes
("SA/M-251/Sindh") dropped. Nothing is paraphrased; canonical_text is never modified.

GRAPH FIELDS (Neo4j)
--------------------
    case_key        "pld-2024-pesh-25" -- the SAME key is produced for this case when another
                    judgment cites it ("PLD 2024 Pesh. 25"), so CITES edges join directly.
    references      {"cases": [{citation, case_key, name, treatment}],
                     "statutes": [{name, year, act_key, provisions, unit_ids}],
                     "constitution_articles": [...], "unit_ids": [...]}
                    unit_ids use the statute / Constitution chunkers' keys:
                    "family-courts-act-1964:s:5", "CONST:art:199".
    document-level  "case" (repaired metadata, parties, counsel, final_orders, statutes,
                    cited_cases), "sections" (label outline).
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import asdict, dataclass, field
from datetime import date
from pathlib import Path

INPUT_DIR = Path("var/rag/structured_gemini_relabeled")
OUTPUT_DIR = Path("var/rag/caselaw_chunked")

EMBED_TOKEN_LIMIT = 512        # hard limit of the embedding model (checked on embedding_text)
MAX_CHUNK_TOKENS = 400         # ceiling for a chunk body; leaves room for header + lead-in
MIN_CHUNK_TOKENS = 50          # smaller label runs are merged with a neighbour
LEAD_IN_MAX_TOKENS = 60
HEADER_TITLE_MAX_WORDS = 8     # per party in the context header

INHERITABLE_METADATA_FIELDS = ("citation", "case_key", "court", "court_location", "case_title", "case_number",
                               "decision_date", "judges", "disposition", "primary_domain")

try:                                                      # deterministic token COUNTING only
    import tiktoken
    _ENCODING = tiktoken.get_encoding("cl100k_base")
except Exception:  # noqa: BLE001 -- package missing or encoding not downloadable
    _ENCODING = None

if _ENCODING is not None:
    def count_tokens(text: str) -> int:
        return len(_ENCODING.encode(text))
else:
    def count_tokens(text: str) -> int:                   # conservative estimate (over-counts)
        words = len(text.split())
        return int(max(words * 1.45, len(text) / 3.4)) + 1


class CaseChunkingError(ValueError):
    pass


SECTION_DISPLAY = {
    "facts": "Facts", "procedural_history": "Procedural history", "issues": "Issues",
    "arguments": "Arguments", "evidence": "Evidence", "applicable_law": "Applicable law",
    "court_reasoning": "Court's reasoning", "findings": "Findings", "quoted_material": "Quoted material",
    "final_order": "Final order", "lower_court_orders": "Lower court orders",
    "authorities_cited": "Authorities cited", "unclassified": "Judgment",
}

# =====================================================================================
# text helpers
# =====================================================================================
_DASH_CHARS = "‐‑‒–—―−⸺⸻﹘﹣－"
_DASH_RE = re.compile(f"[{_DASH_CHARS}]")
_ANY_DASH = rf"[-{_DASH_CHARS}]"
_QUOTES = str.maketrans({"“": '"', "”": '"', "„": '"', "‘": "'", "’": "'",
                         "´": "'", " ": " "})
_EDITOR_CODE_RE = re.compile(r"^[A-Z][A-Za-z.&]*\.?(?:\s?/\s?[\w.&\-]+){1,4}\.?,?\s*")


def _plain(text: str) -> str:
    """Dashes / quotes unified (offset-free helpers use this; lengths may change)."""
    return _DASH_RE.sub("-", text.translate(_QUOTES))


def normalize(text: str) -> str:
    t = _plain(text)
    t = re.sub(r"\?{2,}|�+", " ", t)                                  # "P??????" OCR junk
    t = t.replace("~", "")
    t = re.sub(r"\b((?:[A-Z] ){1,4}[A-Z])\b(?=\s+\d)", lambda m: m.group(1).replace(" ", ""), t)   # P L D 2024
    t = re.sub(r"(?<=\d )((?:[A-Z] ){1,4}[A-Z])\b", lambda m: m.group(1).replace(" ", ""), t)       # 2020 Y L R
    t = re.sub(r"\bP\s?Cr\.?\s?L\.?\s?J\b\.?", "PCrLJ", t)
    t = re.sub(r":[ \t]*-{1,}[ \t]*", ": ", t)                               # "as under:--"
    t = re.sub(r"[ \t]*-{2,}[ \t]*", " — ", t)                               # catch-phrase separators
    t = re.sub(r"(?m)^[ \t]*—[ \t]*", "", t)                                 # leading separator of a line
    t = re.sub(r"[ \t]*—[ \t]*\n+(?=\S)", " — ", t)
    t = re.sub(r"(?<=\))[ \t]*-[ \t]*\n+(?=\S)", " — ", t)                  # "Constitution of Pakistan (1973)-"                            # "Act (XXXV of 1964) —" + next line
    t = re.sub(r"[ \t]{2,}", " ", t)
    t = re.sub(r"[ \t]+\n|\n[ \t]+", "\n", t)
    t = re.sub(r"\n{2,}", "\n", t)
    return t.strip()


def _truncate_tokens(text: str, limit: int, from_end: bool = False) -> str:
    if count_tokens(text) <= limit:
        return text
    words = text.split()
    if from_end:
        out: list[str] = []
        for w in reversed(words):
            if count_tokens(" ".join([w] + out)) > limit:
                break
            out.insert(0, w)
        return "… " + " ".join(out)
    out = []
    for w in words:
        if count_tokens(" ".join(out + [w])) > limit:
            break
        out.append(w)
    return " ".join(out) + " …"


def _slug(text: str) -> str:
    return re.sub(r"[^a-z0-9&]+", "-", text.lower()).strip("-").replace("&", "and")


def paragraph_offsets(full_text: str) -> list[tuple[int, int]]:
    """(start, end) of every paragraph of full_text.split('\\n\\n'), from lengths only."""
    out, cursor = [], 0
    for p in full_text.split("\n\n"):
        out.append((cursor, cursor + len(p)))
        cursor += len(p) + 2
    return out


# ---- sentences (abbreviation-safe) ----
_ABBREVIATIONS = {
    "no", "nos", "s", "ss", "sec", "secs", "art", "arts", "r", "rr", "o", "ord", "cl", "cls", "sched", "sch",
    "para", "paras", "p", "pp", "vol", "ed", "mst", "mr", "mrs", "ms", "dr", "rs", "v", "vs", "j", "jj", "cj",
    "c.j", "ltd", "co", "pvt", "govt", "sr", "jr", "messrs", "m/s", "hon", "ch", "st", "e.g", "i.e", "viz",
    "cf", "ibid", "id", "approx", "dist", "distt", "tehsil", "mauza", "maj", "col", "capt", "gen", "lt",
    "rtd", "brig", "prof", "engr", "sh", "mohd", "muhd", "addl", "asstt", "dy", "w.e.f", "u/s", "u/a",
    "corp", "inc", "deptt", "adj", "rev", "misc", "crl", "civ", "w.p", "c.p", "f.a.o", "r.f.a", "i.c.a",
    "c.r", "c.m", "c.m.a", "s.a", "pw", "dw", "ex", "exh", "approx", "nr", "km", "sq", "ft",
}
_SENT_END_RE = re.compile(r"[.?!][\"')\]]*\s+(?=[\"'(\[]?[A-Z0-9])")


def sentence_spans(text: str, base: int = 0) -> list[tuple[int, int]]:
    """Contiguous (start, end) sentence ranges covering text (absolute offsets via base)."""
    cuts = [0]
    for m in _SENT_END_RE.finditer(text):
        dot = m.start()
        j = dot
        while j > 0 and not text[j - 1].isspace():
            j -= 1
        token = text[j:dot].lstrip("(\"'[").lower()
        if token in _ABBREVIATIONS or re.fullmatch(r"[a-z]", token) or re.fullmatch(r"(?:[a-z]\.)+[a-z]", token) \
                or re.fullmatch(r"\d{1,3}", token) and text[dot] == "." and j > 0 and \
                re.search(r"(?:No|Nos|S|Ss|Art|Arts|R|para|section|Article|order)\.?\s*$", text[max(0, j - 10):j]):
            continue
        if re.fullmatch(r"\d{1,3}", token) and (j == 0 or text[j - 1] == "\n"):
            continue                                       # "12. It is ..." paragraph number
        cuts.append(m.end())
    cuts.append(len(text))
    spans = []
    for a, b in zip(cuts, cuts[1:]):
        seg = text[a:b]
        e = a + len(seg.rstrip())
        if e > a:
            spans.append((base + a, base + e))
    return spans


# =====================================================================================
# citations, statutes, provisions
# =====================================================================================
_PLD_COURTS = {
    "sc": "SC", "s.c.": "SC", "s.c": "SC", "s c": "SC", "s. c.": "SC", "supreme court": "SC", "fsc": "FSC", "federal shariat court": "FSC",
    "lah": "Lah", "lahore": "Lah", "kar": "Kar", "karachi": "Kar", "pesh": "Pesh", "peshawar": "Pesh",
    "quetta": "Quetta", "bal": "Bal", "balochistan": "Bal", "baluchistan": "Bal", "isl": "Isl", "islamabad": "Isl",
    "sindh": "Sindh", "dacca": "Dacca", "dhaka": "Dacca", "aj&k": "AJ&K", "azad j&k": "AJ&K", "gb": "GB",
    "gilgit-baltistan": "GB", "note": "Note", "journal": "Journal", "trib": "Trib", "rev": "Rev",
}
_PLD_COURT_ALT = "|".join(sorted((re.escape(k).replace(r"\ ", r"\s+") for k in _PLD_COURTS), key=len, reverse=True))
_REPORTERS = {
    "SCMR": r"S\s?C\s?M\s?R", "CLC": r"C\s?L\s?C", "YLR": r"Y\s?L\s?R", "MLD": r"M\s?L\s?D", "PLC": r"P\s?L\s?C",
    "PSC": r"P\s?S\s?C", "CLD": r"C\s?L\s?D", "PTD": r"P\s?T\s?D", "SCR": r"S\s?C\s?R", "PLJ": r"P\s?L\s?J",
    "PCrLJ": r"P\s?\.?\s?Cr\.?\s?L\.?\s?J\.?", "KLR": r"K\s?L\s?R", "GBLR": r"G\s?B\s?L\s?R",
    "PLR": r"P\s?L\s?R", "MLJ": r"M\s?L\s?J",
}
_REPORTER_ALT = "|".join(f"(?P<r_{k}>{v})" for k, v in _REPORTERS.items())
_CITATION_RE = re.compile(
    rf"\b(?:P\s?\.?\s?L\s?\.?\s?D\.?)\s+(?P<pyear>1[89]\d\d|20\d\d)\s+(?:\((?:W\.\s?P\.|AJ&K|Pak\.)\)\s+)?"
    rf"(?P<pcourt>{_PLD_COURT_ALT})\.?\s+(?P<ppage>\d{{1,5}})\b"
    rf"|\b(?P<year>1[89]\d\d|20\d\d)\s+(?:{_REPORTER_ALT})\s*(?:\([A-Za-z.&\s]{{1,20}}\)\s*)?(?P<note>Note\s+)?(?P<page>\d{{1,5}})\b"
    rf"|\bNLR\s+(?P<nyear>1[89]\d\d|20\d\d)\s+(?P<nkind>Civil|Criminal|Revenue|Service|Tax|AC|SCJ|UC|Labour)\s+(?P<npage>\d{{1,5}})\b",
    re.I)
_TREATMENTS = ("ref", "rel", "fol", "distinguished", "distinguishable", "dissented from", "not followed",
               "overruled", "approved", "considered", "applied", "explained", "discussed", "cited",
               "affirmed", "reversed", "disapproved", "per incuriam", "held per incuriam", "doubted",
               "not applicable", "quoted", "relied upon", "referred")
_CITELIST_END_RE = re.compile(rf"\b(?P<t>{'|'.join(_TREATMENTS)})\.?\s*$", re.I)


def normalize_citation(m: re.Match) -> str | None:
    if m.group("pyear"):
        court = _PLD_COURTS.get(re.sub(r"\s+", " ", m.group("pcourt").lower().rstrip(".")))
        return f"PLD {m.group('pyear')} {court} {int(m.group('ppage'))}" if court else None
    if m.group("year"):
        rep = next(k[2:] for k, v in m.groupdict().items() if k.startswith("r_") and v)
        return f"{m.group('year')} {rep}{' Note' if m.group('note') else ''} {int(m.group('page'))}"
    return f"NLR {m.group('nyear')} {m.group('nkind').title()} {int(m.group('npage'))}"


def citation_key(citation: str | None) -> str | None:
    """'P L D 2024 Peshawar 25' / 'PLD 2024 Pesh. 25' -> 'pld-2024-pesh-25'; '2020 Y L R 2523' -> '2020-ylr-2523'."""
    if not citation:
        return None
    m = _CITATION_RE.search(_plain(citation))
    norm = normalize_citation(m) if m else None
    return _slug(norm) if norm else None


def extract_case_citations(text: str, self_key: str | None = None) -> list[dict]:
    plain = _plain(text)
    out: dict[str, dict] = {}
    for m in _CITATION_RE.finditer(plain):
        c = normalize_citation(m)
        if not c:
            continue
        k = _slug(c)
        if k != self_key and k not in out:
            out[k] = {"citation": c, "case_key": k}
    return list(out.values())


def parse_cited_list(text: str, self_key: str | None = None) -> list[dict]:
    """'A v. B PLD 2011 SC 260; C v. D 2023 SCMR 413 and E v. F 2001 SCMR 338 rel.' ->
    [{name, citation, case_key, treatment}, ...]"""
    plain = _plain(text).strip()
    mt = _CITELIST_END_RE.search(plain)
    treatment = mt.group("t").lower() if mt else None
    body = plain[:mt.start()] if mt else plain
    items = re.split(r";\s*|(?<=\d)\.?\s+and\s+(?=[A-Z0-9\"'(])|(?<=\d),\s+(?=[A-Z])", body)
    out = []
    for it in items:
        m = None
        for m in _CITATION_RE.finditer(it):
            pass
        if not m:
            continue
        c = normalize_citation(m)
        if not c or _slug(c) == self_key:
            continue
        name = it[:m.start()].strip(" ,.;:()")
        name = re.sub(r"^\s*(?:and|in)\s+", "", name)
        out.append({"name": name if len(name) > 3 else None, "citation": c, "case_key": _slug(c), "treatment": treatment})
    return out


# ---- statutes ----
_ALIASES = {  # lower-case name -> canonical name (matches statute_chunker act keys)
    "civil procedure code": "Code of Civil Procedure", "code of civil procedure": "Code of Civil Procedure",
    "criminal procedure code": "Code of Criminal Procedure", "code of criminal procedure": "Code of Criminal Procedure",
    "qanun-e-shahadat": "Qanun-e-Shahadat Order", "qanun-e-shahadat order": "Qanun-e-Shahadat Order",
    "qanun e shahadat": "Qanun-e-Shahadat Order", "penal code": "Pakistan Penal Code",
    "pakistan penal code": "Pakistan Penal Code",
}
_DEFAULT_YEARS = {"Code of Civil Procedure": 1908, "Code of Criminal Procedure": 1898, "Pakistan Penal Code": 1860,
                  "Qanun-e-Shahadat Order": 1984, "Muslim Family Laws Ordinance": 1961,
                  "Family Courts Act": 1964, "Guardians and Wards Act": 1890,
                  "Dissolution of Muslim Marriages Act": 1939, "Dowry and Bridal Gifts (Restriction) Act": 1976,
                  "Child Marriage Restraint Act": 1929}
_ABBR_ACTS = {"C.P.C.": "Code of Civil Procedure", "CPC": "Code of Civil Procedure",
              "Cr.P.C.": "Code of Criminal Procedure", "CrPC": "Code of Criminal Procedure",
              "Cr. P.C.": "Code of Criminal Procedure", "P.P.C.": "Pakistan Penal Code", "PPC": "Pakistan Penal Code",
              "MFLO": "Muslim Family Laws Ordinance"}
_NAME_WORD = r"(?:[A-Z][\w']*(?:-[\w']+)*|\([A-Z][\w .'\-]{0,30}\))"
_NAME_RE = (rf"(?P<name>(?:West\s+Pakistan\s+|W\.\s?P\.\s+)?{_NAME_WORD}(?:\s+(?:(?:and|of|the|for|on|in|to|e)\s+)*{_NAME_WORD})*?"
            rf"\s+(?:Act|Ordinance|Order|Code|Rules|Regulations?|Shahadat))")
_ACT_ROMAN_RE = re.compile(rf"{_NAME_RE}(?:,?\s*1[6-9]\d\d)?\s*\(\s*(?:[IVXLCDM]+|\d+)\s+of\s+(?P<year>1[6-9]\d\d|20\d\d)\s*\)")
_ACT_YEAR_RE = re.compile(rf"{_NAME_RE},?\s+(?P<year>1[6-9]\d\d|20\d\d)\b")
_ACT_BARE_RE = re.compile(rf"{_NAME_RE}\b")
_CONSTITUTION_RE = re.compile(r"\bConstitution(?:\s+of\s+(?:the\s+)?(?:Islamic\s+Republic\s+of\s+)?Pakistan)?"
                              r"(?:\s*\(\s*1973\s*\)|,?\s+1973)?")
_PROV_TOKEN = r"\d{1,3}(?:\s?-?\s?[A-Z]{1,2}\b)?(?:\s?\([0-9a-zA-Z]{1,5}\))*"
_PROV_LIST = rf"{_PROV_TOKEN}(?:\s*(?:,|&|and|to|/)\s*(?:{_PROV_TOKEN}|Sched\.?))*"
_PROV_AFTER_RE = re.compile(  # ", S.3" / ", Ss. 5 & 17" / ", Arts.199 & 4" / ", R. 6" / ", O. XXXIX, R. 1"
    rf"^\s*,?\s*(?P<kind>Ss?\.|Sections?|Arts?\.|Articles?|Rr?\.|Rules?|O\.\s?[IVXLC]+,?\s*Rr?\.)\s*(?P<list>{_PROV_LIST})")
_PROV_BEFORE_RE = re.compile(  # "section 5 of the" / "Sections 10 and 14 of" / "Article 199 of the"
    rf"\b(?P<kind>[Ss]ections?|[Ss]s?\.|Articles?|Arts?\.|[Rr]ules?|R\.)\s*(?P<list>{_PROV_LIST})\s+of\s+(?:the\s+)?$")
_SCHED_RE = re.compile(r"\bSched(?:ule)?\.?(?![a-z])")
_STOP_NAME_WORDS = {"the", "this", "that", "said", "such", "under", "of", "in", "and", "provisions", "provision",
                    "section", "sections", "whereas", "where", "if", "by", "for", "per", "learned", "honourable",
                    "schedule", "sched", "part", "chapter", "to", "a", "an", "article", "articles", "rule", "rules"}
_DMMA = "Dissolution of Muslim Marriages Act"
_COMPACT_ALIASES = {  # letters-only lower-case name -> canonical name (typos / variants seen in judgments)
    "familycourtact": "Family Courts Act", "familycourtsact": "Family Courts Act",
    "guardianandwardsact": "Guardians and Wards Act", "guardiansandwardact": "Guardians and Wards Act",
    "guardianandwardact": "Guardians and Wards Act", "guardiansandwardsact": "Guardians and Wards Act",
    "muslimfamilylawordinance": "Muslim Family Laws Ordinance", "muslimfamilylawsordinance": "Muslim Family Laws Ordinance",
    "qanuneshahadat": "Qanun-e-Shahadat Order", "qanuneshahadatorder": "Qanun-e-Shahadat Order",
    "qanunishahadatorder": "Qanun-e-Shahadat Order", "muslimmarriagesact": _DMMA, "muslimmarriageact": _DMMA,
    "dissolutionofmuslimmarriagesact": _DMMA, "dissolutionofmuslimmarriageact": _DMMA,
    "muslimdissolutionofmarriageact": _DMMA, "muslimdissolutionofmarriagesact": _DMMA,
    "civilprocedurecode": "Code of Civil Procedure", "codeofcivilprocedure": "Code of Civil Procedure",
    "criminalprocedurecode": "Code of Criminal Procedure", "codeofcriminalprocedure": "Code of Criminal Procedure",
    "pakistanpenalcode": "Pakistan Penal Code", "penalcode": "Pakistan Penal Code",
    "dowryandbridalgiftsrestrictionact": "Dowry and Bridal Gifts (Restriction) Act",
    "childmarriagerestraintact": "Child Marriage Restraint Act",
}


def _canonical_act(name: str, year: int | None) -> tuple[str, int | None] | None:
    n = re.sub(r"\s+", " ", _plain(name)).strip(" ,.;:")
    words = n.split()
    while len(words) > 1 and (words[0].lower().strip(".,") in _STOP_NAME_WORDS or words[0].startswith("(")
                              or re.fullmatch(r"[A-Z]|[IVXLC]+(?:-[A-Z])?|(?i:part)-?[IVXLC\d]+", words[0])):
        words.pop(0)
    n = " ".join(words)
    if len(words) < 2 or n.lower() in ("the act", "this act", "said act", "an act"):
        return None
    if not any(len(w.strip(".()")) >= 3 for w in words[:-1]):
        return None
    if re.match(r"^(?:West\s+Pakistan|W\.\s?P\.)\s+", n):
        n_key = re.sub(r"^(?:West\s+Pakistan|W\.\s?P\.)\s+", "", n)
    else:
        n_key = n
    canon = _ALIASES.get(n_key.lower()) or _COMPACT_ALIASES.get(re.sub(r"[^a-z]", "", n_key.lower()), n_key)
    if canon.lower().startswith("code of") and canon not in ("Code of Civil Procedure", "Code of Criminal Procedure"):
        return None
    year = year or _DEFAULT_YEARS.get(canon)
    return canon, year


def act_key(name: str, year: int | None) -> str:
    """Same rule as statute_chunker.act_key (West Pakistan / W.P. prefix dropped)."""
    n = re.sub(r"^\s*the\s+", "", name.strip(), flags=re.I)
    n = re.sub(r"^(?:West\s+Pakistan|W\.\s?P\.)\s+", "", n)
    slug = re.sub(r"[^a-z0-9]+", "-", n.lower()).strip("-")
    return f"{slug}-{year}" if year else slug


def parse_provisions(kind: str, plist: str) -> tuple[str, list[str], bool]:
    """('Ss.', '5, Sched. & 17-A') -> ('section', ['5', '17A'], schedule=True)."""
    k = kind.lower()
    pk = "article" if k.startswith("art") else "rule" if k.startswith("r") else "order_rule" if k.startswith("o") \
        else "section"
    sched = bool(_SCHED_RE.search(plist))
    clean = re.sub(r"\([^)]*\)", "", _SCHED_RE.sub("", plist))
    toks = re.findall(r"\d{1,3}(?:\s?-?\s?[A-Z]{1,2}\b)?|\bto\b", clean)
    nums: list[str] = []
    for i, t in enumerate(toks):
        t = re.sub(r"[\s-]", "", t)
        if t == "to" and nums and i + 1 < len(toks) and nums[-1].isdigit() and toks[i + 1].strip().isdigit():
            nums += [str(n) for n in range(int(nums[-1]) + 1, int(toks[i + 1]))]
        elif t != "to" and t not in nums:
            nums.append(t)
    return pk, nums, sched


class _Refs:
    """Accumulates statute / Constitution references for one chunk."""

    def __init__(self):
        self.acts: dict[str, dict] = {}
        self.articles: list[str] = []

    def act(self, name: str, year: int | None) -> dict | None:
        ca = _canonical_act(name, year)
        if not ca:
            return None
        canon, yr = ca
        key = act_key(canon, yr)
        if yr is None:                                   # merge into a dated entry of the same Act
            for k, e in self.acts.items():
                if e["name"].lower() == canon.lower():
                    return e
        return self.acts.setdefault(key, {"name": canon, "year": yr, "act_key": key, "provisions": [],
                                          "schedule": False})

    def provisions(self, entry: dict | None, kind: str, plist: str):
        pk, nums, sched = parse_provisions(kind, plist)
        if entry is not None:                            # "S." belongs to an Act, "R." to Rules, "Art." to an Order
            last = entry["name"].split()[-1].lower()
            ok = {"section": last in ("act", "ordinance", "code"), "rule": last in ("rules", "regulations"),
                  "article": last in ("order", "shahadat"), "order_rule": entry["name"] == "Code of Civil Procedure"}
            if not ok.get(pk, False):
                return
        else:                                            # Constitution: only Articles
            if pk == "article":
                self.articles += [n for n in nums if n not in self.articles]
            return
        if sched:
            entry["schedule"] = True
        for n in nums:
            p = f"{pk}:{n}"
            if p not in entry["provisions"]:
                entry["provisions"].append(p)

    def to_dict(self) -> dict:
        statutes, unit_ids = [], []
        for e in self.acts.values():
            ids = []
            for p in e["provisions"]:
                pk, n = p.split(":", 1)
                if pk == "section":
                    ids.append(f"{e['act_key']}:s:{n}")
                elif pk == "rule":
                    ids.append(f"{e['act_key']}:r:{n}")
            if e["schedule"]:
                ids.append(f"{e['act_key']}:schedule")
            statutes.append({**e, "unit_ids": ids})
            unit_ids += ids
        unit_ids += [f"CONST:art:{a}" for a in self.articles]
        return {"statutes": statutes, "constitution_articles": list(self.articles), "unit_ids": unit_ids}


def topic_provisions(refs: _Refs, topic_act: tuple[str, int | None] | None, first_body: str) -> None:
    """Headnote '(a) Family Courts Act (XXXV of 1964)---' + '----Ss. 5, Sched. & 17-A---...':
    the first catch-phrase of the body lists provisions of the topic Act."""
    if topic_act is None:
        return
    m = re.match(rf"^\s*-*\s*(?P<kind>Ss?\.|Arts?\.|Rr?\.|O\.\s?[IVXLC]+,?\s*Rr?\.)\s*(?P<list>{_PROV_LIST})",
                 _plain(first_body))
    if topic_act[0] == "__constitution__":
        if m:
            refs.provisions(None, m.group("kind"), m.group("list"))
        return
    e = refs.act(*topic_act)
    if e and m:
        refs.provisions(e, m.group("kind"), m.group("list"))


def extract_statute_refs(text: str, refs: _Refs | None = None) -> _Refs:
    refs = refs or _Refs()
    plain = _plain(text)
    taken: list[tuple[int, int]] = []

    def free(a: int, b: int) -> bool:
        return all(b <= x or a >= y for x, y in taken)

    def handle(entry, a: int, b: int, is_const: bool = False):
        taken.append((a, b))
        tail = plain[b:b + 120]
        m = _PROV_AFTER_RE.match(tail)
        if m:
            refs.provisions(None if is_const else entry, m.group("kind"), m.group("list"))
        m = _PROV_BEFORE_RE.search(plain[max(0, a - 90):a])
        if m:
            refs.provisions(None if is_const else entry, m.group("kind"), m.group("list"))

    for rx in (_ACT_ROMAN_RE, _ACT_YEAR_RE):
        for m in rx.finditer(plain):
            if not free(m.start(), m.end()):
                continue
            e = refs.act(m.group("name"), int(m.group("year")))
            if e:
                handle(e, m.start("name"), m.end())
    for m in _ACT_BARE_RE.finditer(plain):
        if not free(m.start(), m.end()):
            continue
        name = m.group("name")
        if not re.search(r"(?:Act|Ordinance|Code|Shahadat)$", name):
            continue
        ca = _canonical_act(name, None)
        if ca and ca[1]:                                     # only well-known Acts without a year
            e = refs.act(name, None)
            if e:
                handle(e, m.start(), m.end())
    for abbr, canon in _ABBR_ACTS.items():
        for m in re.finditer(rf"(?<![\w.]){re.escape(abbr)}(?![\w])", plain):
            if free(m.start(), m.end()):
                e = refs.act(canon, None)
                handle(e, m.start(), m.end())
    for m in _CONSTITUTION_RE.finditer(plain):
        if free(m.start(), m.end()):
            handle(None, m.start(), m.end(), is_const=True)
    return refs


# =====================================================================================
# front matter: caption / headnotes / cited cases / counsel  (deterministic)
# =====================================================================================
_MARKER_RE = re.compile(r"^\s*(?:SHORT\s+)?(?:J\s?U\s?D\s?G\s?M\s?E\s?N\s?T|O\s?R\s?D\s?E\s?R)(?:\s+OF\s+THE\s+COURT)?\s*$", re.I)
_DECIDED_RE = re.compile(r"\b(?:decided|heard)\s+on\b", re.I)
_VERSUS_RE = re.compile(r"^\s*(?:versus|vs?\.?|v/s)\s*\.?\s*$", re.I)
_BENCH_RE = re.compile(r"^\s*(?:Before|Present|Coram)\b\s*:?", re.I)
_COURT_LINE_RE = re.compile(r"^\s*\[(?P<court>[^\[\]]{2,80})\]\s*$")
_AGAINST_RE = re.compile(r"^\s*\((?:Against|On\s+appeal|Appeal\s+from|From)\b", re.I)
_COUNSEL_RE = re.compile(
    r"\bfor\s+(?:the\s+)?(?:Petitioner|Respondent|Appellant|Applicant|Complainant|State|Plaintiff|Defendant|Accused|"
    r"Opposite|Federation|Province|Government|Caveator|Intervener|Objector|Appellee|Non-applicant|Department|"
    r"Official|Proforma)|^\s*Nemo\b|\bin[- ]person\b|Advocate[- ]Gen|Attorney[- ]Gen|Amicus|Standing\s+Counsel|"
    r"Law\s+Officer|Prosecutor|A\.\s?A\.\s?G|D\.\s?A\.\s?G|A\.\s?G\.|^\s*Dates?\s+of\s+(?:hearing|decision|order|judgment|announcement)|^\s*(?:Heard|Hearing)\s+on\b|"
    r"^\s*Ex\s+parte\b|^\s*None\s+(?:present|appeared)|\bfor\s*$|"
    r"^\s*(?:the\s+)?(?:Petitioner|Respondent|Appellant|Applicant|Plaintiff|Defendant)s?\b[\w .,]{0,40}\.?\s*$", re.I)
_TOPIC_RE = re.compile(rf"^\s*(?:\((?P<letter>[a-z]{{1,2}})\)\s*)?(?P<topic>[A-Z\"'(].{{1,220}}?){_ANY_DASH}+\s*$")
_LETTER_START_RE = re.compile(r"^\s*\((?P<letter>[a-z]{1,2})\)\s+\S")


def _is_counsel(p: str) -> bool:
    return len(p) < 320 and bool(_COUNSEL_RE.search(_plain(p)))


def _is_citelist(p: str) -> bool:
    pl = _plain(p).strip()
    return bool(_CITELIST_END_RE.search(pl)) and bool(_CITATION_RE.search(pl) or " v. " in pl or " vs. " in pl)


def _is_topic(p: str) -> bool:
    pl = _plain(p)
    return bool(_TOPIC_RE.match(pl)) and not re.match(rf"^\s*{_ANY_DASH}", pl) and len(pl) < 240


@dataclass
class Headnote:
    letter: str | None
    topic: str
    paragraphs: list[int]
    body_paragraphs: list[int]
    citelist_paragraphs: list[int]
    cited_cases: list[dict] = field(default_factory=list)


@dataclass
class FrontMatter:
    end: int                         # first paragraph index of the judgment (marker or body)
    caption: list[int]
    headnotes: list[Headnote]
    counsel: list[int]
    markers: list[int]
    warnings: list[str] = field(default_factory=list)


def segment_front_matter(paragraphs: list[str], spans: list[dict]) -> FrontMatter:
    warnings: list[str] = []
    n = len(paragraphs)
    jm = next((s["paragraph_start"] for s in sorted(spans, key=lambda s: s["paragraph_start"])
               if s.get("type") == "judgment_marker"), None)
    if jm is None:
        jm = next((i for i, p in enumerate(paragraphs[:80]) if _MARKER_RE.match(p)), None)
    if jm is None:  # no marker: judgment starts at the first body span after caption/headnote spans
        jm = next((s["paragraph_start"] for s in sorted(spans, key=lambda s: s["paragraph_start"])
                   if s.get("type") not in ("case_caption", "headnotes")), n)
        warnings.append("no judgment marker; front matter ends at the first body span")
        body_start = jm
    else:
        body_start = jm + 1
    front = list(range(0, jm))

    # caption head: up to the "decided on" line (+ "(Against order ...)" lines)
    h_end = next((i for i in front[:20] if _DECIDED_RE.search(paragraphs[i])), None)
    if h_end is None:
        h_end = next((i for i in front[:20] if re.search(rf"{_ANY_DASH}+\s*(?:Respondents?|Appellees?|Opposite)",
                                                          _plain(paragraphs[i]))), None)
        if h_end is not None and h_end + 1 < jm and re.search(r"\bNo\.|\bNos\.", paragraphs[h_end + 1]) \
                and not _is_topic(paragraphs[h_end + 1]):
            h_end += 1
        warnings.append("caption: no 'decided on' line")
    if h_end is None:
        cap = [s for s in spans if s.get("type") == "case_caption"]
        h_end = cap[0]["paragraph_end"] if cap and cap[0]["paragraph_end"] < jm else jm - 1
        warnings.append("caption: boundary taken from the structuring spans")
    while h_end + 1 < jm and _AGAINST_RE.match(paragraphs[h_end + 1]):
        h_end += 1

    # counsel tail: walk back from the marker
    t = jm
    while t - 1 > h_end and _is_counsel(paragraphs[t - 1]):
        t -= 1
    counsel = list(range(t, jm))
    region = list(range(h_end + 1, t))

    headnotes: list[Headnote] = []
    cur: Headnote | None = None
    for i in region:
        p = paragraphs[i]
        if _is_citelist(p) and cur is not None:
            cur.paragraphs.append(i)
            cur.citelist_paragraphs.append(i)
            continue
        lm = _LETTER_START_RE.match(_plain(p))
        if cur is None or lm or (_is_topic(p) and cur.body_paragraphs):
            tm = _TOPIC_RE.match(_plain(p))
            letter = lm.group("letter") if lm else None
            topic = (tm.group("topic") if tm else re.sub(r"^\s*\([a-z]{1,2}\)\s*", "", _plain(p))[:200]).strip(" -")
            cur = Headnote(letter, topic, [i], [], [])
            headnotes.append(cur)
            if not tm:                          # "(a) Topic---text on the same line": topic paragraph has body
                cur.body_paragraphs.append(i)
            continue
        cur.paragraphs.append(i)
        cur.body_paragraphs.append(i)
    if any(h.letter for h in headnotes) and len(headnotes) > 1:   # "(a)" lost in OCR -> infer from position
        for k, h in enumerate(headnotes):
            if h.letter is None:
                h.letter = chr(ord("a") + k)
    for h in headnotes:
        if not h.body_paragraphs:
            warnings.append(f"headnote ({h.letter}) has a topic line but no text")
    markers = [s["paragraph_start"] for s in spans if s.get("type") == "judgment_marker"]
    if jm < n and _MARKER_RE.match(paragraphs[jm]) and jm not in markers:
        markers.append(jm)
    return FrontMatter(end=body_start, caption=list(range(0, h_end + 1)), headnotes=headnotes,
                       counsel=counsel, markers=sorted(set(markers)), warnings=warnings)


# =====================================================================================
# metadata repair
# =====================================================================================
_MONTHS = {m: i for i, m in enumerate(["january", "february", "march", "april", "may", "june", "july", "august",
                                       "september", "october", "november", "december"], 1)}
_MONTHS.update({m[:3]: i for m, i in list(_MONTHS.items())})
_MONTHS["sept"] = 9
_HIGH_COURTS = {"lahore": "Lahore High Court", "sindh": "Sindh High Court", "karachi": "Sindh High Court",
                "peshawar": "Peshawar High Court", "balochistan": "Balochistan High Court",
                "baluchistan": "Balochistan High Court", "quetta": "Balochistan High Court",
                "islamabad": "Islamabad High Court"}
_CITE_COURT = {"SC": "Supreme Court of Pakistan", "FSC": "Federal Shariat Court", "Lah": "Lahore High Court",
               "Kar": "Sindh High Court", "Sindh": "Sindh High Court", "Pesh": "Peshawar High Court",
               "Bal": "Balochistan High Court", "Quetta": "Balochistan High Court", "Isl": "Islamabad High Court"}


def parse_date(text: str) -> str | None:
    t = _plain(text)
    m = re.search(r"(\d{1,2})(?:st|nd|rd|th)?\s+(?:of\s+)?([A-Za-z]{3,9})\.?,?\s+(\d{4})", t)
    if m and m.group(2).lower() in _MONTHS:
        d, mo, y = int(m.group(1)), _MONTHS[m.group(2).lower()], int(m.group(3))
    else:
        m = re.search(r"([A-Za-z]{3,9})\.?\s+(\d{1,2})(?:st|nd|rd|th)?,\s+(\d{4})", t)
        if m and m.group(1).lower() in _MONTHS:
            d, mo, y = int(m.group(2)), _MONTHS[m.group(1).lower()], int(m.group(3))
        else:
            m = re.search(r"\b(\d{1,2})[-./](\d{1,2})[-./](\d{4})\b", t)
            if not m:
                return None
            d, mo, y = int(m.group(1)), int(m.group(2)), int(m.group(3))
    try:
        return date(y, mo, d).isoformat()
    except ValueError:
        return None


def court_name(bracket: str | None, citation_norm: str | None) -> tuple[str | None, str | None]:
    """'[Sindh (Hyderabad Bench)]' -> ('Sindh High Court (Hyderabad Bench)', 'Hyderabad')."""
    if bracket:
        b = re.sub(r"\s+", " ", _plain(bracket)).strip()
        m = re.match(r"^(?P<base>[^()]+?)\s*(?:\((?P<bench>[^()]+)\))?$", b)
        base, bench = (m.group("base").strip(), m.group("bench")) if m else (b, None)
        if re.search(r"\b(?:Court|Bench|Tribunal|Council)\b", base):
            name = base
        else:
            name = _HIGH_COURTS.get(base.lower(), f"{base} High Court")
        if bench and "Bench" in bench:
            location = re.sub(r"\s*Bench$", "", bench).strip()
        else:
            location = {"sindh": "Karachi", "balochistan": "Quetta"}.get(base.lower(), base)
            if re.search(r"\b(?:Court|Bench|Tribunal)\b", base):
                location = None
        return (f"{name} ({bench})" if bench else name), location
    if citation_norm:
        parts = citation_norm.split()
        if parts[0] == "PLD" and len(parts) >= 4:
            return _CITE_COURT.get(parts[2]), None
        if len(parts) >= 2 and parts[1] == "SCMR":
            return "Supreme Court of Pakistan", None
    return None, None


def _party_name(p: str) -> tuple[str, str | None]:
    pl = re.sub(r"\s+", " ", _plain(p)).strip()
    m = re.match(r"^(?P<name>.+?)\s*-{2,}\s*(?P<role>[A-Za-z][A-Za-z ./-]*?)\.?\s*$", pl) or \
        re.match(r"^(?P<name>.+?)\s*[-\s]\s*(?P<role>(?:Petitioner|Respondent|Appellant|Applicant|Plaintiff|Defendant|"
                 r"Complainant|Accused|Appellee|Opposite part)[a-z]*)\.?\s*$", pl, re.I)
    name, role = (m.group("name"), m.group("role").strip().title()) if m else (pl, None)
    name = name.strip(" .,-")
    if re.fullmatch(r"(?:[A-Z] )+[A-Z]", name):                      # "B H A R O O" -> "BHAROO"
        name = name.replace(" ", "")
    return name, role


def _short(name: str, n: int = HEADER_TITLE_MAX_WORDS) -> str:
    w = name.split()
    return name if len(w) <= n else " ".join(w[:n]) + " …"


_OPERATIVE_RE = re.compile(
    r"\b(?:is|are|stands?|hereby|hence|thus|accordingly)\s+(?:hereby\s+|accordingly\s+|"
    r"therefore\s+|also\s+|partly\s+|partially\s+|in\s+limine\s+)*(?P<v>dismissed|allowed|accepted|disposed\s+of|"
    r"set\s+aside|remanded|remitted|converted|modified|maintained|upheld|decreed|withdrawn|quashed|refused|declined|"
    r"struck\s+off|returned|restored|disposed)\b", re.I)
_TAIL_OUTCOME_RE = re.compile(
    r"^(?:Petitions?|Appeals?|Revisions?|Applications?|Cases?|Suits?|Orders?|References?|Reviews?|Writs?|Matters?|"
    r"Leave|Constitutional\s+petitions?|Order)\b.{0,60}$", re.I)
_DISPOSES_RE = re.compile(r"\b(?:petitions?|appeals?|revisions?|suits?|applications?|cases?|matters?|writ|reference|"
                          r"review|leave)\b", re.I)
_LABEL_OF = [(r"partly|partially", "partly_allowed"), (r"dismiss|refus|declin", "dismissed"),
             (r"allow|accept", "allowed"), (r"remand|remit", "remanded"), (r"set\s+aside|quash", "set_aside"),
             (r"disposed", "disposed_of"), (r"withdrawn", "withdrawn"), (r"convert", "converted"),
             (r"modif", "modified"), (r"maintain|upheld", "upheld"), (r"decreed", "decreed"),
             (r"accordingly", "order_accordingly"), (r"restor", "restored"), (r"return", "returned")]


_ACTIVE_RE = re.compile(  # "I ... therefore, dismiss this petition" / "proceed to dismiss it" (present tense only)
    r"\b(?:I|we|We|this\s+Court)\b[^.;]{0,120}?\b(?P<v>dismiss|allow|accept|remand|set\s+aside)(?:es)?\s+"
    r"(?:the|this|both|these|instant|present|said|it|them)\b")


def outcome_labels(sentence: str) -> list[str]:
    """Labels from the operative verbs ('petition is dismissed', 'I dismiss this petition');
    a reporter line ('Petition partly accepted.') falls back to a keyword scan."""
    s = _plain(sentence)
    out: list[str] = []
    for m in list(_OPERATIVE_RE.finditer(s)) + list(_ACTIVE_RE.finditer(s)):
        v = m.group("v").lower()
        lab = "partly_allowed" if re.search(r"partly|partially", m.group(0), re.I) and re.search(r"allow|accept", v) \
            else next((lab for rx, lab in _LABEL_OF if re.search(rx, v)), None)
        if lab and lab not in out:
            out.append(lab)
    if not out:
        out = [lab for rx, lab in _LABEL_OF if re.search(rx, s.lower())][:1]
    return out


def outcome_label(sentence: str) -> str | None:
    labs = outcome_labels(sentence)
    return labs[0] if labs else None


def reporter_tail(p: str) -> str | None:
    """'SA/M-251/Sindh Petition dismissed.' -> 'Petition dismissed.' (None if not a reporter line)."""
    pl = re.sub(r"\s+", " ", re.sub(r"\?{2,}|\ufffd+", " ", _plain(p))).strip()
    if len(pl.split()) > 16:
        return None
    if "/" in pl[:30] and (m := _EDITOR_CODE_RE.match(pl)):
        rest = pl[m.end():].strip()
        return rest if rest and _TAIL_OUTCOME_RE.match(rest) else (rest or "")
    m = re.match(r"^(?:[A-Z]\s?\.\s?){2,5}(?=[A-Z][a-z])", pl)       # "H . B . T . Petition allowed."
    if m and _TAIL_OUTCOME_RE.match(pl[m.end():].strip()):
        return pl[m.end():].strip()
    return None


# =====================================================================================
# chunk record
# =====================================================================================
@dataclass
class CaseChunk:
    doc_id: str
    chunk_id: str
    parent_id: str
    unit_id: str
    unit_type: str                   # caption | headnote | headnote_part | judgment | judgment_part | counsel | footnotes
    is_parent: bool
    retrievable: bool
    chunk_index: int
    case_key: str
    citation: str | None
    part: str                        # front | judgment
    section: str                     # primary label of the chunk
    labels: list[str]
    headnote_letter: str | None
    headnote_topic: str | None
    opinion_author: str | None
    paragraph_start: int
    paragraph_end: int
    source_start: int
    source_end: int
    token_count: int
    canonical_text: str
    context_header: str
    lead_in: str | None
    embedding_text: str
    bm25_text: str
    references: dict
    flags: dict
    metadata: dict
    prev_chunk_id: str | None = None
    next_chunk_id: str | None = None

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "CaseChunk":
        return cls(**{k: d.get(k) for k in cls.__dataclass_fields__})


@dataclass
class _Atom:
    start: int
    end: int
    p_start: int
    p_end: int
    labels: list[str]
    tokens: int
    lead_in: str | None = None       # used only if the atom opens a chunk
    mid_paragraph: bool = False
    contains_final_order: bool = False
    relabeled_from: str | None = None


def _partition(tokens: list[int], max_tok: int) -> list[list[int]]:
    """Contiguous partition into the FEWEST groups with every group <= max_tok, then the most
    even one (minimal sum of squared sizes) -- avoids a 390-token chunk next to a 40-token one."""
    n = len(tokens)
    if n == 0:
        return []
    k, cur = 1, 0
    for t in tokens:
        if cur and cur + t > max_tok:
            k, cur = k + 1, 0
        cur += t
    pre = [0]
    for t in tokens:
        pre.append(pre[-1] + t)
    inf = float("inf")
    best = [[inf] * (n + 1) for _ in range(k + 1)]
    cut = [[0] * (n + 1) for _ in range(k + 1)]
    best[0][0] = 0
    for g in range(1, k + 1):
        for i in range(g, n + 1):
            for j in range(i - 1, g - 2, -1):
                size = pre[i] - pre[j]
                if size > max_tok and i - j > 1:
                    break                                 # groups only grow as j decreases
                if best[g - 1][j] == inf:
                    continue
                v = best[g - 1][j] + size * size
                if v < best[g][i]:
                    best[g][i], cut[g][i] = v, j
    if best[k][n] == inf:                                 # cannot happen with capped atoms; fall back to greedy
        groups, cur_g, cur = [], [], 0
        for i, t in enumerate(tokens):
            if cur_g and cur + t > max_tok:
                groups.append(cur_g)
                cur_g, cur = [], 0
            cur_g.append(i)
            cur += t
        return groups + ([cur_g] if cur_g else [])
    groups, i = [], n
    for g in range(k, 0, -1):
        j = cut[g][i]
        groups.insert(0, list(range(j, i)))
        i = j
    return [g for g in groups if g]


_LEAD_IN_END_RE = re.compile(r"(?:[:;]\s*-*|\b(?:as\s+under|as\s+follows|follows|reads|below|hereunder|"
                             r"herein\s*-?\s*below|infra|thus|terms)\s*[:;.,-]*)\s*$", re.I)
_AUTHOR_RE = re.compile(r"^\s*(?P<name>[A-Z][A-Za-z.'\s-]{2,60}?),\s*(?:(?:Actg\.\s*)?C\.\s?J|A\.?C\.?J|J|JJ)\.?\s*-{2,}")


# =====================================================================================
# builder
# =====================================================================================
class _Builder:
    def __init__(self, doc: dict):
        self.doc = doc
        self.doc_id = doc["doc_id"]
        self.text = doc["full_text"]
        self.md = dict(doc.get("metadata") or {})
        self.structure = doc.get("structure") or {}
        self.spans = sorted(self.structure.get("spans") or [], key=lambda s: s.get("paragraph_start", -1))
        self.paragraphs = self.text.split("\n\n")
        self.offsets = paragraph_offsets(self.text)
        self.n = len(self.paragraphs)
        self._validate_spans()
        self.front = segment_front_matter(self.paragraphs, self.spans)
        self.warnings = list(self.front.warnings)
        self.case = self._repair_metadata()
        self.case_key = self.case["case_key"]
        self.chunks: list[CaseChunk] = []
        self.ids: set[str] = set()

    # ---- inputs ----
    def _validate_spans(self):
        for s in self.spans:
            for f in ("type", "paragraph_start", "paragraph_end"):
                if f not in s:
                    raise CaseChunkingError(f"span missing '{f}': {s!r}")
            a, b = s["paragraph_start"], s["paragraph_end"]
            if not (isinstance(a, int) and isinstance(b, int) and 0 <= a <= b < self.n):
                raise CaseChunkingError(f"span range [{a},{b}] invalid for {self.n} paragraphs")
        if "\n\n".join(self.paragraphs) != self.text:
            raise CaseChunkingError("full_text does not round-trip through split('\\n\\n')")

    def _para_label(self) -> dict[int, tuple[str, str]]:
        lab: dict[int, tuple[str, str]] = {}
        for s in self.spans:
            t = s["type"]
            if t in ("semantic_section",):
                label = s.get("label") or "unclassified"
            elif t == "quoted_material":
                label = "quoted_material"
            elif t == "final_order":
                label = "final_order"
            elif t == "paragraph_group":
                label = s.get("label") or "unclassified"
            else:
                label = "unclassified"
            if label not in SECTION_DISPLAY:
                label = "unclassified"
            for i in range(s["paragraph_start"], s["paragraph_end"] + 1):
                lab[i] = (t, label)
        return lab

    # ---- metadata ----
    def _repair_metadata(self) -> dict:
        P, md = self.paragraphs, self.md
        cap = self.front.caption
        cit_raw = md.get("citation") or (P[0] if cap else None)
        m = _CITATION_RE.search(_plain(cit_raw or ""))
        cit_norm = normalize_citation(m) if m else None
        case_key = _slug(cit_norm) if cit_norm else self.doc_id
        bracket = next((_COURT_LINE_RE.match(P[i]).group("court") for i in cap if _COURT_LINE_RE.match(P[i])), None)
        court, location = court_name(bracket, cit_norm)
        bench_i = next((i for i in cap if _BENCH_RE.match(P[i])), None)
        decided_i = next((i for i in cap if _DECIDED_RE.search(P[i])), None)
        judges = [re.sub(r"\s+", " ", _plain(j)).strip() for j in (md.get("judges") or [])]
        if not judges and bench_i is not None:
            b = re.sub(r"^\s*(?:Before|Present|Coram)\s*:?\s*", "", _plain(P[bench_i]))
            b = re.sub(r",?\s*(?:(?:Actg\.\s*)?C\.\s?J\.?|JJ?\.?)(?=\s|,|$)", "", b)
            judges = [j.strip(" ,.") for j in re.split(r",|\band\b", b) if j.strip(" ,.")]
        # parties
        lo = (bench_i + 1) if bench_i is not None else (1 + (1 if bracket else 0))
        hi = decided_i if decided_i is not None else (cap[-1] + 1 if cap else 0)
        party_ps = [i for i in range(lo, hi) if not _COURT_LINE_RE.match(P[i])]
        vs = next((i for i in party_ps if _VERSUS_RE.match(P[i])), None)
        def side(idx: list[int]) -> list[tuple[str, str | None]]:
            out, buf = [], ""
            for i in idx:                                 # "THE JUDGE FAMILY COURT, MURREE and" + "another--Respondents"
                buf = (buf + " " + P[i]).strip()
                if re.search(r"(?:-|\s)(?:Petitioner|Respondent|Appellant|Applicant|Plaintiff|Defendant|Complainant|"
                             r"Accused|Appellee|Opposite)[a-z]*\.?\s*$", _plain(buf), re.I) or \
                        not re.search(r"\b(?:and|&|,)\s*$", buf):
                    out.append(_party_name(buf))
                    buf = ""
            if buf:
                out.append(_party_name(buf))
            return out
        petitioners = side([i for i in party_ps if vs is not None and i < vs])
        respondents = side([i for i in party_ps if vs is not None and i > vs])
        title = md.get("case_title")
        title_ok = bool(title) and " vs " in title and not re.search(r"-\s*(?:vs|$)", title)
        if petitioners and respondents:
            title = f"{petitioners[0][0]} v. {respondents[0][0]}"
        elif not title_ok:
            self.warnings.append("caption: parties not found; case_title kept from metadata")
        case_number = decision_date = date_kind = None
        if decided_i is not None:
            cn = re.split(r",?\s*(?:decided|heard)\s+on\b", _plain(P[decided_i]), flags=re.I)[0]
            case_number = re.sub(r"\s+", " ", cn).strip(" ,.")
            dm = _DECIDED_RE.search(P[decided_i])
            decision_date = parse_date(P[decided_i][dm.end():])
            date_kind = "heard_on" if dm.group(0).lower().startswith("heard") else "decided_on"
        if decision_date is None:
            self.warnings.append("caption: decision date not parsed")
        hearing = next((P[i] for i in self.front.counsel if re.match(r"^\s*Dates?\s+of\s+hearing", P[i], re.I)), None)
        counsel = [re.sub(r"\s+", " ", _plain(P[i])).strip() for i in self.front.counsel
                   if not re.match(r"^\s*Dates?\s+of\s+hearing", P[i], re.I)]
        return {
            "doc_id": self.doc_id, "case_key": case_key, "citation": cit_norm or cit_raw,
            "citation_as_printed": cit_raw, "court": court or md.get("court"),
            "court_location": md.get("court_location") or location, "bench": judges and len(judges),
            "judges": judges, "case_title": title, "case_number": case_number, "decision_date": decision_date,
            "decision_date_source": date_kind,
            "hearing_date": parse_date(hearing) if hearing else None,
            "petitioners": [{"name": n, "role": r} for n, r in petitioners],
            "respondents": [{"name": n, "role": r} for n, r in respondents],
            "counsel": counsel, "primary_domain": md.get("primary_domain"),
            "disposition": md.get("disposition"),
            "original_metadata": {k: md.get(k) for k in ("case_title", "court", "decision_date", "case_number",
                                                          "statutes_cited", "provisions_cited", "disposition")},
        }

    # ---- helpers ----
    def _uid(self, base: str) -> str:
        cid, k = base, 2
        while cid in self.ids:
            cid, k = f"{base}_{k}", k + 1
        self.ids.add(cid)
        return cid

    def _header(self, part: str) -> str:
        c = self.case
        seg = [c.get("citation") or self.doc_id]
        if c.get("court"):
            seg.append(c["court"])
        pet = c["petitioners"][0]["name"] if c["petitioners"] else None
        res = c["respondents"][0]["name"] if c["respondents"] else None
        if pet and res:
            seg.append(f"{_short(pet)} v. {_short(res)}")
        elif c.get("case_title"):
            seg.append(_short(c["case_title"], 2 * HEADER_TITLE_MAX_WORDS))
        seg.append(part)
        return "[" + " | ".join(seg) + "]"

    def _inherit(self) -> dict:
        return {k: self.case.get(k) for k in INHERITABLE_METADATA_FIELDS}

    def _p(self, i: int) -> tuple[int, int]:
        return self.offsets[i]

    def _make(self, *, unit_id: str, unit_type: str, part: str, section: str, labels: list[str], start: int,
              end: int, p_start: int, p_end: int, header_part: str, retrievable: bool = True,
              body: str | None = None, bm25_extra: str | None = None, lead_in: str | None = None,
              hn: Headnote | None = None, author: str | None = None, refs: dict | None = None,
              flags: dict | None = None) -> CaseChunk:
        canonical = self.text[start:end]
        header = self._header(header_part)
        body = normalize(canonical) if body is None else body
        lead = normalize(lead_in) if lead_in else None
        if lead and not re.search(r"[.?!…\"']$", lead):
            lead += " …"
        emb = header + "\n" + (f"{lead}\n" if lead else "") + body
        bm25 = emb + ("\n" + normalize(bm25_extra) if bm25_extra else "")
        if refs is None:
            r = extract_statute_refs(canonical)
            refs = {"cases": extract_case_citations(canonical, self.case_key), **r.to_dict()}
        cid = self._uid(f"{self.doc_id}:{len(self.chunks):04d}")
        return CaseChunk(
            doc_id=self.doc_id, chunk_id=cid, parent_id=self.doc_id, unit_id=self._uid(unit_id),
            unit_type=unit_type, is_parent=False, retrievable=retrievable, chunk_index=len(self.chunks),
            case_key=self.case_key, citation=self.case.get("citation"), part=part, section=section,
            labels=labels, headnote_letter=hn.letter if hn else None, headnote_topic=hn.topic if hn else None,
            opinion_author=author, paragraph_start=p_start, paragraph_end=p_end, source_start=start,
            source_end=end, token_count=count_tokens(emb), canonical_text=canonical, context_header=header,
            lead_in=lead, embedding_text=emb, bm25_text=bm25, references=refs, flags=flags or {},
            metadata=self._inherit())

    # ---- front matter ----
    def caption_chunks(self):
        f = self.front
        cap = f.caption
        counsel_text = "\n".join(self.paragraphs[i] for i in f.counsel)
        if not cap:
            return
        if f.headnotes or not f.counsel:
            s, e = self._p(cap[0])[0], self._p(cap[-1])[1]
            self.chunks.append(self._make(
                unit_id=f"{self.case_key}:caption", unit_type="caption", part="front", section="caption",
                labels=["caption"], start=s, end=e, p_start=cap[0], p_end=cap[-1], header_part="Caption",
                body=normalize(self.text[s:e] + ("\n" + counsel_text if counsel_text else ""))))
            if f.counsel:
                s, e = self._p(f.counsel[0])[0], self._p(f.counsel[-1])[1]
                self._pending_counsel = (s, e)
        else:                                             # caption + counsel are contiguous: one chunk
            last = f.counsel[-1]
            s, e = self._p(cap[0])[0], self._p(last)[1]
            self.chunks.append(self._make(
                unit_id=f"{self.case_key}:caption", unit_type="caption", part="front", section="caption",
                labels=["caption"], start=s, end=e, p_start=cap[0], p_end=last, header_part="Caption"))

    def counsel_chunk(self):
        f = self.front
        if not (f.headnotes and f.counsel):
            return
        s, e = self._p(f.counsel[0])[0], self._p(f.counsel[-1])[1]
        self.chunks.append(self._make(
            unit_id=f"{self.case_key}:counsel", unit_type="counsel", part="front", section="counsel",
            labels=["counsel"], start=s, end=e, p_start=f.counsel[0], p_end=f.counsel[-1],
            header_part="Counsel", retrievable=False))

    def headnote_chunks(self):
        for k, hn in enumerate(self.front.headnotes):
            letter = hn.letter
            topic_par = hn.paragraphs[0]
            tm = _ACT_ROMAN_RE.search(_plain(hn.topic)) or _ACT_YEAR_RE.search(_plain(hn.topic))
            if tm:
                topic_act = (tm.group("name"), int(tm.group("year")))
            elif re.match(r"^\s*Constitution\b", hn.topic):
                topic_act = ("__constitution__", None)
            else:
                topic_act = None
            for i in hn.citelist_paragraphs:
                hn.cited_cases += parse_cited_list(self.paragraphs[i], self.case_key)
            # atoms: topic paragraph, body split at "---" separators (sentences if a segment is long)
            atoms: list[_Atom] = []
            for i in hn.paragraphs:
                s0, e0 = self._p(i)
                if i in hn.citelist_paragraphs:
                    atoms.append(_Atom(s0, e0, i, i, ["citelist"], 0))
                    continue
                if i == topic_par and i not in hn.body_paragraphs:
                    atoms.append(_Atom(s0, e0, i, i, ["topic"], count_tokens(normalize(self.text[s0:e0]))))
                    continue
                para = self.text[s0:e0]
                seps = [m.start() for m in re.finditer(rf"{_ANY_DASH}{{2,}}", para) if m.start() > 0]
                cuts = [0] + seps + [len(para)]
                for a, b in zip(cuts, cuts[1:]):
                    seg = para[a:b]
                    if not seg.strip(" -" + _DASH_CHARS):
                        continue
                    if count_tokens(normalize(seg)) > MAX_CHUNK_TOKENS // 2:
                        for x, y in sentence_spans(seg, s0 + a):
                            atoms.append(_Atom(x, y, i, i, ["body"], count_tokens(normalize(self.text[x:y])),
                                               mid_paragraph=True))
                    else:
                        atoms.append(_Atom(s0 + a, s0 + b, i, i, ["body"], count_tokens(normalize(seg)),
                                           mid_paragraph=True))
            groups = _partition([a.tokens for a in atoms], MAX_CHUNK_TOKENS)
            lead = None
            if len(groups) > 1:
                first_body = _plain(self.paragraphs[hn.body_paragraphs[0]]) if hn.body_paragraphs else ""
                seg1 = re.split(r"-{2,}", first_body.strip(" -"))[0] if first_body else ""
                lead = _truncate_tokens(f"{hn.topic} --- {seg1}".strip(" -"), LEAD_IN_MAX_TOKENS)
            for g_i, g in enumerate(groups):
                ga = [atoms[j] for j in g]
                s, e = ga[0].start, ga[-1].end
                body_raw, prev_end = "", None
                for a in ga:
                    if a.labels == ["citelist"]:
                        continue
                    if prev_end is not None:
                        body_raw += self.text[prev_end:a.start] if prev_end == a.start or \
                            not self.text[prev_end:a.start].strip() else "\n\n"
                    body_raw += self.text[a.start:a.end]
                    prev_end = a.end
                cite_parts = [self.text[a.start:a.end] for a in ga if a.labels == ["citelist"]]
                body = normalize(body_raw) if body_raw else ""
                part = f"Headnote ({letter})" if letter else "Headnote"
                if len(groups) > 1:
                    part += f" part {g_i + 1}/{len(groups)}"
                unit = f"{self.case_key}:hn:{letter or k + 1}" + (f":part:{g_i + 1}" if len(groups) > 1 else "")
                r = extract_statute_refs(self.text[s:e])
                if hn.body_paragraphs:
                    topic_provisions(r, topic_act, self.paragraphs[hn.body_paragraphs[0]])
                cited = [c for c in hn.cited_cases if any(c["citation"] == normalize_citation(m)
                                                          for m in _CITATION_RE.finditer(_plain(" ".join(cite_parts))))]
                others = [c for c in extract_case_citations(self.text[s:e], self.case_key)
                          if c["case_key"] not in {x["case_key"] for x in cited}]
                refs = {"cases": cited + others, **r.to_dict()}
                self.chunks.append(self._make(
                    unit_id=unit, unit_type="headnote_part" if len(groups) > 1 else "headnote", part="front",
                    section="headnotes", labels=["headnotes"], start=s, end=e, p_start=ga[0].p_start,
                    p_end=ga[-1].p_end, header_part=part, body=body or normalize(hn.topic),
                    bm25_extra="\n".join(cite_parts) or None, lead_in=lead if g_i > 0 else None, hn=hn,
                    refs=refs, flags={"has_cited_cases": bool(cite_parts)}))

    # ---- judgment body ----
    def body_chunks(self):
        P = self.paragraphs
        lab = self._para_label()
        markers = set(self.front.markers)
        body = [i for i in range(self.front.end, self.n) if i not in markers]
        if not body:
            return
        # reporter line + editorial footnotes after it
        tail_i = next((i for i in reversed(body[-4:]) if reporter_tail(P[i]) is not None), None)
        footnotes = []
        if tail_i is not None:
            footnotes = [i for i in body if i > tail_i and re.match(r"^\s*\d{1,2}\s+\S", P[i])]
        body = [i for i in body if i not in footnotes]
        # labels, fixes, authors
        labels: dict[int, str] = {}
        relabeled: dict[int, str] = {}
        final_long: set[int] = set()
        own_num = None
        for i in body:
            t, l = lab.get(i, ("paragraph_group", "unclassified"))
            pl = _plain(P[i]).lstrip()
            num = re.match(r"^(\d{1,3})\s*(?:&\s*\d{1,3})?\.\s", pl)
            if l == "quoted_material" and num and own_num is not None and int(num.group(1)) == own_num + 1:
                relabeled[i] = l
                l = "court_reasoning"             # the court's own numbered paragraph, not a quotation
            if l == "final_order" and len(P[i].split()) > 60 and len(sentence_spans(P[i])) >= 3:
                relabeled[i] = l
                final_long.add(i)
                l = "court_reasoning"             # reasoning paragraph that ends with the order
            if num and not pl.startswith('"') and l != "quoted_material":
                own_num = int(num.group(1))
            labels[i] = l
        if tail_i is not None:
            labels[tail_i] = "final_order"
        authors: dict[int, str | None] = {}
        cur_author = None
        for i in body:
            m = _AUTHOR_RE.match(_plain(P[i]))
            if m:
                cur_author = re.sub(r"\s+", " ", m.group("name")).strip().title()
            authors[i] = cur_author
        # bonded units: lead-in + quotation, multi-paragraph quotations
        units: list[list[int]] = []
        quote_open = False
        labelled = any(labels[i] not in ("unclassified", "final_order") for i in body)
        for idx, i in enumerate(body):
            pl = _plain(P[i]).strip()
            prev = body[idx - 1] if idx else None
            bond = False
            if prev is not None and prev + 1 == i:          # adjacent, no marker in between
                pprev = _plain(P[prev]).strip()
                if _LEAD_IN_END_RE.search(pprev) and (labels[i] == "quoted_material" or pl[:1] in "\"'(" or
                                                     re.match(r"^(?:\d{1,2}|[ivx]{1,4}|[a-z])[-.)]\s", pl)):
                    bond = True                             # "... reads as under:--" + the quotation
                elif quote_open and (not labelled or labels[i] == "quoted_material"):
                    bond = True                             # still inside an open quotation
                elif labels[i] == "quoted_material" and labels[prev] == "quoted_material":
                    bond = True
            if bond and units and len(units[-1]) < 8:
                units[-1].append(i)
            else:
                units.append([i])
            # quotation balance (straight quotes after normalisation)
            q = pl.count('"')
            if pl.startswith('"') and not quote_open:
                quote_open = q % 2 == 1
            elif quote_open:
                quote_open = q % 2 == 0
            if len(units[-1]) >= 8:
                quote_open = False
        # runs: consecutive units with the same label (a bonded unit takes its lead-in's label)
        runs: list[dict] = []
        for u in units:
            l = labels[u[0]]
            if runs and runs[-1]["label"] == l and runs[-1]["units"][-1][-1] + 1 >= u[0] - 1 \
                    and not any(j in markers for j in range(runs[-1]["units"][-1][-1] + 1, u[0])):
                runs[-1]["units"].append(u)
            else:
                runs.append({"label": l, "units": [u]})

        def ntok(i: int) -> int:
            return count_tokens(normalize(P[i]))

        for r in runs:
            r["tokens"] = sum(ntok(i) for u in r["units"] for i in u)
        # tiny runs merge into a neighbour (next if possible, else previous)
        merged: list[dict] = []
        carry: list[list[int]] = []
        for k, r in enumerate(runs):
            if carry:
                r = {"label": r["label"], "units": carry + r["units"],
                     "tokens": r["tokens"] + sum(ntok(i) for u in carry for i in u)}
                carry = []
            boundary_after = k + 1 < len(runs) and any(
                j in markers for j in range(r["units"][-1][-1] + 1, runs[k + 1]["units"][0][0]))
            if r["tokens"] < MIN_CHUNK_TOKENS and k + 1 < len(runs) and not boundary_after \
                    and labels[r["units"][0][0]] != "final_order":
                carry = r["units"]
                continue
            if r["tokens"] < MIN_CHUNK_TOKENS and merged and not any(
                    j in markers for j in range(merged[-1]["units"][-1][-1] + 1, r["units"][0][0])):
                merged[-1]["units"] += r["units"]
                merged[-1]["tokens"] += r["tokens"]
                continue
            merged.append(r)
        if carry:
            if merged:
                merged[-1]["units"] += carry
            else:
                merged.append({"label": labels[carry[0][0]], "units": carry, "tokens": 0})
        for r in merged:
            atoms: list[_Atom] = []
            for u in r["units"]:
                utok = sum(ntok(i) for i in u)
                if utok <= MAX_CHUNK_TOKENS:
                    s, e = self._p(u[0])[0], self._p(u[-1])[1]
                    atoms.append(_Atom(s, e, u[0], u[-1], [labels[i] for i in u], utok,
                                       contains_final_order=any(i in final_long for i in u),
                                       relabeled_from=next((relabeled[i] for i in u if i in relabeled), None)))
                    continue
                lead_par = u[0] if len(u) > 1 and _LEAD_IN_END_RE.search(_plain(P[u[0]]).strip()) else None
                for i in u:
                    s0, e0 = self._p(i)
                    lead = (_truncate_tokens(normalize(P[lead_par]), LEAD_IN_MAX_TOKENS, from_end=True)
                            if lead_par is not None and i != lead_par else None)
                    alabels = [labels[lead_par], labels[i]] if lead is not None and labels[lead_par] != labels[i] \
                        else [labels[i]]
                    if ntok(i) <= MAX_CHUNK_TOKENS:
                        atoms.append(_Atom(s0, e0, i, i, alabels, ntok(i), lead_in=lead,
                                           contains_final_order=i in final_long, relabeled_from=relabeled.get(i)))
                        continue
                    sents = sentence_spans(P[i], s0)
                    sents = self._cap_sentences(sents)
                    for si, (x, y) in enumerate(sents):
                        sl = lead
                        if si > 0:
                            px, py = sents[si - 1]
                            sl = _truncate_tokens(normalize(self.text[px:py]), LEAD_IN_MAX_TOKENS, from_end=True)
                        atoms.append(_Atom(x, y, i, i, alabels, count_tokens(normalize(self.text[x:y])),
                                           lead_in=sl, mid_paragraph=True, contains_final_order=i in final_long,
                                           relabeled_from=relabeled.get(i)))
            budget = MAX_CHUNK_TOKENS - (LEAD_IN_MAX_TOKENS if any(a.lead_in for a in atoms) else 0)
            groups = _partition([a.tokens for a in atoms], max(budget, MAX_CHUNK_TOKENS // 2))
            gt = lambda g: sum(atoms[j].tokens for j in g)          # noqa: E731
            k = 0
            while k < len(groups) and len(groups) > 1:        # a tiny group joins a neighbour if it fits
                if gt(groups[k]) < MIN_CHUNK_TOKENS:
                    nb = [x for x in (k - 1, k + 1) if 0 <= x < len(groups)
                          and gt(groups[x]) + gt(groups[k]) <= MAX_CHUNK_TOKENS + LEAD_IN_MAX_TOKENS // 2]
                    if nb:
                        x = min(nb, key=lambda x: gt(groups[x]))
                        a, b = sorted((k, x))
                        groups[a:b + 1] = [groups[a] + groups[b]]
                        continue
                k += 1
            for g in groups:
                ga = [atoms[j] for j in g]
                tok_by: dict[str, int] = {}
                for a in ga:
                    for l in a.labels:
                        tok_by[l] = tok_by.get(l, 0) + a.tokens // max(1, len(a.labels))
                labs = list(dict.fromkeys(l for a in ga for l in a.labels))
                main = [l for l in labs if l != "quoted_material"] or labs
                section = max(main, key=lambda l: (tok_by.get(l, 0), -labs.index(l)))
                if r["label"] in main and tok_by.get(r["label"], 0) * 2 >= sum(tok_by[x] for x in main):
                    section = r["label"]
                mid = ga[0].mid_paragraph and (ga[0].start != self._p(ga[0].p_start)[0]) or \
                    ga[-1].mid_paragraph and (ga[-1].end != self._p(ga[-1].p_end)[1])
                ps, pe = ga[0].p_start, ga[-1].p_end
                base = f"{self.case_key}:p:tmp{len(self.chunks)}"            # renamed in _name_units()
                hp = SECTION_DISPLAY.get(section, "Judgment")
                author = authors.get(ps)
                flags = {"split_paragraph": bool(mid),
                         "contains_final_order": any(a.contains_final_order for a in ga) or "final_order" in labs,
                         "relabeled_from": next((a.relabeled_from for a in ga if a.relabeled_from), None),
                         "reporter_note": tail_i is not None and ps <= tail_i <= pe}
                body_txt = normalize(self.text[ga[0].start:ga[-1].end])
                if tail_i is not None and ps <= tail_i <= pe:
                    body_txt = self._strip_editor_code(body_txt, P[tail_i])
                self.chunks.append(self._make(
                    unit_id=base, unit_type="judgment_part" if mid else "judgment", part="judgment",
                    section=section, labels=labs, start=ga[0].start, end=ga[-1].end, p_start=ps, p_end=pe,
                    header_part=hp, body=body_txt, lead_in=ga[0].lead_in, author=author, flags=flags))
        if footnotes:
            s, e = self._p(footnotes[0])[0], self._p(footnotes[-1])[1]
            self.chunks.append(self._make(
                unit_id=f"{self.case_key}:footnotes", unit_type="footnotes", part="judgment", section="footnotes",
                labels=["footnotes"], start=s, end=e, p_start=footnotes[0], p_end=footnotes[-1],
                header_part="Footnotes", retrievable=False))
        self.tail_i = tail_i
        self.labels = labels
        self.final_long = final_long

    def _cap_sentences(self, sents: list[tuple[int, int]]) -> list[tuple[int, int]]:
        """A 'sentence' above the budget (a long quotation without full stops) is cut at ';' / ',' / words."""
        out = []
        for x, y in sents:
            if count_tokens(normalize(self.text[x:y])) <= MAX_CHUNK_TOKENS - LEAD_IN_MAX_TOKENS:
                out.append((x, y))
                continue
            seg = self.text[x:y]
            pieces, cur = [], 0
            for m in re.finditer(r"[;:]\s+|,\s+(?=and\b|but\b|whereas\b|which\b)", seg):
                pieces.append((cur, m.end()))
                cur = m.end()
            pieces.append((cur, len(seg)))
            buf_s = None
            for a, b in pieces:
                if buf_s is None:
                    buf_s = a
                if count_tokens(normalize(seg[buf_s:b])) > (MAX_CHUNK_TOKENS - LEAD_IN_MAX_TOKENS) // 2:
                    out.append((x + buf_s, x + len(seg[:b].rstrip())))
                    buf_s = None
            if buf_s is not None:
                out.append((x + buf_s, y))
        final = []                                        # last resort: word windows
        for a, b in out:
            if count_tokens(normalize(self.text[a:b])) <= MAX_CHUNK_TOKENS - LEAD_IN_MAX_TOKENS:
                final.append((a, b))
                continue
            words = list(re.finditer(r"\S+", self.text[a:b]))
            st = 0
            while st < len(words):
                en = st
                while en + 1 < len(words) and count_tokens(
                        self.text[a + words[st].start():a + words[en + 1].end()]) <= (MAX_CHUNK_TOKENS - LEAD_IN_MAX_TOKENS) // 2:
                    en += 1
                final.append((a + words[st].start(), a + words[en].end()))
                st = en + 1
        return final

    @staticmethod
    def _strip_editor_code(body: str, tail_par: str) -> str:
        t = normalize(tail_par)
        rest = reporter_tail(tail_par)
        if rest is None:
            return body
        return body.replace(t, rest).rstrip() if rest else body.replace(t, "").rstrip()

    # ---- final orders ----
    def final_orders(self) -> list[dict]:
        P = self.paragraphs
        out = []
        body = [i for i in range(self.front.end, self.n) if i not in self.front.markers and i in self.labels]
        last2 = [i for i in body if i != self.tail_i][-2:]
        cand = sorted({i for i in body if self.labels.get(i) == "final_order" or i in self.final_long} | set(last2))
        for i in cand:
            if i == self.tail_i:
                continue
            for x, y in sentence_spans(P[i]):
                s = P[i][x:y]
                if (_OPERATIVE_RE.search(_plain(s)) or _ACTIVE_RE.search(_plain(s))) and not re.search(r"\b(?:prayed|prays|praying|sought|seeking|"
                                                                      r"contended|submitted|argued)\b", s, re.I):
                    out.append({"paragraph": i, "sentence": s.strip(), "label": outcome_label(s),
                                "labels": outcome_labels(s), "source": "judgment",
                                "disposes_case": bool(_DISPOSES_RE.search(_plain(s)))})
        if self.tail_i is not None:
            rest = reporter_tail(P[self.tail_i])
            if rest:
                out.append({"paragraph": self.tail_i, "sentence": rest, "label": outcome_label(rest),
                            "labels": outcome_labels(rest), "source": "reporter_note", "disposes_case": True})
        return out

    # ---- build ----
    def build(self) -> list[CaseChunk]:
        self.caption_chunks()
        self.headnote_chunks()
        self.counsel_chunk()
        self.body_chunks()
        self.chunks.sort(key=lambda c: c.source_start)
        self._name_units()
        for k, c in enumerate(self.chunks):
            c.chunk_index = k
            c.chunk_id = f"{self.doc_id}:{k:04d}"
        retr = [c for c in self.chunks if c.retrievable]
        for k, c in enumerate(retr):
            c.prev_chunk_id = retr[k - 1].chunk_id if k else None
            c.next_chunk_id = retr[k + 1].chunk_id if k + 1 < len(retr) else None
        # case-level aggregates
        fo = self.final_orders()
        self.case["final_orders"] = fo
        tail = next((f for f in fo if f["source"] == "reporter_note" and f["label"]
                     and f["label"] != "order_accordingly"), None)
        judged = [f for f in fo if f["source"] == "judgment" and f["label"] and f["disposes_case"]]
        last = judged[-1] if judged else None
        if tail and (not judged or any(tail["label"] in f["labels"] for f in judged)):
            pick = tail                                   # reporter line confirmed by the judgment text
        else:
            pick = last or tail
            if tail and last and (tail["label"] == "dismissed") != (last["label"] == "dismissed"):
                self.warnings.append(f"reporter line '{tail['sentence']}' disagrees with the judgment "
                                     f"('{last['label']}'); judgment used")
        if pick:
            self.case["disposition"] = pick["label"]
            self.case["disposition_source"] = pick["source"]
        jl = {lab for f in judged for lab in f["labels"]}
        if {"allowed", "dismissed"} <= jl:                 # "to that extent accepted" + "to that extent dismissed"
            self.case["disposition"] = "partly_allowed"
            self.case["disposition_source"] = "judgment"

        stat: dict[str, dict] = {}
        arts: list[str] = []
        cases: dict[str, dict] = {}
        for c in self.chunks:
            for s in c.references.get("statutes", []):
                e = stat.setdefault(s["act_key"], {"name": s["name"], "year": s["year"], "act_key": s["act_key"],
                                                   "provisions": [], "unit_ids": []})
                e["provisions"] += [p for p in s["provisions"] if p not in e["provisions"]]
                e["unit_ids"] += [u for u in s["unit_ids"] if u not in e["unit_ids"]]
            arts += [a for a in c.references.get("constitution_articles", []) if a not in arts]
            for x in c.references.get("cases", []):
                cur = cases.setdefault(x["case_key"], dict(x))
                for k in ("name", "treatment"):
                    if x.get(k) and not cur.get(k):
                        cur[k] = x[k]
        self.case["statutes"] = list(stat.values())
        self.case["statutes_cited"] = [f"{s['name']}, {s['year']}" if s["year"] else s["name"] for s in stat.values()]
        self.case["constitution_articles"] = arts
        self.case["provisions"] = [u for s in stat.values() for u in s["unit_ids"]] + [f"CONST:art:{a}" for a in arts]
        self.case["cited_cases"] = list(cases.values())
        self.case["headnotes"] = [{"letter": h.letter, "topic": h.topic, "paragraphs": h.paragraphs}
                                  for h in self.front.headnotes]
        if len(self.case["judges"]) == 1:                  # single bench: the judge wrote the judgment
            for c in self.chunks:
                if c.part == "judgment" and not c.opinion_author:
                    c.opinion_author = self.case["judges"][0]
        self.case["opinion_authors"] = list(dict.fromkeys(c.opinion_author for c in self.chunks if c.opinion_author))
        inh = self._inherit()
        for c in self.chunks:
            c.metadata = dict(inh)
        return self.chunks

    def _name_units(self):
        """Judgment unit ids from paragraph indices: 'p:13-15'; a paragraph cut into pieces is
        numbered 'p:35.1', 'p:35.2', and a chunk starting in piece 3 of paragraph 35 and ending
        with paragraph 37 is 'p:35.3-37'."""
        body = [c for c in self.chunks if c.unit_type.startswith("judgment")]
        starts_in: dict[int, list[int]] = {}
        for c in body:
            starts_in.setdefault(c.paragraph_start, []).append(c.source_start)
            if c.paragraph_end != c.paragraph_start:
                starts_in.setdefault(c.paragraph_end, [])
        for c in body:
            if c.source_end != self._p(c.paragraph_end)[1]:
                starts_in.setdefault(c.paragraph_end, []).append(self._p(c.paragraph_end)[0])

        def piece(i: int, pos: int) -> str:
            cuts = sorted(set(starts_in.get(i, [])) | {self._p(i)[0]})
            if len(cuts) == 1 and all(c.source_end == self._p(c.paragraph_end)[1] or c.paragraph_end != i
                                      for c in body):
                return str(i)
            k = sum(1 for x in cuts if x <= pos)
            return f"{i}.{k}" if len(cuts) > 1 else str(i)

        for c in body:
            a = piece(c.paragraph_start, c.source_start)
            b = piece(c.paragraph_end, c.source_end - 1)
            self.ids.discard(c.unit_id)
            c.unit_id = self._uid(f"{self.case_key}:p:{a}" + (f"-{b}" if b != a else ""))

    def sections_outline(self) -> list[dict]:
        out: list[dict] = []
        for c in self.chunks:
            if not c.retrievable:
                continue
            if out and out[-1]["section"] == c.section and out[-1]["part"] == c.part:
                out[-1]["paragraph_end"] = c.paragraph_end
                out[-1]["chunk_ids"].append(c.chunk_id)
            else:
                out.append({"part": c.part, "section": c.section, "paragraph_start": c.paragraph_start,
                            "paragraph_end": c.paragraph_end, "chunk_ids": [c.chunk_id]})
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


def validate_chunks(chunks: list[CaseChunk], full_text: str, marker_paragraphs: list[int] | None = None,
                    case: dict | None = None, front_warnings: list[str] | None = None) -> ChunkValidationResult:
    errors: list[str] = []
    warnings: list[str] = list(front_warnings or [])
    offsets = paragraph_offsets(full_text)
    ids = [c.chunk_id for c in chunks]
    if len(set(ids)) != len(ids):
        errors.append("duplicate chunk_id")
    units = [c.unit_id for c in chunks]
    if len(set(units)) != len(units):
        errors.append(f"duplicate unit_id: {sorted({u for u in units if units.count(u) > 1})[:5]}")
    cov = bytearray(len(full_text))
    rcov = bytearray(len(full_text))
    for c in chunks:
        if not (0 <= c.source_start < c.source_end <= len(full_text)):
            errors.append(f"{c.chunk_id}: bad offsets")
            continue
        if full_text[c.source_start:c.source_end] != c.canonical_text:
            errors.append(f"{c.chunk_id}: canonical_text != full_text[source_start:source_end]")
        if not c.embedding_text[len(c.context_header):].strip():
            errors.append(f"{c.chunk_id}: empty body")
        if c.retrievable and c.token_count > EMBED_TOKEN_LIMIT:
            errors.append(f"{c.chunk_id}: {c.token_count} tokens > embedding limit {EMBED_TOKEN_LIMIT}")
        if c.unit_type in ("counsel", "footnotes") and c.retrievable:
            errors.append(f"{c.chunk_id}: {c.unit_type} must not be retrievable")
        if re.search(rf"{_ANY_DASH}{{2,}}|\?{{3,}}", c.embedding_text[len(c.context_header):]):
            errors.append(f"{c.chunk_id}: un-normalized dashes / junk in embedding_text")
        seg = cov[c.source_start:c.source_end]
        if any(seg[i] and not full_text[c.source_start + i].isspace() for i in range(len(seg))):
            errors.append(f"{c.chunk_id}: overlaps another chunk")
        cov[c.source_start:c.source_end] = b"\x01" * (c.source_end - c.source_start)
        if c.retrievable:
            rcov[c.source_start:c.source_end] = b"\x01" * (c.source_end - c.source_start)
    for i in marker_paragraphs or []:
        a, b = offsets[i]
        cov[a:b] = b"\x01" * (b - a)
    lost = [i for i, ch in enumerate(full_text) if not cov[i] and not ch.isspace()]
    if lost:
        errors.append(f"{len(lost)} non-space characters not in any chunk, e.g. {full_text[lost[0]:lost[0] + 60]!r}")
    nonidx = [c for c in chunks if not c.retrievable]
    for c in nonidx:
        rcov[c.source_start:c.source_end] = b"\x01" * (c.source_end - c.source_start)
    for i in marker_paragraphs or []:
        a, b = offsets[i]
        rcov[a:b] = b"\x01" * (b - a)
    miss = [i for i, ch in enumerate(full_text) if not rcov[i] and not ch.isspace()]
    if miss:
        errors.append(f"{len(miss)} characters of legal text not in a retrievable chunk")
    # order / links
    starts = [c.source_start for c in chunks]
    if starts != sorted(starts):
        errors.append("chunks are not in document order")
    retr = [c for c in chunks if c.retrievable]
    for k, c in enumerate(retr):
        if c.prev_chunk_id != (retr[k - 1].chunk_id if k else None) or \
                c.next_chunk_id != (retr[k + 1].chunk_id if k + 1 < len(retr) else None):
            errors.append(f"{c.chunk_id}: prev/next links broken")
            break
    # warnings
    if case is not None:
        if not case.get("decision_date"):
            warnings.append("decision_date not found")
        if not case.get("case_number"):
            warnings.append("case_number not found")
        if not case.get("court"):
            warnings.append("court not found")
        if not case.get("final_orders"):
            warnings.append("no final order sentence found")
    hn = [c for c in chunks if c.unit_type.startswith("headnote")]
    if not hn:
        warnings.append("no headnote chunk (unreported order or headnotes not detected)")
    small = [c.chunk_id for c in retr if c.unit_type.startswith("judgment")
             and c.token_count - count_tokens(c.context_header) < 25]
    if small:
        warnings.append(f"{len(small)} judgment chunks under 25 body tokens: {small[:5]}")
    body = [c for c in retr if c.part == "judgment"]
    uncl = [c for c in body if c.section == "unclassified"]
    if body and len(uncl) == len(body):
        warnings.append("judgment is entirely 'unclassified' (relabel this document)")
    toks = sorted(c.token_count for c in retr)
    diagnostics = {
        "total_chunks": len(chunks), "retrievable_chunks": len(retr),
        "by_unit_type": {t: sum(c.unit_type == t for c in chunks) for t in dict.fromkeys(c.unit_type for c in chunks)},
        "headnotes": len({c.headnote_letter or c.unit_id for c in hn}),
        "sections": list(dict.fromkeys(c.section for c in body)),
        "unclassified_judgment_chunks": len(uncl),
        "split_paragraph_chunks": sum(bool(c.flags.get("split_paragraph")) for c in retr),
        "relabeled_paragraph_chunks": sum(bool(c.flags.get("relabeled_from")) for c in retr),
        "retrievable_token_min_median_max": [toks[0], toks[len(toks) // 2], toks[-1]] if toks else None,
        "embed_token_limit": EMBED_TOKEN_LIMIT,
        "token_counter": "tiktoken cl100k_base" if _ENCODING else "fallback estimator (over-counts)",
    }
    return ChunkValidationResult(ok=not errors, errors=errors, warnings=warnings, diagnostics=diagnostics)


# =====================================================================================
# entry points
# =====================================================================================
def chunk_document(doc: dict) -> dict:
    if "full_text" not in doc or "doc_id" not in doc:
        raise CaseChunkingError("input is not a structured case (no doc_id / full_text)")
    b = _Builder(doc)
    chunks = b.build()
    if not chunks:
        raise CaseChunkingError("no content found to chunk")
    result = validate_chunks(chunks, b.text, b.front.markers, b.case, b.warnings)
    return {
        "doc_id": b.doc_id,
        "case": b.case,
        "metadata": doc.get("metadata", {}),
        "full_text": b.text,
        "markers": b.front.markers,
        "sections": b.sections_outline(),
        "chunks": [c.to_dict() for c in chunks],
        "validation": result.to_dict(),
    }


def is_valid_existing_output(out: dict) -> bool:
    try:
        chunks = [CaseChunk.from_dict(c) for c in out["chunks"]]
        return validate_chunks(chunks, out["full_text"], out.get("markers", [])).ok
    except Exception:  # noqa: BLE001 -- malformed or old-schema output is regenerated
        return False


def chunk_directory(input_dir: Path = INPUT_DIR, output_dir: Path = OUTPUT_DIR, limit: int | None = None,
                    force: bool = False) -> dict:
    """Chunks every <doc_id>.json in input_dir. An existing output is skipped only if it
    re-validates (unless force). Never modifies input_dir."""
    input_dir, output_dir = Path(input_dir), Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    processed = regenerated = skipped_valid = 0
    failed_docs: list[dict] = []
    warn_counts: dict[str, int] = {}
    for n, path in enumerate(sorted(input_dir.glob("*.json"))):
        if limit is not None and n >= limit:
            break
        out_path = output_dir / path.name
        was_present = out_path.exists()
        if was_present and not force:
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
        except (OSError, json.JSONDecodeError, KeyError, CaseChunkingError) as exc:
            failed_docs.append({"doc_id": path.stem, "error": str(exc)})
            continue
        for w in out["validation"]["warnings"]:
            key = re.sub(r"\d+|\[.*?\]|\(.*?\)|:.*$", "", w).strip()
            warn_counts[key] = warn_counts.get(key, 0) + 1
        if not out["validation"]["ok"]:
            failed_docs.append({"doc_id": path.stem, "error": out["validation"]["errors"]})
            continue
        tmp = out_path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
        tmp.replace(out_path)
        processed += 1
        regenerated += was_present
    return {"processed": processed, "skipped_valid": skipped_valid, "regenerated": regenerated,
            "failed": len(failed_docs), "failed_docs": failed_docs, "warning_counts": warn_counts}


def main() -> None:  # pragma: no cover
    ap = argparse.ArgumentParser(description="Chunk structured case-law JSON for hybrid RAG.")
    ap.add_argument("--in", dest="inp", default=str(INPUT_DIR))
    ap.add_argument("--out", default=str(OUTPUT_DIR))
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--force", action="store_true", help="re-chunk even if a valid output exists")
    a = ap.parse_args()
    s = chunk_directory(Path(a.inp), Path(a.out), a.limit, a.force)
    print(f"Case-law chunking: {s['processed']} processed ({s['regenerated']} regenerated), "
          f"{s['skipped_valid']} skipped (valid), {s['failed']} failed")
    for k, v in sorted(s["warning_counts"].items(), key=lambda x: -x[1]):
        print(f"  warning x{v}: {k}")
    for f in s["failed_docs"][:20]:
        print(f"  FAILED {f['doc_id']}: {str(f['error'])[:300]}")


if __name__ == "__main__":  # pragma: no cover
    main()