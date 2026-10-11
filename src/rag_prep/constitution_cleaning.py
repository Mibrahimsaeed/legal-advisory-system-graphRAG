# """
# Constitution of Pakistan Cleaning Pipeline (End-to-End v3)

# Pipeline Flow:
#     Raw ingested JSON (var/rag/constitution_ingested/<doc_id>.json)
#         -> clean_constitution_document()
#         -> quality_gate()
#         -> Cleaned constitution JSON (var/rag/cleaned_constitution/<doc_id>.json)
#            (documents failing the gate go to var/rag/cleaned_constitution/_rejected/)

# Key Capabilities:
# 1. Preamble Anchor Detection: targets the operational Preamble to drop front matter / TOC.
# 2. Page-Aware Footnote Processing: isolates footnote blocks per page and assigns page-scoped
#    references [^<printed_page>.<num>].
# 3. Marginal Note Extraction: re-locates marginal titles as article headings (## <Num>. <Title>).
#    Articles without a marginal note (omitted / repealed ones) get their heading from the TOC.
# 4. Structural Headings: PART / CHAPTER / SCHEDULE / ANNEX lines become '# ' headings
#    ('### ' for PARTs inside a Schedule).
# 5. Table Reconstruction: re-assembles the Art. 51 and Art. 106 seat tables as Markdown tables.
# 6. Paragraph Unwrap & Hyphenation: rejoins wrapped lines into paragraphs (one clause /
#    sub-clause / proviso per line) without breaking legal compound terms.
# 7. Footnote Enrichment: amendment action, amending instrument, 'ibid.' resolution,
#    'See footnote N on page P' cross-references, OCR fixes.
# 8. Verification + Quality Gate: letter-multiset preservation against the raw input, plus
#    structural checks that fail the run instead of silently writing bad output.

# Output text conventions (consumed by the chunker):
#     # PART II Fundamental Rights and Principles of Policy     <- structural heading
#     # CHAPTER 1.–FUNDAMENTAL RIGHTS
#     ## 25A. Right to education.                              <- article heading
#     [^15.2]25A. The State shall provide ...                  <- article body, one paragraph per line
#     [^29.3][three]                                           <- amendment: footnote ref + bracketed text
# """

# from __future__ import annotations

# import json
# import re
# import sys
# from collections import Counter
# from dataclasses import dataclass, field
# from difflib import SequenceMatcher
# from pathlib import Path

# INPUT_DIR = Path("var/rag/constitution_ingested")
# OUTPUT_DIR = Path("var/rag/cleaned_constitution")

# LARGE_REMOVAL_FLAG_THRESHOLD = 0.15
# MAX_LINE_CHARS = 2500                 # quality gate: no single paragraph line longer than this

# # Private-use sentinels for footnote references during processing
# _REF_OPEN, _REF_CLOSE = "", ""
# _REF_SENTINEL_RE = re.compile(f"{_REF_OPEN}([^{_REF_CLOSE}]*){_REF_CLOSE}")
# _REF = f"(?:{_REF_OPEN}[^{_REF_CLOSE}]*{_REF_CLOSE})"          # one sentinel ref, as a regex fragment


# def _ref(fid: str) -> str:
#     return f"{_REF_OPEN}{fid}{_REF_CLOSE}"


# # --------------------------------------------------------------------------
# # Regex Patterns
# # --------------------------------------------------------------------------
# _PREAMBLE_OPERATIONAL_MARKER = "WHEREAS sovereignty over the entire Universe belongs"
# _HEADER_RE = re.compile(r"^[ \t]*(\d{1,3})?[ \t]*CONSTITUTION OF PAKISTAN\d*[ \t]*(\d{1,3})?[ \t]*$")
# _LONE_PAGE_NUMBER_RE = re.compile(r"^[ \t]*(?:\(?[ivxlcdm]{1,6}\)?|\d{1,3})[ \t]*$")
# _PRINTER_LINE_RE = re.compile(
#     r"^[ \t]*(?:\d+\(\d+\)[A-Za-z&]+---by[ \t]+\S+---PC-\d+[ \t]*\([^)\n]*\)"
#     r"|PCPPI\S*.*\(PC-\d+\))[ \t]*$"
# )
# _SEPARATOR_RE = re.compile(r"^[ \t]*[_‐-―-]{15,}(?:[ \t]+[_‐-―-]{10,})?[ \t]*$")
# _DIVIDER_LINE_RE = re.compile(r"^[ \t]*(?:_{3,}(?:[ \t]+_{3,})?|[‐-―]{2,})[ \t]*$")

# _FN_START_RE = re.compile(r"^[ \t]*(\d{1,2})(?:\.(?=[A-Za-z])|[ \t](?=[A-Z])|(?=[A-Za-z‘’“\"']))")
# _ABBREV_END_RE = re.compile(
#     r"\b(?:Amdt|subs|ins|Subs|Ins|s|ss|Art|Arts|No|Nos|Pt|Ext|Gaz|Pak|p|pp|Sch|viz|Ord|O|P|"
#     r"Govt|Vol|Para|Cl|Ch|Notfn|Min|Deptt|Rs|w\.e\.f|i\.e|e\.g)\.$"
# )

# _ARTICLE_HEAD_RE = re.compile(
#     rf"^(?P<ind>[ \t]*)(?P<pre>(?:{_REF}?\[)*{_REF}?)"
#     rf"(?P<num>\d{{1,4}}[A-Z]{{0,3}})\.(?=[ \t]|$|[–—-]|\d{{1,2}}\[|{_REF_OPEN})"
# )

# _KNOWN_HYPHENATED_COMPOUNDS = frozenset({
#     "majlise", "nondiscrimination", "selfincrimination", "subparagraph", "noconfidence",
#     "wellbeing", "hydroelectric", "fullstop", "twentythird",
# })

# _BRACKET_CONTENT_IS_CLAUSE_NUMBER_RE = re.compile(
#     r"^(?:\(\d+[A-Za-z]{0,2}\)|\([a-zA-Z0-9]{1,4}\)|\d+[A-Za-z]{0,2}\.)"
# )

# # Broken small-caps in the PDF text layer, e.g. "GOvErNOr", "ASSEmBly" (capital, lowercase, capital)
# _BROKEN_CAPS_WORD_RE = re.compile(r"\b(?=[A-Za-z]*[A-Z][a-z]+[A-Z])[A-Za-z]{4,}\b")
# _BROKEN_INITIALS = "mrlvy"
# _SMALL_WORDS = {"of", "and", "the", "by", "for", "to", "in", "on", "a", "an", "or", "from", "under", "with"}

# # Omitted / repealed status from the TOC title, e.g. "[Omitted.]", "[Repealed.]", "[Cease to inforce.]"
# _STATUS_RE = re.compile(r"^\[\s*(Omitted|Repealed|Cease)", re.I)

# # Structural lines (work both on sentinel text and on final [^id] text)
# _LEAD_REFS = rf"(?:{_REF}|\[\^[^\]]+\]|\[)*"
# _STRUCT_KW = (r"(?:PART[ \t]+[IVXL]+\b|CHAPTER[ \t]+\d+[A-Z]?[.–—-]"
#               r"|(?:FIRST|SECOND|THIRD|FOURTH|FIFTH|SIXTH|SEVENTH)[ \t]+SCHEDULE\b|ANNEX\b)")
# _STRUCT_LINE_RE = re.compile(rf"^[ \t]*{_LEAD_REFS}{_STRUCT_KW}")
# _SCHEDULE_OR_ANNEX_RE = re.compile(rf"^[ \t]*{_LEAD_REFS}(?:(?:FIRST|SECOND|THIRD|FOURTH|FIFTH|SIXTH|SEVENTH)"
#                                    rf"[ \t]+SCHEDULE\b|ANNEX\b)")
# _PART_RE = re.compile(rf"^[ \t]*{_LEAD_REFS}PART[ \t]+[IVXL]+\b")

# # Short standalone lines that must not be glued to neighbours
# _STANDALONE_LINE_RE = re.compile(
#     rf"^[ \t]*{_LEAD_REFS}(?:"
#     r"[\[(](?:Article|Articles|see|See)\b[^\])]*[\])]\]?"          # [Article 42]  (Article 2A)
#     r"|[A-Z][A-Z0-9 ,.'’()&\-–—]{2,70}\]?"                          # ALL-CAPS short line
#     r")[ \t]*$"
# )

# _PARA_START_RE = re.compile(
#     rf"^(?:[ \t]{{4,}}\S"                                           # indented = new paragraph in this PDF
#     rf"|[ \t]*{_LEAD_REFS}"
#     r"(?:\d{1,4}[A-Z]{0,3}\.(?:\s|$)"                               # article head
#     r"|\((?:\d+[A-Z]{0,2}|[a-z]{1,2}|[ivxl]+)\)"                    # (1) (a) (iv)
#     r"|Provided\b|Explanation\b|\(In the name\b))"
# )


# # --------------------------------------------------------------------------
# # Data Structures
# # --------------------------------------------------------------------------
# @dataclass
# class Footnote:
#     id: str
#     pdf_page: int
#     printed_page: str
#     number: int
#     text: str
#     see_also: str | None = None
#     referenced: bool = False
#     actions: list = field(default_factory=list)
#     amending_instrument: str | None = None
#     amending_year: int | None = None
#     instrument_inherited: bool = False

#     def to_dict(self) -> dict:
#         return {
#             "id": self.id, "pdf_page": self.pdf_page, "printed_page": self.printed_page,
#             "number": self.number, "text": self.text, "see_also": self.see_also,
#             "referenced": self.referenced, "actions": self.actions,
#             "amending_instrument": self.amending_instrument, "amending_year": self.amending_year,
#             "instrument_inherited": self.instrument_inherited,
#         }


# @dataclass
# class Diagnostics:
#     raw_length: int = 0
#     cleaned_length: int = 0
#     pages_processed: int = 0
#     footnotes: int = 0
#     footnote_refs: int = 0
#     unresolved_markers: list = field(default_factory=list)
#     unreferenced_footnotes: list = field(default_factory=list)
#     displaced_body_blocks_restored: list = field(default_factory=list)
#     glued_article_numbers_fixed: list = field(default_factory=list)
#     articles_found: int = 0
#     articles_in_toc: int = 0
#     articles_missing: list = field(default_factory=list)
#     articles_missing_unexpected: list = field(default_factory=list)
#     marginal_notes_attached: int = 0
#     headings_from_toc: list = field(default_factory=list)
#     articles_without_heading: list = field(default_factory=list)
#     structural_headings: int = 0
#     table_repairs: list = field(default_factory=list)
#     broken_caps_pages: list = field(default_factory=list)
#     capitalisation_repairs: int = 0
#     footnote_ocr_fixes: list = field(default_factory=list)
#     footnotes_ibid_resolved: int = 0
#     footnotes_see_also_resolved: int = 0
#     footnotes_without_instrument: int = 0
#     letter_preservation_ok: bool | None = None
#     letter_preservation_diff: dict = field(default_factory=dict)
#     max_line_chars: int = 0
#     quality_gate_errors: list = field(default_factory=list)

#     def to_dict(self) -> dict:
#         d = dict(self.__dict__)
#         d["chars_removed"] = self.raw_length - self.cleaned_length
#         d["pct_removed"] = round(d["chars_removed"] / self.raw_length, 4) if self.raw_length else 0.0
#         d["large_removal_flag"] = d["pct_removed"] > LARGE_REMOVAL_FLAG_THRESHOLD
#         return d


# # --------------------------------------------------------------------------
# # Parsing Helpers
# # --------------------------------------------------------------------------
# def _norm_letters(s: str) -> str:
#     return re.sub(r"[^a-z]", "", s.lower())


# def _collapse(s: str) -> str:
#     return re.sub(r"[ \t]{2,}", " ", s).strip()


# def _is_terminated(line: str) -> bool:
#     s = line.rstrip()
#     return bool(re.search(r"[.;:][’”'\"]*$", s)) and not _ABBREV_END_RE.search(s)


# def _find_body_preamble_start(pages: list[dict]) -> int:
#     matches = []
#     for idx, p in enumerate(pages):
#         if _PREAMBLE_OPERATIONAL_MARKER in p["text"]:
#             return idx
#         if "PREAMBLE" in p["text"]:
#             matches.append(idx)
#     return matches[-1] if matches else 0


# def _parse_toc(front_pages: list[str]) -> list[tuple[str, str]]:
#     toc: list[list[str]] = []
#     cur = None
#     for text in front_pages:
#         for line in text.split("\n"):
#             m = re.match(r"^\s*(\d+[A-Z]{0,3})\.\s+(\S.*)$", line)
#             if m:
#                 cur = [m.group(1), m.group(2).strip()]
#                 toc.append(cur)
#                 continue
#             if cur and re.match(r"^\s{12,}\S", line) and not re.search(
#                 r"PART|CHAPTER|SCHEDULE|ARTICLES|PAGES|\d+–\d+", line
#             ):
#                 cur[1] += " " + line.strip()
#             elif line.strip() == "" or re.search(r"PART|CHAPTER|––––|CONSTITUTION", line):
#                 cur = None
#     return [(a, t) for a, t in toc]


# def _article_status(toc_title: str | None) -> str:
#     m = _STATUS_RE.match(toc_title or "")
#     if not m:
#         return "in_force"
#     return {"omitted": "omitted", "repealed": "repealed", "cease": "ceased"}[m.group(1).lower()]


# def _detect_broken_caps_pages(pages: list[dict]) -> set[int]:
#     """Pages whose text layer has broken small-caps ("GOvErNOr"). The span between the first
#     and last such page is treated as affected, since lowercase-only damage has no visible marker."""
#     hits = [p["page_number"] for p in pages if _BROKEN_CAPS_WORD_RE.search(p["text"])]
#     return set(range(min(hits), max(hits) + 2)) if hits else set()


# def _strip_page_furniture(lines: list[str]) -> tuple[list[str], str | None]:
#     out: list[str] = []
#     printed = None
#     i = 0
#     while i < len(lines):
#         line = lines[i]
#         m = _HEADER_RE.match(line)
#         if m:
#             printed = m.group(1) or m.group(2)
#             if not printed and i + 1 < len(lines) and _LONE_PAGE_NUMBER_RE.match(lines[i + 1]):
#                 printed = lines[i + 1].strip().strip("()")
#                 i += 1
#             i += 1
#             continue
#         if _PRINTER_LINE_RE.match(line):
#             i += 1
#             continue
#         out.append(line)
#         i += 1
#     return out, printed


# def _split_footnote_zone(lines: list[str]):
#     starts = [(i, int(m.group(1))) for i, l in enumerate(lines) if (m := _FN_START_RE.match(l))]
#     ones = [i for i, n in starts if n == 1]
#     z = ones[-1] if ones else None
#     if z is None:
#         for i, l in enumerate(lines):
#             if _SEPARATOR_RE.match(l):
#                 j = next((k for k in range(i + 1, len(lines)) if lines[k].strip()), None)
#                 if j is not None and _FN_START_RE.match(lines[j]):
#                     z = j
#     if z is None:
#         return lines, [], []

#     body = lines[:z]
#     while body and (not body[-1].strip() or _SEPARATOR_RE.match(body[-1])):
#         body.pop()

#     entries: list[tuple[int, list[str]]] = []
#     exp = None
#     for idx in range(z, len(lines)):
#         l = lines[idx]
#         m = _FN_START_RE.match(l)
#         if m and (exp is None or int(m.group(1)) == exp):
#             entries.append((int(m.group(1)), [l]))
#             exp = int(m.group(1)) + 1
#         elif not l.strip() or _SEPARATOR_RE.match(l):
#             continue
#         else:
#             entries[-1][1].append(l)

#     displaced: list[str] = []
#     num, ent = entries[-1]
#     for j in range(1, len(ent)):
#         rest = [_collapse(x) for x in ent[j:] if x.strip()]
#         if not _is_terminated(ent[j - 1]) or not rest:
#             continue
#         short_block = len(rest) >= 3 and all(len(x) <= 72 for x in rest)
#         margin_block = len(rest) >= 2 and all(len(x) <= 30 for x in rest)
#         if short_block or margin_block:
#             displaced = ent[j:]
#             entries[-1] = (num, ent[:j])
#             break
#     return body, entries, displaced


# def _convert_markers(line: str, page_id: str, fn_numbers: set[int], used: set[int]) -> str:
#     def conv(m):
#         n = int(m.group(1))
#         if n in fn_numbers:
#             used.add(n)
#             return _ref(f"{page_id}.{n}")
#         return m.group(0)

#     line = re.sub(r"^([ \t]*\d{1,3}[A-Z]{0,3}\.)(\d{1,2})(?=\[)",
#                   lambda m: m.group(1) + conv(re.match(r"(\d+)", m.group(2))), line)
#     line = re.sub(r"(?<![\d.])(\d{1,2})(?=\[)", conv, line)
#     line = re.sub(r"(?<![\w.])(\d{1,2})(?=[ \t]*\*)", conv, line)
#     line = re.sub(r"(?<![\w.,/\-])(\d{1,2})(?=[A-Za-z]{2,})(?!(?:st|nd|rd|th)\b)", conv, line)
#     line = re.sub(r"(?<=[A-Za-z]{2})(\d{1,2})(?=[\s,;:.)\]]|$)", conv, line)
#     line = re.sub(r"(?<=[,;:])(\d{1,2})(?=[ \t])", conv, line)
#     return line


# # --------------------------------------------------------------------------
# # Class Trackers & Evidence Helpers
# # --------------------------------------------------------------------------
# class _ArticleTracker:
#     def __init__(self, toc: list[tuple[str, str]]):
#         self.ids = [a for a, _ in toc]
#         self.titles = dict(toc)
#         self.ptr = 0
#         self.found: list[str] = []

#     def match(self, line: str, page_id: str, fn_numbers: set[int], used: set[int], diag: Diagnostics):
#         if self.ptr >= len(self.ids):
#             return line, None
#         m = _ARTICLE_HEAD_RE.match(line)
#         if not m:
#             return line, None
#         num = m.group("num")
#         window = self.ids[self.ptr:self.ptr + 6]
#         art = None
#         if num in window:
#             art = num
#         else:
#             for k in (1, 2):
#                 head, tail = num[:k], num[k:]
#                 if tail and tail in window and int(head) in fn_numbers:
#                     used.add(int(head))
#                     diag.glued_article_numbers_fixed.append(
#                         {"raw": num, "article": tail, "printed_page": page_id})
#                     line = (m.group("ind") + m.group("pre") + _ref(f"{page_id}.{int(head)}")
#                             + tail + line[m.end("num"):])
#                     art = tail
#                     break
#         if art is None:
#             return line, None
#         self.ptr = self.ids.index(art, self.ptr) + 1
#         self.found.append(art)
#         return line, art


# class _HyphenEvidence:
#     def __init__(self, full_text: str):
#         flat = re.sub(r"-[ \t]*\n[ \t]*", "", full_text)
#         self.hyph = Counter(m.lower() for m in re.findall(r"\b[A-Za-z]+-[A-Za-z]+\b", full_text.replace("-\n", "~")))
#         self.joined = Counter(w.lower() for w in re.findall(r"\b[A-Za-z]+\b", flat))

#     def keep_hyphen(self, prefix: str, suffix: str) -> bool:
#         if suffix[:1].isupper():
#             return True
#         p, s = prefix.lower(), suffix.lower()
#         if (p + s) in _KNOWN_HYPHENATED_COMPOUNDS:
#             return True
#         h, j = self.hyph.get(f"{p}-{s}", 0), self.joined.get(p + s, 0)
#         return h > 0 and h >= j


# class _CapitalsRepair:
#     def __init__(self, good_text: str):
#         self.cap = Counter()
#         self.low = Counter()
#         for w in re.findall(r"\b[A-Za-z][a-z]+\b", good_text):
#             (self.cap if w[0].isupper() else self.low)[w.lower()] += 1
#         bi = Counter(re.findall(r"\b([A-Za-z][a-z]+ [A-Za-z][a-z]+)\b", good_text))
#         self.cap_bigrams = {k.lower(): k for k, v in bi.items()
#                             if v >= 2 and k.split()[0][0].isupper() and k.split()[1][0].isupper()
#                             and k.split()[1][0].lower() in _BROKEN_INITIALS
#                             and bi.get(k.lower(), 0) == 0 and bi.get(k.split()[0] + " " + k.split()[1].lower(), 0) == 0}
#         self.changes = 0

#     def _fix_word(self, m: re.Match) -> str:
#         w = m.group(0)
#         k = w.lower()
#         c, l = self.cap.get(k, 0), self.low.get(k, 0)
#         if c >= 3 and c >= 9 * l:
#             self.changes += 1
#             return w[0].upper() + w[1:]
#         return w

#     def fix_small_caps(self, line: str) -> str:
#         def up(m):
#             self.changes += 1
#             return m.group(0).upper()
#         line = re.sub(rf"\b(?=[A-Za-z]*[A-Z][A-Za-z]*[A-Z])(?=[A-Za-z]*[{_BROKEN_INITIALS}])[A-Z{_BROKEN_INITIALS}]{{3,}}\b",
#                       up, line)

#         def or_up(m):
#             self.changes += 1
#             return m.group(1) + "OR"
#         return re.sub(rf"((?:[A-Z]{{2,}}|\]|\[|{_REF_CLOSE})[ \t]+|\[|{_REF_CLOSE})Or(?=[ \t]+[A-Z\[]|[ \t]*$)",
#                       or_up, line)

#     def fix_page_line(self, line: str, starts_paragraph: bool, is_heading: bool) -> str:
#         line = re.sub(rf"\b[{_BROKEN_INITIALS}][a-z]+\b", self._fix_word, line)

#         def bigram(m):
#             fixed = self.cap_bigrams.get(m.group(0).lower())
#             if fixed and fixed != m.group(0):
#                 self.changes += 1
#                 return fixed
#             return m.group(0)
#         line = re.sub(r"\b[A-Za-z][a-z]+ [A-Za-z][a-z]+\b", bigram, line)

#         def cap(m):
#             self.changes += 1
#             return m.group(1) + m.group(2).upper()
#         ref = rf"(?:{_REF}|[\[(])*"
#         if starts_paragraph:
#             line = re.sub(rf"^([ \t]*{ref}(?:\d+[A-Z]?\.[ \t]+{ref})?)([{_BROKEN_INITIALS}])(?=[a-z]+\b)", cap, line)
#         line = re.sub(rf"([.?!][’”'\"\]]?[ \t]+{ref})([{_BROKEN_INITIALS}])(?=[a-z]+\b)", cap, line)
#         if is_heading:
#             line = re.sub(rf"(\b)(?!(?:{'|'.join(_SMALL_WORDS)})\b)([{_BROKEN_INITIALS}])(?=[a-z]+\b)", cap, line)
#         n = len(re.findall(r"\brs\.", line))
#         if n:
#             self.changes += n
#             line = re.sub(r"\brs\.", "Rs.", line)
#         return line


# # --------------------------------------------------------------------------
# # Marginal Notes & Tables Repair
# # --------------------------------------------------------------------------
# def _attach_marginal_notes(lines: list[str], heads: list[tuple[int, str]], titles: dict[str, str],
#                            diag: Diagnostics, ev: _HyphenEvidence | None = None) -> list[str]:
#     if not heads:
#         return lines
#     cand = [i for i, l in enumerate(lines)
#             if l.strip() and len(_collapse(_REF_SENTINEL_RE.sub("", l))) <= 25
#             and not _ARTICLE_HEAD_RE.match(l) and not _DIVIDER_LINE_RE.match(l)
#             and not _SEPARATOR_RE.match(l)]
#     cand_set = set(cand)
#     taken: set[int] = set()
#     headings: dict[int, str] = {}

#     def runs_from(i):
#         acc, idxs = "", []
#         j = i
#         while j in cand_set and j not in taken:
#             acc += _norm_letters(lines[j])
#             idxs.append(j)
#             yield acc, list(idxs)
#             j += 1

#     for exact in (True, False):
#         min_start = 0
#         for line_idx, art in heads:
#             if line_idx in headings or art not in titles:
#                 continue
#             if _article_status(titles[art]) != "in_force":
#                 continue                                   # omitted articles have no marginal note
#             target = _norm_letters(titles[art])
#             if not target:
#                 continue
#             best = None
#             for i in cand:
#                 if i in taken or (exact and i < min_start):
#                     continue
#                 for acc, idxs in runs_from(i):
#                     if len(acc) > len(target) * 1.3:
#                         break
#                     if exact and acc == target:
#                         best = (1.0, idxs)
#                         break
#                     if not exact and abs(len(acc) - len(target)) <= max(4, len(target) * 0.2):
#                         r = SequenceMatcher(None, acc, target, autojunk=False).ratio()
#                         if r >= 0.85 and (best is None or r > best[0]):
#                             best = (r, idxs)
#                 if exact and best:
#                     break
#             if best:
#                 idxs = best[1]
#                 taken.update(idxs)
#                 min_start = idxs[-1] + 1
#                 headings[line_idx] = _join_wrapped([lines[k] for k in idxs], ev)

#     head_map = dict(heads)
#     out: list[str] = []
#     for i, l in enumerate(lines):
#         if i in taken:
#             continue
#         if i in head_map:
#             art = head_map[i]
#             if i in headings:
#                 out.append(f"## {art}. {headings[i]}")
#                 diag.marginal_notes_attached += 1
#             elif titles.get(art):
#                 # No marginal note (omitted / repealed article): synthesise the heading from the TOC.
#                 synthetic = f"## {art}. {_collapse(titles[art])}"
#                 out.append(synthetic)
#                 diag.headings_from_toc.append(synthetic)
#         out.append(l)
#     return out


# def _repair_seat_table(lines: list[str], diag: Diagnostics, page: int) -> list[str]:
#     hdr = ["General seats", "Women", "Non-Muslims", "Total"]
#     for i in range(len(lines) - 4):
#         if [lines[i + k].strip() for k in range(4)] != hdr:
#             continue
#         j = i + 4
#         while j < len(lines) and (not lines[j].strip() or _DIVIDER_LINE_RE.match(lines[j])):
#             j += 1
#         rows = []
#         while j + 4 < len(lines) + 1:
#             name = lines[j].strip() if j < len(lines) else ""
#             nums = [lines[j + k].strip() for k in range(1, 5)] if j + 4 < len(lines) else []
#             if name and len(nums) == 4 and all(re.fullmatch(r"\d+\]?", x) for x in nums) \
#                     and not re.fullmatch(r"\d+", name):
#                 closing = "]" if nums[-1].endswith("]") else ""
#                 nums[-1] = nums[-1].rstrip("]")
#                 rows.append(f"| {_collapse(name)}{closing} | " + " | ".join(nums) + " |")
#                 j += 5
#             else:
#                 break
#         if len(rows) < 2:
#             continue
#         while j < len(lines) and _DIVIDER_LINE_RE.match(lines[j]):
#             j += 1
#         table = ["", "|  | " + " | ".join(hdr) + " |", "|---|---|---|---|---|", *rows, ""]
#         diag.table_repairs.append({"pdf_page": page, "table": "Art. 106 seat table", "rows": len(rows)})
#         start = i
#         while start > 0 and _DIVIDER_LINE_RE.match(lines[start - 1]):
#             start -= 1
#         return lines[:start] + table + lines[j:]
#     return lines


# def _repair_column_table(lines: list[str], diag: Diagnostics, page: int) -> list[str]:
#     cells = [re.split(r"[ \t]{3,}", l.strip()) if l.strip() else [] for l in lines]
#     numeric = lambda c: all(re.fullmatch(r"(?:[\d,]+|[–-])\]?", x) for x in c[1:]) and len(c) >= 3
#     i = 0
#     while i < len(lines):
#         if len(cells[i]) >= 3 and all(len(x) <= 20 for x in cells[i]) and not numeric(cells[i]):
#             j = i + 1
#             rows = []
#             while j < len(lines) and numeric(cells[j]) and len(cells[j]) == len(cells[i]) + 1:
#                 rows.append(cells[j])
#                 j += 1
#             if len(rows) >= 2:
#                 hdr = ["", *cells[i]]
#                 table = ["", "| " + " | ".join(hdr) + " |", "|" + "---|" * len(hdr),
#                          *["| " + " | ".join(r) + " |" for r in rows], ""]
#                 diag.table_repairs.append({"pdf_page": page, "table": "column table", "rows": len(rows)})
#                 return lines[:i] + table + lines[j:]
#         i += 1
#     return lines


# # --------------------------------------------------------------------------
# # Line Unwrapping & Hyphenation
# # --------------------------------------------------------------------------
# def _join_wrapped(parts: list[str], ev: _HyphenEvidence | None = None) -> str:
#     out = ""
#     for p in parts:
#         p = _collapse(p)
#         if not p:
#             continue
#         if out.endswith("-") and re.search(r"[A-Za-z]-$", out) and re.match(r"[A-Za-z]", p):
#             pre = re.search(r"([A-Za-z]+)-$", out).group(1)
#             suf = re.match(r"[A-Za-z]+", p).group(0)
#             keep = ev.keep_hyphen(pre, suf) if ev else (suf[:1].isupper())
#             out = (out if keep else out[:-1]) + p
#         else:
#             out = f"{out} {p}" if out else p
#     return out


# def _unwrap(lines: list[str], ev: _HyphenEvidence) -> list[str]:
#     """Group wrapped PDF lines into paragraphs. The raw text has no blank lines between
#     paragraphs, so a new paragraph starts at indentation, an article number, a clause label,
#     a proviso / explanation, or a structural / standalone heading line. A false break only
#     adds a newline; a false join merges unrelated provisions, so breaks are preferred."""
#     paras: list[list[str]] = []
#     force_break = False
#     for line in lines:
#         if not line.strip():
#             if paras and paras[-1]:
#                 paras.append([])
#             force_break = False
#             continue
#         if line.startswith(("## ", "|")):                    # headings and table rows stand alone
#             paras.append([line])
#             paras.append([])
#             continue
#         standalone = bool(_STRUCT_LINE_RE.match(line) or _STANDALONE_LINE_RE.match(line))
#         if not paras or not paras[-1] or force_break or standalone or _PARA_START_RE.match(line):
#             paras.append([line])
#         else:
#             paras[-1].append(line)
#         force_break = standalone
#     return [_join_wrapped(p, ev) if p else "" for p in paras]


# def _normalize_amendment_brackets(text: str) -> str:
#     opener = re.compile(f"{_REF}\\[")
#     result: list[str] = []
#     i, n = 0, len(text)
#     while i < n:
#         m = opener.match(text, i)
#         if m:
#             depth, j = 1, m.end()
#             start_inner = j
#             while j < n and depth > 0:
#                 if text[j] == "[":
#                     depth += 1
#                 elif text[j] == "]":
#                     depth -= 1
#                 j += 1
#             if depth == 0:
#                 ref = text[m.start():m.end() - 1]
#                 inner = _normalize_amendment_brackets(text[start_inner:j - 1])
#                 if _BRACKET_CONTENT_IS_CLAUSE_NUMBER_RE.match(inner):
#                     result.append(ref + inner)
#                 else:
#                     result.append(f"{ref}[{inner}]")
#                 i = j
#                 continue
#         result.append(text[i])
#         i += 1
#     return "".join(result)


# def _mark_structure(text: str, diag: Diagnostics) -> str:
#     """PART / CHAPTER / SCHEDULE / ANNEX lines -> '# ' headings ('### ' for a PART inside a
#     Schedule). A short title on the next line is folded into the heading. No letters are
#     added, so letter preservation still holds."""
#     lines = text.split("\n")
#     out: list[str] = []
#     in_schedules = False
#     i = 0
#     while i < len(lines):
#         line = lines[i]
#         if _STRUCT_LINE_RE.match(line) and not line.startswith("#"):
#             if _SCHEDULE_OR_ANNEX_RE.match(line):
#                 in_schedules = True
#             level = "###" if (in_schedules and _PART_RE.match(line)) else "#"
#             heading = line.strip()
#             j = i + 1
#             while j < len(lines) and not lines[j].strip():
#                 j += 1
#             nxt = lines[j].strip() if j < len(lines) else ""
#             if (nxt and len(nxt) <= 100 and not nxt.startswith(("#", "|", "(", "[Article"))
#                     and not re.match(r"\d", nxt) and not _STRUCT_LINE_RE.match(nxt)
#                     and not nxt.endswith((";", ":", ",", "–", "—")) and not _ARTICLE_HEAD_RE.match(nxt)):
#                 heading = f"{heading} {nxt}"
#                 i = j
#             out.append(f"{level} {heading}")
#             diag.structural_headings += 1
#         else:
#             out.append(line)
#         i += 1
#     return "\n".join(out)


# # --------------------------------------------------------------------------
# # Footnote Enrichment
# # --------------------------------------------------------------------------
# _FN_OCR_FIXES = [
#     (re.compile(r"\blns\."), "Ins."),                # lowercase L read instead of capital I
#     (re.compile(r"\bProvisios\b"), "Provisos"),
# ]
# _ACTION_RE = re.compile(
#     r"\b(subs(?:tituted)?|ins(?:erted)?|added|omitted|renumbered|deleted|repealed|re-?lettered|"
#     r"re-?numbered)\b\.?", re.I)
# _ACTION_CANON = {"subs": "substituted", "substituted": "substituted", "ins": "inserted",
#                  "inserted": "inserted", "added": "added", "omitted": "omitted", "deleted": "omitted",
#                  "repealed": "repealed", "renumbered": "renumbered", "re-numbered": "renumbered",
#                  "relettered": "relettered", "re-lettered": "relettered"}
# _INSTRUMENT_PATTERNS = [
#     # Constitution (Eighteenth Amdt.) Act, 2010  /  Constitution (Amdt.) Order, 1982
#     (re.compile(r"Constitution\s*\(((?:[\w-]+\s+)?)(?:Amdt\.?|Amendment)\)\s*(Act|Order)\s*,?\s*(\d{4})"),
#      lambda m: (f"Constitution ({(m.group(1).strip().title().replace('Sixthy', 'Sixth') + ' ') if m.group(1).strip() else ''}Amdt.) "
#                 f"{m.group(2)}, {m.group(3)}", m.group(3))),
#     # Act No. 10 of 2010  /  Act 10 of 2010  /  Act. No. 25 of 2016
#     (re.compile(r"\bAct\.?\s*(?:No\.\s*)?(\d+)\s+of\s+(\d{4})"),
#      lambda m: (f"Act No. {m.group(1)} of {m.group(2)}", m.group(2))),
#     # P.O. No. 14 of 1985  /  P. O. No. 5 of 1982
#     (re.compile(r"\bP\.\s*O\.\s*(?:No\.\s*)?(\d+)\s+of\s+(\d{4})"),
#      lambda m: (f"P.O. No. {m.group(1)} of {m.group(2)}", m.group(2))),
#     # Named instrument after "by": Revival of the Constitution of 1973 Order, 1985
#     (re.compile(r"\bby\s+(?:the\s+)?([A-Z][\w().,'’ -]{3,90}?(?:Order|Ordinance|Act))\s*,?\s+(\d{4})"),
#      lambda m: (f"{_collapse(m.group(1))}, {m.group(2)}", m.group(2))),
# ]


# def _find_instrument(text: str) -> tuple[str | None, int | None]:
#     for rx, build in _INSTRUMENT_PATTERNS:
#         m = rx.search(text)
#         if m:
#             name, year = build(m)
#             return name, int(year)
#     return None, None


# _SEE_FOOTNOTE_RE = re.compile(r"\bSee\s+(?:footnote|f\.\s*n\.)\s*(\d+)\s+on\s+page\s+(\d+)", re.I)


# def _enrich_footnotes(footnotes: list[Footnote], diag: Diagnostics) -> None:
#     by_id = {f.id: f for f in footnotes}
#     prev_inst, prev_year = None, None
#     for f in footnotes:
#         for rx, repl in _FN_OCR_FIXES:
#             new = rx.sub(repl, f.text)
#             if new != f.text:
#                 diag.footnote_ocr_fixes.append({"id": f.id, "before": f.text[:60], "after": new[:60]})
#                 f.text = new

#         head = re.split(r"\bby\b|\bibid\b", f.text, maxsplit=1)[0]
#         f.actions = sorted({_ACTION_CANON[a.lower().replace("-", "")] if a.lower().replace("-", "") in _ACTION_CANON
#                             else a.lower() for a in _ACTION_RE.findall(head)})

#         inst, year = _find_instrument(f.text)
#         if inst:
#             f.amending_instrument, f.amending_year = inst, year
#         elif re.search(r"\bibid\b", f.text) and prev_inst:
#             f.amending_instrument, f.amending_year = prev_inst, prev_year
#             f.instrument_inherited = True
#             diag.footnotes_ibid_resolved += 1

#         sm = _SEE_FOOTNOTE_RE.search(f.text)
#         if sm:
#             target = f"{sm.group(2)}.{sm.group(1)}"
#             f.see_also = target
#             t = by_id.get(target)
#             if t is not None:
#                 diag.footnotes_see_also_resolved += 1
#                 if not f.amending_instrument:
#                     f.amending_instrument, f.amending_year = t.amending_instrument, t.amending_year
#                     f.actions = f.actions or t.actions
#                     f.instrument_inherited = True

#         if f.amending_instrument:
#             prev_inst, prev_year = f.amending_instrument, f.amending_year
#     diag.footnotes_without_instrument = sum(1 for f in footnotes if not f.amending_instrument)


# # --------------------------------------------------------------------------
# # Index text (for BM25 / embeddings; canonical text stays untouched)
# # --------------------------------------------------------------------------
# def normalize_for_index(text: str) -> str:
#     """Strip footnote refs and amendment brackets, drop '* * *' omission runs and unify
#     dashes / quotes. Use for BM25 and embedding text; never for display or citation."""
#     prev = None
#     while prev != text:                                      # unwrap [^id][content] inner-first
#         prev = text
#         text = re.sub(r"\[\^[^\]]+\]\[([^\[\]]*)\]", r"\1", text)
#     text = re.sub(r"\[\^[^\]]+\]", "", text)
#     text = re.sub(r"(?:\*[ \t]*){3,}", " ", text)
#     text = text.replace("–", "-").replace("—", "-").replace("―", "-")
#     text = text.replace("“", '"').replace("”", '"').replace("‘", "'").replace("’", "'")
#     text = re.sub(r"[ \t]{2,}", " ", text)
#     return re.sub(r"[ \t]+\n", "\n", text).strip()


# # --------------------------------------------------------------------------
# # Main Core Cleaning Process
# # --------------------------------------------------------------------------
# def clean_constitution_pages(pages: list[dict]) -> tuple[str, list[Footnote], list[dict], Diagnostics]:
#     diag = Diagnostics()
#     pages = sorted(pages, key=lambda p: p["page_number"])
#     full_raw = "\n\n".join(p["text"] for p in pages)
#     diag.raw_length = len(full_raw)

#     start = _find_body_preamble_start(pages)
#     toc = _parse_toc([p["text"] for p in pages[:start]])
#     titles = dict(toc)
#     tracker = _ArticleTracker(toc)
#     diag.articles_in_toc = len(toc)
#     ev = _HyphenEvidence(full_raw)
#     broken_pages = _detect_broken_caps_pages(pages[start:])
#     diag.broken_caps_pages = sorted(broken_pages)
#     caps = _CapitalsRepair("\n".join(p["text"] for p in pages[start:]
#                                       if p["page_number"] not in broken_pages))

#     body_lines: list[str] = []
#     footnotes: list[Footnote] = []
#     article_index: list[dict] = []
#     prev_printed = None

#     for p in pages[start:]:
#         n = p["page_number"]
#         lines, printed = _strip_page_furniture(p["text"].split("\n"))
#         if printed is None:
#             printed = str(int(prev_printed) + 1) if prev_printed and prev_printed.isdigit() else "1"
#         prev_printed = printed
#         if not any(l.strip() for l in lines):
#             continue
#         diag.pages_processed += 1

#         body, entries, displaced = _split_footnote_zone(lines)
#         fn_numbers = {num for num, _ in entries}
#         used: set[int] = set()

#         body = [_convert_markers(l, printed, fn_numbers, used) for l in body]
#         displaced = [_convert_markers(l, printed, fn_numbers, used) for l in displaced]
#         if displaced:
#             diag.displaced_body_blocks_restored.append(n)
#             body = body + displaced

#         heads = []
#         for i, l in enumerate(body):
#             new, art = tracker.match(l, printed, fn_numbers, used, diag)
#             if art:
#                 body[i] = new
#                 heads.append((i, art))
#                 article_index.append({"article": art, "title_toc": titles.get(art),
#                                       "status": _article_status(titles.get(art)),
#                                       "present_in_body": True,
#                                       "printed_page": printed, "pdf_page": n})
#         body = _attach_marginal_notes(body, heads, titles, diag, ev)
#         body = _repair_seat_table(body, diag, n)
#         body = _repair_column_table(body, diag, n)
#         body = [caps.fix_small_caps(l) for l in body]

#         if n in broken_pages:
#             fixed = []
#             for k, l in enumerate(body):
#                 prev_line = fixed[-1] if fixed else (body_lines[-1] if body_lines else "")
#                 ends_sentence = bool(re.search(r"[.?!][’”'\")\]]*\s*$", prev_line)) \
#                     and not _ABBREV_END_RE.search(prev_line.rstrip())
#                 starts_para = bool(_PARA_START_RE.match(l)) or not prev_line.strip() or ends_sentence
#                 fixed.append(caps.fix_page_line(l, starts_para, False))
#             body = fixed

#         body = [l for l in body if not _DIVIDER_LINE_RE.match(l) and not _SEPARATOR_RE.match(l)]
#         body_lines.extend(body)

#         for num, ent in entries:
#             first = _FN_START_RE.sub("", ent[0], count=1)
#             text = _join_wrapped([first] + ent[1:], ev)
#             fid = f"{printed}.{num}"
#             footnotes.append(Footnote(id=fid, pdf_page=n, printed_page=printed, number=num, text=text,
#                                       referenced=num in used))

#     paras = _unwrap(body_lines, ev)
#     text = "\n".join(paras)
#     text = _normalize_amendment_brackets(text)
#     text = re.sub(r"[ \t]{2,}", " ", text)
#     text = re.sub(r"\n{3,}", "\n\n", text).strip()
#     for _ in range(2):                                           # keep table rows contiguous
#         text = re.sub(r"(?m)^(\|.*\|)\n\n(?=\|)", r"\1\n", text)
#     text = text.replace(" ", " ").replace("‘‘", "“").replace("’’", "”")

#     text = _REF_SENTINEL_RE.sub(lambda m: f"[^{m.group(1)}]", text)
#     text = _mark_structure(text, diag)

#     diag.footnote_refs = len(re.findall(r"\[\^[^\]]+\]", text))
#     referenced = set(re.findall(r"\[\^([^\]]+)\]", text))
#     for f in footnotes:
#         f.referenced = f.id in referenced
#     diag.footnotes = len(footnotes)
#     diag.unreferenced_footnotes = [f.id for f in footnotes if not f.referenced]
#     diag.unresolved_markers = sorted(referenced - {f.id for f in footnotes})

#     # Article bookkeeping ------------------------------------------------------------
#     found = set(tracker.found)
#     diag.articles_found = len(tracker.found)
#     diag.articles_missing = [a for a in tracker.ids if a not in found]
#     diag.articles_missing_unexpected = [a for a in diag.articles_missing
#                                         if _article_status(titles.get(a)) == "in_force"]
#     for a in diag.articles_missing:                              # e.g. 270C: only '* * *' + a footnote
#         article_index.append({"article": a, "title_toc": titles.get(a),
#                               "status": _article_status(titles.get(a)), "present_in_body": False,
#                               "printed_page": None, "pdf_page": None})
#     order = {a: i for i, a in enumerate(tracker.ids)}
#     article_index.sort(key=lambda r: order.get(r["article"], 10**6))

#     heading_titles = {m.group(1): m.group(2) for m in re.finditer(r"(?m)^## (\d+[A-Z]{0,3})\. (.*)$", text)}
#     synthetic = set(diag.headings_from_toc)
#     for row in article_index:
#         h = heading_titles.get(row["article"])
#         row["heading"] = h
#         row["title"] = normalize_for_index(h) if h else (_collapse(row["title_toc"] or "") or None)
#         row["title_source"] = (None if not h else
#                                "toc" if f"## {row['article']}. {h}" in synthetic else "marginal_note")
#     diag.articles_without_heading = [a for a in tracker.found if a not in heading_titles]
#     diag.capitalisation_repairs = caps.changes
#     diag.max_line_chars = max((len(l) for l in text.split("\n") if not l.startswith("|")), default=0)

#     return text, footnotes, article_index, diag


# def verify_letter_preservation(pages: list[dict], body: str, footnotes: list[Footnote],
#                                synthetic_lines: list[str] | None = None) -> tuple[bool, dict]:
#     """Every letter of the raw body + footnotes must survive cleaning, and no letters may be
#     invented. Headings synthesised from the TOC are excluded (they are not page text)."""
#     pages = sorted(pages, key=lambda p: p["page_number"])
#     start = _find_body_preamble_start(pages)
#     raw = []
#     for p in pages[start:]:
#         lines, _ = _strip_page_furniture(p["text"].split("\n"))
#         raw.append("\n".join(lines))
#     a = Counter(re.sub(r"[^a-z]", "", "\n".join(raw).lower()))
#     synthetic = set(synthetic_lines or [])
#     body_lines = [l for l in body.split("\n") if l not in synthetic]
#     body_clean = "\n".join(body_lines)
#     body_clean = re.sub(r"(?m)^## \d+[A-Z]{0,3}\. ", "", body_clean)
#     body_clean = re.sub(r"(?m)^#{1,3} ", "", body_clean)
#     cleaned = re.sub(r"\[\^[^\]]+\]", "", body_clean) + "".join(f.text for f in footnotes)
#     b = Counter(re.sub(r"[^a-z]", "", cleaned.lower()))
#     missing, extra = dict(a - b), dict(b - a)
#     return (not missing and not extra), {"missing": missing, "extra": extra}


# def quality_gate(doc: dict) -> list[str]:
#     """Return a list of blocking problems; empty means the document is safe to index."""
#     d = doc["cleaning_diagnostics"]
#     body = doc["body_text"]
#     errors = []
#     if not d["letter_preservation_ok"]:
#         errors.append(f"letter preservation failed: {d['letter_preservation_diff']}")
#     n_heads = len(re.findall(r"(?m)^## \d+[A-Z]{0,3}\. ", body))
#     expected = d["marginal_notes_attached"] + len(d["headings_from_toc"])
#     if n_heads != expected:
#         errors.append(f"{n_heads} article headings at line start, expected {expected}")
#     if re.search(r"[^\n]## \d+[A-Z]{0,3}\. ", body):
#         errors.append("article heading found mid-line (paragraph unwrap regression)")
#     if d["articles_without_heading"]:
#         errors.append(f"articles without heading: {d['articles_without_heading']}")
#     if d["articles_missing_unexpected"]:
#         errors.append(f"in-force articles missing from body: {d['articles_missing_unexpected']}")
#     if d["max_line_chars"] > MAX_LINE_CHARS:
#         errors.append(f"longest line is {d['max_line_chars']} chars (> {MAX_LINE_CHARS})")
#     if d["unresolved_markers"]:
#         errors.append(f"footnote refs without a footnote: {d['unresolved_markers'][:10]}")
#     if d["unreferenced_footnotes"]:
#         errors.append(f"footnotes never referenced: {d['unreferenced_footnotes'][:10]}")
#     if d["large_removal_flag"]:
#         errors.append(f"removed {d['pct_removed']:.1%} of raw text")
#     return errors


# def clean_constitution_document(raw_doc: dict) -> dict:
#     body, footnotes, article_index, diag = clean_constitution_pages(raw_doc["pages"])
#     ok, d = verify_letter_preservation(raw_doc["pages"], body, footnotes, diag.headings_from_toc)
#     diag.letter_preservation_ok, diag.letter_preservation_diff = ok, d

#     # Enrichment runs after verification: OCR fixes intentionally change letters ("lns." -> "Ins.").
#     _enrich_footnotes(footnotes, diag)

#     block = "\n".join(f"[^{f.id}]: {f.text}" for f in footnotes)
#     cleaned_text = f"{body}\n\n[FOOTNOTES]\n{block}\n[/FOOTNOTES]" if footnotes else body
#     diag.cleaned_length = len(cleaned_text)

#     doc = {
#         "doc_id": raw_doc["doc_id"],
#         "metadata": dict(raw_doc["metadata"]),
#         "raw_text": raw_doc["raw_text"],
#         "cleaned_text": cleaned_text,
#         "body_text": body,
#         "index_text": normalize_for_index(body),
#         "footnotes": [f.to_dict() for f in footnotes],
#         "article_index": article_index,
#         "cleaning_diagnostics": diag.to_dict(),
#     }
#     errors = quality_gate(doc)
#     doc["cleaning_diagnostics"]["quality_gate_errors"] = errors
#     return doc


# def strip_footnote_refs(text: str) -> str:
#     """Drops page-scoped footnote-reference markers ("[^5.2]", "[^91.8]")
#     from `text`, for use by downstream consumers that need a reference-
#     free string (e.g. constitution_chunker.py's breadcrumb titles, which
#     come from `article_index`/`## <id>. <title>` heading text that can
#     carry one of these markers -- see clean_constitution_pages' own
#     "[^15.2]25A. The State shall provide ..." convention documented in
#     this module's docstring). Only the "[^<id>]" sentinel wrapper itself
#     is removed; a substantive amendment bracket with no leading caret,
#     e.g. "[the Federal Constitutional Court,]", is a different shape
#     (`\\[\\^` vs `\\[`) and is left untouched -- confirmed against the
#     real Article 175A heading, which contains both forms adjacently:
#     "... to [^91.8][the Federal Constitutional Court,] the Supreme
#     Court..." strips to "... to [the Federal Constitutional Court,] the
#     Supreme Court...", never losing the bracketed legal text itself.
#     """
#     return re.sub(r"\[\^[^\]]+\]", "", text)


# def clean_directory(input_dir: Path = INPUT_DIR, output_dir: Path = OUTPUT_DIR) -> dict:
#     input_dir, output_dir = Path(input_dir), Path(output_dir)
#     output_dir.mkdir(parents=True, exist_ok=True)
#     rejected_dir = output_dir / "_rejected"
#     processed, flagged, failed_docs, rejected_docs = 0, 0, [], []

#     for raw_path in sorted(input_dir.glob("*.json")):
#         try:
#             raw_doc = json.loads(raw_path.read_text(encoding="utf-8"))
#             cleaned_doc = clean_constitution_document(raw_doc)
#         except (OSError, json.JSONDecodeError, KeyError, ValueError) as exc:
#             failed_docs.append({"source_file": raw_path.name, "error": str(exc)})
#             continue

#         errors = cleaned_doc["cleaning_diagnostics"]["quality_gate_errors"]
#         target_dir = output_dir
#         if errors:
#             rejected_dir.mkdir(parents=True, exist_ok=True)
#             target_dir = rejected_dir
#             rejected_docs.append({"source_file": raw_path.name, "errors": errors})

#         out_path = target_dir / f"{cleaned_doc['doc_id']}.json"
#         tmp_path = out_path.with_suffix(".json.tmp")
#         tmp_path.write_text(json.dumps(cleaned_doc, ensure_ascii=False, indent=2), encoding="utf-8")
#         tmp_path.replace(out_path)

#         processed += 1
#         if cleaned_doc["cleaning_diagnostics"]["large_removal_flag"]:
#             flagged += 1

#     return {"processed": processed, "failed": len(failed_docs), "rejected": len(rejected_docs),
#             "large_removal_flagged": flagged, "failed_docs": failed_docs, "rejected_docs": rejected_docs}


# def main() -> None:
#     summary = clean_directory()
#     print(f"Constitution cleaning completed: {summary['processed']} processed, {summary['failed']} failed, "
#           f"{summary['rejected']} rejected by quality gate, "
#           f"{summary['large_removal_flagged']} flagged for large removal.")
#     for f in summary["failed_docs"]:
#         print(f"  FAILED {f['source_file']}: {f['error']}")
#     for r in summary["rejected_docs"]:
#         print(f"  REJECTED {r['source_file']}:")
#         for e in r["errors"]:
#             print(f"    - {e}")
#     if summary["failed"] or summary["rejected"]:
#         sys.exit(1)


# if __name__ == "__main__":
#     main()




"""
Constitution of Pakistan Cleaning Pipeline (End-to-End v3)

Pipeline Flow:
    Raw ingested JSON (var/rag/constitution_ingested/<doc_id>.json)
        -> clean_constitution_document()
        -> quality_gate()
        -> Cleaned constitution JSON (var/rag/cleaned_constitution/<doc_id>.json)
           (documents failing the gate go to var/rag/cleaned_constitution/_rejected/)

Key Capabilities:
1. Preamble Anchor Detection: targets the operational Preamble to drop front matter / TOC.
2. Page-Aware Footnote Processing: isolates footnote blocks per page and assigns page-scoped
   references [^<printed_page>.<num>].
3. Marginal Note Extraction: re-locates marginal titles as article headings (## <Num>. <Title>).
   Articles without a marginal note (omitted / repealed ones) get their heading from the TOC.
4. Structural Headings: PART / CHAPTER / SCHEDULE / ANNEX lines become '# ' headings
   ('### ' for PARTs inside a Schedule).
5. Table Reconstruction: re-assembles the Art. 51 and Art. 106 seat tables as Markdown tables.
6. Paragraph Unwrap & Hyphenation: rejoins wrapped lines into paragraphs (one clause /
   sub-clause / proviso per line) without breaking legal compound terms.
7. Footnote Enrichment: amendment action, amending instrument, 'ibid.' resolution,
   'See footnote N on page P' cross-references, OCR fixes.
8. Verification + Quality Gate: letter-multiset preservation against the raw input, plus
   structural checks that fail the run instead of silently writing bad output.

Output text conventions (consumed by the chunker):
    # PART II Fundamental Rights and Principles of Policy     <- structural heading
    # CHAPTER 1.–FUNDAMENTAL RIGHTS
    ## 25A. Right to education.                              <- article heading
    [^15.2]25A. The State shall provide ...                  <- article body, one paragraph per line
    [^29.3][three]                                           <- amendment: footnote ref + bracketed text
"""

from __future__ import annotations

import json
import re
import sys
from collections import Counter
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from pathlib import Path

INPUT_DIR = Path("var/rag/constitution_ingested")
OUTPUT_DIR = Path("var/rag/cleaned_constitution")

LARGE_REMOVAL_FLAG_THRESHOLD = 0.15
MAX_LINE_CHARS = 2500                 # quality gate: no single paragraph line longer than this

# Private-use sentinels for footnote references during processing
_REF_OPEN, _REF_CLOSE = "", ""
_REF_SENTINEL_RE = re.compile(f"{_REF_OPEN}([^{_REF_CLOSE}]*){_REF_CLOSE}")
_REF = f"(?:{_REF_OPEN}[^{_REF_CLOSE}]*{_REF_CLOSE})"          # one sentinel ref, as a regex fragment


def _ref(fid: str) -> str:
    return f"{_REF_OPEN}{fid}{_REF_CLOSE}"


# --------------------------------------------------------------------------
# Regex Patterns
# --------------------------------------------------------------------------
_PREAMBLE_OPERATIONAL_MARKER = "WHEREAS sovereignty over the entire Universe belongs"
_HEADER_RE = re.compile(r"^[ \t]*(\d{1,3})?[ \t]*CONSTITUTION OF PAKISTAN\d*[ \t]*(\d{1,3})?[ \t]*$")
_LONE_PAGE_NUMBER_RE = re.compile(r"^[ \t]*(?:\(?[ivxlcdm]{1,6}\)?|\d{1,3})[ \t]*$")
_PRINTER_LINE_RE = re.compile(
    r"^[ \t]*(?:\d+\(\d+\)[A-Za-z&]+---by[ \t]+\S+---PC-\d+[ \t]*\([^)\n]*\)"
    r"|PCPPI\S*.*\(PC-\d+\))[ \t]*$"
)
_SEPARATOR_RE = re.compile(r"^[ \t]*[_‐-―-]{15,}(?:[ \t]+[_‐-―-]{10,})?[ \t]*$")
_DIVIDER_LINE_RE = re.compile(r"^[ \t]*(?:_{3,}(?:[ \t]+_{3,})?|[‐-―]{2,})[ \t]*$")

_FN_START_RE = re.compile(r"^[ \t]*(\d{1,2})(?:\.(?=[A-Za-z])|[ \t](?=[A-Z])|(?=[A-Za-z‘’“\"']))")
_ABBREV_END_RE = re.compile(
    r"\b(?:Amdt|subs|ins|Subs|Ins|s|ss|Art|Arts|No|Nos|Pt|Ext|Gaz|Pak|p|pp|Sch|viz|Ord|O|P|"
    r"Govt|Vol|Para|Cl|Ch|Notfn|Min|Deptt|Rs|w\.e\.f|i\.e|e\.g)\.$"
)

_ARTICLE_HEAD_RE = re.compile(
    rf"^(?P<ind>[ \t]*)(?P<pre>(?:{_REF}?\[)*{_REF}?)"
    rf"(?P<num>\d{{1,4}}[A-Z]{{0,3}})\.(?=[ \t]|$|[–—-]|\d{{1,2}}\[|{_REF_OPEN})"
)

_KNOWN_HYPHENATED_COMPOUNDS = frozenset({
    "majlise", "nondiscrimination", "selfincrimination", "subparagraph", "noconfidence",
    "wellbeing", "hydroelectric", "fullstop", "twentythird",
})

_BRACKET_CONTENT_IS_CLAUSE_NUMBER_RE = re.compile(
    r"^(?:\(\d+[A-Za-z]{0,2}\)|\([a-zA-Z0-9]{1,4}\)|\d+[A-Za-z]{0,2}\.)"
)

# Broken small-caps in the PDF text layer, e.g. "GOvErNOr", "ASSEmBly" (capital, lowercase, capital)
_BROKEN_CAPS_WORD_RE = re.compile(r"\b(?=[A-Za-z]*[A-Z][a-z]+[A-Z])[A-Za-z]{4,}\b")
_BROKEN_INITIALS = "mrlvy"
_SMALL_WORDS = {"of", "and", "the", "by", "for", "to", "in", "on", "a", "an", "or", "from", "under", "with"}

# Omitted / repealed status from the TOC title, e.g. "[Omitted.]", "[Repealed.]", "[Cease to inforce.]"
_STATUS_RE = re.compile(r"^\[\s*(Omitted|Repealed|Cease)", re.I)

# Structural lines (work both on sentinel text and on final [^id] text)
_LEAD_REFS = rf"(?:{_REF}|\[\^[^\]]+\]|\[)*"
_STRUCT_KW = (r"(?:PART[ \t]+[IVXL]+\b|CHAPTER[ \t]+\d+[A-Z]?[.–—-]"
              r"|(?:FIRST|SECOND|THIRD|FOURTH|FIFTH|SIXTH|SEVENTH)[ \t]+SCHEDULE\b|ANNEX\b)")
_STRUCT_LINE_RE = re.compile(rf"^[ \t]*{_LEAD_REFS}\*?{_STRUCT_KW}")          # "*FIFTH SCHEDULE" occurs
_SCHEDULE_OR_ANNEX_RE = re.compile(rf"^[ \t]*{_LEAD_REFS}\*?(?:(?:FIRST|SECOND|THIRD|FOURTH|FIFTH|SIXTH|SEVENTH)"
                                   rf"[ \t]+SCHEDULE\b|ANNEX\b)")
_PART_RE = re.compile(rf"^[ \t]*{_LEAD_REFS}PART[ \t]+[IVXL]+\b")

# Short standalone lines that must not be glued to neighbours
_STANDALONE_LINE_RE = re.compile(
    rf"^[ \t]*{_LEAD_REFS}(?:"
    r"[\[(](?:Article|Articles|see|See)\b[^\])]*[\])]\]?"          # [Article 42]  (Article 2A)
    r"|[A-Z][A-Z0-9 ,.'’()&\-–—]{2,70}\]?"                          # ALL-CAPS short line
    r")[ \t]*$"
)

_PARA_START_RE = re.compile(
    rf"^(?:[ \t]{{4,}}\S"                                           # indented = new paragraph in this PDF
    rf"|[ \t]*{_LEAD_REFS}"
    r"(?:\d{1,4}[A-Z]{0,3}\.\]?(?:\s|$)"                            # article head / list entry, also "1[14.] ..."
    r"|\((?:\d+[A-Z]{0,2}|[a-z]{1,2}|[ivxl]+)\)"                    # (1) (a) (iv)
    r"|Provided\b|Explanation\b|\(In the name\b))"
)


# --------------------------------------------------------------------------
# Data Structures
# --------------------------------------------------------------------------
@dataclass
class Footnote:
    id: str
    pdf_page: int
    printed_page: str
    number: int
    text: str
    see_also: str | None = None
    referenced: bool = False
    actions: list = field(default_factory=list)
    amending_instrument: str | None = None
    amending_year: int | None = None
    instrument_inherited: bool = False

    def to_dict(self) -> dict:
        return {
            "id": self.id, "pdf_page": self.pdf_page, "printed_page": self.printed_page,
            "number": self.number, "text": self.text, "see_also": self.see_also,
            "referenced": self.referenced, "actions": self.actions,
            "amending_instrument": self.amending_instrument, "amending_year": self.amending_year,
            "instrument_inherited": self.instrument_inherited,
        }


@dataclass
class Diagnostics:
    raw_length: int = 0
    cleaned_length: int = 0
    pages_processed: int = 0
    footnotes: int = 0
    footnote_refs: int = 0
    unresolved_markers: list = field(default_factory=list)
    unreferenced_footnotes: list = field(default_factory=list)
    displaced_body_blocks_restored: list = field(default_factory=list)
    glued_article_numbers_fixed: list = field(default_factory=list)
    articles_found: int = 0
    articles_in_toc: int = 0
    articles_missing: list = field(default_factory=list)
    articles_missing_unexpected: list = field(default_factory=list)
    marginal_notes_attached: int = 0
    headings_from_toc: list = field(default_factory=list)
    articles_without_heading: list = field(default_factory=list)
    structural_headings: int = 0
    table_repairs: list = field(default_factory=list)
    broken_caps_pages: list = field(default_factory=list)
    capitalisation_repairs: int = 0
    footnote_ocr_fixes: list = field(default_factory=list)
    footnotes_ibid_resolved: int = 0
    footnotes_see_also_resolved: int = 0
    footnotes_without_instrument: int = 0
    letter_preservation_ok: bool | None = None
    letter_preservation_diff: dict = field(default_factory=dict)
    max_line_chars: int = 0
    quality_gate_errors: list = field(default_factory=list)

    def to_dict(self) -> dict:
        d = dict(self.__dict__)
        d["chars_removed"] = self.raw_length - self.cleaned_length
        d["pct_removed"] = round(d["chars_removed"] / self.raw_length, 4) if self.raw_length else 0.0
        d["large_removal_flag"] = d["pct_removed"] > LARGE_REMOVAL_FLAG_THRESHOLD
        return d


# --------------------------------------------------------------------------
# Parsing Helpers
# --------------------------------------------------------------------------
def _norm_letters(s: str) -> str:
    return re.sub(r"[^a-z]", "", s.lower())


def _collapse(s: str) -> str:
    return re.sub(r"[ \t]{2,}", " ", s).strip()


def _is_terminated(line: str) -> bool:
    s = line.rstrip()
    return bool(re.search(r"[.;:][’”'\"]*$", s)) and not _ABBREV_END_RE.search(s)


def _find_body_preamble_start(pages: list[dict]) -> int:
    matches = []
    for idx, p in enumerate(pages):
        if _PREAMBLE_OPERATIONAL_MARKER in p["text"]:
            return idx
        if "PREAMBLE" in p["text"]:
            matches.append(idx)
    return matches[-1] if matches else 0


def _parse_toc(front_pages: list[str]) -> list[tuple[str, str]]:
    toc: list[list[str]] = []
    cur = None
    for text in front_pages:
        for line in text.split("\n"):
            m = re.match(r"^\s*(\d+[A-Z]{0,3})\.\s+(\S.*)$", line)
            if m:
                cur = [m.group(1), m.group(2).strip()]
                toc.append(cur)
                continue
            if cur and re.match(r"^\s{12,}\S", line) and not re.search(
                r"PART|CHAPTER|SCHEDULE|ARTICLES|PAGES|\d+–\d+", line
            ):
                cur[1] += " " + line.strip()
            elif line.strip() == "" or re.search(r"PART|CHAPTER|––––|CONSTITUTION", line):
                cur = None
    return [(a, t) for a, t in toc]


def _article_status(toc_title: str | None) -> str:
    m = _STATUS_RE.match(toc_title or "")
    if not m:
        return "in_force"
    return {"omitted": "omitted", "repealed": "repealed", "cease": "ceased"}[m.group(1).lower()]


def _detect_broken_caps_pages(pages: list[dict]) -> set[int]:
    """Pages whose text layer has broken small-caps ("GOvErNOr"). The span between the first
    and last such page is treated as affected, since lowercase-only damage has no visible marker."""
    hits = [p["page_number"] for p in pages if _BROKEN_CAPS_WORD_RE.search(p["text"])]
    return set(range(min(hits), max(hits) + 2)) if hits else set()


def _strip_page_furniture(lines: list[str]) -> tuple[list[str], str | None]:
    out: list[str] = []
    printed = None
    i = 0
    while i < len(lines):
        line = lines[i]
        m = _HEADER_RE.match(line)
        if m:
            printed = m.group(1) or m.group(2)
            if not printed and i + 1 < len(lines) and _LONE_PAGE_NUMBER_RE.match(lines[i + 1]):
                printed = lines[i + 1].strip().strip("()")
                i += 1
            i += 1
            continue
        if _PRINTER_LINE_RE.match(line):
            i += 1
            continue
        out.append(line)
        i += 1
    return out, printed


def _split_footnote_zone(lines: list[str]):
    starts = [(i, int(m.group(1))) for i, l in enumerate(lines) if (m := _FN_START_RE.match(l))]
    ones = [i for i, n in starts if n == 1]
    z = ones[-1] if ones else None
    if z is None:
        for i, l in enumerate(lines):
            if _SEPARATOR_RE.match(l):
                j = next((k for k in range(i + 1, len(lines)) if lines[k].strip()), None)
                if j is not None and _FN_START_RE.match(lines[j]):
                    z = j
    if z is None:
        return lines, [], []

    body = lines[:z]
    while body and (not body[-1].strip() or _SEPARATOR_RE.match(body[-1])):
        body.pop()

    entries: list[tuple[int, list[str]]] = []
    exp = None
    for idx in range(z, len(lines)):
        l = lines[idx]
        m = _FN_START_RE.match(l)
        if m and (exp is None or int(m.group(1)) == exp):
            entries.append((int(m.group(1)), [l]))
            exp = int(m.group(1)) + 1
        elif not l.strip() or _SEPARATOR_RE.match(l):
            continue
        else:
            entries[-1][1].append(l)

    displaced: list[str] = []
    num, ent = entries[-1]
    for j in range(1, len(ent)):
        rest = [_collapse(x) for x in ent[j:] if x.strip()]
        if not _is_terminated(ent[j - 1]) or not rest:
            continue
        short_block = len(rest) >= 3 and all(len(x) <= 72 for x in rest)
        margin_block = len(rest) >= 2 and all(len(x) <= 30 for x in rest)
        if short_block or margin_block:
            displaced = ent[j:]
            entries[-1] = (num, ent[:j])
            break
    return body, entries, displaced


def _convert_markers(line: str, page_id: str, fn_numbers: set[int], used: set[int]) -> str:
    def conv(m):
        n = int(m.group(1))
        if n in fn_numbers:
            used.add(n)
            return _ref(f"{page_id}.{n}")
        return m.group(0)

    line = re.sub(r"^([ \t]*\d{1,3}[A-Z]{0,3}\.)(\d{1,2})(?=\[)",
                  lambda m: m.group(1) + conv(re.match(r"(\d+)", m.group(2))), line)
    line = re.sub(r"(?<![\d.])(\d{1,2})(?=\[)", conv, line)
    line = re.sub(r"(?<![\w.])(\d{1,2})(?=[ \t]*\*)", conv, line)
    line = re.sub(r"(?<![\w.,/\-])(\d{1,2})(?=[A-Za-z]{2,})(?!(?:st|nd|rd|th)\b)", conv, line)
    line = re.sub(r"(?<=[A-Za-z]{2})(\d{1,2})(?=[\s,;:.)\]]|$)", conv, line)
    line = re.sub(r"(?<=[,;:])(\d{1,2})(?=[ \t])", conv, line)
    return line


# --------------------------------------------------------------------------
# Class Trackers & Evidence Helpers
# --------------------------------------------------------------------------
class _ArticleTracker:
    def __init__(self, toc: list[tuple[str, str]]):
        self.ids = [a for a, _ in toc]
        self.titles = dict(toc)
        self.ptr = 0
        self.found: list[str] = []

    def match(self, line: str, page_id: str, fn_numbers: set[int], used: set[int], diag: Diagnostics):
        if self.ptr >= len(self.ids):
            return line, None
        m = _ARTICLE_HEAD_RE.match(line)
        if not m:
            return line, None
        num = m.group("num")
        window = self.ids[self.ptr:self.ptr + 6]
        art = None
        if num in window:
            art = num
        else:
            for k in (1, 2):
                head, tail = num[:k], num[k:]
                if tail and tail in window and int(head) in fn_numbers:
                    used.add(int(head))
                    diag.glued_article_numbers_fixed.append(
                        {"raw": num, "article": tail, "printed_page": page_id})
                    line = (m.group("ind") + m.group("pre") + _ref(f"{page_id}.{int(head)}")
                            + tail + line[m.end("num"):])
                    art = tail
                    break
        if art is None:
            return line, None
        self.ptr = self.ids.index(art, self.ptr) + 1
        self.found.append(art)
        return line, art


class _HyphenEvidence:
    def __init__(self, full_text: str):
        flat = re.sub(r"-[ \t]*\n[ \t]*", "", full_text)
        self.hyph = Counter(m.lower() for m in re.findall(r"\b[A-Za-z]+-[A-Za-z]+\b", full_text.replace("-\n", "~")))
        self.joined = Counter(w.lower() for w in re.findall(r"\b[A-Za-z]+\b", flat))

    def keep_hyphen(self, prefix: str, suffix: str) -> bool:
        if suffix[:1].isupper():
            return True
        p, s = prefix.lower(), suffix.lower()
        if (p + s) in _KNOWN_HYPHENATED_COMPOUNDS:
            return True
        h, j = self.hyph.get(f"{p}-{s}", 0), self.joined.get(p + s, 0)
        return h > 0 and h >= j


class _CapitalsRepair:
    def __init__(self, good_text: str):
        self.cap = Counter()
        self.low = Counter()
        for w in re.findall(r"\b[A-Za-z][a-z]+\b", good_text):
            (self.cap if w[0].isupper() else self.low)[w.lower()] += 1
        bi = Counter(re.findall(r"\b([A-Za-z][a-z]+ [A-Za-z][a-z]+)\b", good_text))
        self.cap_bigrams = {k.lower(): k for k, v in bi.items()
                            if v >= 2 and k.split()[0][0].isupper() and k.split()[1][0].isupper()
                            and k.split()[1][0].lower() in _BROKEN_INITIALS
                            and bi.get(k.lower(), 0) == 0 and bi.get(k.split()[0] + " " + k.split()[1].lower(), 0) == 0}
        self.changes = 0

    def _fix_word(self, m: re.Match) -> str:
        w = m.group(0)
        k = w.lower()
        c, l = self.cap.get(k, 0), self.low.get(k, 0)
        if c >= 3 and c >= 9 * l:
            self.changes += 1
            return w[0].upper() + w[1:]
        return w

    def fix_small_caps(self, line: str) -> str:
        def up(m):
            self.changes += 1
            return m.group(0).upper()
        line = re.sub(rf"\b(?=[A-Za-z]*[A-Z][A-Za-z]*[A-Z])(?=[A-Za-z]*[{_BROKEN_INITIALS}])[A-Z{_BROKEN_INITIALS}]{{3,}}\b",
                      up, line)

        def or_up(m):
            self.changes += 1
            return m.group(1) + "OR"
        return re.sub(rf"((?:[A-Z]{{2,}}|\]|\[|{_REF_CLOSE})[ \t]+|\[|{_REF_CLOSE})Or(?=[ \t]+[A-Z\[]|[ \t]*$)",
                      or_up, line)

    def fix_page_line(self, line: str, starts_paragraph: bool, is_heading: bool) -> str:
        line = re.sub(rf"\b[{_BROKEN_INITIALS}][a-z]+\b", self._fix_word, line)

        def bigram(m):
            fixed = self.cap_bigrams.get(m.group(0).lower())
            if fixed and fixed != m.group(0):
                self.changes += 1
                return fixed
            return m.group(0)
        line = re.sub(r"\b[A-Za-z][a-z]+ [A-Za-z][a-z]+\b", bigram, line)

        def cap(m):
            self.changes += 1
            return m.group(1) + m.group(2).upper()
        ref = rf"(?:{_REF}|[\[(])*"
        if starts_paragraph:
            line = re.sub(rf"^([ \t]*{ref}(?:\d+[A-Z]?\.[ \t]+{ref})?)([{_BROKEN_INITIALS}])(?=[a-z]+\b)", cap, line)
        line = re.sub(rf"([.?!][’”'\"\]]?[ \t]+{ref})([{_BROKEN_INITIALS}])(?=[a-z]+\b)", cap, line)
        if is_heading:
            line = re.sub(rf"(\b)(?!(?:{'|'.join(_SMALL_WORDS)})\b)([{_BROKEN_INITIALS}])(?=[a-z]+\b)", cap, line)
        n = len(re.findall(r"\brs\.", line))
        if n:
            self.changes += n
            line = re.sub(r"\brs\.", "Rs.", line)
        return line


# --------------------------------------------------------------------------
# Marginal Notes & Tables Repair
# --------------------------------------------------------------------------
def _attach_marginal_notes(lines: list[str], heads: list[tuple[int, str]], titles: dict[str, str],
                           diag: Diagnostics, ev: _HyphenEvidence | None = None) -> list[str]:
    if not heads:
        return lines
    cand = [i for i, l in enumerate(lines)
            if l.strip() and len(_collapse(_REF_SENTINEL_RE.sub("", l))) <= 25
            and not _ARTICLE_HEAD_RE.match(l) and not _DIVIDER_LINE_RE.match(l)
            and not _SEPARATOR_RE.match(l)]
    cand_set = set(cand)
    taken: set[int] = set()
    headings: dict[int, str] = {}

    def runs_from(i):
        acc, idxs = "", []
        j = i
        while j in cand_set and j not in taken:
            acc += _norm_letters(lines[j])
            idxs.append(j)
            yield acc, list(idxs)
            j += 1

    for exact in (True, False):
        min_start = 0
        for line_idx, art in heads:
            if line_idx in headings or art not in titles:
                continue
            if _article_status(titles[art]) != "in_force":
                continue                                   # omitted articles have no marginal note
            target = _norm_letters(titles[art])
            if not target:
                continue
            best = None
            for i in cand:
                if i in taken or (exact and i < min_start):
                    continue
                for acc, idxs in runs_from(i):
                    if len(acc) > len(target) * 1.3:
                        break
                    if exact and acc == target:
                        best = (1.0, idxs)
                        break
                    if not exact and abs(len(acc) - len(target)) <= max(4, len(target) * 0.2):
                        r = SequenceMatcher(None, acc, target, autojunk=False).ratio()
                        if r >= 0.85 and (best is None or r > best[0]):
                            best = (r, idxs)
                if exact and best:
                    break
            if best:
                idxs = best[1]
                taken.update(idxs)
                min_start = idxs[-1] + 1
                headings[line_idx] = _join_wrapped([lines[k] for k in idxs], ev)

    head_map = dict(heads)
    out: list[str] = []
    for i, l in enumerate(lines):
        if i in taken:
            continue
        if i in head_map:
            art = head_map[i]
            if i in headings:
                out.append(f"## {art}. {headings[i]}")
                diag.marginal_notes_attached += 1
            elif titles.get(art):
                # No marginal note (omitted / repealed article): synthesise the heading from the TOC.
                synthetic = f"## {art}. {_collapse(titles[art])}"
                out.append(synthetic)
                diag.headings_from_toc.append(synthetic)
        out.append(l)
    return out


def _repair_seat_table(lines: list[str], diag: Diagnostics, page: int) -> list[str]:
    hdr = ["General seats", "Women", "Non-Muslims", "Total"]
    for i in range(len(lines) - 4):
        if [lines[i + k].strip() for k in range(4)] != hdr:
            continue
        j = i + 4
        while j < len(lines) and (not lines[j].strip() or _DIVIDER_LINE_RE.match(lines[j])):
            j += 1
        rows = []
        while j + 4 < len(lines) + 1:
            name = lines[j].strip() if j < len(lines) else ""
            nums = [lines[j + k].strip() for k in range(1, 5)] if j + 4 < len(lines) else []
            if name and len(nums) == 4 and all(re.fullmatch(r"\d+\]?", x) for x in nums) \
                    and not re.fullmatch(r"\d+", name):
                closing = "]" if nums[-1].endswith("]") else ""
                nums[-1] = nums[-1].rstrip("]")
                rows.append(f"| {_collapse(name)}{closing} | " + " | ".join(nums) + " |")
                j += 5
            else:
                break
        if len(rows) < 2:
            continue
        while j < len(lines) and _DIVIDER_LINE_RE.match(lines[j]):
            j += 1
        table = ["", "|  | " + " | ".join(hdr) + " |", "|---|---|---|---|---|", *rows, ""]
        diag.table_repairs.append({"pdf_page": page, "table": "Art. 106 seat table", "rows": len(rows)})
        start = i
        while start > 0 and _DIVIDER_LINE_RE.match(lines[start - 1]):
            start -= 1
        return lines[:start] + table + lines[j:]
    return lines


def _repair_column_table(lines: list[str], diag: Diagnostics, page: int) -> list[str]:
    cells = [re.split(r"[ \t]{3,}", l.strip()) if l.strip() else [] for l in lines]
    numeric = lambda c: all(re.fullmatch(r"(?:[\d,]+|[–-])\]?", x) for x in c[1:]) and len(c) >= 3
    i = 0
    while i < len(lines):
        if len(cells[i]) >= 3 and all(len(x) <= 20 for x in cells[i]) and not numeric(cells[i]):
            j = i + 1
            rows = []
            while j < len(lines) and numeric(cells[j]) and len(cells[j]) == len(cells[i]) + 1:
                rows.append(cells[j])
                j += 1
            if len(rows) >= 2:
                hdr = ["", *cells[i]]
                table = ["", "| " + " | ".join(hdr) + " |", "|" + "---|" * len(hdr),
                         *["| " + " | ".join(r) + " |" for r in rows], ""]
                diag.table_repairs.append({"pdf_page": page, "table": "column table", "rows": len(rows)})
                return lines[:i] + table + lines[j:]
        i += 1
    return lines


# --------------------------------------------------------------------------
# Line Unwrapping & Hyphenation
# --------------------------------------------------------------------------
def _join_wrapped(parts: list[str], ev: _HyphenEvidence | None = None) -> str:
    out = ""
    for p in parts:
        p = _collapse(p)
        if not p:
            continue
        if out.endswith("-") and re.search(r"[A-Za-z]-$", out) and re.match(r"[A-Za-z]", p):
            pre = re.search(r"([A-Za-z]+)-$", out).group(1)
            suf = re.match(r"[A-Za-z]+", p).group(0)
            keep = ev.keep_hyphen(pre, suf) if ev else (suf[:1].isupper())
            out = (out if keep else out[:-1]) + p
        else:
            out = f"{out} {p}" if out else p
    return out


def _unwrap(lines: list[str], ev: _HyphenEvidence) -> list[str]:
    """Group wrapped PDF lines into paragraphs. The raw text has no blank lines between
    paragraphs, so a new paragraph starts at indentation, an article number, a clause label,
    a proviso / explanation, or a structural / standalone heading line. A false break only
    adds a newline; a false join merges unrelated provisions, so breaks are preferred."""
    paras: list[list[str]] = []
    force_break = False
    for line in lines:
        if not line.strip():
            if paras and paras[-1]:
                paras.append([])
            force_break = False
            continue
        if line.startswith(("## ", "|")):                    # headings and table rows stand alone
            paras.append([line])
            paras.append([])
            continue
        standalone = bool(_STRUCT_LINE_RE.match(line) or _STANDALONE_LINE_RE.match(line))
        if not paras or not paras[-1] or force_break or standalone or _PARA_START_RE.match(line):
            paras.append([line])
        else:
            paras[-1].append(line)
        force_break = standalone
    return [_join_wrapped(p, ev) if p else "" for p in paras]


def _normalize_amendment_brackets(text: str) -> str:
    opener = re.compile(f"{_REF}\\[")
    result: list[str] = []
    i, n = 0, len(text)
    while i < n:
        m = opener.match(text, i)
        if m:
            depth, j = 1, m.end()
            start_inner = j
            while j < n and depth > 0:
                if text[j] == "[":
                    depth += 1
                elif text[j] == "]":
                    depth -= 1
                j += 1
            if depth == 0:
                ref = text[m.start():m.end() - 1]
                inner = _normalize_amendment_brackets(text[start_inner:j - 1])
                if _BRACKET_CONTENT_IS_CLAUSE_NUMBER_RE.match(inner):
                    result.append(ref + inner)
                else:
                    result.append(f"{ref}[{inner}]")
                i = j
                continue
        result.append(text[i])
        i += 1
    return "".join(result)


def _mark_structure(text: str, diag: Diagnostics) -> str:
    """PART / CHAPTER / SCHEDULE / ANNEX lines -> '# ' headings ('### ' for a PART inside a
    Schedule). A short title on the next line is folded into the heading. No letters are
    added, so letter preservation still holds."""
    lines = text.split("\n")
    out: list[str] = []
    in_schedules = False
    i = 0
    while i < len(lines):
        line = lines[i]
        if _STRUCT_LINE_RE.match(line) and not line.startswith("#"):
            if _SCHEDULE_OR_ANNEX_RE.match(line):
                in_schedules = True
            level = "###" if (in_schedules and _PART_RE.match(line)) else "#"
            heading = line.strip()
            j = i + 1
            while j < len(lines) and not lines[j].strip():
                j += 1
            nxt = lines[j].strip() if j < len(lines) else ""
            if (nxt and len(nxt) <= 100 and not nxt.startswith(("#", "|", "(", "[Article"))
                    and not re.match(r"\d", nxt) and not _STRUCT_LINE_RE.match(nxt)
                    and not nxt.endswith((";", ":", ",", "–", "—")) and not _ARTICLE_HEAD_RE.match(nxt)):
                heading = f"{heading} {nxt}"
                i = j
            out.append(f"{level} {heading}")
            diag.structural_headings += 1
        else:
            out.append(line)
        i += 1
    return "\n".join(out)


# --------------------------------------------------------------------------
# Footnote Enrichment
# --------------------------------------------------------------------------
_FN_OCR_FIXES = [
    (re.compile(r"\blns\."), "Ins."),                # lowercase L read instead of capital I
    (re.compile(r"\bProvisios\b"), "Provisos"),
]
_ACTION_RE = re.compile(
    r"\b(subs(?:tituted)?|ins(?:erted)?|added|omitted|renumbered|deleted|repealed|re-?lettered|"
    r"re-?numbered)\b\.?", re.I)
_ACTION_CANON = {"subs": "substituted", "substituted": "substituted", "ins": "inserted",
                 "inserted": "inserted", "added": "added", "omitted": "omitted", "deleted": "omitted",
                 "repealed": "repealed", "renumbered": "renumbered", "re-numbered": "renumbered",
                 "relettered": "relettered", "re-lettered": "relettered"}
_INSTRUMENT_PATTERNS = [
    # Constitution (Eighteenth Amdt.) Act, 2010  /  Constitution (Amdt.) Order, 1982
    (re.compile(r"Constitution\s*\(((?:[\w-]+\s+)?)(?:Amdt\.?|Amendment)\)\s*(Act|Order)\s*,?\s*(\d{4})"),
     lambda m: (f"Constitution ({(m.group(1).strip().title().replace('Sixthy', 'Sixth') + ' ') if m.group(1).strip() else ''}Amdt.) "
                f"{m.group(2)}, {m.group(3)}", m.group(3))),
    # Act No. 10 of 2010  /  Act 10 of 2010  /  Act. No. 25 of 2016
    (re.compile(r"\bAct\.?\s*(?:No\.\s*)?(\d+)\s+of\s+(\d{4})"),
     lambda m: (f"Act No. {m.group(1)} of {m.group(2)}", m.group(2))),
    # P.O. No. 14 of 1985  /  P. O. No. 5 of 1982
    (re.compile(r"\bP\.\s*O\.\s*(?:No\.\s*)?(\d+)\s+of\s+(\d{4})"),
     lambda m: (f"P.O. No. {m.group(1)} of {m.group(2)}", m.group(2))),
    # Named instrument after "by": Revival of the Constitution of 1973 Order, 1985
    (re.compile(r"\bby\s+(?:the\s+)?([A-Z][\w().,'’ -]{3,90}?(?:Order|Ordinance|Act))\s*,?\s+(\d{4})"),
     lambda m: (f"{_collapse(m.group(1))}, {m.group(2)}", m.group(2))),
]


def _find_instrument(text: str) -> tuple[str | None, int | None]:
    for rx, build in _INSTRUMENT_PATTERNS:
        m = rx.search(text)
        if m:
            name, year = build(m)
            return name, int(year)
    return None, None


_SEE_FOOTNOTE_RE = re.compile(r"\bSee\s+(?:footnote|f\.\s*n\.)\s*(\d+)\s+on\s+page\s+(\d+)", re.I)


def _enrich_footnotes(footnotes: list[Footnote], diag: Diagnostics) -> None:
    by_id = {f.id: f for f in footnotes}
    prev_inst, prev_year = None, None
    for f in footnotes:
        for rx, repl in _FN_OCR_FIXES:
            new = rx.sub(repl, f.text)
            if new != f.text:
                diag.footnote_ocr_fixes.append({"id": f.id, "before": f.text[:60], "after": new[:60]})
                f.text = new

        head = re.split(r"\bby\b|\bibid\b", f.text, maxsplit=1)[0]
        f.actions = sorted({_ACTION_CANON[a.lower().replace("-", "")] if a.lower().replace("-", "") in _ACTION_CANON
                            else a.lower() for a in _ACTION_RE.findall(head)})

        inst, year = _find_instrument(f.text)
        if inst:
            f.amending_instrument, f.amending_year = inst, year
        elif re.search(r"\bibid\b", f.text) and prev_inst:
            f.amending_instrument, f.amending_year = prev_inst, prev_year
            f.instrument_inherited = True
            diag.footnotes_ibid_resolved += 1

        sm = _SEE_FOOTNOTE_RE.search(f.text)
        if sm:
            target = f"{sm.group(2)}.{sm.group(1)}"
            f.see_also = target
            t = by_id.get(target)
            if t is not None:
                diag.footnotes_see_also_resolved += 1
                if not f.amending_instrument:
                    f.amending_instrument, f.amending_year = t.amending_instrument, t.amending_year
                    f.actions = f.actions or t.actions
                    f.instrument_inherited = True

        if f.amending_instrument:
            prev_inst, prev_year = f.amending_instrument, f.amending_year
    diag.footnotes_without_instrument = sum(1 for f in footnotes if not f.amending_instrument)


# --------------------------------------------------------------------------
# Index text (for BM25 / embeddings; canonical text stays untouched)
# --------------------------------------------------------------------------
def normalize_for_index(text: str) -> str:
    """Strip footnote refs and amendment brackets, drop '* * *' omission runs and unify
    dashes / quotes. Use for BM25 and embedding text; never for display or citation."""
    prev = None
    while prev != text:                                      # unwrap [^id][content] inner-first
        prev = text
        text = re.sub(r"\[\^[^\]]+\]\[([^\[\]]*)\]", r"\1", text)
    text = re.sub(r"\[\^[^\]]+\]", "", text)
    text = re.sub(r"(?:\*[ \t]*){3,}", " ", text)
    text = text.replace("–", "-").replace("—", "-").replace("―", "-")
    text = text.replace("“", '"').replace("”", '"').replace("‘", "'").replace("’", "'")
    text = re.sub(r"[ \t]{2,}", " ", text)
    return re.sub(r"[ \t]+\n", "\n", text).strip()


# --------------------------------------------------------------------------
# Main Core Cleaning Process
# --------------------------------------------------------------------------
def clean_constitution_pages(pages: list[dict]) -> tuple[str, list[Footnote], list[dict], Diagnostics]:
    diag = Diagnostics()
    pages = sorted(pages, key=lambda p: p["page_number"])
    full_raw = "\n\n".join(p["text"] for p in pages)
    diag.raw_length = len(full_raw)

    start = _find_body_preamble_start(pages)
    toc = _parse_toc([p["text"] for p in pages[:start]])
    titles = dict(toc)
    tracker = _ArticleTracker(toc)
    diag.articles_in_toc = len(toc)
    ev = _HyphenEvidence(full_raw)
    broken_pages = _detect_broken_caps_pages(pages[start:])
    diag.broken_caps_pages = sorted(broken_pages)
    caps = _CapitalsRepair("\n".join(p["text"] for p in pages[start:]
                                      if p["page_number"] not in broken_pages))

    body_lines: list[str] = []
    footnotes: list[Footnote] = []
    article_index: list[dict] = []
    prev_printed = None

    for p in pages[start:]:
        n = p["page_number"]
        lines, printed = _strip_page_furniture(p["text"].split("\n"))
        if printed is None:
            printed = str(int(prev_printed) + 1) if prev_printed and prev_printed.isdigit() else "1"
        prev_printed = printed
        if not any(l.strip() for l in lines):
            continue
        diag.pages_processed += 1

        body, entries, displaced = _split_footnote_zone(lines)
        fn_numbers = {num for num, _ in entries}
        used: set[int] = set()

        body = [_convert_markers(l, printed, fn_numbers, used) for l in body]
        displaced = [_convert_markers(l, printed, fn_numbers, used) for l in displaced]
        if displaced:
            diag.displaced_body_blocks_restored.append(n)
            body = body + displaced

        heads = []
        for i, l in enumerate(body):
            new, art = tracker.match(l, printed, fn_numbers, used, diag)
            if art:
                body[i] = new
                heads.append((i, art))
                article_index.append({"article": art, "title_toc": titles.get(art),
                                      "status": _article_status(titles.get(art)),
                                      "present_in_body": True,
                                      "printed_page": printed, "pdf_page": n})
        body = _attach_marginal_notes(body, heads, titles, diag, ev)
        body = _repair_seat_table(body, diag, n)
        body = _repair_column_table(body, diag, n)
        body = [caps.fix_small_caps(l) for l in body]

        if n in broken_pages:
            fixed = []
            for k, l in enumerate(body):
                prev_line = fixed[-1] if fixed else (body_lines[-1] if body_lines else "")
                ends_sentence = bool(re.search(r"[.?!][’”'\")\]]*\s*$", prev_line)) \
                    and not _ABBREV_END_RE.search(prev_line.rstrip())
                starts_para = bool(_PARA_START_RE.match(l)) or not prev_line.strip() or ends_sentence
                fixed.append(caps.fix_page_line(l, starts_para, False))
            body = fixed

        body = [l for l in body if not _DIVIDER_LINE_RE.match(l) and not _SEPARATOR_RE.match(l)]
        body_lines.extend(body)

        for num, ent in entries:
            first = _FN_START_RE.sub("", ent[0], count=1)
            text = _join_wrapped([first] + ent[1:], ev)
            fid = f"{printed}.{num}"
            footnotes.append(Footnote(id=fid, pdf_page=n, printed_page=printed, number=num, text=text,
                                      referenced=num in used))

    paras = _unwrap(body_lines, ev)
    text = "\n".join(paras)
    text = _normalize_amendment_brackets(text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    for _ in range(2):                                           # keep table rows contiguous
        text = re.sub(r"(?m)^(\|.*\|)\n\n(?=\|)", r"\1\n", text)
    text = text.replace(" ", " ").replace("‘‘", "“").replace("’’", "”")

    text = _REF_SENTINEL_RE.sub(lambda m: f"[^{m.group(1)}]", text)
    text = _mark_structure(text, diag)

    diag.footnote_refs = len(re.findall(r"\[\^[^\]]+\]", text))
    referenced = set(re.findall(r"\[\^([^\]]+)\]", text))
    for f in footnotes:
        f.referenced = f.id in referenced
    diag.footnotes = len(footnotes)
    diag.unreferenced_footnotes = [f.id for f in footnotes if not f.referenced]
    diag.unresolved_markers = sorted(referenced - {f.id for f in footnotes})

    # Article bookkeeping ------------------------------------------------------------
    found = set(tracker.found)
    diag.articles_found = len(tracker.found)
    diag.articles_missing = [a for a in tracker.ids if a not in found]
    diag.articles_missing_unexpected = [a for a in diag.articles_missing
                                        if _article_status(titles.get(a)) == "in_force"]
    for a in diag.articles_missing:                              # e.g. 270C: only '* * *' + a footnote
        article_index.append({"article": a, "title_toc": titles.get(a),
                              "status": _article_status(titles.get(a)), "present_in_body": False,
                              "printed_page": None, "pdf_page": None})
    order = {a: i for i, a in enumerate(tracker.ids)}
    article_index.sort(key=lambda r: order.get(r["article"], 10**6))

    heading_titles = {m.group(1): m.group(2) for m in re.finditer(r"(?m)^## (\d+[A-Z]{0,3})\. (.*)$", text)}
    synthetic = set(diag.headings_from_toc)
    for row in article_index:
        h = heading_titles.get(row["article"])
        row["heading"] = h
        row["title"] = normalize_for_index(h) if h else (_collapse(row["title_toc"] or "") or None)
        row["title_source"] = (None if not h else
                               "toc" if f"## {row['article']}. {h}" in synthetic else "marginal_note")
    diag.articles_without_heading = [a for a in tracker.found if a not in heading_titles]
    diag.capitalisation_repairs = caps.changes
    diag.max_line_chars = max((len(l) for l in text.split("\n") if not l.startswith("|")), default=0)

    return text, footnotes, article_index, diag


def verify_letter_preservation(pages: list[dict], body: str, footnotes: list[Footnote],
                               synthetic_lines: list[str] | None = None) -> tuple[bool, dict]:
    """Every letter of the raw body + footnotes must survive cleaning, and no letters may be
    invented. Headings synthesised from the TOC are excluded (they are not page text)."""
    pages = sorted(pages, key=lambda p: p["page_number"])
    start = _find_body_preamble_start(pages)
    raw = []
    for p in pages[start:]:
        lines, _ = _strip_page_furniture(p["text"].split("\n"))
        raw.append("\n".join(lines))
    a = Counter(re.sub(r"[^a-z]", "", "\n".join(raw).lower()))
    synthetic = set(synthetic_lines or [])
    body_lines = [l for l in body.split("\n") if l not in synthetic]
    body_clean = "\n".join(body_lines)
    body_clean = re.sub(r"(?m)^## \d+[A-Z]{0,3}\. ", "", body_clean)
    body_clean = re.sub(r"(?m)^#{1,3} ", "", body_clean)
    cleaned = re.sub(r"\[\^[^\]]+\]", "", body_clean) + "".join(f.text for f in footnotes)
    b = Counter(re.sub(r"[^a-z]", "", cleaned.lower()))
    missing, extra = dict(a - b), dict(b - a)
    return (not missing and not extra), {"missing": missing, "extra": extra}


def quality_gate(doc: dict) -> list[str]:
    """Return a list of blocking problems; empty means the document is safe to index."""
    d = doc["cleaning_diagnostics"]
    body = doc["body_text"]
    errors = []
    if not d["letter_preservation_ok"]:
        errors.append(f"letter preservation failed: {d['letter_preservation_diff']}")
    n_heads = len(re.findall(r"(?m)^## \d+[A-Z]{0,3}\. ", body))
    expected = d["marginal_notes_attached"] + len(d["headings_from_toc"])
    if n_heads != expected:
        errors.append(f"{n_heads} article headings at line start, expected {expected}")
    if re.search(r"[^\n]## \d+[A-Z]{0,3}\. ", body):
        errors.append("article heading found mid-line (paragraph unwrap regression)")
    if d["articles_without_heading"]:
        errors.append(f"articles without heading: {d['articles_without_heading']}")
    if d["articles_missing_unexpected"]:
        errors.append(f"in-force articles missing from body: {d['articles_missing_unexpected']}")
    if d["max_line_chars"] > MAX_LINE_CHARS:
        errors.append(f"longest line is {d['max_line_chars']} chars (> {MAX_LINE_CHARS})")
    if d["unresolved_markers"]:
        errors.append(f"footnote refs without a footnote: {d['unresolved_markers'][:10]}")
    if d["unreferenced_footnotes"]:
        errors.append(f"footnotes never referenced: {d['unreferenced_footnotes'][:10]}")
    if d["large_removal_flag"]:
        errors.append(f"removed {d['pct_removed']:.1%} of raw text")
    return errors


def clean_constitution_document(raw_doc: dict) -> dict:
    body, footnotes, article_index, diag = clean_constitution_pages(raw_doc["pages"])
    ok, d = verify_letter_preservation(raw_doc["pages"], body, footnotes, diag.headings_from_toc)
    diag.letter_preservation_ok, diag.letter_preservation_diff = ok, d

    # Enrichment runs after verification: OCR fixes intentionally change letters ("lns." -> "Ins.").
    _enrich_footnotes(footnotes, diag)

    block = "\n".join(f"[^{f.id}]: {f.text}" for f in footnotes)
    cleaned_text = f"{body}\n\n[FOOTNOTES]\n{block}\n[/FOOTNOTES]" if footnotes else body
    diag.cleaned_length = len(cleaned_text)

    doc = {
        "doc_id": raw_doc["doc_id"],
        "metadata": dict(raw_doc["metadata"]),
        "raw_text": raw_doc["raw_text"],
        "cleaned_text": cleaned_text,
        "body_text": body,
        "index_text": normalize_for_index(body),
        "footnotes": [f.to_dict() for f in footnotes],
        "article_index": article_index,
        "cleaning_diagnostics": diag.to_dict(),
    }
    errors = quality_gate(doc)
    doc["cleaning_diagnostics"]["quality_gate_errors"] = errors
    return doc


def strip_footnote_refs(text: str) -> str:
    """Drops page-scoped footnote-reference markers ("[^5.2]", "[^91.8]")
    from `text`, for use by downstream consumers that need a reference-
    free string (e.g. constitution_chunker.py's breadcrumb titles, which
    come from `article_index`/`## <id>. <title>` heading text that can
    carry one of these markers -- see clean_constitution_pages' own
    "[^15.2]25A. The State shall provide ..." convention documented in
    this module's docstring). Only the "[^<id>]" sentinel wrapper itself
    is removed; a substantive amendment bracket with no leading caret,
    e.g. "[the Federal Constitutional Court,]", is a different shape
    (`\\[\\^` vs `\\[`) and is left untouched -- confirmed against the
    real Article 175A heading, which contains both forms adjacently:
    "... to [^91.8][the Federal Constitutional Court,] the Supreme
    Court..." strips to "... to [the Federal Constitutional Court,] the
    Supreme Court...", never losing the bracketed legal text itself.
    """
    return re.sub(r"\[\^[^\]]+\]", "", text)


def clean_directory(input_dir: Path = INPUT_DIR, output_dir: Path = OUTPUT_DIR) -> dict:
    input_dir, output_dir = Path(input_dir), Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rejected_dir = output_dir / "_rejected"
    processed, flagged, failed_docs, rejected_docs = 0, 0, [], []

    for raw_path in sorted(input_dir.glob("*.json")):
        try:
            raw_doc = json.loads(raw_path.read_text(encoding="utf-8"))
            cleaned_doc = clean_constitution_document(raw_doc)
        except (OSError, json.JSONDecodeError, KeyError, ValueError) as exc:
            failed_docs.append({"source_file": raw_path.name, "error": str(exc)})
            continue

        errors = cleaned_doc["cleaning_diagnostics"]["quality_gate_errors"]
        target_dir = output_dir
        if errors:
            rejected_dir.mkdir(parents=True, exist_ok=True)
            target_dir = rejected_dir
            rejected_docs.append({"source_file": raw_path.name, "errors": errors})

        out_path = target_dir / f"{cleaned_doc['doc_id']}.json"
        tmp_path = out_path.with_suffix(".json.tmp")
        tmp_path.write_text(json.dumps(cleaned_doc, ensure_ascii=False, indent=2), encoding="utf-8")
        tmp_path.replace(out_path)

        processed += 1
        if cleaned_doc["cleaning_diagnostics"]["large_removal_flag"]:
            flagged += 1

    return {"processed": processed, "failed": len(failed_docs), "rejected": len(rejected_docs),
            "large_removal_flagged": flagged, "failed_docs": failed_docs, "rejected_docs": rejected_docs}


def main() -> None:
    summary = clean_directory()
    print(f"Constitution cleaning completed: {summary['processed']} processed, {summary['failed']} failed, "
          f"{summary['rejected']} rejected by quality gate, "
          f"{summary['large_removal_flagged']} flagged for large removal.")
    for f in summary["failed_docs"]:
        print(f"  FAILED {f['source_file']}: {f['error']}")
    for r in summary["rejected_docs"]:
        print(f"  REJECTED {r['source_file']}:")
        for e in r["errors"]:
            print(f"    - {e}")
    if summary["failed"] or summary["rejected"]:
        sys.exit(1)


if __name__ == "__main__":
    main()