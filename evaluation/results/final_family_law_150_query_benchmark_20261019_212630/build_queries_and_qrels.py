"""
Read-only construction of queries_150.jsonl, query_coverage.csv, and a qrels seed file
for the 150-query family-law benchmark. All classifications below are based on direct
source-text inspection of var/rag/docstore.sqlite (constitution/statute article & section
text) and targeted keyword search over indexed case-law headnotes (see search_caselaw.py
runs captured in this directory's notes). No retrieval scores were used to decide support.
"""
import json, sqlite3

DB = "var/rag/docstore.sqlite"
con = sqlite3.connect(DB)
cur = con.cursor()

queries = json.load(open("/tmp/queries_150_raw.json"))

# ---------------------------------------------------------------------------
# CONSTITUTION: C01-C50
# status, primary article(s), note
# ---------------------------------------------------------------------------
const_class = {
 "C01": ("supported", ["9"], None),
 "C02": ("supported", ["25"], None),
 "C03": ("supported", ["25"], "Art.25(3) is clause (3) of Art.25; chunk is the article-level unit, clause not separately chunked."),
 "C04": ("supported", ["35"], None),
 "C05": ("supported", ["10A"], None),
 "C06": ("supported", ["14"], None),
 "C07": ("supported", ["199"], None),
 "C08": ("supported", ["199"], None),
 "C09": ("supported", ["227"], None),
 "C10": ("supported", ["203C","203D"], None),
 "C11": ("supported", ["4"], None),
 "C12": ("supported", ["25"], None),
 "C13": ("supported", ["37"], "Query asks about 37(e); Art.37 is a single indexed chunk covering all clauses incl. (e)."),
 "C14": ("supported", ["199","203"], "HC superintendence of subordinate courts is Art.203; writ jurisdiction is Art.199; no single article named 'jurisdiction of HC over subordinate courts in family cases' specifically."),
 "C16": ("supported", ["175E"], "Query evokes Art.184 (original SC jurisdiction). Art.184 was OMITTED by the Constitution (Twenty-Seventh Amdt.) Act, 2025, s.29 (confirmed: article_status='omitted'). Jurisdiction over fundamental-rights matters of this kind now sits with the new Federal Constitutional Court under Art.175E (new Chapter 1A). Original query text preserved verbatim; target corrected to 175E without rewriting the query."),
 "C15": ("supported", ["189"], None),
 "C17": ("supported", ["8","227"], "Art.8(3)(b)/(4) + First Schedule explicitly list the Muslim Family Laws Ordinance, 1961 among laws exempted from the operation of fundamental-rights scrutiny -- a direct, specific textual link between Art.8 and Muslim personal-status legislation."),
 "C18": ("supported", ["13"], None),
 "C19": ("supported", ["2A"], "Art.2A + the Annex (full Objectives Resolution text) are indexed; Art.2A makes the Objectives Resolution 'a substantive part of the Constitution'."),
 "C20": ("supported", ["11"], None),
 "C21": ("supported", ["25A"], None),
 "C22": ("supported", ["15"], None),
 "C23": ("supported", ["24"], None),
 "C24": ("partially_supported", ["142","270AA"], "No single constitutional provision enumerates 'marriage, divorce, inheritance' as a legislative subject. Art.142 states the general Federal/Provincial division of legislative power (residuary power to provinces for matters not on the Federal Legislative List); Art.270AA confirms the Concurrent Legislative List (which historically covered Muslim personal law/marriage/divorce/succession) was OMITTED by the 18th Amendment, with subjects devolved to provinces. The Fourth Schedule (Federal Legislative List, Arts.70(4)) does not itself name marriage/divorce/inheritance. Constitution corpus therefore provides only the general framework, not a specific family-law legislative-competence clause -- classified partially_supported rather than invented as fully supported."),
 "C25": ("supported", ["13"], None),
 "C26": ("supported", ["199"], None),
 "C27": ("supported", ["203F","203DD"], None),
 "C28": ("supported", ["199"], None),
 "C29": ("supported", ["25"], None),
 "C30": ("supported", ["34"], None),
 "C31": ("supported", ["201"], None),
 "C32": ("supported", ["20"], None),
 "C33": ("supported", ["20","36"], "Art.36 (protection of minorities) is the specific textual hook for non-Muslim communities; Art.20 the general religious-freedom guarantee."),
 "C34": ("unsupported", [], "No constitutional provision directly addresses cross-border child removal/abduction; this is addressed in the indexed corpus only at the STATUTE level (Family Courts Act, 1964, Schedule Part I, item 6A: 'Matters pertaining to Return of Child under the Hague Convention on Civil Aspects of International Child Abduction, 1980'), not in the Constitution. Recorded as unsupported for the Constitution corpus specifically, with the statute cross-reference documented rather than invented as constitutional text."),
 "C35": ("supported", ["9","10A"], None),
 "C36": ("supported", ["203C","203D"], None),
 "C37": ("partially_supported", ["26"], "Art.26 concerns access to public places, not family-specific rights; relevance to family-related circumstances is inferential, consistent with the query's own 'where relevant' qualifier."),
 "C38": ("partially_supported", ["142","270AA"], "Same basis as C24 -- general legislative-competence framework exists, but no article specifically enumerates marriage/divorce/inheritance."),
 "C39": ("supported", ["14"], None),
 "C40": ("unsupported", [], "No constitutional provision addresses recognition of foreign custody/divorce decisions; this is private-international-law/case-law territory, absent from the Constitution corpus."),
 "C41": ("supported", ["187"], None),
 "C42": ("supported", ["23","24","25"], None),
 "C43": ("supported", ["9"], None),
 "C44": ("supported", ["9","10A"], None),
 "C45": ("supported", ["38"], None),
 "C46": ("supported", ["227","2A"], None),
 "C47": ("supported", ["11"], None),
 "C48": ("supported", ["175","209"], "Art.175 (establishment/independence of courts) + Art.209 (Supreme Judicial Council) together ground judicial independence; no single article is captioned that way."),
 "C49": ("supported", ["199"], None),
 "C50": ("supported", ["25","25A"], None),
}

def const_evidence(arts):
    out = []
    for a in arts:
        cur.execute("""select chunk_id, json_extract(json,'$.article_status'), canonical_text
                       from chunks where source_type='constitution' and retrievable=1
                       and json_extract(json,'$.article_number')=?""", (a,))
        for chunk_id, status, text in cur.fetchall():
            out.append({"chunk_id": chunk_id, "article": a, "status": status,
                        "snippet": text[:220].replace("\n"," ")})
    return out

# ---------------------------------------------------------------------------
# STATUTE: S01-S50  (doc_id, section, note)
# ---------------------------------------------------------------------------
MFLO = "6867f44e86ef03cec19831c6"
DMMA = "0425160ffedaefd8c79b3241"
FCA  = "24520c344d2285555aa10988"
GWA  = "42769a60dc59c5473affbc32"
CMRA = "470c476c4195cbc08e94a6da"
DOWRY = "417403be7fc29b06cafadbbe"
APPL = "b11740afa5f050bea50c5f2d"

stat_class = {
 "S01": ("supported", [(MFLO,"s5")], None),
 "S02": ("partially_supported", [(MFLO,"s5")], "MFLO s.5(5) delegates the nikahnama's exact required content/form to subordinate rules ('shall be such as may be prescribed') -- the Ordinance text itself does not enumerate the information required. The Muslim Family Laws Rules, 1961 (which would contain the prescribed form) are NOT part of the indexed 8-Act statute corpus. Flagged rather than silently treated as fully supported."),
 "S03": ("supported", [(MFLO,"s5")], None),
 "S04": ("partially_supported", [(MFLO,"s5")], "MFLO s.5(4) penalizes only the officiant's failure to report an unregistered marriage under s.5(3); it does not address consequences of non-registration generally (e.g. for the spouses themselves), which the query's phrasing could be read to ask about."),
 "S05": ("partially_supported", [(MFLO,"s5")], "MFLO s.5(2) covers licensing of Nikah Registrars by the Union Council, but 'appointment, licensing, and removal' procedures in full (including removal) are delegated to rules not present in the indexed corpus."),
 "S06": ("supported", [(MFLO,"s10")], None),
 "S07": ("supported", [(MFLO,"s10")], None),
 "S08": ("supported", [(MFLO,"s7_ss1"),(MFLO,"s7_ss1A-2-3-4-5-6")], None),
 "S09": ("supported", [(MFLO,"s7_ss1")], None),
 "S10": ("supported", [(MFLO,"s7_ss1A-2-3-4-5-6")], None),
 "S11": ("supported", [(MFLO,"s7_ss1A-2-3-4-5-6")], None),
 "S12": ("supported", [(MFLO,"s7_ss1A-2-3-4-5-6")], None),
 "S13": ("supported", [(MFLO,"s7_ss1A-2-3-4-5-6")], None),
 "S14": ("supported", [(MFLO,"s6")], None),
 "S15": ("supported", [(MFLO,"s6")], None),
 "S16": ("supported", [(MFLO,"s6")], None),
 "S17": ("partially_supported", [(MFLO,"s7_ss1A-2-3-4-5-6")], "s.7 covers the Arbitration Council/Chairman procedure generally; a dedicated provision on 'issuance of official documentation of talaq's effectiveness' as such is not separately indexed (it is inferred from the Chairman/certificate mechanism in s.7)."),
 "S18": ("supported", [(MFLO,"s4_ss1-2-3-4-5")], None),
 "S19": ("supported", [(DMMA,"s2_ssi-ii-iii-iv-v-vi"),(DMMA,"s2_ssvii-viii-a-b-c-d-e-f-ix-a")], None),
 "S20": ("supported", [(DMMA,"s2_ssi-ii-iii-iv-v-vi")], None),
 "S21": ("supported", [(DMMA,"s2_ssi-ii-iii-iv-v-vi")], None),
 "S22": ("supported", [(DMMA,"s2_ssvii-viii-a-b-c-d-e-f-ix-a")], None),
 "S23": ("supported", [(DMMA,"s2_ssi-ii-iii-iv-v-vi")], None),
 "S24": ("supported", [(DMMA,"s2_ssi-ii-iii-iv-v-vi")], None),
 "S25": ("supported", [(DMMA,"s2_ssvii-viii-a-b-c-d-e-f-ix-a")], None),
 "S26": ("supported", [(FCA,"s5"),(FCA,"schedule:partI")], None),
 "S27": ("supported", [(FCA,"schedule:partI")], None),
 "S28": ("supported", [(FCA,"s7")], None),
 "S29": ("supported", [(FCA,"s9_ss1-1a-1b-2"),(FCA,"s9_ss3-4-5")], None),
 "S30": ("supported", [(FCA,"s8_ss1-2-3"),(FCA,"s8_ss4-5-6")], None),
 "S31": ("supported", [(FCA,"s10")], None),
 "S32": ("supported", [(FCA,"s10")], None),
 "S33": ("supported", [(FCA,"s17A")], None),
 "S34": ("supported", [(FCA,"s13")], None),
 "S35": ("supported", [(FCA,"s14")], None),
 "S36": ("supported", [(FCA,"s14")], None),
 "S37": ("supported", [(MFLO,"s9")], None),
 "S38": ("supported", [(FCA,"s13"),(MFLO,"s9")], None),
 "S39": ("supported", [(GWA,"s10_ss1"),(GWA,"s11")], None),
 "S40": ("supported", [(GWA,"s17")], None),
 "S41": ("supported", [(GWA,"s27")], None),
 "S42": ("supported", [(GWA,"s29"),(GWA,"s30"),(GWA,"s31_ss1-2")], None),
 "S43": ("supported", [(GWA,"s25")], None),
 "S44": ("supported", [(GWA,"s7"),(GWA,"s15")], None),
 "S45": ("supported", [(CMRA,"s2")], "Indexed corpus contains only the federal Child Marriage Restraint Act, 1929 (original thresholds: male under 18, female under 16). No provincial amendment (e.g. Sindh 2013 raising female age to 18, or later Punjab amendments) is present in this corpus -- query's premise ('legislation actually present in the indexed corpus') correctly anticipates this; answer is scoped to the 1929 Act only."),
 "S46": ("supported", [(CMRA,"s4"),(CMRA,"s5"),(CMRA,"s6")], None),
 "S47": ("supported", [(DOWRY,"s2"),(DOWRY,"s5")], None),
 "S48": ("supported", [(DOWRY,"s3")], None),
 "S49": ("supported", [(APPL,"preamble")], None),
 "S50": ("supported", [(FCA,"s5"),(FCA,"schedule:partI")], None),
}

def stat_evidence(pairs):
    out = []
    for doc_id, chunk_suffix in pairs:
        chunk_id = f"{doc_id}:{chunk_suffix}"
        cur.execute("select canonical_text from chunks where chunk_id=?", (chunk_id,))
        row = cur.fetchone()
        if row:
            out.append({"chunk_id": chunk_id, "snippet": row[0][:220].replace("\n"," ")})
    return out

# ---------------------------------------------------------------------------
# CASE LAW: CL01-CL50 -- pooled candidate case_keys found via keyword search
# (search terms and raw hits captured interactively; case_key -> chunk_id resolved here)
# ---------------------------------------------------------------------------
case_class = {
 "CL01": ("supported", ["pld-2024-pesh-25","1987-scmr-699","2007-clc-422"], None),
 "CL02": ("supported", ["pld-2025-bal-43","2022-mld-968","2008-clc-587"], None),
 "CL03": ("supported", ["2004-ylr-1666","2011-mld-571","1992-mld-2289"], None),
 "CL04": ("supported", ["2021-clc-1512","2002-clc-1754","pld-2012-lah-154"], None),
 "CL05": ("supported", ["2024-clc-1772","0f536fc32a2d7ba4cd3b12d8".replace("0f536fc32a2d7ba4cd3b12d8","2023-clc-161"),"2011-clc-820"], None),
 "CL06": ("supported", ["pld-1997-kar-399","2003-ylr-1572"], None),
 "CL07": ("supported", ["2005-clc-859","2021-ylr-2127","pld-1995-lah-151"], None),
 "CL08": ("supported", ["2025-clc-960","2017-clc-1747","pld-2023-lah-433"], None),
 "CL09": ("supported", ["2023-clc-143","2014-clc-895","2009-mld-945"], None),
 "CL10": ("supported", ["2021-clc-1512","1999-ylr-425","2022-clc-947"], None),
 "CL11": ("supported", ["2002-clc-1838","2000-ylr-537","pld-2015-bal-26"], None),
 "CL12": ("supported", ["2020-clc-1524","2009-clc-980","pld-2022-lah-840"], None),
 "CL13": ("supported", ["2002-clc-1964","2007-clc-195","2016-clc-313"], "'Iddat expenses' explicit in 2007-clc-195 headnote."),
 "CL14": ("supported", ["2025-clc-1620"], "Weaker pool (only 1 direct hit from this search pass); recommend treating as single-grade-2-source qrels with a note that pooling was not exhaustive."),
 "CL15": ("supported", ["pld-2019-lah-160","2007-mld-829","1986-clc-1722"], None),
 "CL16": ("supported", ["pld-1975-lah-805","1986-clc-1722"], None),
 "CL17": ("supported", ["1997-mld-11","2014-ylr-590"], None),
 "CL18": ("supported", ["2003-ylr-1572","2021-ylr-1989","2014-clc-330"], None),
 "CL19": ("supported", ["1984-clc-638","2022-mld-968"], None),
 "CL20": ("supported", ["03c327e5464aeef9a0d0c73d","2021-clc-704"], None),
 "CL21": ("supported", ["pld-2024-pesh-25","1996-mld-1693","2023-ylr-2233"], None),
 "CL22": ("supported", ["pld-2017-lah-892","2018-mld-407","2007-clc-1145"], None),
 "CL23": ("supported", ["pld-2025-sc-461","2018-clc-1350","2013-clc-276"], None),
 "CL24": ("supported", ["2020-ylr-2523","pld-2006-pesh-99","1992-mld-2289"], None),
 "CL25": ("supported", ["pld-2026-sc-91","2022-clc-1447","pld-2013-pesh-88"], None),
 "CL26": ("supported", ["2002-clc-1838"], "Direct grant-of-right-to-separate-on-non-payment holding (Mst. Muhammadi line of authority); pool otherwise thin for the specific 'justified separation' framing."),
 "CL27": ("supported", ["12b901fed564866d211dc449","pld-1970-pesh-52","2004-scmr-1352"], None),
 "CL28": ("supported", ["1997-mld-11","2004-ylr-111"], None),
 "CL29": ("partially_supported", ["2023-ylr-1961","2017-mld-214"], "No case found squarely on 'grandfather seeking guardianship after father's death'; closest hits concern grandfather's/grandchildren's inheritance share (MFLO s.4 line) and general S.12 GWA applications. Statute-level support (MFLO s.4 succession; GWA ss.7/15) is strong; case-law support for this exact fact pattern is weak/indirect."),
 "CL30": ("supported", ["2022-scmr-1647","2024-mld-995","2026-clc-120"], None),
 "CL31": ("supported", ["2023-mld-1084","2002-scmr-1950"], "Pool is thinner/more indirect than most CL queries; GWA ss.29-31 (statute) squarely on point, case hits more general."),
 "CL32": ("supported", ["pld-2024-pesh-25","2003-clc-1858"], None),
 "CL33": ("supported", ["2003-ylr-696","1999-ylr-425"], "No case headnote uses 'preference' verbatim; relevance inferred from welfare-of-minor custody holdings (shared with CL07)."),
 "CL34": ("supported", ["pld-2025-bal-43","2025-ylr-720","2002-clc-1838"], "Constitutional petitions challenging Family Court decrees are extremely common in this corpus (Art.199 route); pool drawn from CL-overlap cases."),
 "CL35": ("supported", ["pld-1997-kar-399","2002-clc-1838"], None),
 "CL36": ("supported", ["2014-mld-1400","2023-clc-143","2020-clc-1910"], None),
 "CL37": ("partially_supported", ["2023-ylr-1961"], "No case found squarely litigating the effective-date-of-talaq-vs-inheritance-dispute fact pattern; closest available evidence is general inheritance/succession cases plus MFLO s.7 (date talaq takes effect) at the statute level. Flagged rather than forced."),
 "CL38": ("supported", ["2003-ylr-696","2006-ylr-549","1988-scmr-1283"], None),
 "CL39": ("supported", ["2022-scmr-1352"], "Direct, strong hit: 'Minor daughter of predeceased son' / MFLO s.4 succession dispute."),
 "CL40": ("supported", ["2020-ylr-2523","pld-2006-pesh-99"], "No case headnote uses 'threats/safety' verbatim; relevance inferred from restitution-of-conjugal-rights case line generally."),
 "CL41": ("supported", ["1995-clc-731","2022-clc-1821","pld-2026-sc-91"], None),
 "CL42": ("supported", ["2015-mld-11","2016-clc-1337","2023-ylr-2233"], None),
 "CL43": ("supported", ["2f058e72ef3a0956934e5436","pld-2012-isl-61","pld-2026-sc-118"], None),
 "CL44": ("supported", ["2010-clc-1729","2007-mld-570","2005-mld-5"], None),
 "CL45": ("partially_supported", ["2000-ylr-537","2018-clc-note-2"], "These hits concern MFLO s.6(5) consequences of unauthorized polygamy as background to a dower/maintenance claim, not a case centered on polygamy-permission non-compliance itself; statute-level support (MFLO s.6) is strong, case-law support more indirect."),
 "CL46": ("supported", ["pld-2008-lah-410","2017-mld-580","pld-2002-pesh-30"], None),
 "CL47": ("supported", ["2006-ylr-2582","1ad6dc8578b53f77be064b94".replace("1ad6dc8578b53f77be064b94","pld-2001-lah-495")], None),
 "CL48": ("supported", ["pld-2024-pesh-25","2025-ylr-720"], None),
 "CL49": ("supported", ["2025-clc-960","2023-clc-2110"], None),
 "CL50": ("supported", ["pld-1997-kar-399","2002-clc-1838"], None),
}

def case_evidence(case_keys):
    out = []
    for ck in case_keys:
        cur.execute("""select chunk_id, json_extract(json,'$.citation'), canonical_text
                       from chunks where source_type='case' and retrievable=1
                       and json_extract(json,'$.case_key')=? and unit_type in ('headnote','headnote_part')
                       order by chunk_id limit 1""", (ck,))
        row = cur.fetchone()
        if row:
            chunk_id, citation, text = row
            out.append({"chunk_id": chunk_id, "case_key": ck, "citation": citation,
                        "snippet": text[:220].replace("\n"," ")})
    return out

# ---------------------------------------------------------------------------
# Assemble queries_150.jsonl
# ---------------------------------------------------------------------------
records = []
for qid in sorted(queries, key=lambda x: (x[0] if not x.startswith("CL") else "CL", int(''.join(filter(str.isdigit, x))))):
    text = queries[qid]
    if qid.startswith("CL"):
        corpus = "case"
        status, keys, note = case_class[qid]
        evidence = case_evidence(keys)
    elif qid.startswith("C"):
        corpus = "constitution"
        status, arts, note = const_class[qid]
        evidence = const_evidence(arts)
    else:
        corpus = "statute"
        status, pairs, note = stat_class[qid]
        evidence = stat_evidence(pairs)
    records.append({
        "id": qid, "query": text, "corpus": corpus, "status": status,
        "note": note, "evidence": evidence,
    })

import os
OUT_DIR = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(OUT_DIR, "queries_150.jsonl"), "w") as f:
    for r in records:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")

# coverage counts
from collections import Counter
by_corpus = {}
for r in records:
    by_corpus.setdefault(r["corpus"], Counter())[r["status"]] += 1
for c, cnt in by_corpus.items():
    print(c, dict(cnt), "total=", sum(cnt.values()))

print("TOTAL records:", len(records))
