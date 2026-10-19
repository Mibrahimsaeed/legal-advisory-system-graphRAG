"""Read-only keyword search helper over the case-law corpus (docstore.sqlite) for query validation."""
import sqlite3, sys, json

DB = "var/rag/docstore.sqlite"
con = sqlite3.connect(DB)
cur = con.cursor()

def search(terms, unit_types=("headnote","headnote_part"), limit=8):
    """terms: list of (AND-required substrings) OR'd across multiple term-sets."""
    seen_cases = {}
    results = []
    placeholders = " OR ".join(["canonical_text LIKE ?"] * len(terms))
    ut_placeholders = ",".join(["?"] * len(unit_types))
    q = f"""select chunk_id, case_key, citation, canonical_text from (
             select chunk_id, json_extract(json,'$.case_key') as case_key,
                    json_extract(json,'$.citation') as citation, canonical_text
             from chunks where source_type='case' and retrievable=1
             and unit_type in ({ut_placeholders}))
           where {placeholders}"""
    params = list(unit_types) + [f"%{t}%" for t in terms]
    cur.execute(q, params)
    rows = cur.fetchall()
    for chunk_id, case_key, citation, text in rows:
        if case_key not in seen_cases:
            seen_cases[case_key] = (chunk_id, citation, text[:220].replace("\n"," "))
    return list(seen_cases.items())[:limit]

if __name__ == "__main__":
    terms = sys.argv[1:]
    for case_key, (chunk_id, citation, snippet) in search(terms):
        print(f"{case_key} | {citation} | {chunk_id}\n    {snippet}\n")
