"""Read-only validation of C01-C50 against the indexed Constitution corpus (docstore.sqlite)."""
import json, sqlite3, re

DB = "var/rag/docstore.sqlite"
queries = json.load(open("/tmp/queries_150_raw.json"))

con = sqlite3.connect(DB)
cur = con.cursor()

def get_article(num):
    cur.execute("""select chunk_id, json_extract(json,'$.article_status'), json_extract(json,'$.article_title'),
                   canonical_text from chunks where source_type='constitution' and retrievable=1
                   and json_extract(json,'$.article_number')=?""", (str(num),))
    return cur.fetchall()

# Map each C-query to the article(s) its text explicitly names
article_map = {
    "C01": ["9"], "C02": ["25"], "C03": ["25"], "C04": ["35"], "C05": ["10A"],
    "C06": ["14"], "C07": ["199"], "C08": ["199"], "C09": ["227"], "C10": ["203C","203D"],
    "C11": ["4"], "C12": ["25"], "C13": ["37"], "C14": ["198","199","203"], "C15": ["189"],
    "C16": ["184","175E"], "C17": ["8","227"], "C18": ["13"], "C19": [], "C20": ["11"],
    "C21": ["25A"], "C22": ["15"], "C23": ["24"], "C24": [], "C25": ["13"],
    "C26": ["199"], "C27": ["203F","203DD"], "C28": ["199"], "C29": ["25"], "C30": ["34"],
    "C31": ["201"], "C32": ["20"], "C33": ["20","36"], "C34": [], "C35": ["9","10A"],
    "C36": ["203D","203C"], "C37": ["26"], "C38": [], "C39": ["14"], "C40": [],
    "C41": ["187"], "C42": ["23","24","25"], "C43": ["9"], "C44": ["9","10A"],
    "C45": ["38"], "C46": ["227","2A"], "C47": ["11"], "C48": ["175","209"],
    "C49": ["199"], "C50": ["25","25A"],
}

out = {}
for qid, text in sorted(queries.items()):
    if not qid.startswith("C") or qid.startswith("CL"):
        continue
    arts = article_map.get(qid, [])
    evidence = []
    for a in arts:
        rows = get_article(a)
        for chunk_id, status, title, ctext in rows:
            evidence.append({
                "article": a, "chunk_id": chunk_id, "status": status, "title": title,
                "snippet": ctext[:300].replace("\n", " ")
            })
    out[qid] = {"query": text, "articles_checked": arts, "evidence": evidence}

json.dump(out, open("/tmp/constitution_validation_raw.json", "w"), indent=2, ensure_ascii=False)
for qid, v in out.items():
    print(qid, "arts=", v["articles_checked"], "hits=", len(v["evidence"]))
