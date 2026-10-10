# """Constitution of Pakistan parent-child chunking -- Stage 4 of the
# independent Constitution pipeline.

#     parsed Constitution AST (var/rag/constitution_parsed/<doc_id>.json)
#         -> build_chunks_for_document()   [deterministic, token-count-driven, no LLM]
#         -> parent-child chunk JSON       (var/rag/constitution_chunked/<doc_id>.json)

# Converts the already-parsed hierarchical AST into retrieval-ready
# parent/child chunks:

#     Article, <=600 tokens   -> exactly one chunk (parent == child)
#     Article, >600 tokens    -> one parent (the complete Article) +
#                                  one child per Clause
#     Clause,  >400 tokens     -> that clause's single child is replaced by
#                                  one child per Sub-clause/Proviso (never
#                                  an arbitrary character/sentence split;
#                                  preserved whole if no such boundary exists)
#     Schedule / Preamble / Annex -> exactly one chunk each (the parser
#                                  identifies no item-level boundary inside
#                                  them, so none is invented here)

# Every chunk's ``canonical_text`` is an exact, untouched slice of the
# parsed document's own ``cleaned_text`` (never normalized or rewritten
# here); ``embedding_text`` adds a breadcrumb line built only from
# structural metadata the parser already captured, never invented.

# PIPELINE ISOLATION
# -------------------
# No import of src.ingestion/, src.classification/, src.rag_prep.
# case_cleaner/structurer*/chunk*/statute_*, or src.extraction.
# case_loader. Independent entry point, independent input/output
# directories. Never modifies the parser's or cleaner's output.
# """

# from __future__ import annotations

# import json
# from dataclasses import dataclass, field
# from pathlib import Path

# import tiktoken

# from src.rag_prep.constitution_cleaning import strip_footnote_refs

# INPUT_DIR = Path("var/rag/constitution_parsed")
# OUTPUT_DIR = Path("var/rag/constitution_chunked")

# # Task-specified thresholds -- not configurable elsewhere; see the
# # module docstring's "Core Chunking Rules" summary for their exact
# # effect at each level.
# ARTICLE_TOKEN_THRESHOLD = 600
# CLAUSE_TOKEN_THRESHOLD = 400

# DOCUMENT_TITLE = "Constitution of Pakistan"

# # cl100k_base: a stable, offline, model-agnostic tokenizer -- used
# # purely as a deterministic token-COUNTING yardstick for the chunking
# # thresholds above, never for any generative/LLM purpose.
# _ENCODING = tiktoken.get_encoding("cl100k_base")


# class ConstitutionChunkingError(ValueError):
#     """Raised when the input AST cannot be safely chunked -- e.g. its
#     own parse validation did not pass, or no Article/Schedule/Preamble/
#     Annex content exists anywhere. Per "fail validation rather than
#     silently dropping text": raised instead of chunking a known-bad or
#     empty structure."""


# def count_tokens(text: str) -> int:
#     return len(_ENCODING.encode(text))


# @dataclass(frozen=True)
# class Chunk:
#     doc_id: str
#     parent_id: str
#     child_id: str | None
#     part_number: str | None
#     part_title: str | None
#     chapter_number: str | None
#     chapter_title: str | None
#     article_number: str | None
#     article_title: str | None
#     clause_number: str | None
#     is_parent: bool
#     token_count: int
#     source_start: int
#     source_end: int
#     canonical_text: str
#     embedding_text: str

#     def to_dict(self) -> dict:
#         return {
#             "doc_id": self.doc_id, "parent_id": self.parent_id, "child_id": self.child_id,
#             "part_number": self.part_number, "part_title": self.part_title,
#             "chapter_number": self.chapter_number, "chapter_title": self.chapter_title,
#             "article_number": self.article_number, "article_title": self.article_title,
#             "clause_number": self.clause_number, "is_parent": self.is_parent,
#             "token_count": self.token_count, "source_start": self.source_start,
#             "source_end": self.source_end, "canonical_text": self.canonical_text,
#             "embedding_text": self.embedding_text,
#         }

#     @classmethod
#     def from_dict(cls, d: dict) -> "Chunk":
#         return cls(**d)


# # -- breadcrumb -----------------------------------------------------------------

# def _clean_title_for_breadcrumb(title: str | None) -> str | None:
#     """Strips the cleaning stage's footnote-reference sentinels
#     ("[^91.8]") out of a title string used ONLY inside the breadcrumb --
#     never applied to canonical_text or to a chunk's own stored
#     *_title metadata field, which stay verbatim/faithful to the parser's
#     own output. A raw "[^91.8]" is internal pipeline plumbing, not
#     constitutional wording, and left in would only add embedding noise;
#     stripping it is text hygiene for this one derived field, not
#     "inventing missing metadata" (the metadata itself is untouched)."""

#     if title is None:
#         return None
#     cleaned = strip_footnote_refs(title).strip()
#     return cleaned or None


# def _breadcrumb(
#     part_number: str | None, part_title: str | None, chapter_number: str | None, chapter_title: str | None,
#     article_number: str | None, article_title: str | None, clause_number: str | None,
# ) -> str:
#     part_title = _clean_title_for_breadcrumb(part_title)
#     chapter_title = _clean_title_for_breadcrumb(chapter_title)
#     article_title = _clean_title_for_breadcrumb(article_title)

#     segments = [DOCUMENT_TITLE]
#     if part_number:
#         segments.append(f"Part {part_number}: {part_title}" if part_title else f"Part {part_number}")
#     if chapter_number:
#         segments.append(f"Chapter {chapter_number}: {chapter_title}" if chapter_title else f"Chapter {chapter_number}")
#     if article_number:
#         segments.append(f"Article {article_number}: {article_title}" if article_title else f"Article {article_number}")
#     if clause_number:
#         segments.append(f"Clause ({clause_number})")
#     return "[" + " | ".join(segments) + "]"


# def _embedding_text(breadcrumb: str, canonical_text: str) -> str:
#     return f"{breadcrumb}\n\n{canonical_text}"


# # -- Part/Chapter title extraction (from the parser's own node text) -----------
# # The parser captures a Chapter's title directly (its own `heading`
# # field), but NOT a Part's -- a Part's subtitle ("Introductory", "The
# # Judicature", ...) is simply the first line of its own body text,
# # right after the bare "PART <roman>" line, with no dedicated capture
# # group in constitution_parser.py's _PART_RE (which only needed the
# # roman numeral to find boundaries, not the title). Deriving it here,
# # from text the parser already produced, is not "inventing missing
# # metadata" -- it is reading metadata the parser already preserved
# # verbatim, just not pre-extracted into its own field.
# def _part_title_from_text(part_text: str) -> str | None:
#     lines = part_text.split("\n", 2)
#     if len(lines) < 2:
#         return None
#     title = lines[1].strip()
#     return title or None


# # -- clause/sub-clause/proviso child construction --------------------------------

# def _chunk_simple_leaf(
#     node: dict, doc_id: str, parent_id: str, child_id: str,
#     part_number: str | None, part_title: str | None, chapter_number: str | None, chapter_title: str | None,
#     article_number: str | None, article_title: str | None, clause_number: str | None,
# ) -> Chunk:
#     breadcrumb = _breadcrumb(part_number, part_title, chapter_number, chapter_title, article_number, article_title, clause_number)
#     text = node["text"]
#     return Chunk(
#         doc_id=doc_id, parent_id=parent_id, child_id=child_id,
#         part_number=part_number, part_title=part_title, chapter_number=chapter_number, chapter_title=chapter_title,
#         article_number=article_number, article_title=article_title, clause_number=clause_number,
#         is_parent=False, token_count=count_tokens(text),
#         source_start=node["source_start"], source_end=node["source_end"],
#         canonical_text=text, embedding_text=_embedding_text(breadcrumb, text),
#     )


# def _flatten_clause_children(clause: dict, cleaned_text: str) -> list[dict]:
#     """Flattens a clause's own children by one level wherever a
#     sub-clause itself has nested children of its own.

#     The parser nests a trailing Proviso (or stray "text") under the
#     LAST sub-clause it immediately follows, rather than as a direct
#     sibling of the clause's sub-clauses -- there is simply no later
#     sub-clause boundary available to stop that sub-clause's own span
#     before the proviso begins. Confirmed real shape: Article 175A's
#     clause (2) ends "... (a) ... (b) ... Provided that ...", and the
#     proviso is parsed as a child of sub-clause (b), not of clause (2)
#     directly. The task's own example output ("Child 2 -> (b) ...",
#     "Child 3 -> Provided that ...") calls for these as separate
#     sibling children, so each sub-clause with nested children here has
#     its own span narrowed to stop at its first nested child's start,
#     and that nested child (promoted to this flat list, at the same
#     level) carries the rest."""

#     flat: list[dict] = []
#     for child in clause["children"]:
#         if child["type"] == "subclause" and child["children"]:
#             own_end = child["children"][0]["source_start"]
#             flat.append({**child, "source_end": own_end, "text": cleaned_text[child["source_start"]:own_end]})
#             flat.extend(child["children"])
#         else:
#             flat.append(child)
#     return flat


# def _chunk_clause(
#     clause: dict, cleaned_text: str, doc_id: str, parent_id: str, article_id: str,
#     part_number: str | None, part_title: str | None, chapter_number: str | None, chapter_title: str | None,
#     article_number: str, article_title: str | None,
# ) -> list[Chunk]:
#     clause_number = clause["number"]
#     clause_id = f"{article_id}_c{clause_number.lower()}"
#     token_count = count_tokens(clause["text"])

#     if token_count <= CLAUSE_TOKEN_THRESHOLD:
#         return [_chunk_simple_leaf(
#             clause, doc_id, parent_id, clause_id,
#             part_number, part_title, chapter_number, chapter_title, article_number, article_title, clause_number,
#         )]

#     # Oversized Clause Rule: split by sub-clause/proviso children the
#     # parser already identified (flattened -- see _flatten_clause_children).
#     # If none exist, the clause has no valid internal legal boundary --
#     # preserved whole rather than guessed at.
#     flat_children = _flatten_clause_children(clause, cleaned_text)
#     splittable = [c for c in flat_children if c["type"] in ("subclause", "proviso")]
#     if not splittable:
#         return [_chunk_simple_leaf(
#             clause, doc_id, parent_id, clause_id,
#             part_number, part_title, chapter_number, chapter_title, article_number, article_title, clause_number,
#         )]

#     chunks: list[Chunk] = []
#     proviso_n = 0
#     seen_ids: dict[str, int] = {}
#     for child in flat_children:  # every child, in order -- union(children) == clause text
#         if child["type"] == "subclause":
#             cid = f"{clause_id}_{child['number'].lower()}"
#         elif child["type"] == "proviso":
#             proviso_n += 1
#             cid = f"{clause_id}_proviso" + (str(proviso_n) if proviso_n > 1 else "")
#         else:  # a "text" gap within the oversized clause -- kept, not dropped
#             cid = f"{clause_id}_text{len(chunks) + 1}"
#         # A single clause can restart its own lettering more than once --
#         # confirmed real case: Article 260 (Definitions) has several
#         # distinct defined terms, each with its own "(a)", "(b)", "(c)"
#         # sub-list, so plain letter-based IDs alone are not always
#         # unique within one clause. The common (non-repeating) case keeps
#         # its natural letter-based id; a repeat gets a disambiguating
#         # numeric suffix so every id still stays deterministic and unique.
#         if cid in seen_ids:
#             seen_ids[cid] += 1
#             cid = f"{cid}{seen_ids[cid]}"
#         else:
#             seen_ids[cid] = 1
#         chunks.append(_chunk_simple_leaf(
#             child, doc_id, parent_id, cid,
#             part_number, part_title, chapter_number, chapter_title, article_number, article_title, clause_number,
#         ))
#     return chunks


# def _chunk_article(
#     article: dict, cleaned_text: str, doc_id: str,
#     part_number: str | None, part_title: str | None, chapter_number: str | None, chapter_title: str | None,
# ) -> list[Chunk]:
#     article_number = article["article_number"]
#     article_title = article.get("heading")
#     article_id = f"art_{article_number.lower()}"
#     token_count = count_tokens(article["text"])

#     if token_count <= ARTICLE_TOKEN_THRESHOLD:
#         # Rule A: exactly one chunk -- parent IS the child.
#         chunk = _chunk_simple_leaf(
#             article, doc_id, article_id, article_id,
#             part_number, part_title, chapter_number, chapter_title, article_number, article_title, None,
#         )
#         return [Chunk(**{**chunk.to_dict(), "is_parent": True})]

#     # Rule B: one parent (the complete Article) + one child per Clause
#     # (each itself split further if oversized -- see _chunk_clause).
#     parent_id = f"{article_id}_parent"
#     breadcrumb = _breadcrumb(part_number, part_title, chapter_number, chapter_title, article_number, article_title, None)
#     parent_chunk = Chunk(
#         doc_id=doc_id, parent_id=parent_id, child_id=None,
#         part_number=part_number, part_title=part_title, chapter_number=chapter_number, chapter_title=chapter_title,
#         article_number=article_number, article_title=article_title, clause_number=None,
#         is_parent=True, token_count=token_count,
#         source_start=article["source_start"], source_end=article["source_end"],
#         canonical_text=article["text"], embedding_text=_embedding_text(breadcrumb, article["text"]),
#     )

#     chunks = [parent_chunk]
#     text_n = 0
#     for child in article["children"]:
#         if child["type"] == "clause":
#             chunks.extend(_chunk_clause(
#                 child, cleaned_text, doc_id, parent_id, article_id,
#                 part_number, part_title, chapter_number, chapter_title, article_number, article_title,
#             ))
#         else:  # a "text" gap directly under the Article (no recognized clause there)
#             text_n += 1
#             chunks.append(_chunk_simple_leaf(
#                 child, doc_id, parent_id, f"{article_id}_text{text_n}",
#                 part_number, part_title, chapter_number, chapter_title, article_number, article_title, None,
#             ))
#     return chunks


# def _chunk_top_level_unit(node: dict, doc_id: str) -> list[Chunk]:
#     """Preamble, Annex (Objectives Resolution), and Schedule chunks.

#     The task's own scope is explicit about Articles (Rule A/B) and
#     Schedules ("where the AST identifies individual items" -- it does
#     not, for this document; see constitution_parser.py's Schedule nodes,
#     each a single undivided span). The Preamble and Annex are handled
#     the SAME conservative way, for the same reason and per this
#     project's standing "never silently lose legal content" principle:
#     with no sub-structure in the AST to split on, each becomes exactly
#     one parent==child chunk, regardless of its own token count, rather
#     than an invented split.
#     """

#     node_id = node["id"]
#     heading = node.get("heading")
#     chunk = _chunk_simple_leaf(node, doc_id, node_id, node_id, None, None, None, None, None, heading, None)
#     return [Chunk(**{**chunk.to_dict(), "is_parent": True})]


# def build_chunks_for_document(parsed_doc: dict) -> list[Chunk]:
#     """Builds the full, ordered, deterministic chunk list for one
#     parsed Constitution document. Raises ConstitutionChunkingError if
#     the input AST's own validation did not pass, or if it contains no
#     chunkable content at all."""

#     if not parsed_doc.get("validation", {}).get("ok"):
#         raise ConstitutionChunkingError(
#             "Refusing to chunk a parsed Constitution document whose own structural "
#             f"validation did not pass: {parsed_doc.get('validation', {}).get('errors')}"
#         )

#     doc_id = parsed_doc["doc_id"]
#     cleaned_text = parsed_doc["cleaned_text"]
#     root = parsed_doc["structure"]

#     chunks: list[Chunk] = []
#     for top in root.get("children", []):
#         if top["type"] == "part":
#             part_number = top["id"].removeprefix("part_")
#             part_title = _part_title_from_text(top["text"])
#             for sub in top["children"]:
#                 if sub["type"] == "chapter":
#                     chapter_number = sub["id"].rsplit("_chapter", 1)[-1]
#                     chapter_title = sub.get("heading")
#                     for art in sub["children"]:
#                         if art["type"] == "article":
#                             chunks.extend(_chunk_article(art, cleaned_text, doc_id, part_number, part_title, chapter_number, chapter_title))
#                 elif sub["type"] == "article":
#                     chunks.extend(_chunk_article(sub, cleaned_text, doc_id, part_number, part_title, None, None))
#                 # "text" gaps directly under a Part are transitional PDF/layout
#                 # artifacts between Chapters/Articles, not an independently
#                 # citable legal unit -- not chunked (this stage's "no text
#                 # loss" guarantee is scoped to Articles/Schedules/Preamble/
#                 # Annex, matching the task's own validation checklist).
#         elif top["type"] in ("schedule", "preamble", "annex"):
#             chunks.extend(_chunk_top_level_unit(top, doc_id))

#     if not chunks:
#         raise ConstitutionChunkingError("No Article, Schedule, Preamble, or Annex content found to chunk.")

#     return chunks


# # -- validation -------------------------------------------------------------------

# @dataclass
# class ChunkValidationResult:
#     ok: bool
#     errors: list[str] = field(default_factory=list)
#     diagnostics: dict = field(default_factory=dict)

#     def to_dict(self) -> dict:
#         return {"ok": self.ok, "errors": self.errors, "diagnostics": self.diagnostics}


# def validate_chunks(chunks: list[Chunk], parsed_doc: dict) -> ChunkValidationResult:
#     errors: list[str] = []
#     cleaned_text = parsed_doc["cleaned_text"]

#     by_parent: dict[str, list[Chunk]] = {}
#     parent_chunks: dict[str, Chunk] = {}
#     for c in chunks:
#         by_parent.setdefault(c.parent_id, []).append(c)
#         if c.is_parent:
#             parent_chunks[c.parent_id] = c

#     for c in chunks:
#         if not (0 <= c.source_start <= c.source_end <= len(cleaned_text)):
#             errors.append(f"{c.child_id or c.parent_id}: out-of-bounds offsets [{c.source_start},{c.source_end})")
#             continue
#         if cleaned_text[c.source_start:c.source_end] != c.canonical_text:
#             errors.append(f"{c.child_id or c.parent_id}: canonical_text does not match cleaned_text[source_start:source_end]")
#         if c.token_count <= 0 and c.canonical_text.strip():
#             errors.append(f"{c.child_id or c.parent_id}: token_count not recorded for non-empty text")
#         if not c.embedding_text.startswith("[" + DOCUMENT_TITLE):
#             errors.append(f"{c.child_id or c.parent_id}: embedding_text does not start with the expected breadcrumb")
#         if not c.is_parent and c.child_id is None:
#             errors.append(f"a non-parent chunk under {c.parent_id} is missing a child_id")
#         if not c.is_parent and c.parent_id not in parent_chunks:
#             errors.append(f"child {c.child_id} has no corresponding parent chunk {c.parent_id!r}")

#     # Rule A: small Article -> parent_id == child_id, exactly one chunk group.
#     article_groups: dict[str, list[Chunk]] = {}
#     for c in chunks:
#         if c.article_number:
#             article_groups.setdefault(c.article_number, []).append(c)

#     for article_number, group in article_groups.items():
#         parents = [c for c in group if c.is_parent]
#         if len(parents) != 1:
#             errors.append(f"Article {article_number}: expected exactly one parent chunk, found {len(parents)}")
#             continue
#         parent = parents[0]
#         if parent.parent_id == parent.child_id:
#             # Rule A shape
#             if len(group) != 1:
#                 errors.append(f"Article {article_number}: Rule A (parent==child) article has extra chunks: {len(group)}")
#         else:
#             # Rule B shape -- children (excluding the parent itself) must
#             # gaplessly, non-overlappingly cover the parent's own span.
#             children = sorted((c for c in group if not c.is_parent), key=lambda c: c.source_start)
#             cursor = parent.source_start
#             for child in children:
#                 if child.source_start < cursor:
#                     errors.append(f"Article {article_number}: child {child.child_id} overlaps a preceding sibling")
#                 elif child.source_start > cursor:
#                     errors.append(f"Article {article_number}: gap in child coverage before {child.child_id} "
#                                    f"-- union(children) != Article text")
#                 cursor = max(cursor, child.source_end)
#             if cursor != parent.source_end:
#                 errors.append(f"Article {article_number}: children do not fully cover the Article's span "
#                                f"(lost text, covered up to {cursor} of {parent.source_end})")

#     real_article_numbers = set()

#     def _collect_real_articles(node: dict):
#         if node.get("type") == "article":
#             real_article_numbers.add(node["article_number"])
#         for c in node.get("children", []):
#             _collect_real_articles(c)

#     _collect_real_articles(parsed_doc["structure"])
#     missing = real_article_numbers - set(article_groups)
#     if missing:
#         errors.append(f"Article(s) present in the parsed AST but not represented in any chunk: {sorted(missing)}")

#     seen_child_ids = [c.child_id for c in chunks if c.child_id is not None]
#     duplicate_child_ids = {cid for cid in seen_child_ids if seen_child_ids.count(cid) > 1}
#     if duplicate_child_ids:
#         errors.append(f"duplicate child_id(s): {sorted(duplicate_child_ids)}")

#     diagnostics = {
#         "total_chunks": len(chunks),
#         "parent_chunks": sum(1 for c in chunks if c.is_parent),
#         "child_chunks": sum(1 for c in chunks if not c.is_parent),
#         "articles_represented": len(article_groups),
#     }
#     return ChunkValidationResult(ok=not errors, errors=errors, diagnostics=diagnostics)


# def is_valid_existing_output(doc_dict: dict) -> bool:
#     """Resumability gate: is this existing chunked output file actually
#     valid, not merely present. Returns False (never raises) for anything
#     unreadable/malformed."""

#     try:
#         chunks = [Chunk.from_dict(c) for c in doc_dict["chunks"]]
#         parsed_doc = {"cleaned_text": doc_dict["cleaned_text"], "structure": doc_dict["structure"]}
#     except (KeyError, TypeError, ValueError):
#         return False
#     try:
#         result = validate_chunks(chunks, parsed_doc)
#     except Exception:  # noqa: BLE001 -- a malformed file must never crash the skip-check
#         return False
#     return result.ok


# def chunk_document(parsed_doc: dict) -> dict:
#     chunks = build_chunks_for_document(parsed_doc)
#     result = validate_chunks(chunks, parsed_doc)
#     return {
#         "doc_id": parsed_doc["doc_id"],
#         "cleaned_text": parsed_doc["cleaned_text"],
#         "structure": parsed_doc["structure"],
#         "chunks": [c.to_dict() for c in chunks],
#         "validation": result.to_dict(),
#     }


# def chunk_directory(input_dir: Path = INPUT_DIR, output_dir: Path = OUTPUT_DIR) -> dict:
#     """Chunks every ``<doc_id>.json`` in ``input_dir`` into
#     ``output_dir/<doc_id>.json``. Resumable: an existing output is
#     skipped only if is_valid_existing_output() confirms it, never on
#     file existence alone. Never modifies ``input_dir``."""

#     input_dir, output_dir = Path(input_dir), Path(output_dir)
#     output_dir.mkdir(parents=True, exist_ok=True)

#     processed = regenerated = skipped_valid = 0
#     failed_docs: list[dict] = []

#     for parsed_path in sorted(input_dir.glob("*.json")):
#         doc_id = parsed_path.stem
#         out_path = output_dir / f"{doc_id}.json"

#         was_present = out_path.exists()
#         if was_present:
#             try:
#                 existing = json.loads(out_path.read_text(encoding="utf-8"))
#             except (OSError, json.JSONDecodeError):
#                 existing = None
#             if existing is not None and is_valid_existing_output(existing):
#                 skipped_valid += 1
#                 continue

#         try:
#             parsed_doc = json.loads(parsed_path.read_text(encoding="utf-8"))
#             chunked_doc = chunk_document(parsed_doc)
#         except (OSError, json.JSONDecodeError, KeyError, ConstitutionChunkingError) as exc:
#             failed_docs.append({"doc_id": doc_id, "error": str(exc)})
#             continue

#         if not chunked_doc["validation"]["ok"]:
#             failed_docs.append({"doc_id": doc_id, "error": chunked_doc["validation"]["errors"]})
#             continue

#         tmp_path = out_path.with_suffix(".json.tmp")
#         tmp_path.write_text(json.dumps(chunked_doc, ensure_ascii=False, indent=2), encoding="utf-8")
#         tmp_path.replace(out_path)
#         processed += 1
#         if was_present:
#             regenerated += 1

#     return {
#         "processed": processed, "skipped_valid": skipped_valid, "regenerated": regenerated,
#         "failed": len(failed_docs), "failed_docs": failed_docs,
#     }


# def main() -> None:  # pragma: no cover -- thin CLI wrapper
#     summary = chunk_directory()
#     print(f"Constitution chunking: {summary['processed']} processed "
#           f"({summary['regenerated']} regenerated), {summary['skipped_valid']} skipped (valid), "
#           f"{summary['failed']} failed")
#     for f in summary["failed_docs"]:
#         print(f"  FAILED {f['doc_id']}: {f['error']}")


# if __name__ == "__main__":  # pragma: no cover
#     main()


"""Constitution of Pakistan parent-child chunking -- Stage 4 of the
independent Constitution pipeline.

    parsed Constitution AST (var/rag/constitution_parsed/<doc_id>.json)
        -> build_chunks_for_document()   [deterministic, token-count-driven, no LLM]
        -> parent-child chunk JSON       (var/rag/constitution_chunked/<doc_id>.json)

Converts the already-parsed hierarchical AST into retrieval-ready
parent/child chunks:

    Article, <=600 tokens   -> exactly one chunk (parent == child)
    Article, >600 tokens    -> one parent (the complete Article) +
                                 one child per Clause
    Clause,  >400 tokens     -> that clause's single child is replaced by
                                 one child per Sub-clause/Proviso (never
                                 an arbitrary character/sentence split;
                                 preserved whole if no such boundary exists)
    Schedule / Preamble / Annex -> exactly one chunk each (the parser
                                 identifies no item-level boundary inside
                                 them, so none is invented here)

Every chunk's ``canonical_text`` is an exact, untouched slice of the
parsed document's own ``cleaned_text`` (never normalized or rewritten
here); ``embedding_text`` adds a breadcrumb line built only from
structural metadata the parser already captured, never invented.

PIPELINE ISOLATION
-------------------
No import of src.ingestion/, src.classification/, src.rag_prep.
case_cleaner/structurer*/chunk*/statute_*, or src.extraction.
case_loader. Independent entry point, independent input/output
directories. Never modifies the parser's or cleaner's output.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import tiktoken

from src.rag_prep.constitution_cleaning import strip_footnote_refs

INPUT_DIR = Path("var/rag/constitution_parsed")
OUTPUT_DIR = Path("var/rag/constitution_chunked")

# Task-specified thresholds -- not configurable elsewhere; see the
# module docstring's "Core Chunking Rules" summary for their exact
# effect at each level.
ARTICLE_TOKEN_THRESHOLD = 600
CLAUSE_TOKEN_THRESHOLD = 400

DOCUMENT_TITLE = "Constitution of Pakistan"

# cl100k_base: a stable, offline, model-agnostic tokenizer -- used
# purely as a deterministic token-COUNTING yardstick for the chunking
# thresholds above, never for any generative/LLM purpose.
_ENCODING = tiktoken.get_encoding("cl100k_base")


class ConstitutionChunkingError(ValueError):
    """Raised when the input AST cannot be safely chunked -- e.g. its
    own parse validation did not pass, or no Article/Schedule/Preamble/
    Annex content exists anywhere. Per "fail validation rather than
    silently dropping text": raised instead of chunking a known-bad or
    empty structure."""


def count_tokens(text: str) -> int:
    return len(_ENCODING.encode(text))


@dataclass(frozen=True)
class Chunk:
    doc_id: str
    parent_id: str
    child_id: str | None
    part_number: str | None
    part_title: str | None
    chapter_number: str | None
    chapter_title: str | None
    article_number: str | None
    article_title: str | None
    clause_number: str | None
    is_parent: bool
    token_count: int
    source_start: int
    source_end: int
    canonical_text: str
    embedding_text: str

    def to_dict(self) -> dict:
        return {
            "doc_id": self.doc_id, "parent_id": self.parent_id, "child_id": self.child_id,
            "part_number": self.part_number, "part_title": self.part_title,
            "chapter_number": self.chapter_number, "chapter_title": self.chapter_title,
            "article_number": self.article_number, "article_title": self.article_title,
            "clause_number": self.clause_number, "is_parent": self.is_parent,
            "token_count": self.token_count, "source_start": self.source_start,
            "source_end": self.source_end, "canonical_text": self.canonical_text,
            "embedding_text": self.embedding_text,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Chunk":
        return cls(**d)


# -- breadcrumb -----------------------------------------------------------------

def _clean_title_for_breadcrumb(title: str | None) -> str | None:
    """Strips the cleaning stage's footnote-reference sentinels
    ("[^91.8]") out of a title string used ONLY inside the breadcrumb --
    never applied to canonical_text or to a chunk's own stored
    *_title metadata field, which stay verbatim/faithful to the parser's
    own output. A raw "[^91.8]" is internal pipeline plumbing, not
    constitutional wording, and left in would only add embedding noise;
    stripping it is text hygiene for this one derived field, not
    "inventing missing metadata" (the metadata itself is untouched)."""

    if title is None:
        return None
    cleaned = strip_footnote_refs(title).strip()
    return cleaned or None


def _breadcrumb(
    part_number: str | None, part_title: str | None, chapter_number: str | None, chapter_title: str | None,
    article_number: str | None, article_title: str | None, clause_number: str | None,
) -> str:
    part_title = _clean_title_for_breadcrumb(part_title)
    chapter_title = _clean_title_for_breadcrumb(chapter_title)
    article_title = _clean_title_for_breadcrumb(article_title)

    segments = [DOCUMENT_TITLE]
    if part_number:
        segments.append(f"Part {part_number}: {part_title}" if part_title else f"Part {part_number}")
    if chapter_number:
        segments.append(f"Chapter {chapter_number}: {chapter_title}" if chapter_title else f"Chapter {chapter_number}")
    if article_number:
        segments.append(f"Article {article_number}: {article_title}" if article_title else f"Article {article_number}")
    if clause_number:
        segments.append(f"Clause ({clause_number})")
    return "[" + " | ".join(segments) + "]"


def _embedding_text(breadcrumb: str, canonical_text: str) -> str:
    return f"{breadcrumb}\n\n{canonical_text}"


# -- Part/Chapter title extraction (from the parser's own node text) -----------
def _part_title_from_text(part_text: str) -> str | None:
    lines = part_text.split("\n", 2)
    if len(lines) < 2:
        return None
    title = lines[1].strip()
    return title or None


# -- clause/sub-clause/proviso child construction --------------------------------

def _chunk_simple_leaf(
    node: dict, doc_id: str, parent_id: str, child_id: str,
    part_number: str | None, part_title: str | None, chapter_number: str | None, chapter_title: str | None,
    article_number: str | None, article_title: str | None, clause_number: str | None,
) -> Chunk:
    breadcrumb = _breadcrumb(part_number, part_title, chapter_number, chapter_title, article_number, article_title, clause_number)
    text = node["text"]
    return Chunk(
        doc_id=doc_id, parent_id=parent_id, child_id=child_id,
        part_number=part_number, part_title=part_title, chapter_number=chapter_number, chapter_title=chapter_title,
        article_number=article_number, article_title=article_title, clause_number=clause_number,
        is_parent=False, token_count=count_tokens(text),
        source_start=node["source_start"], source_end=node["source_end"],
        canonical_text=text, embedding_text=_embedding_text(breadcrumb, text),
    )


def _flatten_clause_children(clause: dict, cleaned_text: str) -> list[dict]:
    """Recursively flattens a clause's children to arbitrary nesting depths.

    Ensures sub-clauses with nested children, trailing provisos, or sub-subclauses
    have their text spans accurately truncated to end at their first child's start,
    and returns a completely flattened list of contiguous, non-overlapping legal sub-units.
    """
    flat: list[dict] = []
    children = clause.get("children", [])

    for child in children:
        nested_children = child.get("children", [])
        if nested_children:
            # Narrow current node's span to end where its first child starts
            own_end = nested_children[0]["source_start"]
            current_node = {
                **child,
                "source_end": own_end,
                "text": cleaned_text[child["source_start"]:own_end]
            }
            flat.append(current_node)
            # Recurse down nested subtree
            flat.extend(_flatten_clause_children(child, cleaned_text))
        else:
            flat.append(child)

    return flat


def _chunk_clause(
    clause: dict, cleaned_text: str, doc_id: str, parent_id: str, article_id: str,
    part_number: str | None, part_title: str | None, chapter_number: str | None, chapter_title: str | None,
    article_number: str, article_title: str | None,
) -> list[Chunk]:
    clause_number = clause["number"]
    clause_id = f"{article_id}_c{clause_number.lower()}"
    token_count = count_tokens(clause["text"])

    if token_count <= CLAUSE_TOKEN_THRESHOLD:
        return [_chunk_simple_leaf(
            clause, doc_id, parent_id, clause_id,
            part_number, part_title, chapter_number, chapter_title, article_number, article_title, clause_number,
        )]

    # Oversized Clause Rule: split by sub-clause/proviso children the
    # parser already identified (flattened -- see _flatten_clause_children).
    # If none exist, the clause has no valid internal legal boundary --
    # preserved whole rather than guessed at.
    flat_children = _flatten_clause_children(clause, cleaned_text)
    splittable = [c for c in flat_children if c["type"] in ("subclause", "proviso")]
    if not splittable:
        return [_chunk_simple_leaf(
            clause, doc_id, parent_id, clause_id,
            part_number, part_title, chapter_number, chapter_title, article_number, article_title, clause_number,
        )]

    chunks: list[Chunk] = []
    proviso_n = 0
    seen_ids: dict[str, int] = {}
    for child in flat_children:  # every child, in order -- union(children) == clause text
        if child["type"] == "subclause":
            cid = f"{clause_id}_{child['number'].lower()}"
        elif child["type"] == "proviso":
            proviso_n += 1
            cid = f"{clause_id}_proviso" + (str(proviso_n) if proviso_n > 1 else "")
        else:  # a "text" gap within the oversized clause -- kept, not dropped
            cid = f"{clause_id}_text{len(chunks) + 1}"

        if cid in seen_ids:
            seen_ids[cid] += 1
            cid = f"{cid}{seen_ids[cid]}"
        else:
            seen_ids[cid] = 1

        chunks.append(_chunk_simple_leaf(
            child, doc_id, parent_id, cid,
            part_number, part_title, chapter_number, chapter_title, article_number, article_title, clause_number,
        ))
    return chunks


def _chunk_article(
    article: dict, cleaned_text: str, doc_id: str,
    part_number: str | None, part_title: str | None, chapter_number: str | None, chapter_title: str | None,
) -> list[Chunk]:
    article_number = article["article_number"]
    article_title = article.get("heading")
    article_id = f"art_{article_number.lower()}"
    token_count = count_tokens(article["text"])

    if token_count <= ARTICLE_TOKEN_THRESHOLD:
        # Rule A: exactly one chunk -- parent IS the child.
        chunk = _chunk_simple_leaf(
            article, doc_id, article_id, article_id,
            part_number, part_title, chapter_number, chapter_title, article_number, article_title, None,
        )
        return [Chunk(**{**chunk.to_dict(), "is_parent": True})]

    # Rule B: one parent (the complete Article) + one child per Clause
    parent_id = f"{article_id}_parent"
    breadcrumb = _breadcrumb(part_number, part_title, chapter_number, chapter_title, article_number, article_title, None)
    parent_chunk = Chunk(
        doc_id=doc_id, parent_id=parent_id, child_id=None,
        part_number=part_number, part_title=part_title, chapter_number=chapter_number, chapter_title=chapter_title,
        article_number=article_number, article_title=article_title, clause_number=None,
        is_parent=True, token_count=token_count,
        source_start=article["source_start"], source_end=article["source_end"],
        canonical_text=article["text"], embedding_text=_embedding_text(breadcrumb, article["text"]),
    )

    chunks = [parent_chunk]
    text_n = 0
    for child in article.get("children", []):
        if child["type"] == "clause":
            chunks.extend(_chunk_clause(
                child, cleaned_text, doc_id, parent_id, article_id,
                part_number, part_title, chapter_number, chapter_title, article_number, article_title,
            ))
        else:  # a "text" gap directly under the Article
            text_n += 1
            chunks.append(_chunk_simple_leaf(
                child, doc_id, parent_id, f"{article_id}_text{text_n}",
                part_number, part_title, chapter_number, chapter_title, article_number, article_title, None,
            ))
    return chunks


def _chunk_top_level_unit(node: dict, doc_id: str) -> list[Chunk]:
    """Preamble, Annex (Objectives Resolution), and Schedule chunks."""
    node_id = node["id"]
    heading = node.get("heading")
    chunk = _chunk_simple_leaf(node, doc_id, node_id, node_id, None, None, None, None, None, heading, None)
    return [Chunk(**{**chunk.to_dict(), "is_parent": True})]


def build_chunks_for_document(parsed_doc: dict) -> list[Chunk]:
    """Builds the full, ordered, deterministic chunk list for one parsed Constitution document."""
    if not parsed_doc.get("validation", {}).get("ok"):
        raise ConstitutionChunkingError(
            "Refusing to chunk a parsed Constitution document whose own structural "
            f"validation did not pass: {parsed_doc.get('validation', {}).get('errors')}"
        )

    doc_id = parsed_doc["doc_id"]
    cleaned_text = parsed_doc["cleaned_text"]
    root = parsed_doc["structure"]

    chunks: list[Chunk] = []
    for top in root.get("children", []):
        if top["type"] == "part":
            part_number = top["id"].removeprefix("part_")
            part_title = _part_title_from_text(top["text"])
            for sub in top.get("children", []):
                if sub["type"] == "chapter":
                    chapter_number = sub["id"].rsplit("_chapter", 1)[-1]
                    chapter_title = sub.get("heading")
                    for art in sub.get("children", []):
                        if art["type"] == "article":
                            chunks.extend(_chunk_article(art, cleaned_text, doc_id, part_number, part_title, chapter_number, chapter_title))
                elif sub["type"] == "article":
                    chunks.extend(_chunk_article(sub, cleaned_text, doc_id, part_number, part_title, None, None))
        elif top["type"] in ("schedule", "preamble", "annex"):
            chunks.extend(_chunk_top_level_unit(top, doc_id))

    if not chunks:
        raise ConstitutionChunkingError("No Article, Schedule, Preamble, or Annex content found to chunk.")

    return chunks


# -- validation -------------------------------------------------------------------

@dataclass
class ChunkValidationResult:
    ok: bool
    errors: list[str] = field(default_factory=list)
    diagnostics: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {"ok": self.ok, "errors": self.errors, "diagnostics": self.diagnostics}


def validate_chunks(chunks: list[Chunk], parsed_doc: dict) -> ChunkValidationResult:
    errors: list[str] = []
    cleaned_text = parsed_doc["cleaned_text"]

    by_parent: dict[str, list[Chunk]] = {}
    parent_chunks: dict[str, Chunk] = {}
    for c in chunks:
        by_parent.setdefault(c.parent_id, []).append(c)
        if c.is_parent:
            parent_chunks[c.parent_id] = c

    for c in chunks:
        if not (0 <= c.source_start <= c.source_end <= len(cleaned_text)):
            errors.append(f"{c.child_id or c.parent_id}: out-of-bounds offsets [{c.source_start},{c.source_end})")
            continue
        if cleaned_text[c.source_start:c.source_end] != c.canonical_text:
            errors.append(f"{c.child_id or c.parent_id}: canonical_text does not match cleaned_text[source_start:source_end]")
        if c.token_count <= 0 and c.canonical_text.strip():
            errors.append(f"{c.child_id or c.parent_id}: token_count not recorded for non-empty text")
        if not c.embedding_text.startswith("[" + DOCUMENT_TITLE):
            errors.append(f"{c.child_id or c.parent_id}: embedding_text does not start with the expected breadcrumb")
        if not c.is_parent and c.child_id is None:
            errors.append(f"a non-parent chunk under {c.parent_id} is missing a child_id")
        if not c.is_parent and c.parent_id not in parent_chunks:
            errors.append(f"child {c.child_id} has no corresponding parent chunk {c.parent_id!r}")

    # Rule A check
    article_groups: dict[str, list[Chunk]] = {}
    for c in chunks:
        if c.article_number:
            article_groups.setdefault(c.article_number, []).append(c)

    for article_number, group in article_groups.items():
        parents = [c for c in group if c.is_parent]
        if len(parents) != 1:
            errors.append(f"Article {article_number}: expected exactly one parent chunk, found {len(parents)}")
            continue
        parent = parents[0]
        if parent.parent_id == parent.child_id:
            if len(group) != 1:
                errors.append(f"Article {article_number}: Rule A (parent==child) article has extra chunks: {len(group)}")
        else:
            children = sorted((c for c in group if not c.is_parent), key=lambda c: c.source_start)
            cursor = parent.source_start
            for child in children:
                if child.source_start < cursor:
                    errors.append(f"Article {article_number}: child {child.child_id} overlaps a preceding sibling")
                elif child.source_start > cursor:
                    errors.append(f"Article {article_number}: gap in child coverage before {child.child_id} "
                                   f"-- union(children) != Article text")
                cursor = max(cursor, child.source_end)
            if cursor != parent.source_end:
                errors.append(f"Article {article_number}: children do not fully cover the Article's span "
                               f"(lost text, covered up to {cursor} of {parent.source_end})")

    real_article_numbers = set()

    def _collect_real_articles(node: dict):
        if node.get("type") == "article":
            real_article_numbers.add(node["article_number"])
        for c in node.get("children", []):
            _collect_real_articles(c)

    _collect_real_articles(parsed_doc["structure"])
    missing = real_article_numbers - set(article_groups)
    if missing:
        errors.append(f"Article(s) present in the parsed AST but not represented in any chunk: {sorted(missing)}")

    seen_child_ids = [c.child_id for c in chunks if c.child_id is not None]
    duplicate_child_ids = {cid for cid in seen_child_ids if seen_child_ids.count(cid) > 1}
    if duplicate_child_ids:
        errors.append(f"duplicate child_id(s): {sorted(duplicate_child_ids)}")

    diagnostics = {
        "total_chunks": len(chunks),
        "parent_chunks": sum(1 for c in chunks if c.is_parent),
        "child_chunks": sum(1 for c in chunks if not c.is_parent),
        "articles_represented": len(article_groups),
    }
    return ChunkValidationResult(ok=not errors, errors=errors, diagnostics=diagnostics)


def is_valid_existing_output(doc_dict: dict) -> bool:
    """Resumability gate: checks validity of existing chunked output file."""
    try:
        chunks = [Chunk.from_dict(c) for c in doc_dict["chunks"]]
        parsed_doc = {"cleaned_text": doc_dict["cleaned_text"], "structure": doc_dict["structure"]}
    except (KeyError, TypeError, ValueError):
        return False
    try:
        result = validate_chunks(chunks, parsed_doc)
    except Exception:  # noqa: BLE001
        return False
    return result.ok


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
    """Chunks every `<doc_id>.json` in input_dir into output_dir/<doc_id>.json."""
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
        if was_present:
            regenerated += 1

    return {
        "processed": processed, "skipped_valid": skipped_valid, "regenerated": regenerated,
        "failed": len(failed_docs), "failed_docs": failed_docs,
    }


def main() -> None:  # pragma: no cover
    summary = chunk_directory()
    print(f"Constitution chunking: {summary['processed']} processed "
          f"({summary['regenerated']} regenerated), {summary['skipped_valid']} skipped (valid), "
          f"{summary['failed']} failed")
    for f in summary["failed_docs"]:
        print(f"  FAILED {f['doc_id']}: {f['error']}")


if __name__ == "__main__":  # pragma: no cover
    main()