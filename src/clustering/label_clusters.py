"""Turn one discovered cluster into a draft domain definition.

Three steps per cluster, matching the Stage 1.2 spec (point 7):

1. Extract representative keywords via a corpus-relative TF-IDF score
   (pure Python -- no sklearn dependency) so a cluster's keywords are
   words that are distinctively *common in this cluster*, not just
   common across every legal document in the sample (e.g. "agreement",
   "party", "shall").
2. Pick representative documents -- highest HDBSCAN membership
   probability if available, else closest to the cluster centroid in
   embedding space.
3. Send keywords + representative excerpts to the LLM
   (:mod:`src.common.llm_client`) and parse the JSON response into a
   :class:`~src.clustering.taxonomy_card.DomainDraft`.

Every step degrades gracefully rather than failing the whole run: a
cluster with no clean keywords still gets labeled from title/TOC text
alone, and an LLM failure produces an "(unlabeled)" draft carrying the
raw keywords/representative docs and an ``error`` field, rather than
losing the cluster's ranking/volume information entirely.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass
from typing import Any

import numpy as np

from src.clustering.taxonomy_card import DomainDraft
from src.common.llm_client import LLMClient, extract_json_object
from src.common.logging_utils import get_logger
from src.extraction.doc_representation import EmbeddableDocument

logger = get_logger(__name__)

_WORD_RE = re.compile(r"[A-Za-z][A-Za-z\-]{2,}")

# Generic English stopwords plus boilerplate that's common across nearly
# every legal document regardless of domain -- keeping these out of the
# TF-IDF vocabulary means the corpus-relative IDF weighting isn't wasted
# discounting words that would've been filtered anyway.
_STOPWORDS = frozenset(
    """
    the a an and or but if then else for of to in on at by with as is are
    was were be been being this that these those it its as from into
    through during before after above below between under again further
    once here there when where why how all any both each few more most
    other some such no nor not only own same so than too very can will
    shall should may might must would could not
    agreement party parties hereby herein hereof hereunder pursuant
    whereas witnesseth document page section
    """.split()
)

DOMAIN_LABEL_SYSTEM_PROMPT = (
    "You are assisting with building a legal-document taxonomy from "
    "clusters of automatically grouped legal documents. Given "
    "keywords and representative document excerpts from ONE cluster, "
    "propose a draft legal domain definition. Respond with ONLY a JSON "
    "object (no prose, no markdown fence) with exactly these keys: "
    '"name" (a short 2-5 word domain name), "description" (1-2 '
    "sentences), \"inclusion_criteria\" (a list of short strings "
    "describing what belongs in this domain), and \"exclusion_criteria\" "
    "(a list of short strings describing boundary cases or what does "
    "NOT belong, especially versus neighboring legal domains)."
)


def _tokenize(text: str) -> list[str]:
    return [
        w.lower() for w in _WORD_RE.findall(text or "") if w.lower() not in _STOPWORDS
    ]


def _document_text(document: EmbeddableDocument) -> str:
    parts = [
        document.title or "",
        "; ".join(document.headings),
        document.body_preview or "",
    ]
    return " ".join(p for p in parts if p)


@dataclass(frozen=True)
class KeywordCorpus:
    """Document-frequency table over the *whole sample* (not one cluster),
    used as the IDF term when scoring a single cluster's keywords."""

    doc_freq: Counter
    n_docs: int


def build_keyword_corpus(documents: list[EmbeddableDocument]) -> KeywordCorpus:
    doc_freq: Counter = Counter()
    for document in documents:
        words = set(_tokenize(_document_text(document)))
        doc_freq.update(words)
    return KeywordCorpus(doc_freq=doc_freq, n_docs=len(documents))


def extract_cluster_keywords(
    cluster_documents: list[EmbeddableDocument],
    corpus: KeywordCorpus,
    top_k: int = 15,
) -> list[str]:
    """Corpus-relative TF-IDF keywords for one cluster.

    ``tf`` is the raw count of a word across the cluster's documents;
    ``idf`` discounts words that are common across the whole sample
    (smoothed so an unseen word doesn't produce a divide-by-zero).
    """

    tf: Counter = Counter()
    for document in cluster_documents:
        tf.update(_tokenize(_document_text(document)))

    if not tf:
        return []

    scores: dict[str, float] = {}
    for word, count in tf.items():
        df = corpus.doc_freq.get(word, 0)
        idf = math.log((corpus.n_docs + 1) / (df + 1)) + 1.0
        scores[word] = count * idf

    ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    return [word for word, _ in ranked[:top_k]]


def select_representative_docs(
    doc_ids: list[str],
    vectors: dict[str, np.ndarray],
    probabilities: dict[str, float] | None = None,
    top_k: int = 8,
) -> list[str]:
    """Pick the ``top_k`` documents that best represent a cluster.

    Prefers HDBSCAN's own membership probability (how confidently a
    point belongs to its cluster) when available; falls back to
    Euclidean distance to the cluster centroid in embedding space.
    """

    if probabilities:
        scored = [(doc_id, probabilities.get(doc_id, 0.0)) for doc_id in doc_ids]
        scored.sort(key=lambda t: t[1], reverse=True)
        return [doc_id for doc_id, _ in scored[:top_k]]

    available = [d for d in doc_ids if d in vectors]
    if not available:
        return doc_ids[:top_k]

    vecs = np.stack([vectors[d] for d in available])
    centroid = vecs.mean(axis=0)
    distances = np.linalg.norm(vecs - centroid, axis=1)
    order = np.argsort(distances)
    return [available[i] for i in order[:top_k]]


def _as_str_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value.strip()] if value.strip() else []
    if isinstance(value, list):
        return [str(v).strip() for v in value if str(v).strip()]
    return [str(value).strip()]


def build_domain_prompt(keywords: list[str], representative_snippets: list[str]) -> str:
    snippet_block = "\n\n".join(
        f"--- Document {i + 1} ---\n{snippet}"
        for i, snippet in enumerate(representative_snippets)
    )
    keyword_block = ", ".join(keywords) if keywords else "(no distinctive keywords found)"
    return (
        f"Cluster keywords (most distinctive first): {keyword_block}\n\n"
        f"Representative documents from this cluster:\n{snippet_block}\n\n"
        "Propose a draft legal domain definition as a JSON object."
    )


def _snippet_for(document: EmbeddableDocument, max_chars: int = 600) -> str:
    parts = []
    if document.title:
        parts.append(f"Title: {document.title}")
    if document.headings:
        parts.append("Headings: " + "; ".join(document.headings[:8]))
    if document.body_preview:
        parts.append("Excerpt: " + document.body_preview[:max_chars])
    return "\n".join(parts) if parts else "(no content)"


def generate_domain_draft(
    cluster_id: int,
    cluster_documents: list[EmbeddableDocument],
    representative_doc_ids: list[str],
    keywords: list[str],
    sample_size: int,
    llm_client: LLMClient,
) -> DomainDraft:
    """Build one cluster's :class:`DomainDraft`, calling the LLM for the
    name/description/inclusion/exclusion fields.

    Never raises: an LLM or parsing failure produces a draft with
    ``name="Cluster {id} (unlabeled)"``, an ``error`` message, and every
    field this function *could* compute without the LLM (keywords,
    representative docs, doc_count) still populated -- so a bad LLM call
    degrades the labeling quality, not the whole run.
    """

    by_id = {d.doc_id: d for d in cluster_documents}
    representative_snippets = [
        _snippet_for(by_id[doc_id]) for doc_id in representative_doc_ids if doc_id in by_id
    ]

    try:
        prompt = build_domain_prompt(keywords, representative_snippets)
        raw = llm_client.complete(system=DOMAIN_LABEL_SYSTEM_PROMPT, prompt=prompt)
        parsed = extract_json_object(raw)

        name = str(parsed.get("name") or f"Cluster {cluster_id}").strip()
        description = str(parsed.get("description") or "").strip()
        inclusion = _as_str_list(parsed.get("inclusion_criteria"))
        exclusion = _as_str_list(parsed.get("exclusion_criteria"))
        error = None
    except Exception as exc:  # noqa: BLE001 - deliberately broad; see docstring
        logger.error("LLM domain labeling failed for cluster %s: %s", cluster_id, exc)
        name = f"Cluster {cluster_id} (unlabeled)"
        description = ""
        inclusion, exclusion = [], []
        error = f"{type(exc).__name__}: {exc}"

    return DomainDraft(
        cluster_id=cluster_id,
        name=name,
        description=description,
        inclusion_criteria=inclusion,
        exclusion_criteria=exclusion,
        keywords=keywords,
        representative_doc_ids=representative_doc_ids,
        doc_count=len(cluster_documents),
        sample_size=sample_size,
        error=error,
    )