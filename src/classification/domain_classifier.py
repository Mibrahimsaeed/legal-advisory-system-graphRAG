"""Assign one document to one frozen domain -- or refuse to.

The classifier is a single LLM call per document against the frozen
taxonomy, and everything interesting here is about not trusting its
answer:

* **The output is validated, not parsed optimistically.** A response is
  accepted only if it is JSON, names a domain that actually exists in the
  frozen taxonomy, carries a confidence in [0, 1], and gives a
  justification. Anything else is a ``failed`` result carrying the reason
  -- never a guessed label. A hallucinated domain id is the failure mode
  that would otherwise corrupt a corpus quietly.
* **Low confidence is routed, not rounded up.** Below
  ``min_confidence`` the proposed label is kept but the row is marked
  ``needs_review``: a human sees the suggestion and the justification
  rather than the pipeline forcing a domain the model was unsure of.
* **Ambiguity is first-class.** Secondary domains are requested
  explicitly, and a document that lands in several domains at once is
  routed to review rather than silently reduced to its first label.

Prompting deliberately mirrors ``src/clustering/label_clusters.py``: the
same LLM client protocol, the same JSON-only instruction, the same
"never raise, return a degraded result" contract, so orchestration code
handles both stages identically.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from src.classification.taxonomy_registry import (
    OTHER_DOMAIN_ID,
    FrozenTaxonomy,
)
from src.common.llm_client import LLMClient, extract_json_object
from src.common.logging_utils import get_logger

logger = get_logger(__name__)

# Bump when the prompt, the validation rules, or the routing policy
# change: it is stamped on every row so results from different classifier
# behaviors are never silently compared as if they were the same thing.
CLASSIFIER_VERSION = "domain_classifier/1.0"

STATUS_CLASSIFIED = "classified"
STATUS_NEEDS_REVIEW = "needs_review"
STATUS_FAILED = "failed"

REVIEW_LOW_CONFIDENCE = "low_confidence"
REVIEW_MULTI_DOMAIN = "multiple_domains"
REVIEW_OTHER_BUCKET = "classified_as_other"

DEFAULT_MIN_CONFIDENCE = 0.6
DEFAULT_BODY_CHARS = 3_000
DEFAULT_MAX_JUSTIFICATION_CHARS = 600

CLASSIFY_SYSTEM_PROMPT = (
    "You are classifying Pakistani case-law documents into a FIXED legal "
    "domain taxonomy. You may only use the domain ids given to you. If a "
    "document does not clearly belong to any listed domain, answer with "
    f'"{OTHER_DOMAIN_ID}" rather than forcing a poor fit. Respond with ONLY '
    "a JSON object (no prose, no markdown fence) with exactly these keys: "
    '"primary_domain" (one domain id), "secondary_domains" (a list of '
    "domain ids that also substantially apply, or an empty list), "
    '"confidence" (a number between 0 and 1 for the primary domain), and '
    '"justification" (one or two sentences citing what in the document '
    "supports the choice)."
)


@dataclass
class ClassificationResult:
    doc_id: str
    primary_domain: str | None = None
    secondary_domains: list[str] = field(default_factory=list)
    confidence: float | None = None
    justification: str | None = None
    status: str = STATUS_CLASSIFIED
    review_reason: str | None = None
    error: str | None = None
    cluster_id: int | None = None

    @property
    def needs_review(self) -> bool:
        return self.status == STATUS_NEEDS_REVIEW

    @property
    def failed(self) -> bool:
        return self.status == STATUS_FAILED


def render_taxonomy_prompt(taxonomy: FrozenTaxonomy) -> str:
    """The domain menu, rendered once per batch rather than per document."""

    lines: list[str] = []
    for domain in taxonomy.domains:
        if domain.is_other:
            continue
        lines.append(f"- id: {domain.domain_id}\n  name: {domain.name}")
        if domain.description:
            lines.append(f"  description: {domain.description}")
        for inclusion in domain.inclusion_criteria[:6]:
            lines.append(f"  include: {inclusion}")
        for exclusion in domain.exclusion_criteria[:6]:
            lines.append(f"  exclude: {exclusion}")
    lines.append(
        f"- id: {OTHER_DOMAIN_ID}\n  name: Other / Uncertain\n"
        "  description: Use when the document does not clearly belong to any "
        "domain above."
    )
    return "\n".join(lines)


def build_classification_prompt(
    document: Any,
    taxonomy_block: str,
    body_chars: int = DEFAULT_BODY_CHARS,
) -> str:
    parts = [f"TAXONOMY:\n{taxonomy_block}", "", "DOCUMENT:"]
    title = getattr(document, "title", None)
    court = getattr(document, "court", None)
    decision_date = getattr(document, "decision_date", None)
    headings = getattr(document, "headings", None) or []
    body = (getattr(document, "body_preview", "") or "")[:body_chars]

    if title:
        parts.append(f"Title: {title}")
    if court:
        parts.append(f"Court: {court}")
    if decision_date:
        parts.append(f"Date: {decision_date}")
    if headings:
        parts.append("Headings: " + "; ".join(headings[:10]))
    parts.append(f"Text:\n{body}")
    parts.append(
        "\nClassify this document into ONE primary domain id from the "
        "taxonomy above. Respond with the JSON object only."
    )
    return "\n".join(parts)


def _coerce_confidence(value: Any) -> float | None:
    """Accept 0-1 floats, and percentages the model may emit as 0-100."""

    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, str):
        try:
            value = float(value.strip().rstrip("%"))
        except ValueError:
            return None
    if not isinstance(value, (int, float)):
        return None
    value = float(value)
    if 1.0 < value <= 100.0:
        # "confidence": 90 plainly means 90%. "confidence": 1.7 is not a
        # percentage, it is a broken number -- only whole values are
        # rescaled, so a malformed float still fails validation instead of
        # quietly becoming a near-zero confidence.
        if not value.is_integer():
            return None
        value = value / 100.0
    if not 0.0 <= value <= 1.0:
        return None
    return value


def validate_classification(
    parsed: dict[str, Any],
    taxonomy: FrozenTaxonomy,
    doc_id: str,
    max_justification_chars: int = DEFAULT_MAX_JUSTIFICATION_CHARS,
) -> ClassificationResult:
    """Turn a parsed LLM response into a result, or into a failure.

    Every rejection reason is explicit, because "the model named a domain
    that does not exist" and "the model was unsure" require completely
    different human follow-up.
    """

    allowed = taxonomy.assignable_ids

    primary = parsed.get("primary_domain") or parsed.get("domain") or parsed.get("primary")
    primary = str(primary).strip() if primary is not None else ""
    if not primary:
        return ClassificationResult(
            doc_id=doc_id, status=STATUS_FAILED, error="missing_primary_domain"
        )
    if primary not in allowed:
        return ClassificationResult(
            doc_id=doc_id,
            status=STATUS_FAILED,
            error=f"unknown_domain_id:{primary[:60]}",
        )

    raw_secondary = parsed.get("secondary_domains") or []
    if isinstance(raw_secondary, str):
        raw_secondary = [raw_secondary]
    secondary: list[str] = []
    dropped: list[str] = []
    for item in raw_secondary if isinstance(raw_secondary, list) else []:
        candidate = str(item).strip()
        if not candidate or candidate == primary or candidate == OTHER_DOMAIN_ID:
            continue
        if candidate in allowed:
            if candidate not in secondary:
                secondary.append(candidate)
        else:
            dropped.append(candidate)
    if dropped:
        # A bad secondary is not worth failing the document over -- the
        # primary label is still validated -- but it is worth saying.
        logger.warning(
            "Document %s: dropped unknown secondary domain(s): %s",
            doc_id, ", ".join(dropped[:5]),
        )

    confidence = _coerce_confidence(parsed.get("confidence"))
    if confidence is None:
        return ClassificationResult(
            doc_id=doc_id,
            status=STATUS_FAILED,
            error=f"invalid_confidence:{str(parsed.get('confidence'))[:40]}",
        )

    justification = str(parsed.get("justification") or "").strip()
    if not justification:
        return ClassificationResult(
            doc_id=doc_id, status=STATUS_FAILED, error="missing_justification"
        )

    return ClassificationResult(
        doc_id=doc_id,
        primary_domain=primary,
        secondary_domains=secondary,
        confidence=confidence,
        justification=justification[:max_justification_chars],
        status=STATUS_CLASSIFIED,
    )


def apply_review_policy(
    result: ClassificationResult,
    min_confidence: float = DEFAULT_MIN_CONFIDENCE,
    review_multi_domain: bool = True,
    review_other_bucket: bool = True,
) -> ClassificationResult:
    """Route weak or ambiguous verdicts to review instead of accepting them."""

    if result.status != STATUS_CLASSIFIED:
        return result

    reasons: list[str] = []
    if result.confidence is not None and result.confidence < min_confidence:
        reasons.append(REVIEW_LOW_CONFIDENCE)
    if review_multi_domain and result.secondary_domains:
        reasons.append(REVIEW_MULTI_DOMAIN)
    if review_other_bucket and result.primary_domain == OTHER_DOMAIN_ID:
        reasons.append(REVIEW_OTHER_BUCKET)

    if reasons:
        result.status = STATUS_NEEDS_REVIEW
        result.review_reason = ",".join(reasons)
    return result


def classify_document(
    document: Any,
    taxonomy: FrozenTaxonomy,
    llm_client: LLMClient,
    taxonomy_block: str | None = None,
    min_confidence: float = DEFAULT_MIN_CONFIDENCE,
    body_chars: int = DEFAULT_BODY_CHARS,
    review_multi_domain: bool = True,
    review_other_bucket: bool = True,
) -> ClassificationResult:
    """Classify one document. Never raises.

    An LLM or parsing failure yields ``status='failed'`` with the reason,
    so one bad document cannot abort a 10k-document batch run and cannot
    masquerade as a classified one either.
    """

    doc_id = getattr(document, "doc_id", "")
    block = taxonomy_block if taxonomy_block is not None else render_taxonomy_prompt(taxonomy)

    try:
        raw = llm_client.complete(
            system=CLASSIFY_SYSTEM_PROMPT,
            prompt=build_classification_prompt(document, block, body_chars=body_chars),
        )
        parsed = extract_json_object(raw)
    except Exception as exc:  # noqa: BLE001 - deliberately broad; see docstring
        logger.error("Classification failed for %s: %s", doc_id, exc)
        return ClassificationResult(
            doc_id=doc_id,
            status=STATUS_FAILED,
            error=f"{type(exc).__name__}: {exc}"[:300],
        )

    result = validate_classification(parsed, taxonomy, doc_id)
    return apply_review_policy(
        result,
        min_confidence=min_confidence,
        review_multi_domain=review_multi_domain,
        review_other_bucket=review_other_bucket,
    )
