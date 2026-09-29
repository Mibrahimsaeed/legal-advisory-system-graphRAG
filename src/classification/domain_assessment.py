"""The LLM's broad, non-binding read on which domain a case belongs to.

This is Phase 3's one non-deterministic step, and it is deliberately the
narrowest possible use of a model: given the case text and the frozen
domain definitions, answer with one domain, a confidence, and a sentence
of evidence. Nothing else.

What it must NOT do, and what the prompt actively discourages:

* no detailed legal analysis or holding summary,
* no exact statute/section extraction (that is a later phase),
* no secondary domains or subdomains,
* no forcing an unclear case into a domain -- ``other_uncertain`` is
  always available and the prompt says so.

The output is *evidence*, not a decision. It is stored beside the keyword
and cluster signals, and Phase 4 weighs all three. That separation is why
a low-confidence or failed assessment here is harmless: it contributes
weak evidence rather than a wrong label.

Validation mirrors :mod:`src.classification.domain_classifier`: a response
is accepted only if it is JSON, names a domain the frozen taxonomy
actually defines, carries a confidence in [0, 1] and gives a reason.
Anything else is a ``failed`` assessment carrying the error -- never a
guessed domain.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from src.classification.case_representation import CaseRepresentation
from src.classification.taxonomy_registry import OTHER_DOMAIN_ID, FrozenTaxonomy
from src.common.llm_client import LLMClient, extract_json_object
from src.common.logging_utils import get_logger

logger = get_logger(__name__)

# Bump when the prompt or the validation rules change: it is stored on
# every row so assessments made under different behaviour are never
# silently compared.
# 2.0: the keyword hint was removed from the prompt (CI-7). A 1.0
# assessment was made by a model that had already been shown the keyword
# verdict, so the two are not interchangeable as evidence -- the version
# stamp is what keeps a mixed corpus detectable.
ASSESSMENT_VERSION = "domain_assessment/2.0"

STATUS_OK = "ok"
STATUS_FAILED = "failed"

DEFAULT_PROMPT_CHARS = 3_000

ASSESSMENT_SYSTEM_PROMPT = (
    "You are giving a BROAD, first-pass reading of a Pakistani case-law "
    "document. Decide only which of the given domains the case belongs to. "
    "Do NOT summarise the holding, do NOT analyse the law, and do NOT list "
    "statutes or section numbers. If the case does not clearly belong to a "
    f'listed domain, answer "{OTHER_DOMAIN_ID}" -- an honest '
    '"uncertain" is more useful here than a forced guess, because a later '
    "stage weighs this against other evidence. Respond with ONLY a JSON "
    "object (no prose, no markdown fence) with exactly these keys: "
    '"domain" (one domain id), "confidence" (a number between 0 and 1), '
    'and "reason" (one short sentence citing what in the text supports it).'
)


@dataclass
class DomainAssessment:
    """The model's broad reading of one case."""

    doc_id: str
    domain: str | None = None
    confidence: float | None = None
    reason: str | None = None
    status: str = STATUS_OK
    error: str | None = None
    model_name: str | None = None

    @property
    def failed(self) -> bool:
        return self.status == STATUS_FAILED

    @property
    def is_uncertain(self) -> bool:
        return self.domain == OTHER_DOMAIN_ID


def render_domain_definitions(taxonomy: FrozenTaxonomy) -> str:
    """The domain menu, rendered once per run rather than per document."""

    lines: list[str] = []
    for domain in taxonomy.domains:
        if domain.is_other:
            continue
        lines.append(f"- id: {domain.domain_id}\n  name: {domain.name}")
        if domain.description:
            lines.append(f"  about: {domain.description}")
        for criterion in domain.inclusion_criteria[:4]:
            lines.append(f"  typically: {criterion}")
    lines.append(
        f"- id: {OTHER_DOMAIN_ID}\n  name: Other / Uncertain\n"
        "  about: Use when the case does not clearly belong to a domain above."
    )
    return "\n".join(lines)


def build_assessment_prompt(
    representation: CaseRepresentation,
    domain_block: str,
    prompt_chars: int = DEFAULT_PROMPT_CHARS,
) -> str:
    """Assemble the prompt for one case: the domain menu and the document.

    **The keyword result is deliberately absent.** It used to be appended
    as a labelled "hint", on the reasoning that a model told it may
    disagree would stay independent. That reasoning does not survive
    contact with how the signals are then used: Phase 5 weights keyword and
    LLM at 0.40 each *as two independent readings*, and a model shown the
    keyword verdict agrees with it more often than it would blind. The
    inflated agreement then makes the keyword/LLM conflict rule -- the main
    safety valve in the decision engine -- fire less often, and inflates
    Phase 7's LLM/human agreement metric too.

    So the model sees only the document: title, headings, and a bounded
    slice of the text. The keyword signal reaches the decision on its own
    weight, which is the only way its 0.40 means what the engine says it
    means.
    """

    parts = [f"DOMAINS:\n{domain_block}", "", "CASE:"]
    if representation.title:
        parts.append(f"Title: {representation.title}")
    if representation.headings:
        parts.append("Headings: " + "; ".join(representation.headings[:10]))
    parts.append(f"Text:\n{representation.prompt_text(prompt_chars)}")
    parts.append(
        "\nWhich domain does this case broadly belong to? Respond with the "
        "JSON object only."
    )
    return "\n".join(parts)


def _coerce_confidence(value: Any) -> float | None:
    """Accept 0-1 floats and whole-number percentages; reject the rest."""

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
        # "confidence": 90 means 90%; "confidence": 1.7 is malformed.
        if not value.is_integer():
            return None
        value = value / 100.0
    if not 0.0 <= value <= 1.0:
        return None
    return value


def validate_assessment(
    parsed: dict[str, Any],
    taxonomy: FrozenTaxonomy,
    doc_id: str,
    max_reason_chars: int = 400,
) -> DomainAssessment:
    """Turn a parsed response into an assessment, or into a failure."""

    raw_domain = parsed.get("domain") or parsed.get("primary_domain")
    domain = str(raw_domain).strip() if raw_domain is not None else ""
    if not domain:
        return DomainAssessment(doc_id=doc_id, status=STATUS_FAILED, error="missing_domain")
    if domain not in taxonomy.assignable_ids:
        return DomainAssessment(
            doc_id=doc_id,
            status=STATUS_FAILED,
            error=f"unknown_domain_id:{domain[:60]}",
        )

    confidence = _coerce_confidence(parsed.get("confidence"))
    if confidence is None:
        return DomainAssessment(
            doc_id=doc_id,
            status=STATUS_FAILED,
            error=f"invalid_confidence:{str(parsed.get('confidence'))[:40]}",
        )

    reason = str(parsed.get("reason") or parsed.get("justification") or "").strip()
    if not reason:
        return DomainAssessment(doc_id=doc_id, status=STATUS_FAILED, error="missing_reason")

    return DomainAssessment(
        doc_id=doc_id,
        domain=domain,
        confidence=confidence,
        reason=reason[:max_reason_chars],
        status=STATUS_OK,
    )


def assess_domain(
    representation: CaseRepresentation,
    taxonomy: FrozenTaxonomy,
    llm_client: LLMClient,
    domain_block: str | None = None,
    prompt_chars: int = DEFAULT_PROMPT_CHARS,
    model_name: str | None = None,
) -> DomainAssessment:
    """Ask the model for a broad domain reading of one case. Never raises.

    An LLM or parsing failure yields ``status='failed'`` with the reason,
    so one bad document cannot abort a corpus run and cannot masquerade as
    a successful assessment either.
    """

    block = (
        domain_block if domain_block is not None else render_domain_definitions(taxonomy)
    )

    try:
        raw = llm_client.complete(
            system=ASSESSMENT_SYSTEM_PROMPT,
            prompt=build_assessment_prompt(
                representation, block, prompt_chars=prompt_chars
            ),
        )
        parsed = extract_json_object(raw)
    except Exception as exc:  # noqa: BLE001 - deliberately broad; see docstring
        logger.error("Domain assessment failed for %s: %s", representation.doc_id, exc)
        return DomainAssessment(
            doc_id=representation.doc_id,
            status=STATUS_FAILED,
            error=f"{type(exc).__name__}: {exc}"[:300],
            model_name=model_name,
        )

    assessment = validate_assessment(parsed, taxonomy, representation.doc_id)
    assessment.model_name = model_name
    return assessment
