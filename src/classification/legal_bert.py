"""Legal-BERT encoder infrastructure for the domain-classification path.

**This module does not classify anything, and deliberately cannot.**

``AutoModel`` gives a transformer *encoder*: it turns legal text into
vectors. It has no notion of the frozen taxonomy's domains. Turning it into
a domain classifier requires a supervised head trained on human-labelled
Pakistani judgments, and this project currently has none --
no review-ledger rows, no labels file, and fixtures organised by *court*
rather than by domain. So:

* :class:`LegalBertEncoder` is real and usable -- it is what a future
  fine-tune will be built on, and it is useful now for feature extraction
  and similarity work.
* :func:`load_domain_classifier` refuses, loudly, until a trained
  checkpoint exists. That refusal is the point of this module's design: an
  encoder with an untrained head would still emit three numbers per
  document, and those numbers would look exactly like predictions while
  being noise. A pipeline that cannot tell the difference between a
  classifier and a random projection is worse than one with no classifier
  at all, because the first silently poisons a corpus.

The model is loaded **once per process** (see
:func:`get_legal_bert_encoder`) rather than per document: loading
``nlpaueb/legal-bert-base-uncased`` costs seconds and ~420MB, which would
be catastrophic 14,000 times over.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from src.common.exceptions import ConfigurationError
from src.common.logging_utils import get_logger

logger = get_logger(__name__)

DEFAULT_MODEL_NAME = "nlpaueb/legal-bert-base-uncased"

# BERT's positional embeddings stop at 512 tokens. This is a hard ceiling,
# not a tunable preference: the case signature is bounded at 20,000
# characters, so the encoder sees roughly its first 2,000 -- less than the
# 3,000 characters the current classifier prompt receives. Any future head
# has to be trained on the same truncation it will see at inference.
MAX_MODEL_TOKENS = 512

# The three domains a head would have to predict come from the frozen
# registry, never from here -- this module holds no domain vocabulary.


@dataclass
class LegalBertEncoder:
    """A loaded Legal-BERT encoder, reused across a whole run.

    Prefer :func:`get_legal_bert_encoder`, which caches instances so the
    weights are read from disk once per process.
    """

    model_name: str = DEFAULT_MODEL_NAME
    max_length: int = MAX_MODEL_TOKENS
    device: str = "cpu"
    batch_size: int = 8
    _tokenizer: Any = field(default=None, repr=False)
    _model: Any = field(default=None, repr=False)
    _load_count: int = field(default=0, repr=False)

    def __post_init__(self) -> None:
        if self.max_length > MAX_MODEL_TOKENS:
            raise ConfigurationError(
                f"legal_bert.max_length={self.max_length} exceeds BERT's "
                f"{MAX_MODEL_TOKENS}-token limit; the model cannot attend "
                "beyond it and transformers would raise at inference"
            )

    # -- loading ---------------------------------------------------------

    def load(self) -> None:
        """Read tokenizer and weights. Idempotent: a second call is a no-op."""

        if self._model is not None:
            return

        try:
            from transformers import AutoModel, AutoTokenizer
        except ImportError as exc:  # pragma: no cover - dependency present
            raise ConfigurationError(
                "transformers is required for the Legal-BERT encoder"
            ) from exc

        logger.info("Loading Legal-BERT encoder %s on %s", self.model_name, self.device)
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        model = AutoModel.from_pretrained(self.model_name)
        model.to(self.device)
        # Inference only: eval() disables dropout, which would otherwise
        # make the same text encode differently on every call.
        model.eval()
        self._model = model
        self._load_count += 1

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    @property
    def load_count(self) -> int:
        """How many times weights were actually read -- asserted by tests."""

        return self._load_count

    # -- encoding --------------------------------------------------------

    def encode(self, texts: Iterable[str]) -> np.ndarray:
        """Encode case signatures into one mean-pooled vector each.

        Mean pooling over the attention mask, not the ``[CLS]`` vector:
        ``[CLS]`` is only meaningful after it has been trained for a
        downstream task, and this encoder has not been. Padding is excluded
        from the mean so a short judgment is not diluted by its padding.

        Returns an array of shape ``(len(texts), hidden_size)``. These are
        **representations, not predictions** -- nothing here maps a vector
        to a domain.
        """

        import torch

        items = [t or "" for t in texts]
        if not items:
            return np.empty((0, 0), dtype=np.float32)

        self.load()
        vectors: list[np.ndarray] = []

        with torch.inference_mode():
            for start in range(0, len(items), max(self.batch_size, 1)):
                batch = items[start : start + max(self.batch_size, 1)]
                encoded = self._tokenizer(
                    batch,
                    padding=True,
                    truncation=True,
                    max_length=self.max_length,
                    return_tensors="pt",
                )
                encoded = {k: v.to(self.device) for k, v in encoded.items()}
                output = self._model(**encoded)

                hidden = output.last_hidden_state              # (B, T, H)
                mask = encoded["attention_mask"].unsqueeze(-1) # (B, T, 1)
                mask = mask.type_as(hidden)
                summed = (hidden * mask).sum(dim=1)
                counts = mask.sum(dim=1).clamp(min=1e-9)
                pooled = summed / counts
                vectors.append(pooled.cpu().numpy().astype(np.float32))

        return np.vstack(vectors)

    def encode_signature(self, signature_text: str) -> np.ndarray:
        """Encode one case signature -- the bounded text the pipeline already uses."""

        return self.encode([signature_text])[0]


@lru_cache(maxsize=4)
def _cached_encoder(
    model_name: str, max_length: int, device: str, batch_size: int
) -> LegalBertEncoder:
    return LegalBertEncoder(
        model_name=model_name,
        max_length=max_length,
        device=device,
        batch_size=batch_size,
    )


def get_legal_bert_encoder(
    model_name: str | None = None,
    max_length: int | None = None,
    device: str | None = None,
    batch_size: int | None = None,
) -> LegalBertEncoder:
    """The shared encoder for this process, created once per configuration.

    Defaults come from ``legal_bert`` in the project configuration, so the
    model name is never hard-coded at a call site.
    """

    from src.common.config import get_settings

    config = get_settings().legal_bert
    encoder = _cached_encoder(
        model_name or config.model_name,
        max_length or config.max_length,
        device or config.device,
        batch_size or config.batch_size,
    )
    encoder.load()
    return encoder


def reset_encoder_cache() -> None:
    """Drop cached encoders. For tests and for a deliberate model switch."""

    _cached_encoder.cache_clear()


# ---------------------------------------------------------------------------
# The classification head that does not exist yet
# ---------------------------------------------------------------------------

# What a trained head would need, recorded here so the requirement is not
# rediscovered later:
#
#   * human-reviewed (doc_id, domain) pairs for Pakistani judgments, from
#     the Phase 6 review ledger or an external labels file;
#   * a train/validation/test split by doc_id, with no case appearing in
#     more than one split;
#   * enough per-class examples that `other_uncertain` is learnable rather
#     than a dumping ground;
#   * the same 512-token truncation at training time as at inference.
MINIMUM_LABELS_PER_DOMAIN = 100


def load_domain_classifier(checkpoint: str | Path | None = None):
    """Load a fine-tuned Legal-BERT domain classifier.

    Raises :class:`ConfigurationError` while no checkpoint exists, which is
    the current state of this project. This is deliberate and is not a
    stub to be filled in with a heuristic: an untrained head emits three
    plausible-looking numbers per document, and Phase 5 would weight them
    at 0.40 exactly as if they meant something.

    Pseudo-labels are not an acceptable substitute. Training on keyword
    output would teach the model the keyword rules -- after which Phase 5
    would be weighting the same signal twice under two names, and the
    keyword/classifier conflict rule, its main safety valve, could never
    fire.
    """

    from src.common.config import get_settings

    checkpoint = checkpoint or get_settings().legal_bert.classifier_checkpoint
    if checkpoint is None:
        raise ConfigurationError(
            "No Legal-BERT domain classifier checkpoint is configured "
            "(legal_bert.classifier_checkpoint is unset), and the base "
            "encoder cannot classify: nlpaueb/legal-bert-base-uncased is an "
            "encoder with no task head, so it has no notion of the frozen "
            "taxonomy's domains.\n\n"
            "A head must be fine-tuned on human-reviewed labels first. This "
            "project currently has none: document_review_decisions is empty, "
            "evaluation.labels_file is unset, and the test fixtures are "
            f"organised by court rather than domain. Roughly "
            f"{MINIMUM_LABELS_PER_DOMAIN} reviewed cases per domain is the "
            "minimum worth training on.\n\n"
            "Until then the configured classifier signal remains in use; see "
            "src/classification/domain_assessment.py."
        )

    path = Path(checkpoint)
    if not path.exists():
        raise ConfigurationError(
            f"legal_bert.classifier_checkpoint points at {path}, which does "
            "not exist"
        )
    raise ConfigurationError(
        f"A checkpoint exists at {path} but no loader is implemented yet: "
        "the training code must be written together with its evaluation, so "
        "that a head is never usable before it has been measured on a held-"
        "out split."
    )
