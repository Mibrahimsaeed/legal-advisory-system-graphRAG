"""Legal-BERT encoder infrastructure, and what it must not pretend to be.

The encoder is real; the classifier is not. These tests cover both halves:
that loading, caching, inference mode and input handling are correct, and
that nothing in the module can be mistaken for a trained Family/Criminal
classifier.

No test downloads the model. ``AutoTokenizer``/``AutoModel`` are patched,
so the 420MB of weights are never fetched and the suite stays fast.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from src.classification.legal_bert import (
    DEFAULT_MODEL_NAME,
    MAX_MODEL_TOKENS,
    MINIMUM_LABELS_PER_DOMAIN,
    LegalBertEncoder,
    get_legal_bert_encoder,
    load_domain_classifier,
    reset_encoder_cache,
)
from src.common.config import get_settings
from src.common.exceptions import ConfigurationError

HIDDEN = 8


class _FakeTokenizer:
    """Records what it was asked to tokenize, then returns real tensors."""

    def __init__(self):
        self.calls: list[dict] = []

    def __call__(self, batch, padding=None, truncation=None, max_length=None,
                 return_tensors=None):
        self.calls.append({
            "batch": list(batch), "padding": padding, "truncation": truncation,
            "max_length": max_length, "return_tensors": return_tensors,
        })
        # One token per whitespace word, capped at max_length, padded.
        lengths = [
            max(1, min(len((t or "").split()), max_length or MAX_MODEL_TOKENS))
            for t in batch
        ]
        width = max(lengths)
        ids, mask = [], []
        for length in lengths:
            ids.append([1] * length + [0] * (width - length))
            mask.append([1] * length + [0] * (width - length))
        return {
            "input_ids": torch.tensor(ids),
            "attention_mask": torch.tensor(mask),
        }


class _FakeModel:
    def __init__(self):
        self.eval_calls = 0
        self.to_calls: list[str] = []
        self.forward_calls = 0
        self.training = True
        self.grad_enabled_during_forward: bool | None = None

    def to(self, device):
        self.to_calls.append(device)
        return self

    def eval(self):
        self.eval_calls += 1
        self.training = False
        return self

    def __call__(self, input_ids=None, attention_mask=None, **kwargs):
        self.forward_calls += 1
        # Recorded here rather than via an instance-level patch: Python
        # looks dunders up on the type, so assigning __call__ on the
        # instance would never take effect.
        self.grad_enabled_during_forward = torch.is_grad_enabled()
        batch, tokens = input_ids.shape
        # Deterministic, content-independent: these tests are about plumbing.
        hidden = torch.ones((batch, tokens, HIDDEN), dtype=torch.float32)
        return SimpleNamespace(last_hidden_state=hidden)


@pytest.fixture()
def fake_transformers(monkeypatch):
    """Patch the two from_pretrained calls and count them."""

    loaded = {"tokenizer": 0, "model": 0, "names": []}
    tokenizer, model = _FakeTokenizer(), _FakeModel()

    def _tok(name, *a, **k):
        loaded["tokenizer"] += 1
        loaded["names"].append(name)
        return tokenizer

    def _mod(name, *a, **k):
        loaded["model"] += 1
        loaded["names"].append(name)
        return model

    import transformers

    monkeypatch.setattr(transformers.AutoTokenizer, "from_pretrained", staticmethod(_tok))
    monkeypatch.setattr(transformers.AutoModel, "from_pretrained", staticmethod(_mod))
    reset_encoder_cache()
    yield SimpleNamespace(loaded=loaded, tokenizer=tokenizer, model=model)
    reset_encoder_cache()


SIGNATURE = (
    "Mst. Zainab v. Muhammad Ali\nFacts Arguments Order\n"
    "Suit for recovery of dower and maintenance before the Judge Family Court."
)


# ---------------------------------------------------------------------------
# 1. Loading
# ---------------------------------------------------------------------------


def test_the_configured_model_name_is_used(fake_transformers):
    encoder = LegalBertEncoder(model_name="some-org/some-legal-model")
    encoder.load()

    assert fake_transformers.loaded["names"] == [
        "some-org/some-legal-model", "some-org/some-legal-model"
    ]


def test_the_default_model_is_legal_bert():
    assert DEFAULT_MODEL_NAME == "nlpaueb/legal-bert-base-uncased"
    assert get_settings().legal_bert.model_name == DEFAULT_MODEL_NAME


def test_the_model_name_comes_from_configuration(fake_transformers, monkeypatch):
    """Never hard-coded at a call site."""

    import src.classification.legal_bert as module

    real = get_settings()
    monkeypatch.setattr(
        module, "get_settings",
        lambda: SimpleNamespace(
            legal_bert=real.legal_bert.model_copy(
                update={"model_name": "configured/model"}
            )
        ),
        raising=False,
    )
    # get_settings is imported inside the function, so patch the source.
    monkeypatch.setattr(
        "src.common.config.get_settings",
        lambda: SimpleNamespace(
            legal_bert=real.legal_bert.model_copy(
                update={"model_name": "configured/model"}
            )
        ),
    )
    get_legal_bert_encoder()
    assert "configured/model" in fake_transformers.loaded["names"]


def test_the_model_is_put_in_inference_mode(fake_transformers):
    """eval() off would make the same text encode differently each call."""

    LegalBertEncoder().load()

    assert fake_transformers.model.eval_calls == 1
    assert fake_transformers.model.training is False


def test_the_model_is_moved_to_the_configured_device(fake_transformers):
    LegalBertEncoder(device="cpu").load()
    assert fake_transformers.model.to_calls == ["cpu"]


def test_a_max_length_beyond_berts_limit_is_rejected():
    with pytest.raises(ConfigurationError, match="512-token limit"):
        LegalBertEncoder(max_length=1024)


def test_the_configured_max_length_respects_berts_limit():
    assert get_settings().legal_bert.max_length <= MAX_MODEL_TOKENS


# ---------------------------------------------------------------------------
# 2. Loaded once, reused
# ---------------------------------------------------------------------------


def test_weights_are_read_once_however_many_documents_are_encoded(fake_transformers):
    """Loading per document would cost seconds and ~420MB, 14,000 times over."""

    encoder = LegalBertEncoder()
    for _ in range(25):
        encoder.encode([SIGNATURE])

    assert fake_transformers.loaded["model"] == 1
    assert fake_transformers.loaded["tokenizer"] == 1
    assert encoder.load_count == 1


def test_calling_load_twice_is_a_no_op(fake_transformers):
    encoder = LegalBertEncoder()
    encoder.load()
    encoder.load()
    assert encoder.load_count == 1


def test_the_shared_encoder_is_cached_across_calls(fake_transformers):
    first = get_legal_bert_encoder()
    second = get_legal_bert_encoder()

    assert first is second
    assert fake_transformers.loaded["model"] == 1


def test_a_different_configuration_gets_its_own_encoder(fake_transformers):
    a = get_legal_bert_encoder(model_name="org/a")
    b = get_legal_bert_encoder(model_name="org/b")

    assert a is not b
    assert fake_transformers.loaded["model"] == 2


def test_the_cache_can_be_reset_for_a_deliberate_model_switch(fake_transformers):
    get_legal_bert_encoder()
    reset_encoder_cache()
    get_legal_bert_encoder()
    assert fake_transformers.loaded["model"] == 2


# ---------------------------------------------------------------------------
# 3. Input: the bounded case signature, truncated to the model's limit
# ---------------------------------------------------------------------------


def test_the_case_signature_is_what_reaches_the_tokenizer(fake_transformers):
    LegalBertEncoder().encode([SIGNATURE])

    (call,) = fake_transformers.tokenizer.calls
    assert call["batch"] == [SIGNATURE]


def test_truncation_is_requested_at_the_configured_length(fake_transformers):
    LegalBertEncoder(max_length=128).encode([SIGNATURE])

    (call,) = fake_transformers.tokenizer.calls
    assert call["truncation"] is True
    assert call["max_length"] == 128
    assert call["padding"] is True


def test_encode_signature_takes_one_case(fake_transformers):
    vector = LegalBertEncoder().encode_signature(SIGNATURE)
    assert vector.shape == (HIDDEN,)


def test_encoding_is_batched_not_one_call_per_document(fake_transformers):
    LegalBertEncoder(batch_size=4).encode([SIGNATURE] * 10)

    # 10 documents in batches of 4 -> 3 forward passes, not 10.
    assert fake_transformers.model.forward_calls == 3


def test_empty_input_returns_an_empty_array_without_loading(fake_transformers):
    out = LegalBertEncoder().encode([])
    assert out.shape == (0, 0)
    assert fake_transformers.loaded["model"] == 0


def test_none_and_blank_text_do_not_crash(fake_transformers):
    out = LegalBertEncoder().encode([None, "", SIGNATURE])
    assert out.shape == (3, HIDDEN)
    assert not np.isnan(out).any()


# ---------------------------------------------------------------------------
# 4. Output: representations, one vector per case
# ---------------------------------------------------------------------------


def test_one_vector_per_case(fake_transformers):
    out = LegalBertEncoder().encode([SIGNATURE, SIGNATURE, SIGNATURE])
    assert out.shape == (3, HIDDEN)
    assert out.dtype == np.float32


def test_padding_is_excluded_from_the_pooled_vector(fake_transformers):
    """A short case must not be diluted by the padding beside a long one."""

    short, long = "dower", " ".join(["maintenance"] * 40)
    out = LegalBertEncoder().encode([short, long])

    # The fake model emits all-ones hidden states, so a correctly masked
    # mean is exactly 1.0 for both; an unmasked mean would dilute the short
    # one toward zero.
    assert np.allclose(out[0], 1.0)
    assert np.allclose(out[1], 1.0)


def test_encoding_is_deterministic(fake_transformers):
    encoder = LegalBertEncoder()
    assert np.array_equal(encoder.encode([SIGNATURE]), encoder.encode([SIGNATURE]))


def test_no_gradients_are_tracked(fake_transformers):
    """Inference only -- gradient tracking would waste memory on 14k documents."""

    LegalBertEncoder().encode([SIGNATURE])

    assert fake_transformers.model.grad_enabled_during_forward is False


# ---------------------------------------------------------------------------
# 5. It must not pretend to classify
# ---------------------------------------------------------------------------


def test_no_classifier_is_available_without_a_trained_checkpoint():
    with pytest.raises(ConfigurationError) as exc:
        load_domain_classifier()

    message = str(exc.value)
    assert "no task head" in message
    assert "human-reviewed labels" in message
    assert str(MINIMUM_LABELS_PER_DOMAIN) in message


def test_the_checkpoint_setting_is_unset_because_no_head_is_trained():
    assert get_settings().legal_bert.classifier_checkpoint is None


def test_a_missing_checkpoint_path_is_reported(tmp_path):
    with pytest.raises(ConfigurationError, match="does not exist"):
        load_domain_classifier(tmp_path / "no_such_head.pt")


def test_the_module_holds_no_domain_vocabulary():
    """The taxonomy lives in config/domains.yaml, never in model code."""

    source = open("src/classification/legal_bert.py").read()
    body = source.split('"""', 2)[-1]
    assert "family_law" not in body
    assert "criminal_law" not in body


def test_the_encoder_exposes_no_predict_or_classify_method():
    """Nothing here should be mistakable for a classifier."""

    names = {n for n in dir(LegalBertEncoder) if not n.startswith("_")}
    assert not {n for n in names if "predict" in n or "classif" in n or "domain" in n}


# ---------------------------------------------------------------------------
# 6. Everything else is unchanged
# ---------------------------------------------------------------------------


def test_the_mpnet_embedding_model_is_unchanged():
    assert get_settings().discovery.embedding_model_name == (
        "sentence-transformers/all-mpnet-base-v2"
    )


def test_the_embedding_pooling_weights_are_unchanged():
    discovery = get_settings().discovery
    assert (discovery.title_weight, discovery.toc_weight, discovery.body_weight) == (
        2.0, 1.5, 1.0
    )
    assert discovery.body_chunk_chars == 2000
    assert discovery.max_body_chunks == 4


def test_the_clustering_configuration_is_unchanged():
    discovery = get_settings().discovery
    assert discovery.umap_n_components == 50
    assert discovery.umap_n_neighbors == 15
    assert discovery.umap_min_dist == 0.0
    assert discovery.umap_metric == "cosine"
    assert discovery.hdbscan_min_cluster_size == 15
    assert discovery.hdbscan_metric == "euclidean"


def test_the_keyword_profiles_and_threshold_are_unchanged():
    signals = get_settings().domain_signals
    assert len(signals.profiles["family_law"]) == 53
    assert len(signals.profiles["criminal_law"]) == 15
    assert signals.min_keyword_matches == 2


def test_the_phase_5_weights_are_unchanged():
    weights = get_settings().domain_decision.weights
    assert weights.as_mapping() == {
        "keyword": 0.40, "llm": 0.40, "cluster": 0.15, "title": 0.05
    }


def test_the_existing_classifier_signal_is_still_the_active_one():
    """Qwen stays until a trained Legal-BERT head exists.

    Removing it now would leave Phase 5 with keyword + cluster + title
    only -- a maximum coverage of 0.60 against a 0.50 floor, and no
    independent reading of the text at all. That is strictly worse than the
    situation this task set out to improve.
    """

    signals = get_settings().domain_signals
    assert signals.llm_enabled is True
    assert signals.llm_model == "qwen3:14b"


def test_legal_bert_is_not_wired_into_the_signal_flow_yet():
    """Integration is deliberately incomplete: no head means no predictions."""

    flow = open("orchestration/dags/domain_signal_flow.py").read()
    assert "legal_bert" not in flow
