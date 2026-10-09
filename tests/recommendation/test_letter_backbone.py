import pytest
import torch
import torch.nn.functional as F
from transformers import PrefixConstrainedLogitsProcessor

from src.recommendation.letter.backbone import LetterBackbone, LetterPrefixLogitsProcessor


def backbone(**kwargs):
    sids = torch.tensor([[a, b, 0, 0] for a in range(2) for b in range(4)])
    return LetterBackbone(
        torch.arange(8) * 3,
        sids,
        codebook_size=4,
        base_vocab_size=8,
        d_model=8,
        d_ff=16,
        d_kv=4,
        num_heads=2,
        num_layers=1,
        dropout=0,
        generation_candidates=4,
        top_k=3,
        **kwargs,
    )


def test_temperature_and_standard_t5_reference():
    model = backbone(temperature=0.7)
    tokens = model.token_rows(torch.tensor([0, 3]))
    inputs = torch.cat((tokens, torch.ones(2, 1, dtype=torch.long)), -1)
    labels = inputs.clone()
    labels[1, -1] = -100
    loss, logits = model(inputs, torch.ones_like(inputs), labels)
    expected = F.cross_entropy(logits.flatten(0, 1) / 0.7, labels.flatten(), ignore_index=-100)
    torch.testing.assert_close(loss, expected)
    assert model.t5.shared.weight is model.t5.lm_head.weight
    model.temperature = 1
    loss, _ = model(inputs, torch.ones_like(inputs), labels)
    official = model.t5(input_ids=inputs, attention_mask=torch.ones_like(inputs), labels=labels).loss
    torch.testing.assert_close(loss, official)
    loss.backward()
    assert model.t5.shared.weight.grad.isfinite().all()


def test_trie_generation_is_legal_unique_and_not_temperature_scaled():
    torch.manual_seed(42)
    model = backbone().eval()
    inputs = torch.cat((model.token_rows(torch.tensor([0])), torch.ones(1, 1, dtype=torch.long)), -1)
    result, scores = model.generate(inputs, torch.ones_like(inputs))
    assert result.shape == (1, 3) and result.unique().numel() == 3
    assert all(value in model.item_keys for value in result.flatten())
    model.temperature = 0.01
    other, other_scores = model.generate(inputs, torch.ones_like(inputs))
    torch.testing.assert_close(result, other)
    torch.testing.assert_close(scores, other_scores)


def test_lexicographic_token_mapping():
    sids = torch.tensor([[code, 0, 0, 0] for code in (0, 2, 10)])
    model = LetterBackbone(
        torch.arange(3),
        sids,
        base_vocab_size=8,
        generation_candidates=2,
        top_k=1,
        d_model=8,
        d_ff=16,
        d_kv=4,
        num_heads=2,
        num_layers=1,
    )
    assert model.catalog_tokens[:, 0].tolist() == [8, 10, 9]


def test_invalid_temperature_and_keys():
    for temperature in (0, -1, float("nan")):
        with pytest.raises(ValueError, match="temperature"):
            backbone(temperature=temperature)
    with pytest.raises(ValueError, match="absent"):
        backbone().token_rows(torch.tensor([1]))


@pytest.mark.parametrize("depth", range(1, 8))
def test_batched_prefix_mask_matches_hf_and_reads_matrix_once(depth, monkeypatch):
    model = backbone()
    paths = torch.cat(
        (
            torch.zeros(8, 1, dtype=torch.long),
            model.catalog_tokens,
            torch.ones(8, 1, dtype=torch.long),
            torch.zeros(8, 1, dtype=torch.long),
        ),
        dim=1,
    )
    prefixes = paths[:, :depth]
    scores = torch.randn(8, model.t5.config.vocab_size)
    scores[0, 0] = -torch.inf
    expected = PrefixConstrainedLogitsProcessor(model.allowed_tokens, 4)(prefixes, scores)
    original = torch.Tensor.tolist
    reads = []

    def counted(tensor):
        reads.append(tensor.shape)
        return original(tensor)

    monkeypatch.setattr(torch.Tensor, "tolist", counted)
    actual = LetterPrefixLogitsProcessor(model.prefixes)(prefixes, scores)
    assert torch.equal(actual, expected)
    assert reads == [prefixes.shape]


@pytest.mark.parametrize("prefixes, message", [({(0,): []}, "empty"), ({}, "unknown")])
def test_batched_prefix_rejects_invalid_paths(prefixes, message):
    with pytest.raises(ValueError, match=message):
        LetterPrefixLogitsProcessor(prefixes)(torch.tensor([[0] if prefixes else [0, 9]]), torch.zeros(1, 16))


@pytest.mark.parametrize("seed", [0, 42, 200, 2026])
def test_generation_matches_original_hf_callback(seed, monkeypatch):
    torch.manual_seed(seed)
    model = backbone().eval()
    if seed == 0:
        # 所有 logits 同分，覆盖 beam tie 与最终稳定排序边界。
        with torch.no_grad():
            for parameter in model.parameters():
                parameter.zero_()
    inputs = torch.cat((model.token_rows(torch.tensor([0, 3, 21])), torch.ones(3, 1, dtype=torch.long)), dim=1)
    mask = torch.ones_like(inputs)
    state = {name: value.clone() for name, value in model.state_dict().items()}
    actual_ids, actual_scores = model.generate(inputs, mask)
    original_generate = model.t5.generate

    def legacy_generate(**kwargs):
        assert "prefix_allowed_tokens_fn" not in kwargs
        kwargs.pop("logits_processor")
        return original_generate(**kwargs, prefix_allowed_tokens_fn=model.allowed_tokens)

    monkeypatch.setattr(model.t5, "generate", legacy_generate)
    expected_ids, expected_scores = model.generate(inputs, mask)
    assert torch.equal(actual_ids, expected_ids)
    assert torch.equal(actual_scores, expected_scores)
    model.load_state_dict(state, strict=True)
