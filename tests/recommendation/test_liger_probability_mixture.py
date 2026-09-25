import pytest
import torch
from test_liger import batch, model

from src.data.components.liger_trace import validate_liger_trace
from src.recommendation.liger.candidate_guidance import ProbabilityMixtureProcessor


@pytest.mark.parametrize("alpha", [0.0, 0.5, 1.0])
def test_probability_matches_direct_item_enumeration(alpha):
    sids = torch.tensor([[0, 0], [0, 1], [0, 2], [1, 0], [2, 1]])
    content = torch.tensor([[1000.0, 999.0, -1000.0, 0.0, 3.0], [-10.0, 0.0, 5.0, 6.0, 7.0]])
    processor = ProbabilityMixtureProcessor(sids, content, 3, alpha)
    paths = torch.cat([sids, sids.flip(0)])
    ids = torch.zeros(10, 1, dtype=torch.long)
    cumulative = torch.zeros(10)
    raw = torch.log_softmax(torch.arange(80.0).reshape(10, 8) / 13, -1)
    for depth in range(2):
        got = processor(ids, raw)
        for row in range(10):
            user = row // 5
            mask = (sids[:, :depth] == paths[row, :depth]).all(-1)
            mass = torch.tensor([content[user][mask & (sids[:, depth] == token)].logsumexp(0) for token in range(3)])
            legal = torch.isfinite(mass)
            start = depth * 3 + 1
            gen = raw[row, start : start + 3].masked_fill(~legal, -torch.inf).softmax(-1)
            expected = (1 - alpha) * gen + alpha * mass.softmax(-1)
            torch.testing.assert_close(got[row, start : start + 3].exp(), expected, atol=2e-6, rtol=2e-6)
        torch.testing.assert_close(got.exp().sum(-1), torch.ones(10))
        assert torch.isneginf(got[:, 0]).all() and torch.isneginf(got[:, 7]).all()
        token = paths[:, depth] + 1 + depth * 3
        cumulative += got.gather(1, token[:, None]).flatten()
        ids = torch.cat([ids, token[:, None]], 1)
    if alpha == 1:
        expected = torch.cat([content.log_softmax(-1)[0], content.log_softmax(-1)[1].flip(0)])
        torch.testing.assert_close(cumulative, expected, atol=3e-5, rtol=2e-6)


def test_dead_beam_is_negative_infinity_without_nan():
    p = ProbabilityMixtureProcessor(torch.tensor([[0, 0], [1, 1]]), torch.tensor([[1.0, 2.0]]), 3, 0.5)
    # 一个无目录前缀，一个错误层级token，一个合法前缀。
    out = p(torch.tensor([[0, 3], [0, 5], [0, 1]]), torch.zeros(3, 8))
    assert torch.isneginf(out[:2]).all()
    assert not torch.isnan(out).any()
    assert out[2, 4] == 0


def test_real_hf_probability_mixture_trace_and_label_independence():
    torch.manual_seed(17)
    mixed = model(candidate_strategy="probability_mixture", content_mixture_alpha=0, candidate_trace=True).eval()
    zero = mixed.predict_step(batch())
    assert (zero.auxiliary["liger_candidates"]["trace"]["generated_unique_count"] == 3).all()
    mixed.content_mixture_alpha = 0.5
    out = mixed.predict_step(batch())
    bundle = dict(keys=out.keys, **out.auxiliary["liger_candidates"])
    validate_liger_trace(bundle)
    assert bundle["metadata"]["content_mixture_alpha"] == 0.5
    assert bundle["metadata"]["candidate_normalization"] == "legal_conditional"
    x, label = batch()
    label.target_ids = label.target_ids.flip(0)
    torch.testing.assert_close(mixed.predict_step((x, label)).predictions, out.predictions)


@pytest.mark.parametrize("alpha", [-0.1, 1.1, float("nan"), float("inf")])
def test_bad_alpha_rejected(alpha):
    with pytest.raises(ValueError, match="alpha"):
        model(candidate_strategy="probability_mixture", content_mixture_alpha=alpha)


def test_non_mixture_alpha_and_retired_guidance_option_rejected():
    with pytest.raises(ValueError, match="alpha"):
        model(candidate_strategy="original", content_mixture_alpha=0.5)
    with pytest.raises(TypeError, match="content_guidance_weight"):
        model(candidate_strategy="probability_mixture", content_guidance_weight=1)
