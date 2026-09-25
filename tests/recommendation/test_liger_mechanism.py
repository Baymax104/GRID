import pytest
import torch
from test_liger import batch
from test_liger_joint import joint

from src.recommendation.liger.candidate_guidance import ProbabilityMixtureProcessor


def test_max_mass_enumeration_and_identical_support():
    ids = torch.tensor([[0, 0], [0, 1], [1, 0]])
    scores = torch.tensor([[0., 0., 0.4]])
    raw = torch.tensor([[0., 0.2, -0.1, 0., 0.]])
    prefix = torch.zeros(1, 1, dtype=torch.long)
    mass = ProbabilityMixtureProcessor(ids, scores, 2, 0.5)
    maximum = ProbabilityMixtureProcessor(ids, scores, 2, 0.5, aggregation="max")
    _, g, c, valid = mass.distributions(prefix, raw)
    _, gm, cm, vm = maximum.distributions(prefix, raw)
    assert torch.equal(valid, vm) and torch.equal(g, gm)
    torch.testing.assert_close(c.exp(), torch.tensor([[2., torch.exp(torch.tensor(.4))]]) / (2 + torch.exp(torch.tensor(.4))))
    torch.testing.assert_close(cm, torch.tensor([[0., .4]]).log_softmax(-1))
    assert c.argmax(-1).item() != cm.argmax(-1).item()
    torch.testing.assert_close(maximum(prefix, raw)[:, 1:3].exp(), (g.exp() + cm.exp()) / 2)
    for p in (torch.tensor([[0, 1]]), torch.tensor([[0, 2]]), torch.tensor([[0, 0]])):
        a, b = mass(p, raw), maximum(p, raw)
        assert torch.equal(torch.isfinite(a), torch.isfinite(b))
        assert not torch.isnan(a).any() and not torch.isnan(b).any()


@pytest.mark.parametrize("control", ["learned_mass", "legal_generation", "max_mixture"])
def test_checkpoint_scores_support_training_and_metadata(control):
    base = joint(candidate_trace=True).eval()
    m = joint(candidate_trace=True, mechanism_control=control).eval()
    ckpt = {}
    base.on_save_checkpoint(ckpt)
    m.on_load_checkpoint(ckpt)
    m.load_state_dict(base.state_dict(), strict=True)
    x, y = batch()
    with torch.no_grad():
        _, _, q = base.encode(x.input_ids, x.attention_mask)
        _, _, qm = m.encode(x.input_ids, x.attention_mask)
        logits = base.dense_logits(q)
        torch.testing.assert_close(logits, m.dense_logits(qm), rtol=0, atol=0)
        prefix = torch.zeros(len(q), 1, dtype=torch.long)
        raw = torch.randn(len(q), base.transformer.config.vocab_size)
        a = base.candidate_processor(logits)(prefix, raw)
        b = m.candidate_processor(logits)(prefix, raw)
        assert torch.equal(torch.isfinite(a), torch.isfinite(b))
        if control == "legal_generation":
            _, g, _, _ = base.candidate_processor(logits).distributions(prefix, raw)
            torch.testing.assert_close(b[:, 1:1 + base.codebook_size], g)
        output = m.predict_step((x, y))
        meta = output.auxiliary["liger_candidates"]["metadata"]
        assert meta["mechanism_control"] == control
        if control == "legal_generation":
            assert meta["content_mixture_alpha"] == 0
        y.target_ids = y.target_ids.flip(0)
        torch.testing.assert_close(output.predictions, m.predict_step((x, y)).predictions)
    if control != "learned_mass":
        with pytest.raises(ValueError, match="inference-only"):
            m.losses(x, y.target_ids, training=True)
        with pytest.raises(ValueError, match="combined"):
            joint(content_only=True, mechanism_control=control)


def test_invalid_control():
    with pytest.raises(ValueError, match="Unknown"):
        joint(mechanism_control="typo")
