from unittest.mock import patch

import pytest
import torch
from test_liger import batch, model

from src.recommendation.liger.candidate_guidance import ProbabilityMixtureProcessor
from src.recommendation.liger.joint_mixture import JointMixtureLiger


def joint(**kwargs):
    with patch("test_liger.Liger", JointMixtureLiger):
        return model(**kwargs)


def test_joint_gradients_initialization_and_cold_support():
    torch.manual_seed(42)
    base = model()
    rng = torch.random.get_rng_state()
    torch.manual_seed(42)
    m = joint()
    assert torch.equal(rng, torch.random.get_rng_state())
    for name, value in base.state_dict().items():
        torch.testing.assert_close(value, m.state_dict()[name], rtol=0, atol=0)
    x, y = batch()
    outputs = m.losses(x, y.target_ids, training=True)
    torch.testing.assert_close(outputs["loss"], sum(outputs[k] for k in ("sid_loss", "content_loss", "mixture_loss")))
    outputs["mixture_loss"].backward()
    for p in [
        m.transformer.decoder.block[0].layer[0].SelfAttention.q.weight,
        m.transformer.encoder.block[0].layer[0].SelfAttention.q.weight,
        m.content_projection.output.weight,
        m.dynamic_gate.bias,
    ]:
        assert p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum() > 0
    assert m.content_bank.grad is None
    # 训练seen mask只影响原内容CE，混合项保持完整目录支持集。
    torch.testing.assert_close(outputs["mixture_loss"], m.losses(x, y.target_ids)["mixture_loss"])


def test_content_aggregation_gradient_matches_enumeration():
    sids = torch.tensor([[0, 0], [0, 1], [1, 0], [2, 1]])
    scores = torch.tensor([[0.2, -0.3, 0.7, 0.4]], dtype=torch.double, requires_grad=True)
    raw = torch.randn(1, 8, dtype=torch.double, requires_grad=True)
    proc = ProbabilityMixtureProcessor(sids, scores, 3, 0.5)
    _, g, c, _ = proc.distributions(torch.zeros(1, 1, dtype=torch.long), raw)
    got = -torch.logaddexp(g[0, 0], c[0, 0]) + torch.log(torch.tensor(2.0, dtype=torch.double))
    mass = torch.stack([scores[0, sids[:, 0] == i].logsumexp(0) for i in range(3)])
    ref = -((raw[0, 1:4].softmax(0)[0] + mass.softmax(0)[0]) / 2).log()
    torch.testing.assert_close(got, ref)
    grads = torch.autograd.grad(got, (scores, raw), retain_graph=True)
    refs = torch.autograd.grad(ref, (scores, raw))
    for a, b in zip(grads, refs, strict=True):
        torch.testing.assert_close(a, b)


def test_teacher_forcing_matches_incremental_mixture():
    m = joint().eval()
    with torch.no_grad():
        m.dynamic_gate.bias.fill_(0.8)
    x, y = batch()
    encoded, mask, query = m.encode(x.input_ids, x.attention_mask)
    processor = ProbabilityMixtureProcessor(
        m.semantic_ids, m.dense_logits(query), m.codebook_size, 0.5, gate=m.dynamic_gate
    )
    prefix = torch.zeros(len(y.target_ids), 1, dtype=torch.long)
    total = []
    for depth in range(m.num_hierarchies):
        out = m.transformer(encoder_outputs=encoded, attention_mask=mask, decoder_input_ids=prefix)
        logits = out.logits[:, -1]
        target = y.target_ids[:, depth, None] + m.offsets[depth]
        total.append(processor(prefix, logits).gather(1, target))
        prefix = torch.cat([prefix, target], dim=1)
    torch.testing.assert_close(-torch.stack(total).mean(), m.losses(x, y.target_ids)["mixture_loss"])


def test_checkpoint_beam_ablation_and_labels():
    m = joint(candidate_trace=True).eval()
    with torch.no_grad():
        m.dynamic_gate.bias.fill_(1.1)
    checkpoint = {}
    m.on_save_checkpoint(checkpoint)
    restored = joint(candidate_trace=True).eval()
    restored.on_load_checkpoint(checkpoint)
    restored.load_state_dict(m.state_dict())
    x, y = batch()
    a = m.predict_step((x, y))
    b = restored.predict_step((x, y))
    torch.testing.assert_close(a.predictions, b.predictions)
    assert a.auxiliary["liger_candidates"]["metadata"]["content_mixture_alpha"] == pytest.approx(
        torch.sigmoid(torch.tensor(1.1)).item()
    )
    y.target_ids = y.target_ids.flip(0)
    torch.testing.assert_close(a.predictions, m.predict_step((x, y)).predictions)
    ablation = joint(content_only=True).eval()
    ablation.load_state_dict(m.state_dict())
    reference = model(candidate_strategy="probability_mixture", content_mixture_alpha=1).eval()
    reference.load_state_dict({k: v for k, v in m.state_dict().items() if not k.startswith("dynamic_gate.")})
    torch.testing.assert_close(ablation.retrieve(x)[0], reference.retrieve(x)[0])
    with pytest.raises(ValueError, match="inference-only"):
        ablation.losses(x, y.target_ids, training=True)
    with pytest.raises(ValueError, match="matching joint"):
        restored.on_load_checkpoint({"liger_catalog_sha256": m.catalog_sha256})


def test_fixed_inference_alpha_is_checkpoint_compatible_and_inference_only():
    trained = joint(candidate_trace=True).eval()
    with torch.no_grad():
        trained.dynamic_gate.bias.fill_(1.1)
    checkpoint = {}
    trained.on_save_checkpoint(checkpoint)
    fixed = joint(candidate_trace=True, inference_mixture_alpha=0.5).eval()
    fixed.on_load_checkpoint(checkpoint)
    fixed.load_state_dict(trained.state_dict(), strict=True)
    reference = model(candidate_strategy="probability_mixture", content_mixture_alpha=0.5).eval()
    reference.load_state_dict({k: v for k, v in trained.state_dict().items() if not k.startswith("dynamic_gate.")})
    x, y = batch()
    torch.testing.assert_close(fixed.retrieve(x)[0], reference.retrieve(x)[0])
    output = fixed.predict_step((x, y))
    metadata = output.auxiliary["liger_candidates"]["metadata"]
    assert metadata["content_mixture_alpha"] == 0.5
    assert metadata["mixture_alpha_source"] == "fixed_inference"
    with pytest.raises(ValueError, match="inference-only"):
        fixed.losses(x, y.target_ids, training=True)
    for value in [-0.1, 1.1, float("nan")]:
        with pytest.raises(ValueError, match="finite"):
            joint(inference_mixture_alpha=value)
    with pytest.raises(ValueError, match="another inference ablation"):
        joint(content_only=True, inference_mixture_alpha=0.5)
