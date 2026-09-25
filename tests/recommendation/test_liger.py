import copy

import pytest
import torch
from torch import nn
from torch.nn import functional as F

from src.data.components.data_models import TigerLabelData, TigerModelInput
from src.recommendation.liger import Liger


def catalog():
    return dict(
        keys=torch.tensor([10, 20, 40, 60, 90, 100]),
        semantic_ids=torch.tensor([[0, 0], [0, 1], [1, 0], [1, 1], [2, 0], [2, 1]]),
        embeddings=torch.arange(30).reshape(6, 5).float() / 30,
        seen_mask=torch.tensor([True, True, True, True, False, False]),
    )


def model(**kwargs):
    options = dict(
        catalog=catalog(),
        num_hierarchies=2,
        codebook_size=3,
        embedding_dim=8,
        num_layers=1,
        num_heads=2,
        d_kv=4,
        d_ff=16,
        dropout=0,
        max_history_items=3,
        projection_hidden_sizes=(7, 6),
        projection_dropout=0,
        input_dropout=0,
        generation_candidates=3,
        top_k=3,
        catalog_chunk_size=2,
    )
    options.update(kwargs)
    return Liger(**options)


def batch():
    x = TigerModelInput(
        torch.tensor([[0, 0, 0, 1, -1, -1], [1, 0, -1, -1, -1, -1]]),
        torch.tensor([[1, 1, 1, 1, 0, 0], [1, 1, 0, 0, 0, 0]]),
        torch.tensor([11, 12]),
    )
    return x, TigerLabelData(torch.tensor([[1, 0], [1, 1]]))


def test_joint_loss_matches_explicit_reference_and_gradients():
    m = model()
    x, y = batch()
    loss = m.training_step((x, y), 0)
    encoded, mask, q = m.encode(x.input_ids, x.attention_mask)
    tokens = y.target_ids + m.offsets
    out = m.transformer(encoder_outputs=encoded, attention_mask=mask, labels=tokens)
    ref_sid = F.cross_entropy(out.logits.flatten(0, 1), tokens.flatten())
    projected = m.content_projection(m.content_bank)
    logits = F.normalize(q, dim=-1) @ F.normalize(projected, dim=-1).T / 0.07
    ref_dense = F.cross_entropy(logits.masked_fill(~m.seen_mask[None], -100), torch.tensor([2, 3]))
    torch.testing.assert_close(loss["sid_loss"], ref_sid)
    torch.testing.assert_close(loss["content_loss"], ref_dense)
    torch.testing.assert_close(loss["loss"], ref_sid + ref_dense)
    assert m.transformer.shared.weight is m.transformer.lm_head.weight
    for name in ["sid_loss", "content_loss"]:
        m.zero_grad()
        m.training_step((x, y), 0)[name].backward()
        assert m.content_projection.output.weight.grad.abs().sum() > 0
    assert m.content_bank.grad is None and "content_bank" not in dict(m.named_parameters())


def test_projection_matches_upstream_conv_residual_batch_one():
    m = model().content_projection.eval()
    x = torch.randn(1, 5)
    ref = m.dropout(x)
    for block, residual in zip(m.blocks, m.residuals, strict=True):
        conv = nn.Conv1d(1, residual.out_features, residual.in_features, bias=False)
        conv.weight.data.copy_(residual.weight[:, None, :])
        ref = block(ref) + conv(ref[:, None, :]).squeeze(-1)
    torch.testing.assert_close(m(x), m.output(ref))
    assert m(x).shape == (1, 8)


def test_last_valid_query_padding_and_content_effect():
    m = model().eval()
    x, _ = batch()
    encoded, _, q = m.encode(x.input_ids, x.attention_mask)
    torch.testing.assert_close(q[0], encoded.last_hidden_state[0, 3])
    torch.testing.assert_close(q[1], encoded.last_hidden_state[1, 1])
    _, _, q_short = m.encode(x.input_ids[:1, :4], x.attention_mask[:1, :4])
    torch.testing.assert_close(q[:1], q_short, atol=1e-6, rtol=1e-5)
    # padding 的占位值不参与商品 lookup，也不影响有效 query。
    padded = x.input_ids.masked_fill(~x.attention_mask.bool(), 999)
    torch.testing.assert_close(q, m.encode(padded, x.attention_mask)[2])
    with torch.no_grad():
        m.content_bank[1].add_(torch.tensor([2.0, -1.0, 3.0, 0.0, 4.0]))
    assert not torch.allclose(q[0], m.encode(x.input_ids, x.attention_mask)[2][0])


@pytest.mark.parametrize("fault", ["duplicate_sid", "duplicate_key", "bad_token", "nan", "unseen_all"])
def test_catalog_fail_closed(fault):
    c = catalog()
    if fault == "duplicate_sid":
        c["semantic_ids"][1] = c["semantic_ids"][0]
    if fault == "duplicate_key":
        c["keys"][1] = c["keys"][0]
    if fault == "bad_token":
        c["semantic_ids"][0, 0] = 3
    if fault == "nan":
        c["embeddings"][0, 0] = float("nan")
    if fault == "unseen_all":
        c["seen_mask"][:] = False
    with pytest.raises(ValueError):
        model(catalog=c)


def test_invalid_history_and_cold_training_label():
    m = model()
    x, _ = batch()
    with pytest.raises(ValueError, match="cold-start"):
        m.training_step((x, TigerLabelData(torch.tensor([[2, 0], [2, 1]]))), 0)
    with pytest.raises(ValueError, match="absent"):
        m.lookup_rows(torch.tensor([[2, 2]]))
    with pytest.raises(ValueError, match="complete"):
        m.encode(torch.tensor([[0, -1]]), torch.tensor([[1, 0]]))
    with pytest.raises(ValueError, match="nonempty"):
        m.encode(torch.tensor([[-1, -1]]), torch.tensor([[0, 0]]))


def test_checkpoint_roundtrip_identity_and_eval():
    m, restored = model().eval(), model().eval()
    state = copy.deepcopy(m.state_dict())
    ckpt = {"state_dict": state}
    m.on_save_checkpoint(ckpt)
    restored.on_load_checkpoint(ckpt)
    restored.load_state_dict(state)
    torch.testing.assert_close(m.retrieve(batch()[0], "dense")[0], restored.retrieve(batch()[0], "dense")[0])
    for field in ["content_bank", "seen_mask", "semantic_ids", "item_keys"]:
        changed = copy.deepcopy(state)
        changed[field].flatten()[0] = 0 if field == "seen_mask" else 99
        with pytest.raises(ValueError, match="mismatch"):
            restored.load_state_dict(changed)
    with pytest.raises(ValueError, match="identity"):
        restored.on_load_checkpoint({})
    result = restored.validation_step(batch(), 0)
    assert result["generated_ids"].shape == (2, 3, 2) and torch.isfinite(result["loss"])


def test_liger_preprocessing_and_optimizer_update():
    from src.data.components.liger import generate_liger_next_item
    from src.data.components.preprocessing import normalize_sequence

    row = generate_liger_next_item({"sequence_data": torch.tensor([0, 0, 0, 1, 1, 0])}, next_k=2)
    row = normalize_sequence(row, sequence_length=6, sid_hierarchy=2, padding_token=-1)
    assert row["attention_mask"].tolist() == [1, 1, 1, 1, 0, 0]
    x = TigerModelInput(row["input_ids"][None], row["attention_mask"][None])
    y = TigerLabelData(row["target_ids"][None])
    m = model()
    before = m.content_projection.output.weight.detach().clone()
    bank = m.content_bank.clone()
    optimizer = torch.optim.AdamW(m.parameters(), lr=0.001)
    loss = m.training_step((x, y), 0)["loss"]
    assert torch.isfinite(loss)
    loss.backward()
    optimizer.step()
    assert not torch.equal(before, m.content_projection.output.weight)
    torch.testing.assert_close(bank, m.content_bank)
