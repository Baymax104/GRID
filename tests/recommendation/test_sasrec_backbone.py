"""按作者 modules.py 的独立 NumPy 公式验证，测试不依赖 TensorFlow 或网络。"""

import numpy as np
import pytest
import torch

from src.recommendation.sasrec import SASRecBackbone


def numpy_official_forward(model, ids):
    """转写官方分头 concat、mask、residual；仅读取被测模型参数。"""
    parameters = {name: value.detach().numpy() for name, value in model.named_parameters()}

    def normalize(values, prefix):
        variance = np.mean((values - values.mean(-1, keepdims=True)) ** 2, axis=-1, keepdims=True)
        return (values - values.mean(-1, keepdims=True)) / np.sqrt(variance + 1e-8) * parameters[
            prefix + ".weight"
        ] + parameters[prefix + ".bias"]

    def dense(values, prefix):
        return values @ parameters[prefix + ".weight"].T + parameters[prefix + ".bias"]

    table = parameters["item_embedding.weight"].copy()
    table[0] = 0
    mask = (ids != 0)[..., None]
    values = (table[ids] * np.sqrt(model.hidden_size) + parameters["position_embedding.weight"]) * mask
    for index in range(len(model.blocks)):
        prefix = f"blocks.{index}"
        queries = normalize(values, prefix + ".attention_norm")
        q = np.concatenate(np.split(dense(queries, prefix + ".query"), model.blocks[index].num_heads, axis=2))
        k = np.concatenate(np.split(dense(values, prefix + ".key"), model.blocks[index].num_heads, axis=2))
        v = np.concatenate(np.split(dense(values, prefix + ".value"), model.blocks[index].num_heads, axis=2))
        scores = q @ k.transpose(0, 2, 1) / np.sqrt(k.shape[-1])
        key_masks = np.tile(np.sign(np.abs(values).sum(-1)), (model.blocks[index].num_heads, 1))
        scores = np.where(key_masks[:, None, :] == 0, -(2**32) + 1, scores)
        scores = np.where(np.tril(np.ones(scores.shape[1:], dtype=bool)), scores, -(2**32) + 1)
        weights = np.exp(scores - scores.max(-1, keepdims=True))
        weights /= weights.sum(-1, keepdims=True)
        query_masks = np.tile(np.sign(np.abs(queries).sum(-1)), (model.blocks[index].num_heads, 1))
        weights *= query_masks[..., None]
        attention = np.concatenate(np.split(weights @ v, model.blocks[index].num_heads, axis=0), axis=2)
        ff_input = normalize(attention + queries, prefix + ".feedforward_norm")
        values = (
            dense(np.maximum(dense(ff_input, prefix + ".feedforward_in"), 0), prefix + ".feedforward_out") + ff_input
        ) * mask
    return normalize(values, "final_norm")


@pytest.mark.parametrize("num_heads", [1, 2, 4])
def test_forward_matches_numpy_transcription_of_official_source(num_heads):
    torch.manual_seed(42)
    model = SASRecBackbone(8, max_history_items=4, hidden_size=8, num_heads=num_heads, dropout=0).double()
    # 非零 LN/linear bias 覆盖训练后 padding 和 query mask，不只检查初始化状态。
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if name.endswith("bias"):
                parameter.copy_(torch.linspace(-0.1, 0.2, parameter.numel(), dtype=parameter.dtype))
    ids = torch.tensor([[0, 0, 1, 3], [2, 4, 6, 8]])
    expected = numpy_official_forward(model, ids.numpy())
    np.testing.assert_allclose(model.encode(ids).detach().numpy(), expected, atol=1e-10, rtol=1e-10)


def test_causal_attention_ignores_future_actions():
    torch.manual_seed(7)
    model = SASRecBackbone(8, max_history_items=4, hidden_size=8, num_heads=2, dropout=0)
    first = model.encode(torch.tensor([[1, 2, 3, 4]]))
    second = model.encode(torch.tensor([[1, 2, 7, 8]]))
    torch.testing.assert_close(first[:, :2], second[:, :2])
    assert not torch.allclose(first[:, 2:], second[:, 2:])


def test_official_bce_regularizes_both_raw_embedding_tables():
    model = SASRecBackbone(8, max_history_items=3, hidden_size=4, dropout=0, l2_emb=0.2).double()
    positive = torch.tensor([[100.0, -2.0, 0.3]], dtype=torch.float64, requires_grad=True)
    negative = torch.tensor([[100.0, 1.0, -0.8]], dtype=torch.float64, requires_grad=True)
    ids = torch.tensor([[0, 2, 3]])
    expected = (
        -np.log(1 / (1 + np.exp(-positive.detach().numpy()[0, 1:])))
        - np.log(1 - 1 / (1 + np.exp(-negative.detach().numpy()[0, 1:])))
    ).mean() + 0.1 * sum(
        np.square(table.detach().numpy()).sum()
        for table in (model.item_embedding.weight, model.position_embedding.weight)
    )
    loss = model.objective(positive, negative, ids)
    assert loss.item() == pytest.approx(expected, abs=1e-12)
    loss.backward()
    assert positive.grad[0, 0] == 0
    assert negative.grad[0, 0] == 0
    torch.testing.assert_close(model.item_embedding.weight.grad, 0.2 * model.item_embedding.weight)
    torch.testing.assert_close(model.position_embedding.weight.grad, 0.2 * model.position_embedding.weight)


def test_epsilon_preserves_official_saturated_probability_objective():
    model = SASRecBackbone(3, max_history_items=1, hidden_size=4, dropout=0).double()
    actual = model.objective(torch.tensor([[-100.0]]), torch.tensor([[100.0]]), torch.tensor([[1]]))
    assert actual.item() == pytest.approx(-2 * np.log(1e-24), rel=1e-6)


def test_shared_item_scoring_and_gradient_paths():
    torch.manual_seed(42)
    model = SASRecBackbone(8, max_history_items=4, hidden_size=8, num_heads=2, dropout=0)
    inputs = torch.tensor([[0, 1, 2, 3], [4, 5, 6, 7]])
    positive = torch.tensor([[0, 2, 3, 4], [5, 6, 7, 8]])
    negative = torch.tensor([[0, 5, 6, 7], [1, 2, 3, 1]])
    pos_logits, neg_logits = model(inputs, positive, negative)
    features = model.encode(inputs)
    torch.testing.assert_close(pos_logits, (features * model.item_table[positive]).sum(-1))
    query = features[:, -1]
    torch.testing.assert_close(model.score_items(query, torch.arange(1, 9)), query @ model.item_table[1:].T)
    model.objective(pos_logits, neg_logits, positive).backward()
    assert torch.count_nonzero(model.item_embedding.weight.grad[0]) == 0
    assert torch.count_nonzero(model.position_embedding.weight.grad) > 0
    for block in model.blocks:
        for projection in (block.query, block.key, block.value):
            assert torch.isfinite(projection.weight.grad).all()
            assert torch.count_nonzero(projection.weight.grad) > 0


@pytest.mark.parametrize("inputs", [[[0, 0, 0]], [[1, 0, 2]], [[0, 1, 9]]])
def test_invalid_histories_rejected(inputs):
    model = SASRecBackbone(8, max_history_items=3)
    with pytest.raises(ValueError):
        model.encode(torch.tensor(inputs))


def test_no_active_labels_rejected():
    model = SASRecBackbone(8, max_history_items=3)
    with pytest.raises(ValueError, match="positive label"):
        model.objective(torch.zeros(1, 3), torch.zeros(1, 3), torch.zeros(1, 3, dtype=torch.long))


def test_dropout_is_disabled_in_evaluation_and_finite_in_training():
    model = SASRecBackbone(8, max_history_items=4, hidden_size=8, num_heads=2, dropout=0.5)
    inputs = torch.tensor([[0, 0, 1, 2]])
    model.eval()
    torch.testing.assert_close(model.encode(inputs), model.encode(inputs))
    model.train()
    assert torch.isfinite(model.encode(inputs)).all()
