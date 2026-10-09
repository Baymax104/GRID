import math

import numpy as np
import pytest
import torch

from src.data.components.letter_cf import LetterCFCatalog, LetterCFPreprocessor, letter_cf_collate
from src.recommendation.letter.cf_teacher import LetterCFBlock, LetterCFTeacher


def test_cf_block_matches_independent_numpy_formula():
    torch.manual_seed(42)
    block = LetterCFBlock(hidden_size=4, dropout=0).double().eval()
    x = torch.randn(2, 5, 4, dtype=torch.float64)
    valid = torch.tensor([[False, False, True, True, True], [False, True, True, True, True]])
    x[~valid] = 0
    p = {name: value.detach().numpy() for name, value in block.named_parameters()}

    def norm(a, name):
        return (a - a.mean(-1, keepdims=True)) / np.sqrt(a.var(-1, keepdims=True) + 1e-8) * p[name + ".weight"] + p[
            name + ".bias"
        ]

    def dense(a, name):
        return a @ p[name + ".weight"].T + p[name + ".bias"]

    a, mask = x.numpy(), valid.numpy()
    q = norm(a, "attention_norm")
    logits = dense(q, "query") @ dense(a, "key").transpose(0, 2, 1) / math.sqrt(4)
    allowed = np.tril(np.ones((5, 5), dtype=bool)) & mask[:, None, :]
    logits = np.where(allowed, logits, -(2**32) + 1)
    weights = np.exp(logits - logits.max(-1, keepdims=True))
    weights /= weights.sum(-1, keepdims=True)
    weights *= (np.abs(q).sum(-1) != 0)[..., None]
    a = weights @ dense(a, "value") + q
    a = norm(a, "ffn_norm")
    expected = (a + dense(np.maximum(dense(a, "ffn.0"), 0), "ffn.3")) * mask[..., None]
    np.testing.assert_allclose(block(x, valid).detach().numpy(), expected, rtol=1e-10, atol=1e-10)


def test_training_negative_scope_and_raw_zero():
    catalog = LetterCFCatalog(torch.arange(20) * 3)
    pre = LetterCFPreprocessor(catalog, max_history_items=2, training=True)
    row = {"sequence_data": torch.tensor([0, 3, 6, 9, 12])}
    batch = pre(row)
    assert batch["input_ids"].tolist() == [3, 4]
    assert batch["positive_ids"].tolist() == [4, 5]
    for _ in range(30):
        assert all(value not in range(6) for value in pre(row)["negative_ids"].tolist())
    assert catalog.model_ids(torch.tensor([0])).item() == 1
    evaluation = LetterCFPreprocessor(catalog, training=False)(row)
    assert evaluation["target"].item() == 12 and "negative_ids" not in evaluation
    with pytest.raises(ValueError, match="no legal negative"):
        pre({"sequence_data": catalog.keys})


def test_causality_loss_gradients_and_export_identity():
    torch.manual_seed(42)
    catalog = LetterCFCatalog(torch.arange(20) * 3)
    model = LetterCFTeacher(catalog, max_history_items=4, dropout=0).eval()
    ids = torch.tensor([[0, 1, 2, 3]])
    changed = ids.clone()
    changed[0, -1] = 7
    torch.testing.assert_close(model.encode(ids)[:, :-1], model.encode(changed)[:, :-1])
    sample = LetterCFPreprocessor(catalog, max_history_items=4, training=True)(
        {"sequence_data": torch.tensor([0, 3, 6, 9])}
    )
    batch = letter_cf_collate([sample])
    loss = model.training_step(batch, 0)["loss"]
    loss.backward()
    assert loss.isfinite() and model.items.weight.grad[1:].abs().sum() > 0
    assert model.items.weight.grad[0].abs().sum() == 0
    bundle = model.predict_step((torch.arange(20),))
    assert torch.equal(bundle.keys, catalog.keys) and bundle.predictions.shape == (20, 32)
    assert bundle.predictions.isfinite().all() and bundle.predictions.device.type == "cpu"
    checkpoint = {}
    model.on_save_checkpoint(checkpoint)
    model.on_load_checkpoint(checkpoint)
    other = LetterCFTeacher(LetterCFCatalog(catalog.keys + 1), max_history_items=4, dropout=0)
    with pytest.raises(ValueError, match="protocol mismatch"):
        other.on_load_checkpoint(checkpoint)


def test_no_other_project_model_imports():
    from pathlib import Path

    source = Path("src/recommendation/letter/cf_teacher.py").read_text(encoding="utf-8")
    assert all(f"src.recommendation.{name}" not in source for name in ("sasrec", "tiger", "liger"))
