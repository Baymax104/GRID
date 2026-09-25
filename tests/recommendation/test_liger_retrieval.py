import types

import pytest
import torch
from test_liger import batch, model


def test_real_generation_and_keyed_output():
    m = model().eval()
    x, _ = batch()
    out = m.predict_step(x)
    assert torch.equal(out.keys, x.output_keys)
    assert out.predictions.shape == (2, 3, 2)
    for row in out.predictions:
        valid = row[(row >= 0).all(-1)]
        assert valid.unique(dim=0).shape[0] == len(valid)
        m.lookup_rows(valid)


def test_cold_union_dedup_pure_dense_ranking():
    m = model().eval()
    m.generate_candidates = types.MethodType(lambda self, e, mask: torch.tensor([[0, 0, -1], [1, 4, -1]]), m)
    m.dense_logits = types.MethodType(
        lambda self, q: torch.tensor([[3.0, 99.0, 1.0, 1.0, 8.0, 7.0], [0.0, 6.0, 99.0, 1.0, 8.0, 7.0]]), m
    )
    sids, scores = m.retrieve(batch()[0], "hybrid")
    torch.testing.assert_close(m.lookup_rows(sids), torch.tensor([[4, 5, 0], [4, 5, 1]]))
    torch.testing.assert_close(scores, torch.tensor([[8.0, 7.0, 3.0], [8.0, 7.0, 6.0]]))
    # 全目录最高分的 seen 商品不能偷偷从 dense 全目录补入。
    assert not (sids[0] == m.semantic_ids[1]).all(-1).any()


def test_invalid_generation_does_not_map_to_first_item():
    m = model().eval()

    def fake_generate(**kwargs):
        return torch.tensor([[0, 0, 0], [0, 1, 4], [0, 7, 0]]).repeat(2, 1)

    m.transformer.generate = fake_generate
    e, mask, _ = m.encode(batch()[0].input_ids, batch()[0].attention_mask)
    torch.testing.assert_close(m.generate_candidates(e, mask), torch.tensor([[-1, 0, -1], [-1, 0, -1]]))
    output, _ = m.retrieve(batch()[0], "generative")
    assert torch.equal(output[:, 0], m.semantic_ids[0].expand(2, -1))
    assert (output[:, 1:] == -1).all()


def test_early_eos_empty_candidates_and_missing_keys():
    m = model().eval()
    m.transformer.generate = lambda **kw: torch.tensor([[0, 7]]).repeat(6, 1)
    output, _ = m.retrieve(batch()[0], "generative")
    assert (output == -1).all()
    x, _ = batch()
    x.output_keys = None
    with pytest.raises(ValueError, match="output_keys"):
        m.predict_step(x)
