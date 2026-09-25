import pickle
import types

import pytest
import torch
from test_liger import batch, model

from src.common.writers.liger_trace_writer import LigerTraceWriter
from src.data.components.liger_trace import summarize_liger_trace, validate_liger_trace


def fixture_model():
    m = model(candidate_trace=True).eval()
    m.generate_candidates = types.MethodType(lambda self, e, mask: torch.tensor([[0, 0, -1], [3, 4, -1]]), m)
    # 第一目标row2全目录第一但未生成；第二目标row3并列分数按catalog row稳定排序。
    m.dense_logits = types.MethodType(
        lambda self, q: torch.tensor([[3.0, 2.0, 9.0, 1.0, 8.0, 7.0], [9.0, 8.0, 7.0, 7.0, 6.0, 5.0]]), m
    )
    return m


def test_trace_preserves_output_and_measures_missing_target_and_ties():
    m = fixture_model()
    out = m.predict_step(batch())
    m.candidate_trace = False
    plain = m.predict_step(batch())
    torch.testing.assert_close(out.predictions, plain.predictions)
    b = dict(keys=out.keys, **out.auxiliary["liger_candidates"])
    validate_liger_trace(b)
    t = b["trace"]
    assert t["dense_rank"].tolist() == [1, 4]
    assert t["hybrid_rank"].tolist() == [0, 1]
    assert t["candidate_count"].tolist() == [3, 3]
    assert t["generated_unique_count"].tolist() == [1, 2]
    assert t["invalid_generated_count"].tolist() == [1, 1]
    assert summarize_liger_trace(b)["dense_only10_target_missing"] == 1
    t["hybrid_rank"][0] = 1
    with pytest.raises(ValueError, match="Coverage/rank"):
        validate_liger_trace(b)


def test_cold_target_is_covered_without_generation_and_fail_closed():
    m = fixture_model()
    x, y = batch()
    y.target_ids[0] = m.semantic_ids[5]
    out = m.predict_step((x, y))
    t = out.auxiliary["liger_candidates"]["trace"]
    assert t["target_cold"][0] and t["target_covered"][0] and not t["target_generated"][0]
    with pytest.raises(ValueError, match="labels"):
        m.predict_step(x)
    m.prediction_mode = "dense"
    with pytest.raises(ValueError, match="hybrid"):
        m.predict_step((x, y))


def test_shared_writer_roundtrip_and_old_checkpoint_compatibility(tmp_path):
    m = fixture_model()
    m.load_state_dict(model().state_dict(), strict=True)
    out = m.predict_step(batch())
    w = LigerTraceWriter(
        output_dir=str(tmp_path),
        payload_name="liger_candidates",
        output_filename="liger_candidates.pt",
        validator=validate_liger_trace,
    )
    with (tmp_path / "auxiliary_liger_candidates_0_test.pkl").open("wb") as f:
        pickle.dump([out], f)
    path, _ = w._merge_files()
    b = torch.load(path, weights_only=True)
    assert b["keys"].tolist() == [11, 12]
    assert summarize_liger_trace(b)["users"] == 2
