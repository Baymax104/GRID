import copy
from pathlib import Path

import pytest
import torch
from hydra import compose, initialize_config_dir
from test_liger import batch
from test_liger_joint import joint

from src.common.writers.auxiliary_tensor_writer import AuxiliaryTensorWriter
from src.data.components.liger_path_trace import validate_liger_path_trace
from src.recommendation.liger.candidate_guidance import ProbabilityMixtureProcessor
from src.recommendation.liger.path_trace import PathTraceProcessor


@pytest.mark.parametrize("control", ["learned_mass"])
@pytest.mark.parametrize("beams", [1, 3])
def test_paths_preserve_predictions_and_checkpoint(control, beams, tmp_path):
    torch.manual_seed(17)
    base = joint(candidate_trace=True, mechanism_control=control, generation_candidates=beams).eval()
    traced = joint(candidate_trace=True, path_trace=True, mechanism_control=control, generation_candidates=beams).eval()
    traced.load_state_dict(base.state_dict(), strict=True)
    with torch.no_grad():
        expected = base.predict_step(batch())
        actual = traced.predict_step(batch())
    assert torch.equal(expected.predictions, actual.predictions)
    assert torch.equal(
        expected.auxiliary["liger_candidates"]["trace"]["generated_rows"],
        actual.auxiliary["liger_candidates"]["trace"]["generated_rows"],
    )
    payload = dict(actual.auxiliary["liger_paths"], keys=actual.keys)
    validate_liger_path_trace(payload)
    assert torch.equal(
        payload["trace"]["target_prefix_survived"][:, -1],
        actual.auxiliary["liger_candidates"]["trace"]["target_generated"],
    )
    x, y = batch()
    y.target_ids = torch.tensor([[0, 0], [2, 1]])
    with torch.no_grad():
        changed = traced.predict_step((x, y))
    assert torch.equal(actual.predictions, changed.predictions)
    writer = AuxiliaryTensorWriter(str(tmp_path), "liger_paths", "liger_paths.pt", validator=validate_liger_path_trace)
    writer.global_rank = 0
    writer.buffer = [actual]
    writer.flush_buffer()
    path, _ = writer._merge_files()
    restored = torch.load(path, weights_only=False)
    validate_liger_path_trace(restored)
    assert restored["metadata"]["mechanism_control"] == control
    broken = copy.deepcopy(restored)
    broken["trace"]["first_failure_depth"].fill_(42)
    with pytest.raises(ValueError, match="observation"):
        validate_liger_path_trace(broken)


def test_path_callback_configuration():
    with initialize_config_dir(config_dir=str(Path(__file__).resolve().parents[2] / "configs"), version_base="1.3"):
        cfg = compose(
            config_name="main",
            overrides=[
                "experiment=liger_inference",
                "model.root.candidate_strategy=probability_mixture",
                "model.root.content_mixture_alpha=0.5",
                "callbacks=liger_path_trace",
                "model.root.candidate_trace=true",
                "model.root.path_trace=true",
            ],
        )
    assert cfg.model.root.path_trace
    assert cfg.callbacks.liger_path_writer.payload_name == "liger_paths"
    assert cfg.callbacks.liger_trace_writer.payload_name == "liger_candidates"


def test_pruned_frontier_is_retained_and_unreachable_probabilities_are_missing():
    sids = torch.tensor([[0, 0], [0, 1], [1, 0], [1, 1], [2, 0], [2, 1]])
    processor = ProbabilityMixtureProcessor(sids, torch.zeros(2, 6), 3, 0.5)
    observer = PathTraceProcessor(processor, torch.tensor([[1, 0], [2, 1]]))
    scores = torch.zeros(4, 8)
    initial = torch.zeros(4, 1, dtype=torch.long)
    assert torch.equal(observer(initial, scores), processor(initial, scores))
    frontier = torch.tensor([[0, 1], [0, 2], [0, 1], [0, 2]])
    assert torch.equal(observer(frontier, scores), processor(frontier, scores))
    trace = observer.finish(torch.tensor([[0, 0], [0, 1], [0, 0], [0, 1]]))
    assert trace["beam_prefixes"][0, 0, :, 0].tolist() == [0, 1]
    assert trace["target_prefix_survived"].tolist() == [[True, False], [False, False]]
    assert trace["first_failure_depth"].tolist() == [2, 1]
    for name in ["target_generation_log_prob", "target_content_log_prob", "target_mixed_log_prob"]:
        torch.testing.assert_close(trace[name][0], -torch.tensor([3.0, 2.0]).log())
        assert torch.isnan(trace[name][1, 1])
    assert trace["target_branch_rank"].tolist() == [[1, 1], [1, 0]]
    validate_liger_path_trace(
        dict(
            schema_version="liger_paths_v1",
            keys=torch.tensor([1, 2]),
            labels=torch.tensor([[1, 0], [2, 1]]),
            trace=trace,
            metadata=dict(num_hierarchies=2, generation_candidates=2, codebook_size=3),
        )
    )
