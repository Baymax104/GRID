from __future__ import annotations

import pytest
import torch

from src.data.components.data_models import ModelOutput
from src.data.components.prefix_trace import (
    PREFIX_TRACE_SCHEMA_VERSION,
    load_prefix_trace,
    validate_prefix_trace_bundle,
)


def make_bundle(num_rows: int = 2, num_hierarchies: int = 3):
    layer_shape = (num_rows, num_hierarchies)
    return {
        "schema_version": PREFIX_TRACE_SCHEMA_VERSION,
        "keys": torch.arange(num_rows),
        "labels": torch.zeros(layer_shape, dtype=torch.long),
        "trace": {
            "teacher_target_probability": torch.full(layer_shape, 0.5),
            "teacher_legal_rank": torch.ones(layer_shape, dtype=torch.long),
            "teacher_target_vs_best_legal_margin": torch.zeros(layer_shape),
            "target_prefix_survived": torch.ones(layer_shape, dtype=torch.bool),
            "target_beam_rank": torch.ones(layer_shape, dtype=torch.long),
            "target_parent_beam_rank": torch.ones(layer_shape, dtype=torch.long),
            "target_path_score": torch.full(layer_shape, 0.5),
            "beam_cutoff_score": torch.full(layer_shape, 0.1),
            "cutoff_margin": torch.full(layer_shape, 0.4),
            "legal_candidate_count": torch.full(layer_shape, 2, dtype=torch.long),
            "first_failure_depth": torch.full((num_rows,), -1, dtype=torch.long),
        },
        "metadata": {
            "data_split": "evaluation",
            "beam_width": 10,
            "num_hierarchies": num_hierarchies,
            "codebook_size": 256,
            "trace_mode": "teacher_forcing_and_constrained_beam",
            "checkpoint_reference": "wandb://checkpoint",
            "semantic_id_reference": "wandb://sid",
        },
    }


def test_model_output_old_constructor_has_empty_auxiliary_payload():
    output = ModelOutput(torch.tensor([1]), torch.tensor([[2]]))

    assert output.auxiliary == {}


def test_prefix_trace_schema_accepts_one_based_ranks_and_minus_one_sentinel(tmp_path):
    bundle = make_bundle()
    bundle["trace"]["target_prefix_survived"][0, 1:] = False
    bundle["trace"]["target_beam_rank"][0, 1:] = -1
    bundle["trace"]["target_parent_beam_rank"][0, 2] = -1
    bundle["trace"]["first_failure_depth"][0] = 2
    path = tmp_path / "prefix_trace.pt"
    torch.save(bundle, path)

    validate_prefix_trace_bundle(bundle)
    loaded = load_prefix_trace(str(path))

    assert loaded.trace["first_failure_depth"].tolist() == [2, -1]
    assert loaded.trace["target_prefix_survived"].dtype == torch.bool


def test_prefix_trace_loader_normalizes_legacy_list_keys(tmp_path):
    bundle = make_bundle()
    bundle["keys"] = bundle["keys"].tolist()
    path = tmp_path / "legacy_prefix_trace.pt"
    torch.save(bundle, path)

    loaded = load_prefix_trace(str(path))

    assert loaded.keys.tolist() == [0, 1]


@pytest.mark.parametrize(
    ("field_name", "replacement", "message"),
    [
        ("target_beam_rank", torch.zeros((2, 3), dtype=torch.long), "1-based or -1"),
        ("target_prefix_survived", torch.ones((2, 2), dtype=torch.bool), "must have shape"),
        ("teacher_target_probability", torch.ones((2, 3), dtype=torch.long), "floating dtype"),
    ],
)
def test_prefix_trace_schema_rejects_invalid_tensor_contract(field_name, replacement, message):
    bundle = make_bundle()
    bundle["trace"][field_name] = replacement

    with pytest.raises((TypeError, ValueError), match=message):
        validate_prefix_trace_bundle(bundle)


def test_prefix_trace_schema_rejects_duplicate_keys():
    bundle = make_bundle()
    bundle["keys"] = torch.tensor([7, 7])

    with pytest.raises(ValueError, match="Duplicate keys"):
        validate_prefix_trace_bundle(bundle)
