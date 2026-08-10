import pytest
import torch

from src.data.components.data_models import ModelOutput
from src.data.utils import gather_predictions_by_keys, load_model_output, load_semantic_id_tensor


def test_load_model_output_sorts_predictions_by_key(tmp_path):
    file_path = tmp_path / "merged_predictions_tensor.pt"
    torch.save(
        {
            "keys": torch.tensor([30, 10, 20]),
            "predictions": torch.tensor([[3, 3], [1, 1], [2, 2]]),
        },
        file_path,
    )

    output = load_model_output(str(file_path))

    assert isinstance(output, ModelOutput)
    assert torch.equal(output.keys, torch.tensor([10, 20, 30]))
    assert torch.equal(output.predictions, torch.tensor([[1, 1], [2, 2], [3, 3]]))


def test_load_model_output_rejects_duplicate_keys(tmp_path):
    file_path = tmp_path / "merged_predictions_tensor.pt"
    torch.save(
        {
            "keys": torch.tensor([10, 10]),
            "predictions": torch.tensor([[1], [2]]),
        },
        file_path,
    )

    with pytest.raises(ValueError, match="Duplicate keys"):
        load_model_output(str(file_path))


def test_load_semantic_id_tensor_returns_long_predictions(tmp_path):
    file_path = tmp_path / "merged_predictions_tensor.pt"
    torch.save(
        {
            "keys": torch.tensor([2, 1]),
            "predictions": torch.tensor([[20.0, 21.0], [10.0, 11.0]]),
        },
        file_path,
    )

    semantic_ids = load_semantic_id_tensor(str(file_path))

    assert semantic_ids.dtype == torch.long
    assert torch.equal(semantic_ids, torch.tensor([[10, 11], [20, 21]]))


def test_gather_predictions_by_keys_preserves_lookup_shape():
    bundle = ModelOutput(
        keys=torch.tensor([10, 20, 30]),
        predictions=torch.tensor([[1, 1], [2, 2], [3, 3]]),
    )

    gathered = gather_predictions_by_keys(bundle, torch.tensor([[30, 10], [20, 30]]))

    assert torch.equal(gathered, torch.tensor([[[3, 3], [1, 1]], [[2, 2], [3, 3]]]))


def test_gather_predictions_by_keys_rejects_missing_key():
    bundle = ModelOutput(
        keys=torch.tensor([10, 20, 30]),
        predictions=torch.tensor([[1], [2], [3]]),
    )

    with pytest.raises(KeyError, match="Missing keys"):
        gather_predictions_by_keys(bundle, torch.tensor([40]))
