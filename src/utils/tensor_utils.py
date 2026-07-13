from typing import Any

import torch

from src.common.components.model_output import ModelOutput
from src.utils.file_utils import open_local_or_remote


def _validate_model_output(bundle: dict[str, Any]):
    if "keys" not in bundle or "predictions" not in bundle:
        raise ValueError("Model output must contain 'keys' and 'predictions'.")

    keys = bundle["keys"]
    predictions = bundle["predictions"]
    if not isinstance(keys, torch.Tensor) or not isinstance(predictions, torch.Tensor):
        raise TypeError("Model output fields 'keys' and 'predictions' must both be torch.Tensor.")

    if keys.ndim != 1:
        raise ValueError(f"Model output 'keys' must be 1-D, got shape {tuple(keys.shape)}.")

    if predictions.ndim == 0:
        raise ValueError("Model output 'predictions' must have at least 1 dimension.")

    if keys.size(0) != predictions.size(0):
        raise ValueError(
            f"Model output first dimension mismatch: len(keys)={keys.size(0)} vs predictions={predictions.size(0)}."
        )


def load_model_output(file_path: str) -> ModelOutput:
    """Load a model output from disk, sorted by key for binary search lookup."""
    bundle: dict[str, torch.Tensor] = torch.load(open_local_or_remote(file_path, mode="rb"), weights_only=False)
    if not isinstance(bundle, dict):
        raise TypeError(f"Expected model output dict at {file_path}, got {type(bundle).__name__}.")

    _validate_model_output(bundle)

    keys = bundle["keys"]
    if keys.unique().numel() != keys.numel():
        raise ValueError("Duplicate keys detected in model output.")

    sort_idx = keys.argsort()
    model_output = ModelOutput(keys=keys[sort_idx], predictions=bundle["predictions"][sort_idx])
    return model_output


def gather_predictions_by_keys(
    bundle: ModelOutput,
    keys: torch.Tensor,
) -> torch.Tensor:
    """Gather prediction rows from a model output by business keys via binary search."""

    keys = torch.as_tensor(keys, dtype=torch.long)
    original_shape = keys.shape
    flat_keys = keys.reshape(-1)

    indices = torch.searchsorted(bundle.keys, flat_keys)
    # Clamp to valid range to avoid index-out-of-bounds on non-existent keys
    indices = indices.clamp(max=bundle.keys.numel() - 1)
    found = bundle.keys[indices]
    missing = flat_keys[found != flat_keys]
    if missing.numel() > 0:
        preview = missing[:5].tolist()
        raise KeyError(f"Missing keys in model output: {preview} (total missing={missing.numel()}).")

    gathered = bundle.predictions[indices]
    prediction_shape = bundle.predictions.shape[1:]
    return gathered.reshape(*original_shape, *prediction_shape)
