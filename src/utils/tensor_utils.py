from typing import Any

import torch

from src.utils.file_utils import open_local_or_remote


def _validate_keyed_prediction_bundle(bundle: dict[str, Any]):
    if "keys" not in bundle or "predictions" not in bundle:
        raise ValueError("Keyed prediction bundle must contain 'keys' and 'predictions'.")

    keys = bundle["keys"]
    predictions = bundle["predictions"]
    if not isinstance(keys, torch.Tensor) or not isinstance(predictions, torch.Tensor):
        raise TypeError("Keyed prediction bundle fields 'keys' and 'predictions' must both be torch.Tensor.")

    if keys.ndim != 1:
        raise ValueError(f"Keyed prediction bundle 'keys' must be 1-D, got shape {tuple(keys.shape)}.")

    if predictions.ndim == 0:
        raise ValueError("Keyed prediction bundle 'predictions' must have at least 1 dimension.")

    if keys.size(0) != predictions.size(0):
        raise ValueError(
            f"Keyed prediction bundle first dimension mismatch: len(keys)={keys.size(0)} vs predictions={predictions.size(0)}."
        )


def _build_key_to_index(keys: torch.Tensor) -> dict[int, int]:
    key_to_index: dict[int, int] = {}
    for idx, key in enumerate(keys.tolist()):
        key = int(key)
        if key in key_to_index:
            raise ValueError(f"Duplicate key detected in keyed prediction bundle: {key}")
        key_to_index[key] = idx
    return key_to_index


def create_keyed_prediction_bundle(
    data: list[dict[str, torch.Tensor | list | int | float]],
    index_key: str,
    value_key: str,
) -> dict[str, torch.Tensor]:
    """Convert row-format keyed predictions into a compact keyed prediction bundle."""
    if len(data) == 0:
        raise ValueError("Cannot create keyed prediction bundle from empty data.")

    keys = torch.tensor([int(row[index_key]) for row in data], dtype=torch.long)
    predictions = torch.stack([torch.as_tensor(row[value_key]) for row in data], dim=0)
    bundle = {"keys": keys, "predictions": predictions}
    _validate_keyed_prediction_bundle(bundle)
    _build_key_to_index(keys)
    return bundle


def load_keyed_prediction_bundle(file_path: str) -> dict[str, Any]:
    """Load a keyed prediction bundle from disk and attach a runtime key->row index map."""
    bundle = torch.load(open_local_or_remote(file_path, mode="rb"), weights_only=False)
    if not isinstance(bundle, dict):
        raise TypeError(f"Expected keyed prediction bundle dict at {file_path}, got {type(bundle).__name__}.")

    _validate_keyed_prediction_bundle(bundle)
    bundle["key_to_index"] = _build_key_to_index(bundle["keys"])
    return bundle


def lookup_values_in_keyed_prediction_bundle(
    bundle: dict[str, Any],
    lookup_keys: torch.Tensor,
) -> torch.Tensor:
    """Lookup prediction rows from a keyed prediction bundle by business keys."""
    _validate_keyed_prediction_bundle(bundle)
    key_to_index = bundle.get("key_to_index")
    if key_to_index is None:
        key_to_index = _build_key_to_index(bundle["keys"])
        bundle["key_to_index"] = key_to_index

    lookup_keys = torch.as_tensor(lookup_keys, dtype=torch.long)
    flat_lookup_keys = lookup_keys.reshape(-1).tolist()
    missing_keys = [int(key) for key in flat_lookup_keys if int(key) not in key_to_index]
    if missing_keys:
        preview = missing_keys[:5]
        raise KeyError(f"Missing keys in keyed prediction bundle: {preview} (total missing={len(missing_keys)}).")

    row_indices = torch.tensor([key_to_index[int(key)] for key in flat_lookup_keys], dtype=torch.long)
    gathered = bundle["predictions"][row_indices]
    prediction_shape = bundle["predictions"].shape[1:]
    return gathered.reshape(*lookup_keys.shape, *prediction_shape)


def merge_list_of_keyed_tensors_to_single_tensor(
    data: list[dict[str, torch.Tensor]],
    index_key: str,
    value_key: str,
) -> dict[str, torch.Tensor]:
    """
    Converts a list of dictionaries of keyed predictions into a compact keyed prediction bundle.
    e.g.,
    data = [
        {
            'user_id': 123,
            'semantic_id': torch.tensor([21, 32, 124]),
            other features.....,
        },
        {
            'user_id': 456,
            'semantic_id': torch.tensor([11, 22, 33]),
            other features.....,
        }
    ]
    output:
    {
        "keys": tensor([123, 456]),
        "predictions": tensor([[21, 32, 124], [11, 22, 33]]),
    }

    Args:
        data (list[dict[str, torch.Tensor]]): A list of dictionaries where each dictionary
            contains an index key and a value key.
        index_key (str): The key in the dictionary that contains the business key for each row.
        value_key (str): The key in the dictionary that contains the tensor to be merged.
    """
    return create_keyed_prediction_bundle(data=data, index_key=index_key, value_key=value_key)


def deduplicate_rows_in_tensor(file_path: str | None = None, return_tensor: bool = False) -> None | dict[str, Any]:
    """
    Identifies and de-duplicate repeated rows in a PyTorch tensor.
    Rows that are not duplicated will have a new column with value 0,
    while rows that are duplicated will have a new column indicating the number of duplicates from 1 to N-1
    where N is the number of duplicates for that row.

    Args:
        file_path: Optional; Path to a file containing the tensor data.
        return_tensor: If True, returns the modified tensor; otherwise, saves it to the file.
    Returns:
        If return_tensor is True, returns the modified tensor with a new column indicating
        the number of duplicates for each row. If False, saves the modified tensor to the file.
    """
    if not file_path.endswith(".pt"):
        return None
    bundle = load_keyed_prediction_bundle(file_path)
    data = bundle["predictions"]
    assert len(data.size()) == 2, "Input data must be a 2D PyTorch tensor."

    # Use torch.unique to get unique rows and their inverse indices
    unique_rows, inverse_indices, counts = torch.unique(data, dim=0, return_inverse=True, return_counts=True)

    output_indices = torch.zeros_like(inverse_indices)

    # Find indices where counts > 1 (meaning duplicates exist)
    duplicate_indices = torch.where(counts > 1)[0]

    for i in range(len(duplicate_indices)):
        # Calculate number of collisions
        num_of_collisions = counts[duplicate_indices[i]]

        # Gather the indices where the collisions occur
        indices_to_change = torch.where(inverse_indices == duplicate_indices[i])[0]

        # Create a range based on the number of collision, starting from 1
        range_to_add = torch.arange(1, num_of_collisions + 1)

        # Scatter to those specific indices
        output_indices = output_indices.scatter(0, indices_to_change, range_to_add)

    # Concatenate the duplicate indicator column to the original data
    result = torch.cat((data, output_indices.unsqueeze(1)), dim=1).long()
    bundle["predictions"] = result
    if return_tensor:
        return bundle
    else:
        torch.save({"keys": bundle["keys"], "predictions": result}, file_path)
        return None
