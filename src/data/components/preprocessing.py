from typing import Any

import numpy as np
import torch

from src.utils.file_utils import load_json
from src.utils.tensor_utils import lookup_values_in_keyed_prediction_bundle
from src.utils.utils import load_tokenize


def convert_bytes_to_string(
    row: dict[str, np.ndarray],
    features_to_apply: list[str] | None = None,
) -> dict[str, np.ndarray]:
    for k in row:
        if is_feature_in_features_to_apply(features_to_apply, k):
            row[k] = row[k].astype(str)
    return row


def is_feature_in_features_to_apply(features_to_apply: list[str] | None, k: str) -> bool:
    if features_to_apply and k not in features_to_apply:
        return False
    return True


def filter_features_to_consider(
    row: dict[str, Any],
    features_to_consider: list[str] | None = None,
    feature_map: dict[str, str] | None = None,
):
    row = map_feature_names(row, feature_map=feature_map)
    features_to_consider_set = set(features_to_consider or [])

    if len(features_to_consider_set) > 0:
        return {k: v for k, v in row.items() if k in features_to_consider_set}
    return row


def convert_to_dense_numpy_array(
    row: dict[str, Any],
    features_to_apply: list[str] | None = None,
) -> dict[str, np.ndarray]:
    """
    Transform a record example to a dictionary of dense numpy arrays.

    The current TFRecord reader already decodes values into Python / numpy values,
    so this function mainly normalizes scalars and lists into at-least-1D numpy arrays
    to preserve compatibility with the downstream preprocessing pipeline.

    Args:
        row: dataset row
        features_to_apply: feature key list for applying function

    Returns:
        row
    """
    for k in row:
        if is_feature_in_features_to_apply(features_to_apply, k):
            value = row[k]
            if isinstance(value, np.ndarray):
                row[k] = np.atleast_1d(value)
            elif isinstance(value, list):
                row[k] = np.asarray(value)
            else:
                row[k] = np.atleast_1d(value)
    return row


def map_feature_names(
    row: dict[str, np.ndarray | torch.Tensor | Any],
    feature_map: dict[str, str] | None = None,
) -> dict[str, np.ndarray | torch.Tensor]:
    """
    Map the feature names to the desired feature names.

    Args:
        row: dataset row
        feature_map: optional rename map

    Returns:
        row
    """
    if feature_map:
        row = {v: row[k] for k, v in feature_map.items() if k in row}
    return row


def convert_fields_to_tensors(
    row: dict[str, np.ndarray],
    field_type_map: dict[str, torch.dtype] | None = None,
    features_to_apply: list[str] | None = None,
) -> dict[str, torch.Tensor]:
    """
    Given a row, convert all fields to torch tensors.
    Uses the field type map to determine the dtype, defaulting to torch.long if no dtype is specified.

    Args:
        row: dataset row
        field_type_map: field to dtype mapping
        features_to_apply: feature key list for applying function

    Returns:
        row
    """
    field_type_map = field_type_map or {}
    tensor_row = {}
    for k, v in row.items():
        if is_feature_in_features_to_apply(features_to_apply, k):
            if isinstance(v, int) or isinstance(v, float):
                v = [int(v)]
            # field_type_map value is like {"id": torch.int32, "text": torch.bytes}
            tensor_row[k] = torch.tensor(v, dtype=field_type_map.get(k, torch.long))
    return tensor_row


def filter_sequence_length_row(
    row: dict[str, torch.Tensor],
    min_sequence_length: int,
) -> dict[str, torch.Tensor] | None:
    """
    This filters out rows that have fields with sequence length smaller than the min threshold.
    Only works for a row right now.

    Args:
        row: dataset row, aka. one sample
        min_sequence_length:

    Returns:
        row or None
    """
    for _, tensor in row.items():
        if len(tensor) < min_sequence_length:
            return None
    return row


def filter_empty_feature(
    row: dict[str, torch.Tensor],
    features_to_apply: list[str] | None = None,
) -> dict | None:
    """
    This filters out rows that have fields with empty tensors.
    Only works for a row right now.

    Args:
        row: dataset row, aka. one sample
        features_to_apply: feature key list for applying function

    Returns:
        row or None
    """
    for k, v in row.items():
        if is_feature_in_features_to_apply(features_to_apply, k):
            if len(v) == 0:
                return None
    return row


def map_sparse_id_to_semantic_id(
    row: dict[str, torch.Tensor],
    semantic_id_bundle: dict[str, Any] | None = None,
    features_to_apply: list[str] | None = None,
    num_hierarchies: int | None = None,
) -> dict[str, torch.Tensor]:
    """
    Given a row of data, maps the sparse ids to semantic ids based on the semantic_id_bundle.

    Args:
        row: dataset row
        semantic_id_bundle: the keyed prediction bundle containing semantic ids
        features_to_apply: feature key list for applying function
        num_hierarchies: semantic id digits

    Returns:
        row
    """
    if semantic_id_bundle is None:
        raise ValueError("Semantic id bundle not provided")

    for k, v in row.items():
        if is_feature_in_features_to_apply(features_to_apply, k):
            # predictions is an N x D tensor generated by residual quantization
            # where N is the number of unique items in the dataset
            # and D is the number of semantic id digits per item
            semantic_ids = lookup_values_in_keyed_prediction_bundle(semantic_id_bundle, v)
            # flatten the semantic id sequence
            if num_hierarchies is None:
                row[k] = semantic_ids.view(-1)
            else:
                assert num_hierarchies <= semantic_ids.size(-1), (
                    "num_hierarchies must be less than or equal to the number of hierarchies in the semantic id map."
                )
                # actually, v is sequence_data in dataset, which is the item id list user interacted with
                # 1. lookup semantic ids for each item id, which is (n_items, num_hierarchies_total)
                # 2. get previous n hierarchies
                # 3. flatten to (n_items x num_hierarchies,)
                row[k] = semantic_ids[..., :num_hierarchies].reshape(-1)
    return row


def trim_sequence_row(
    row: dict[str, Any],
    sequence_length: int,
    should_trim_left: bool,
    features_to_apply: list[str] | None = None,
) -> dict[str, Any]:
    """
    Trim the sequences in the row to the sequence_length.

    This function handles only rows (not batches) and assumes that the sequences are not
    padded in the first dimension (the dimension to truncate).

    Args:
        row (dict[str, Any]): A dictionary representing a row of data where each key
            is a feature name and, if the feature is being trimmed, the corresponding
            value is a sequential object to be truncated. The value will be trimmed on
            the side determined by should_trim_left to the specified sequence_length in
            the first dimension.
        sequence_length (int): The desired length to trim the sequences to.
        should_trim_left (bool): If True, trim the left side of the sequence.
            If False, trim the right side of the sequence.
        features_to_apply (list[str] | None): A list of feature names to apply the
            trimming to. If empty, all features in the row will be trimmed.
    Returns:
        dict[str, Any]:
            A dictionary identical to the input row, but with the sequences trimmed to
            the specified sequence_length on the specified side. Sequences that are
            shorter than sequence_length will remain unchanged.
    """
    if should_trim_left:
        for k, v in row.items():
            if is_feature_in_features_to_apply(features_to_apply, k):
                v = v[-sequence_length:]
                row[k] = v
    else:
        for k, v in row.items():
            if is_feature_in_features_to_apply(features_to_apply, k):
                v = v[:sequence_length]
                row[k] = v
    return row


def tokenize_text_features(
    row: dict[str, Any],
    tokenizer_config: Any,
    features_to_apply: list[str] | None = None,
) -> dict[str, Any]:
    """
    Tokenize text features. features_to_apply must contain only text features.

    Args:
        row: dataset row
        features_to_apply: feature key list for applying function
        tokenizer_config: tokenizer config

    Returns:
        row containing text input_ids and attention_mask
    """
    if not tokenizer_config:
        raise AttributeError("Tokenizer config not provided")
    tokenize = load_tokenize(config=tokenizer_config)
    row_masks = {}
    for k, v in row.items():
        if is_feature_in_features_to_apply(features_to_apply, k):
            k_mask = "_".join([k, "mask"])
            if isinstance(v, np.ndarray) or isinstance(v, list):
                v = str(v[0])
            # one row
            tokenized_seq = tokenize(v)
            row[k] = tokenized_seq["input_ids"].flatten()
            row_masks[k_mask] = tokenized_seq["attention_mask"].flatten()
    row.update(row_masks)
    return row


def preprocess_categorical_feature_to_idx(
    row: dict[str, Any],
    features_to_apply: list[str] | None = None,
    mapping_file: str | None = "",
) -> dict[str, Any]:
    # Translate categorical features to indices by looking at the mapping provided.
    # features_to_apply must contain name of the categorical features whose mapping is available in the mapping_file.
    # This operates on a single row.

    # Load the mapping if a mapping file is provided
    if mapping_file:
        category_to_idx = load_json(mapping_file)
    else:
        raise ValueError("A valid path to the mapping file must be provided.")

    # Helper function to translate feature values to index
    def translate_to_index(value: str | list[str]) -> int | list[int]:
        if isinstance(value, list):
            return [category_to_idx.get(v, 0) for v in value]
        else:
            return category_to_idx.get(value, 0)

    features_to_apply = features_to_apply if features_to_apply else []
    for feature in features_to_apply:
        if feature in row:
            row[feature] = translate_to_index(row[feature])
    return row


def map_sparse_id_to_embedding(
    row: dict[str, Any],
    embedding_bundle: dict[str, Any] | None = None,
    sparse_id_field: str = "id",
    embedding_field_to_add: str = "embedding",
) -> dict[str, Any]:
    # Map sparse id to pre-computed embedding

    # predictions is an N x d tensor
    # where N is the number of unique items in the dataset
    # and d is the dimension of the embedding
    if embedding_bundle is not None:
        row[embedding_field_to_add] = lookup_values_in_keyed_prediction_bundle(
            embedding_bundle, row[sparse_id_field]
        ).squeeze()
    else:
        raise ValueError("Embedding map not found")
    return row


def squeeze_tensor_in_place(
    row: dict[str, Any],
    features_to_apply: list[str] | None = None,
) -> dict[str, Any]:
    # Squeeze row field values in place when they carry unnecessary extra dimensions.
    for k, v in row.items():
        if is_feature_in_features_to_apply(features_to_apply, k):
            if isinstance(v, torch.Tensor):
                if len(v.shape) == 1:
                    continue
                row[k] = v.squeeze_()
            elif isinstance(v, np.ndarray):
                if len(v.shape) == 1:
                    continue
                row[k] = v.squeeze()
            elif isinstance(v, list):
                row[k] = [
                    item.squeeze_() if isinstance(item, torch.Tensor) and len(item.shape) > 1 else item for item in v
                ]
            else:
                raise ValueError(f"Unsupported type for feature {k}: {type(v)}. Expected torch.Tensor or list.")
    return row
