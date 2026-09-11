import torch

from src.data.components.data_models import (
    DiagnosisBatch,
    ItemBatch,
    TigerLabelData,
    TigerModelInput,
)
from src.data.utils import combine_list_of_tensor_dicts


def collate_fn_sequence(
    rows: list[dict[str, torch.Tensor]],
    input_field_name: str = "input_ids",
    attention_mask_field_name: str = "attention_mask",
    target_field_name: str | None = "target_ids",
    output_key_field_name: str | None = None,
) -> tuple[TigerModelInput, TigerLabelData | None]:
    """
    Assemble preprocessed TIGER sequence rows for training/evaluation or inference.

    Args:
        rows: Row dictionaries emitted by ``SequenceDataset`` preprocessing.
        input_field_name: Field containing preprocessed encoder input IDs.
        attention_mask_field_name: Field containing preprocessed encoder attention masks.
        target_field_name: Field containing preprocessed target semantic IDs. If ``None``, label data is not
            returned.
        output_key_field_name: Optional field containing output keys for mapping predictions back to source rows.

    Returns:
        model input data, and label data when ``target_field_name`` is configured.
    """

    if not rows:
        raise ValueError("TIGER sequence collate requires at least one row.")

    batch = combine_list_of_tensor_dicts(rows)
    batch: dict[str, torch.Tensor] = {field_name: torch.stack(field_values, dim=0) for field_name, field_values in batch.items()}

    if input_field_name not in batch:
        raise ValueError(f"TIGER sequence collate requires field '{input_field_name}'.")
    if attention_mask_field_name not in batch:
        raise ValueError(f"TIGER sequence collate requires field '{attention_mask_field_name}'.")
    if target_field_name is not None and target_field_name not in batch:
        raise ValueError(f"TIGER sequence collate requires field '{target_field_name}'.")
    if output_key_field_name is not None and output_key_field_name not in batch:
        raise ValueError(f"TIGER sequence collate requires field '{output_key_field_name}'.")

    input_ids = batch[input_field_name]
    attention_mask = batch[attention_mask_field_name]
    output_keys = None
    if output_key_field_name is not None:
        output_keys = batch[output_key_field_name]
        if output_keys.numel() != len(rows):
            raise ValueError(
                f"TIGER sequence collate requires one scalar output key per row in "
                f"'{output_key_field_name}', got shape {tuple(output_keys.shape)} for {len(rows)} rows."
            )
        output_keys = output_keys.reshape(len(rows))

    model_input = TigerModelInput(
        input_ids=input_ids,
        attention_mask=attention_mask,
        output_keys=output_keys
    )
    if target_field_name is None:
        return model_input, None

    target_ids = batch[target_field_name]
    return model_input, TigerLabelData(target_ids=target_ids)


def collate_fn_items(
    rows: list[dict[str, torch.Tensor]],
    item_id_field: str,
    feature_to_input_name: dict[str, str]
) -> ItemBatch:
    """
    The collate function passed to the item dataloader.

    Args:
        rows: The batch of row to be collated.
        item_id_field: The name of the field in the batch that contains the item IDs.
        feature_to_input_name: The mapping from raw feature name to input feature name in ItemData.

    Returns:
        model_input_data: An ItemData object, which stores a batch of item features via a list of item IDs
            in the field `item_ids` and a dictionary mapping feature names to value tensors
            stacked along the batch dimension.
    """

    # In this case, value is a list of tensors, each representing the
    # features of a single item. We stack these tensors along the batch
    # dimension to create a single tensor for the batch of items.
    batch: dict[str, list[torch.Tensor]] = combine_list_of_tensor_dicts(rows)
    batch: dict[str, torch.Tensor] = {k: torch.stack(v, dim=0) for k, v in batch.items()}

    if item_id_field not in batch:
        raise AttributeError(f"Item ID field not found in batch: {item_id_field}")

    item_ids = batch[item_id_field]
    features = {}
    for field_name, field_value in batch.items():
        if field_name == item_id_field:
            continue
        new_name = feature_to_input_name[field_name]
        features[new_name] = field_value

    return ItemBatch(item_ids=item_ids, features=features)


def collate_fn_single_diagnosis_batch(rows: list[DiagnosisBatch]) -> DiagnosisBatch:
    if len(rows) != 1:
        raise ValueError(f"Diagnosis dataloader expects one full analysis batch, got {len(rows)}.")
    return rows[0]
