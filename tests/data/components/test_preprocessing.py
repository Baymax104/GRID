import torch

from src.data.components.preprocessing import generate_next_k_labels, normalize_sequence


def test_generate_next_k_labels_hides_last_item_and_preserves_output_key():
    user_id = torch.tensor(17)
    row = {
        "sequence_data": torch.arange(1, 13),
        "user_id": user_id,
    }

    labeled = generate_next_k_labels(
        row,
        sequence_field_name="sequence_data",
        next_k=4,
        masking_token=0,
        padding_token=-1,
    )

    assert labeled["input_ids"].tolist() == [1, 2, 3, 4, 5, 6, 7, 8, 0, -1, -1, -1]
    assert labeled["target_ids"].tolist() == [9, 10, 11, 12]
    assert labeled["user_id"] is user_id
    assert "sequence_data" not in labeled


def test_normalize_sequence_trims_sid_input_on_item_boundary():
    input_ids = torch.tensor([*range(20), 0, -1, -1, -1])
    row = {"input_ids": input_ids, "target_ids": torch.tensor([20, 21, 22, 23])}

    normalized = normalize_sequence(
        row,
        sequence_length=10,
        padding_token=-1,
        sid_hierarchy=4,
    )

    assert normalized["input_ids"].tolist() == [16, 17, 18, 19, 0, -1, -1, -1, -1, -1]


def test_normalize_sequence_pads_after_non_divisible_item_aligned_trim():
    row = {"input_ids": torch.arange(18)}

    normalized = normalize_sequence(
        row,
        sequence_length=10,
        padding_token=-1,
        sid_hierarchy=3,
    )

    assert normalized["input_ids"].tolist() == [9, 10, 11, 12, 13, 14, 15, 16, 17, -1]


def test_normalize_sequence_preserves_targets_and_builds_attention_mask():
    target_ids = torch.tensor([20, 21, 22, 23])
    row = {"input_ids": torch.arange(24), "target_ids": target_ids}

    normalized = normalize_sequence(
        row,
        sequence_length=10,
        padding_token=-1,
        sid_hierarchy=4,
    )

    assert normalized["target_ids"] is target_ids
    assert normalized["target_ids"].tolist() == [20, 21, 22, 23]
    assert normalized["attention_mask"].tolist() == [1, 1, 1, 1, 1, 1, 1, 1, 0, 0]


def test_normalize_sequence_keeps_token_level_trim_without_sid_hierarchy():
    row = {"input_ids": torch.arange(12)}

    normalized = normalize_sequence(
        row,
        sequence_length=10,
        padding_token=-1,
    )

    assert normalized["input_ids"].tolist() == [2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
    assert normalized["attention_mask"].tolist() == [1] * 10


def test_normalize_sequence_rejects_invalid_sid_hierarchy():
    row = {"input_ids": torch.arange(12)}

    try:
        normalize_sequence(
            row,
            sequence_length=10,
            padding_token=-1,
            sid_hierarchy=0,
        )
    except ValueError as error:
        assert "sid_hierarchy must be a positive integer" in str(error)
    else:
        raise AssertionError("Expected invalid sid_hierarchy to raise ValueError.")
