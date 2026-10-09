import numpy as np
import pytest
import torch

from src.data.components.collate import collate_fn_sasrec
from src.data.components.sasrec import ItemCatalog, SASRecPreprocessor, load_sasrec_catalog, sample_sasrec_negatives


def test_sparse_keys_and_real_item_zero_are_reversible():
    catalog = ItemCatalog(torch.tensor([90, 0, 11, 3]))
    raw = torch.tensor([[0, 90], [3, 11]])
    torch.testing.assert_close(catalog.encode(raw), torch.tensor([[1, 4], [2, 3]]))
    torch.testing.assert_close(catalog.decode(catalog.encode(raw)), raw)
    assert catalog.sha256 == ItemCatalog(torch.tensor([3, 90, 11, 0])).sha256
    assert catalog.sha256 != ItemCatalog(torch.tensor([0, 3, 11, 91])).sha256
    with pytest.raises(ValueError, match="absent"):
        catalog.encode(torch.tensor([7]))
    with pytest.raises(ValueError, match="nonpadding"):
        catalog.decode(torch.tensor([0]))


@pytest.mark.parametrize("keys", [[0, 0], [-1, 2], []])
def test_invalid_catalog_keys_rejected(keys):
    with pytest.raises(ValueError):
        ItemCatalog(torch.tensor(keys, dtype=torch.long))


def test_catalog_loader_uses_bundle_keys_only(tmp_path):
    path = tmp_path / "catalog.pt"
    torch.save({"keys": torch.tensor([3, 0, 8]), "predictions": torch.randn(3, 4)}, path)
    catalog = load_sasrec_catalog(str(path))
    torch.testing.assert_close(catalog.keys, torch.tensor([0, 3, 8]))


def test_training_shift_and_left_padding_follow_official_sampler():
    catalog = ItemCatalog(torch.arange(8))
    preprocess = SASRecPreprocessor(catalog, max_history_items=5, training=True)
    row = preprocess({"sequence_data": np.array([0, 1, 2]), "user_id": np.array([0])})
    torch.testing.assert_close(row["input_ids"], torch.tensor([0, 0, 0, 1, 2]))
    torch.testing.assert_close(row["target_ids"], torch.tensor([0, 0, 0, 2, 3]))
    assert (row["negative_ids"][:3] == 0).all()
    assert (row["negative_ids"][3:] >= 4).all()
    assert row["user_id"].item() == 0


def test_truncated_history_and_target_remain_excluded_from_negative_sampling():
    catalog = ItemCatalog(torch.arange(7))
    preprocess = SASRecPreprocessor(catalog, max_history_items=2, training=True)
    row = preprocess({"sequence_data": torch.arange(6)})
    torch.testing.assert_close(row["input_ids"], torch.tensor([4, 5]))
    torch.testing.assert_close(row["target_ids"], torch.tensor([5, 6]))
    torch.testing.assert_close(row["negative_ids"], torch.tensor([7, 7]))


def test_negatives_are_reproducible_uniform_and_handle_dense_exclusion():
    excluded = torch.tensor([1, 2, 3, 5, 6, 8])
    torch.manual_seed(42)
    first = sample_sasrec_negatives(8, excluded, 8000)
    torch.manual_seed(42)
    second = sample_sasrec_negatives(8, excluded, 8000)
    assert torch.equal(first, second)
    assert not torch.isin(first, excluded).any()
    assert 0.47 < first.eq(4).float().mean() < 0.53
    assert first.unique().tolist() == [4, 7]
    assert sample_sasrec_negatives(1000, torch.arange(1, 1000), 10).eq(1000).all()


def test_training_filters_short_rows_and_rejects_no_legal_negatives():
    preprocess = SASRecPreprocessor(ItemCatalog(torch.arange(3)), training=True)
    assert preprocess({"sequence_data": torch.tensor([0])}) is None
    with pytest.raises(ValueError, match="no valid negative"):
        preprocess({"sequence_data": torch.arange(3)})


def test_evaluation_is_target_free_and_does_not_sample_negatives():
    preprocess = SASRecPreprocessor(ItemCatalog(torch.arange(5)), max_history_items=4)
    state = torch.random.get_rng_state().clone()
    rows = [
        preprocess({"sequence_data": torch.tensor([0, 2, 4]), "user_id": torch.tensor([0])}),
        preprocess({"sequence_data": torch.tensor([1, 3]), "user_id": torch.tensor([12])}),
    ]
    assert torch.equal(state, torch.random.get_rng_state())
    inputs, labels = collate_fn_sasrec(rows)
    torch.testing.assert_close(inputs.input_ids, torch.tensor([[0, 0, 1, 3], [0, 0, 0, 2]]))
    torch.testing.assert_close(labels.target_ids, torch.tensor([5, 4]))
    torch.testing.assert_close(inputs.output_keys, torch.tensor([0, 12]))
    assert labels.negative_ids is None


def test_evaluation_never_silently_filters_users():
    preprocess = SASRecPreprocessor(ItemCatalog(torch.arange(3)))
    with pytest.raises(ValueError, match="nonempty history"):
        preprocess({"sequence_data": torch.tensor([0]), "user_id": torch.tensor([0])})
    with pytest.raises(ValueError, match="user_id"):
        preprocess({"sequence_data": torch.tensor([0, 1])})


def test_training_collate_preserves_sequence_supervision():
    preprocess = SASRecPreprocessor(ItemCatalog(torch.arange(6)), max_history_items=3, training=True)
    inputs, labels = collate_fn_sasrec([preprocess({"sequence_data": torch.tensor([0, 1, 2])})])
    assert inputs.output_keys is None
    assert labels.target_ids.shape == labels.negative_ids.shape == inputs.input_ids.shape == (1, 3)
    with pytest.raises(ValueError, match="consistent"):
        collate_fn_sasrec([{"input_ids": torch.zeros(3)}])
