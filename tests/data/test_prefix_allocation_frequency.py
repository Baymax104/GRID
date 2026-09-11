from types import SimpleNamespace

import pytest
import torch

from src.data import utils as data_utils


class _Reader:
    rows = []

    def __init__(self, list_of_file_paths, shuffle_rows=False):
        assert list_of_file_paths
        assert shuffle_rows is False

    def iterrows(self):
        yield from self.rows


def _patch_inputs(monkeypatch, tmp_path, *, keys=(30, 10, 20), rows=None):
    training_file = tmp_path / "training" / "part.tfrecord.gz"
    training_file.parent.mkdir()
    training_file.touch()
    bundle = SimpleNamespace(
        keys=torch.tensor(keys),
        predictions=torch.zeros(len(keys), 2, dtype=torch.long),
    )
    monkeypatch.setattr(data_utils, "load_model_output", lambda *args, **kwargs: bundle)
    _Reader.rows = rows if rows is not None else []
    monkeypatch.setattr(data_utils, "TFRecordReader", _Reader)
    return training_file.parent


def test_training_frequencies_follow_semantic_id_key_order_and_keep_zero_items(monkeypatch, tmp_path):
    training_dir = _patch_inputs(
        monkeypatch,
        tmp_path,
        rows=[
            {"sequence_data": torch.tensor([10, 10, 30, 999])},
            {"sequence_data": torch.tensor([10])},
        ],
    )

    frequencies = data_utils.load_training_item_frequency_tensor("semantic.pt", str(training_dir))

    assert torch.equal(frequencies, torch.tensor([1, 3, 0]))


def test_training_frequency_rejects_non_training_source(monkeypatch, tmp_path):
    training_dir = _patch_inputs(monkeypatch, tmp_path, rows=[{"sequence_data": [10]}])

    with pytest.raises(ValueError, match="source_split='training'"):
        data_utils.load_training_item_frequency_tensor(
            "semantic.pt",
            str(training_dir),
            source_split="evaluation",
        )


def test_training_frequency_rejects_duplicate_semantic_id_keys(monkeypatch, tmp_path):
    training_dir = _patch_inputs(monkeypatch, tmp_path, keys=(10, 10), rows=[{"sequence_data": [10]}])

    with pytest.raises(ValueError, match="Duplicate keys"):
        data_utils.load_training_item_frequency_tensor("semantic.pt", str(training_dir))


def test_training_frequency_rejects_empty_training_reader(monkeypatch, tmp_path):
    training_dir = _patch_inputs(monkeypatch, tmp_path, rows=[])

    with pytest.raises(ValueError, match="No training sequence records"):
        data_utils.load_training_item_frequency_tensor("semantic.pt", str(training_dir))


def test_training_frequency_requires_training_files(monkeypatch, tmp_path):
    monkeypatch.setattr(
        data_utils,
        "load_model_output",
        lambda *args, **kwargs: SimpleNamespace(
            keys=torch.tensor([10]),
            predictions=torch.zeros(1, 2, dtype=torch.long),
        ),
    )

    with pytest.raises(FileNotFoundError, match="No training TFRecord files"):
        data_utils.load_training_item_frequency_tensor("semantic.pt", str(tmp_path / "training"))


def test_training_frequency_rejects_negative_item_ids(monkeypatch, tmp_path):
    training_dir = _patch_inputs(monkeypatch, tmp_path, rows=[{"sequence_data": [10, -1]}])

    with pytest.raises(ValueError, match="negative item id"):
        data_utils.load_training_item_frequency_tensor("semantic.pt", str(training_dir))
