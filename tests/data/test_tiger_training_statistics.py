import pytest
import torch

from src.data.components import tiger_training_statistics as statistics
from src.data.components.data_models import ModelOutput


def test_expected_counts_follow_causal_position_and_preserve_rng():
    before = torch.random.get_rng_state()
    counts = statistics.expected_target_counts(torch.tensor([10, 20, 30, 40]), [[10, 20, 30]])
    torch.testing.assert_close(counts, torch.tensor([0.0, 1.0, 2.0, 0.0], dtype=torch.float64))
    assert torch.equal(before, torch.random.get_rng_state())


def test_sampling_is_with_replacement_then_unique_not_uniform_item_counts():
    counts = statistics.expected_target_counts(torch.arange(5), [list(range(5))], max_num_sequences=2)
    q = 1 - (1 - 1 / 10) ** 2
    torch.testing.assert_close(counts, torch.arange(5, dtype=torch.float64) * q)
    assert counts.sum() < 2


@pytest.mark.parametrize(
    "keys,rows,limit",
    [([1, 1], [[1, 1]], 32), ([1], [[2, 1]], 32), ([1], [], 32), ([1], [[1]], 32), ([1], [[1, 1]], 0)],
)
def test_bad_counts_input(keys, rows, limit):
    with pytest.raises(ValueError):
        statistics.expected_target_counts(torch.tensor(keys), rows, limit)


def test_file_statistics_identity_and_training_only(tmp_path, monkeypatch):
    directory = tmp_path / "training"
    directory.mkdir()
    (directory / "x.tfrecord.gz").write_bytes(b"fixture")
    monkeypatch.setattr(
        statistics,
        "load_model_output",
        lambda *a, **k: ModelOutput(torch.tensor([10, 20]), torch.tensor([[0, 0], [0, 1]])),
    )

    class Reader:
        def __init__(self, **kwargs):
            assert kwargs["shuffle_rows"] is False

        def iterrows(self):
            return iter([{"sequence_data": [10, 20]}])

    monkeypatch.setattr(statistics, "TFRecordReader", Reader)
    result = statistics.load_training_statistics("sid.pt", str(directory), 2)
    assert result["metadata"]["expected_targets"] == 1
    assert len(result["metadata"]["files"][0]["sha256"]) == 64
    assert len(result["metadata"]["semantic_id_sha256"]) == 64
    with pytest.raises(ValueError, match="training"):
        statistics.load_training_statistics("sid.pt", str(directory), 2, source_split="testing")
