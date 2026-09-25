import numpy as np
import pytest
import torch

from src.data.components import liger


def test_loader_training_only_key_alignment(monkeypatch, tmp_path):
    catalog = dict(
        keys=torch.tensor([4, 7, 9]), semantic_ids=torch.tensor([[0, 0], [0, 1], [1, 0]]), embeddings=torch.ones(3, 5)
    )
    monkeypatch.setattr(liger, "load_catalog_content", lambda *a, **k: dict(catalog))
    training = tmp_path / "training"
    training.mkdir()
    (training / "part.tfrecord.gz").touch()
    seen_files = []

    class Reader:
        def __init__(self, files, shuffle_rows):
            seen_files.extend(files)
            assert not shuffle_rows

        def iterrows(self):
            return iter([{"sequence_data": np.array([7, 4, 7])}])

    monkeypatch.setattr(liger, "TFRecordReader", Reader)
    result = liger.load_liger_catalog("sid", "emb", str(training))
    assert result["seen_mask"].tolist() == [True, True, False]
    assert seen_files == [str(training / "part.tfrecord.gz")]
    catalog["keys"] = torch.tensor([4, 8, 9])
    with pytest.raises(ValueError, match="missing"):
        liger.load_liger_catalog("sid", "emb", str(training))
    with pytest.raises(ValueError, match="nonempty"):
        liger.load_liger_catalog("sid", "emb", str(tmp_path / "absent"))
