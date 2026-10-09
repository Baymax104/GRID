import pytest
import torch

from src.common.configs.data import DatasetConfig
from src.data.components.letter import (
    LetterCatalog,
    LetterPreprocessor,
    LetterSequenceDataset,
    letter_collate,
    load_letter_embeddings,
)


def test_prefixes_and_eos():
    catalog = LetterCatalog(torch.tensor([0, 8, 2]), torch.tensor([[0, 0, 0, 0], [1, 1, 1, 1], [2, 2, 2, 2]]), 3, 8)
    rows = LetterPreprocessor(catalog, 1, True)({"sequence_data": [0, 8, 2], "user_id": [4]})
    assert len(rows) == 2
    assert rows[0]["target"] == 8 and rows[1]["target"] == 2
    assert torch.equal(rows[1]["input_ids"][:-1], catalog.tokens[catalog.positions(torch.tensor([8]))].flatten())
    batch = letter_collate(rows)
    assert batch["input_ids"].shape == (2, 5) and (batch["labels"][:, -1] == 1).all()
    eval_rows = LetterPreprocessor(catalog)({"sequence_data": [0, 8, 2], "user_id": [4]})
    assert len(eval_rows) == 1 and len(eval_rows[0]["input_ids"]) == 9
    with pytest.raises(ValueError):
        LetterPreprocessor(catalog)({"sequence_data": [0, 7], "user_id": [4]})


def test_keyed_cf_alignment(tmp_path):
    content, cf = tmp_path / "content.pt", tmp_path / "cf.pt"
    torch.save({"keys": torch.tensor([8, 0]), "predictions": torch.tensor([[8.0], [0.0]])}, content)
    torch.save({"keys": torch.tensor([0, 8]), "predictions": torch.stack([torch.zeros(32), torch.ones(32)])}, cf)
    result = load_letter_embeddings(str(content), str(cf))
    assert result["keys"].tolist() == [0, 8] and result["cf"][1, 0] == 1
    torch.save({"keys": torch.tensor([0, 9]), "predictions": torch.ones(2, 32)}, cf)
    with pytest.raises(ValueError, match="exactly"):
        load_letter_embeddings(str(content), str(cf))
    torch.save({"keys": torch.tensor([0.0, 8.0]), "predictions": torch.ones(2, 32)}, cf)
    with pytest.raises(ValueError, match="integer"):
        load_letter_embeddings(str(content), str(cf))


def test_empty_training_shard_fails_instead_of_spinning():
    class EmptyReader:
        def __init__(self, list_of_file_paths):
            pass

        def iterrows(self):
            return iter([])

    dataset = LetterSequenceDataset(DatasetConfig(EmptyReader), "unused", [], 0, True)
    with pytest.raises(ValueError, match="no usable"):
        next(iter(dataset))
    dataset.is_for_training = False
    assert list(dataset) == []
