from io import BytesIO

import pytest
import torch
from omegaconf import OmegaConf

from src.quantization.letter.module import LetterTokenizerModule


def test_tokenizer_checkpoint_has_safe_plain_identity():
    data = {"keys": torch.arange(8), "content": torch.randn(8, 2), "cf": torch.randn(8, 32)}
    model = LetterTokenizerModule(data, input_dim=2, codebook_size=4, num_groups=2, hidden_sizes=OmegaConf.create([8]))
    checkpoint = {"state_dict": model.state_dict()}
    model.on_save_checkpoint(checkpoint)
    stream = BytesIO()
    torch.save(checkpoint, stream)
    stream.seek(0)
    restored = torch.load(stream, weights_only=True)
    model.on_load_checkpoint(restored)
    assert isinstance(restored["letter_tokenizer_identity"]["tokenizer"]["hidden_sizes"], list)
    data["cf"][0, 0] += 1
    changed = LetterTokenizerModule(data, input_dim=2, codebook_size=4, num_groups=2, hidden_sizes=[8])
    with pytest.raises(ValueError, match="mismatch"):
        changed.on_load_checkpoint(restored)
