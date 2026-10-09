"""引用与 SHA 必须共同锁定实际 checkpoint 文件。"""

import hashlib

import pytest
import torch

from src.common.callbacks.checkpoint_identity import CheckpointIdentityCallback
from src.data.components.artifacts import load_audited_checkpoint, load_m3_checkpoint


def test_checkpoint_identity_and_no_checkpoint_analysis(tmp_path):
    path = tmp_path / "own.ckpt"
    torch.save({"state_dict": {"weight": torch.ones(2)}, "global_step": 9}, path)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    loaded = load_audited_checkpoint(str(path), digest)
    assert loaded["global_step"] == 9
    CheckpointIdentityCallback(str(path), digest).setup(None, None, "predict")
    with pytest.raises(ValueError, match="mismatch"):
        load_audited_checkpoint(str(path), "0" * 64)
    with pytest.raises(ValueError, match="audited"):
        load_audited_checkpoint(str(path), "pending")
    assert load_m3_checkpoint(None, None, "hits") is None
