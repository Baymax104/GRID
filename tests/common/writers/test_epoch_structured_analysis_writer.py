"""完整 epoch 产物一次落盘，沿用共享 manifest/bundle 协议。"""

import json
from types import SimpleNamespace
from unittest.mock import Mock

import torch

from src.common.writers.epoch_structured_analysis_writer import EpochStructuredAnalysisWriter
from src.common.writers.structured_analysis import StructuredAnalysisOutput


def test_epoch_writer_defers_output_and_serializes_all_accumulated_business_keys(tmp_path):
    output_dir = tmp_path / "analysis"
    payload = StructuredAnalysisOutput(
        documents={"summary.json": {"n": 2}},
        tables={"users.csv": [{"key": 11}, {"key": 12}]},
        bundles={"view.pt": {"keys": torch.tensor([11, 12]), "predictions": torch.tensor([[1], [2]])}},
        metadata={"n": 2},
    )
    model = SimpleNamespace(structured_analysis_output=Mock(return_value=payload))
    writer = EpochStructuredAnalysisWriter(str(output_dir))
    trainer = SimpleNamespace(global_rank=0)
    for i in range(2):
        writer.on_test_batch_end(trainer, model, {}, None, i)
    assert not output_dir.exists()
    writer.on_test_epoch_end(trainer, model)
    model.structured_analysis_output.assert_called_once()
    assert json.loads((output_dir / "manifest.json").read_text())["complete"]
    stored = torch.load(output_dir / "view.pt", weights_only=False)
    assert stored["keys"].tolist() == [11, 12]
    writer.on_test_epoch_end(SimpleNamespace(global_rank=1), model)
    model.structured_analysis_output.assert_called_once()
