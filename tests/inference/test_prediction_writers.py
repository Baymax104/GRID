import inspect
from types import SimpleNamespace

import torch
from lightning.pytorch.callbacks import BasePredictionWriter, Callback

import src.inference.prediction_writers as prediction_writers
from src.inference.model_output import ModelOutput
from src.inference.prediction_writers import LocalPickleWriter


def test_local_pickle_writer_is_batch_only_callback(tmp_path):
    writer = LocalPickleWriter(output_dir=str(tmp_path))

    assert isinstance(writer, Callback)
    assert not isinstance(writer, BasePredictionWriter)
    assert not hasattr(writer, "write_on_batch_end")
    assert not hasattr(writer, "write_on_epoch_end")
    assert "write_interval" not in inspect.signature(LocalPickleWriter).parameters


def test_local_pickle_writer_flushes_batch_outputs_and_merges_bundle(tmp_path, monkeypatch):
    monkeypatch.setattr(prediction_writers, "sync_file", lambda _: None)
    monkeypatch.setattr(prediction_writers, "distributed_barrier", lambda: None)
    writer = LocalPickleWriter(output_dir=str(tmp_path), flush_frequency=2)
    trainer = SimpleNamespace(global_rank=0)
    output = ModelOutput(
        keys=torch.tensor([10, 20]),
        predictions=torch.tensor([[1, 2], [3, 4]]),
    )

    writer.setup(trainer=trainer, pl_module=None, stage="predict")
    writer.on_predict_batch_end(
        trainer=trainer,
        pl_module=None,
        outputs=output,
        batch=None,
        batch_idx=0,
    )

    assert writer.buffer == []
    assert len(list(tmp_path.glob("*.pkl"))) == 1

    writer.on_predict_end(trainer=trainer, pl_module=None)

    bundle = torch.load(tmp_path / "merged_predictions_tensor.pt")
    assert torch.equal(bundle["keys"], torch.tensor([10, 20]))
    assert torch.equal(bundle["predictions"], torch.tensor([[1, 2], [3, 4]]))
    assert not list(tmp_path.glob("*.pkl"))
