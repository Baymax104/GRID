from __future__ import annotations

import sys
from types import SimpleNamespace

import pytest
import torch

from src.common.writers.auxiliary_tensor_writer import AuxiliaryTensorWriter
from src.common.writers.local_pickle_writer import LocalPickleWriter
from src.common.writers.wandb_artifact_writer import WandbArtifactWriter
from src.data.components.data_models import ModelOutput
from src.data.components.prefix_trace import PREFIX_TRACE_SCHEMA_VERSION, load_prefix_trace


def make_output(keys: list[int]) -> ModelOutput:
    num_rows = len(keys)
    shape = (num_rows, 2)
    payload = {
        "schema_version": PREFIX_TRACE_SCHEMA_VERSION,
        "labels": torch.zeros(shape, dtype=torch.long),
        "trace": {
            "teacher_target_probability": torch.full(shape, 0.5),
            "teacher_legal_rank": torch.ones(shape, dtype=torch.long),
            "teacher_target_vs_best_legal_margin": torch.zeros(shape),
            "target_prefix_survived": torch.ones(shape, dtype=torch.bool),
            "target_beam_rank": torch.ones(shape, dtype=torch.long),
            "target_parent_beam_rank": torch.ones(shape, dtype=torch.long),
            "target_path_score": torch.full(shape, 0.5),
            "beam_cutoff_score": torch.full(shape, 0.1),
            "cutoff_margin": torch.full(shape, 0.4),
            "legal_candidate_count": torch.full(shape, 2, dtype=torch.long),
            "first_failure_depth": torch.full((num_rows,), -1, dtype=torch.long),
        },
        "metadata": {
            "data_split": "evaluation",
            "beam_width": 10,
            "num_hierarchies": 2,
            "codebook_size": 256,
            "trace_mode": "teacher_forcing_and_constrained_beam",
            "checkpoint_reference": "wandb://checkpoint",
            "semantic_id_reference": "wandb://sid",
        },
    }
    return ModelOutput(
        keys=torch.tensor(keys),
        predictions=torch.zeros((num_rows, 10, 2), dtype=torch.long),
        auxiliary={"prefix_trace": payload},
    )


def test_auxiliary_writer_merges_keyed_trace_without_changing_standard_bundle(tmp_path):
    trace_dir = tmp_path / "trace"
    writer = AuxiliaryTensorWriter(
        output_dir=str(trace_dir),
        payload_name="prefix_trace",
        output_filename="prefix_trace.pt",
    )
    writer.global_rank = 0
    writer.buffer = [make_output([2]), make_output([1])]
    writer.flush_buffer()
    path, _ = writer._merge_files()

    trace = load_prefix_trace(path)
    assert trace.keys.tolist() == [1, 2]

    prediction_dir = tmp_path / "prediction"
    prediction_writer = LocalPickleWriter(str(prediction_dir))
    prediction_writer.global_rank = 0
    prediction_writer.buffer = [make_output([1])]
    prediction_writer.flush_buffer()
    prediction_writer._merge_files()
    standard_bundle = torch.load(prediction_dir / "merged_predictions_tensor.pt", weights_only=False)
    assert set(standard_bundle) == {"keys", "predictions"}


def test_wandb_prediction_writer_ignores_auxiliary_payload(tmp_path, monkeypatch):
    monkeypatch.setattr("src.common.writers.wandb_artifact_writer.sync_file", lambda _: None)
    writer = WandbArtifactWriter(
        output_dir=str(tmp_path),
        artifact_name="recommendations",
        artifact_type="recommendation_output",
        role="recommendation_output",
        task_name="tiger_prefix_trace",
    )
    writer.global_rank = 0
    writer.buffer = [make_output([1])]
    writer.flush_buffer()

    output_path = writer._merge_files()
    bundle = torch.load(output_path, weights_only=False)

    assert set(bundle) == {"keys", "predictions"}


def test_auxiliary_writer_rejects_duplicate_keys(tmp_path):
    writer = AuxiliaryTensorWriter(str(tmp_path), "prefix_trace", "prefix_trace.pt")
    writer.global_rank = 0
    writer.buffer = [make_output([1]), make_output([1])]
    writer.flush_buffer()

    with pytest.raises(ValueError, match="Duplicate keys"):
        writer._merge_files()


def test_auxiliary_writer_local_mode_does_not_require_wandb(tmp_path):
    writer = AuxiliaryTensorWriter(str(tmp_path), "prefix_trace", "prefix_trace.pt")
    writer.global_rank = 0
    writer.buffer = [make_output([1])]

    writer.on_predict_end(SimpleNamespace(global_rank=0), SimpleNamespace())

    assert (tmp_path / "prefix_trace.pt").is_file()


def test_auxiliary_writer_propagates_publication_failure(tmp_path, monkeypatch):
    writer = AuxiliaryTensorWriter(
        str(tmp_path),
        "prefix_trace",
        "prefix_trace.pt",
        publish_wandb=True,
        artifact_name="trace",
        artifact_type="prefix_trace",
        role="prefix_trace",
    )
    writer.global_rank = 0
    writer.buffer = [make_output([1])]
    monkeypatch.setattr(writer, "_publish_file", lambda *args: (_ for _ in ()).throw(RuntimeError("publish failed")))

    with pytest.raises(RuntimeError, match="publish failed"):
        writer.on_predict_end(SimpleNamespace(global_rank=0), SimpleNamespace())


def test_auxiliary_writer_publishes_prefix_trace_with_logger_owned_run(tmp_path, monkeypatch):
    path = tmp_path / "prefix_trace.pt"
    path.write_bytes(b"trace")
    artifacts = []

    class FakeArtifact:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            self.files = []

        def add_file(self, file_path, name):
            self.files.append((file_path, name))

    run = SimpleNamespace(log_artifact=lambda artifact, aliases: artifacts.append((artifact, aliases)))
    monkeypatch.setitem(sys.modules, "wandb", SimpleNamespace(Artifact=FakeArtifact))
    monkeypatch.setattr(
        "src.common.writers.auxiliary_tensor_writer.require_wandb_logger_run",
        lambda trainer, purpose: run,
    )
    writer = AuxiliaryTensorWriter(
        str(tmp_path),
        "prefix_trace",
        "prefix_trace.pt",
        publish_wandb=True,
        artifact_name="tiger-prefix-trace",
        artifact_type="prefix_trace",
        role="prefix_trace",
    )

    writer._publish_file(
        SimpleNamespace(),
        str(path),
        {
            "schema_version": PREFIX_TRACE_SCHEMA_VERSION,
            "data_split": "evaluation",
            "beam_width": 10,
            "checkpoint_reference": "wandb://checkpoint",
            "semantic_id_reference": "wandb://sid",
        },
    )

    artifact, aliases = artifacts[0]
    assert artifact.kwargs["type"] == "prefix_trace"
    assert artifact.kwargs["metadata"]["role"] == "prefix_trace"
    assert artifact.kwargs["metadata"]["schema_version"] == PREFIX_TRACE_SCHEMA_VERSION
    assert artifact.files[0][1] == "prefix_trace.pt"
    assert aliases == ["latest"]
