from types import SimpleNamespace

import pytest
import torch

import src.data.components.artifacts as artifacts
import src.utils.wandb as wandb_utils
from src.data.components.artifacts import (
    ResolvedArtifactReference,
    get_resolved_artifact_registry,
    load_model_output,
    resolve_reference,
)
from src.utils.wandb import parse_wandb_uri


def test_parse_short_wandb_uri():
    parsed = parse_wandb_uri("wandb://abc123?role=semantic_id&alias=v3&file=merged_predictions_tensor.pt")

    assert parsed.run_id == "abc123"
    assert parsed.entity is None
    assert parsed.project is None
    assert parsed.role == "semantic_id"
    assert parsed.alias == "v3"
    assert parsed.file == "merged_predictions_tensor.pt"


def test_parse_cross_project_wandb_uri():
    parsed = parse_wandb_uri("wandb://baymaxam/GRID/abc123?role=checkpoint")

    assert parsed.entity == "baymaxam"
    assert parsed.project == "GRID"
    assert parsed.run_id == "abc123"
    assert parsed.role == "checkpoint"


def test_resolve_reference_bypasses_local_path(monkeypatch):
    def fail_if_called(**kwargs):
        raise AssertionError("W&B resolver should not run for local paths.")

    monkeypatch.setattr(artifacts, "resolve_wandb_artifact", fail_if_called)

    assert resolve_reference("local/file.pt", field_name="semantic_id_path") == "local/file.pt"


def test_resolve_reference_records_wandb_artifact(monkeypatch, tmp_path):
    resolved_file = tmp_path / "merged_predictions_tensor.pt"
    resolved_file.write_bytes(b"data")

    def fake_resolve(**kwargs):
        return ResolvedArtifactReference(
            field_name=kwargs["field_name"],
            original_uri=kwargs["uri"].original_uri,
            resolved_path=str(resolved_file),
            producer_run_id=kwargs["uri"].run_id,
            entity=kwargs["entity"],
            project=kwargs["project"],
            artifact_name="semantic-id:v0",
            artifact_version="v0",
            artifact_type="semantic_id",
            artifact_path="baymaxam/GRID/semantic-id:v0",
            role=kwargs["role"],
            file=kwargs["target_file"],
        )

    registry = get_resolved_artifact_registry()
    registry.clear()
    monkeypatch.setattr(artifacts, "resolve_wandb_artifact", fake_resolve)

    resolved = resolve_reference(
        "wandb://abc123",
        field_name="semantic_id_path",
        default_entity="baymaxam",
        default_project="GRID",
    )

    assert resolved == str(resolved_file)
    references = registry.records()
    assert len(references) == 1
    assert references[0].field_name == "semantic_id_path"
    assert references[0].role == "semantic_id"


def test_resolve_short_uri_requires_entity_and_project(monkeypatch):
    monkeypatch.delenv("WANDB_ENTITY", raising=False)
    monkeypatch.delenv("WANDB_PROJECT", raising=False)
    monkeypatch.setattr(artifacts, "active_wandb_attr", lambda _: None)

    with pytest.raises(ValueError, match="short W&B URI"):
        resolve_reference("wandb://abc123", field_name="semantic_id_path")


def test_select_output_artifact_rejects_ambiguous_matches():
    artifact_a = SimpleNamespace(
        name="semantic-id-a:v0",
        type="semantic_id",
        metadata={"role": "semantic_id"},
        aliases=["latest"],
        files=lambda: [SimpleNamespace(name="merged_predictions_tensor.pt")],
    )
    artifact_b = SimpleNamespace(
        name="semantic-id-b:v0",
        type="semantic_id",
        metadata={"role": "semantic_id"},
        aliases=["latest"],
        files=lambda: [SimpleNamespace(name="merged_predictions_tensor.pt")],
    )
    run = SimpleNamespace(logged_artifacts=lambda: [artifact_a, artifact_b])

    with pytest.raises(ValueError, match="Multiple W&B output artifacts"):
        wandb_utils.select_output_artifact(
            run=run,
            role="semantic_id",
            alias=None,
            target_file="merged_predictions_tensor.pt",
        )


def test_load_model_output_from_wandb_resolved_path(monkeypatch, tmp_path):
    bundle_path = tmp_path / "merged_predictions_tensor.pt"
    torch.save(
        {
            "keys": torch.tensor([2, 1]),
            "predictions": torch.tensor([[20], [10]]),
        },
        bundle_path,
    )
    monkeypatch.setattr(artifacts, "resolve_reference", lambda *args, **kwargs: str(bundle_path))

    output = load_model_output("wandb://abc123", field_name="semantic_id_path")

    assert torch.equal(output.keys, torch.tensor([1, 2]))
    assert torch.equal(output.predictions, torch.tensor([[10], [20]]))


class _DownloadTrackingArtifact:
    def __init__(self, local_dir):
        self.local_dir = local_dir
        self.download_calls = []

    def download(self, root: str):
        self.download_calls.append(root)
        return str(self.local_dir)


def test_download_artifact_calls_wandb_when_not_distributed(monkeypatch, tmp_path):
    artifact = _DownloadTrackingArtifact(tmp_path)
    monkeypatch.setattr(wandb_utils, "is_distributed_initialized", lambda: False)

    local_dir = wandb_utils.download_artifact_once_per_distributed_run(artifact, tmp_path)

    assert local_dir == tmp_path
    assert artifact.download_calls == [str(tmp_path)]


def test_download_artifact_calls_wandb_on_rank_zero(monkeypatch, tmp_path):
    artifact = _DownloadTrackingArtifact(tmp_path)
    barrier_calls = []
    monkeypatch.setattr(wandb_utils, "is_distributed_initialized", lambda: True)
    monkeypatch.setattr(wandb_utils, "get_distributed_rank", lambda: 0)
    monkeypatch.setattr(wandb_utils, "distributed_barrier", lambda: barrier_calls.append("barrier"))

    local_dir = wandb_utils.download_artifact_once_per_distributed_run(artifact, tmp_path)

    assert local_dir == tmp_path
    assert artifact.download_calls == [str(tmp_path)]
    assert barrier_calls == ["barrier"]


def test_download_artifact_skips_wandb_on_non_zero_rank(monkeypatch, tmp_path):
    artifact = _DownloadTrackingArtifact(tmp_path)
    barrier_calls = []
    monkeypatch.setattr(wandb_utils, "is_distributed_initialized", lambda: True)
    monkeypatch.setattr(wandb_utils, "get_distributed_rank", lambda: 1)
    monkeypatch.setattr(wandb_utils, "distributed_barrier", lambda: barrier_calls.append("barrier"))

    local_dir = wandb_utils.download_artifact_once_per_distributed_run(artifact, tmp_path)

    assert local_dir == tmp_path
    assert artifact.download_calls == []
    assert barrier_calls == ["barrier"]
