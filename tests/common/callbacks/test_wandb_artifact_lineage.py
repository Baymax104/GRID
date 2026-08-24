from types import SimpleNamespace

import pytest

from src.common.callbacks.wandb_artifact_lineage import WandbArtifactLineageCallback
from src.data.components.artifacts import (
    ResolvedArtifactReference,
    ResolvedArtifactRegistry,
)


class _Run:
    def __init__(self):
        self.used = []

    def use_artifact(self, artifact_path):
        self.used.append(artifact_path)


class WandbLogger:
    def __init__(self, run):
        self._run = run

    @property
    def experiment(self):
        return self._run


def _trainer_with_run(run):
    return SimpleNamespace(logger=WandbLogger(run), loggers=[WandbLogger(run)])


def _trainer_without_run():
    return SimpleNamespace(logger=None, loggers=[])


def _reference(path: str = "baymaxam/GRID/semantic-id:v0") -> ResolvedArtifactReference:
    return ResolvedArtifactReference(
        field_name="semantic_id_path",
        original_uri="wandb://abc123",
        resolved_path="local.pt",
        producer_run_id="abc123",
        entity="baymaxam",
        project="GRID",
        artifact_name="semantic-id:v0",
        artifact_version="v0",
        artifact_type="semantic_id",
        artifact_path=path,
        role="semantic_id",
        file="merged_predictions_tensor.pt",
    )


def test_lineage_callback_records_resolved_artifacts():
    registry = ResolvedArtifactRegistry()
    registry.add(_reference())
    active_run = _Run()
    callback = WandbArtifactLineageCallback(registry=registry)

    callback.on_fit_start(trainer=_trainer_with_run(active_run), pl_module=None)
    callback.on_fit_start(trainer=_trainer_with_run(active_run), pl_module=None)

    assert active_run.used == ["baymaxam/GRID/semantic-id:v0"]


def test_lineage_callback_skips_missing_run_by_default():
    registry = ResolvedArtifactRegistry()
    registry.add(_reference())
    callback = WandbArtifactLineageCallback(registry=registry)

    callback.on_fit_start(trainer=_trainer_without_run(), pl_module=None)


def test_lineage_callback_can_fail_on_missing_run(monkeypatch):
    registry = ResolvedArtifactRegistry()
    registry.add(_reference())
    monkeypatch.setitem(
        __import__("sys").modules,
        "wandb",
        SimpleNamespace(init=lambda **kwargs: (_ for _ in ()).throw(AssertionError("wandb.init must not be called"))),
    )
    callback = WandbArtifactLineageCallback(fail_on_missing_run=True, registry=registry)

    with pytest.raises(RuntimeError, match="requires a configured Lightning WandbLogger"):
        callback.on_fit_start(trainer=_trainer_without_run(), pl_module=None)
