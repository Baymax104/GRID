import inspect
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from lightning.pytorch.callbacks import BasePredictionWriter, Callback, ModelCheckpoint
from omegaconf import OmegaConf

import src.common.writers.local_pickle_writer as local_pickle_writer
import src.common.writers.wandb_artifact_writer as wandb_artifact_writer
from src.common.writers.local_pickle_writer import LocalPickleWriter
from src.common.writers.wandb_artifact_writer import WandbArtifactWriter
from src.common.writers.wandb_checkpoint_writer import WandbCheckpointWriter
from src.data.components.data_models import ModelOutput

PROJECT_ROOT = Path(__file__).resolve().parents[3]
FORBIDDEN_WRITER_RUN_FIELDS = {
    "project",
    "entity",
    "group",
    "run_name",
    "job_type",
    "tags",
    "notes",
    "mode",
    "finish_run",
    "fail_on_error",
}


class _Artifact:
    def __init__(self, name, type, metadata):
        self.name = name
        self.type = type
        self.metadata = metadata
        self.files = []

    def add_file(self, path, name=None):
        self.files.append((path, name))


class _Run:
    def __init__(self):
        self.logged = []
        self.finished = False

    def log_artifact(self, artifact, aliases=None):
        self.logged.append((artifact, aliases))

    def finish(self):
        self.finished = True


class _WandbModule:
    def __init__(self, run=None):
        self.run = run
        self.Artifact = _Artifact
        self.init_calls = []

    def init(self, **kwargs):
        self.init_calls.append(kwargs)
        self.run = _Run()
        return self.run


class WandbLogger:
    def __init__(self, run):
        self._run = run

    @property
    def experiment(self):
        return self._run


def _prediction_output() -> ModelOutput:
    return ModelOutput(
        keys=torch.tensor([10, 20]),
        predictions=torch.tensor([[1, 2], [3, 4]]),
    )


def _run_wandb_writer_prediction(writer: WandbArtifactWriter, run: _Run | None = None):
    loggers = [WandbLogger(run)] if run is not None else []
    trainer = SimpleNamespace(global_rank=0, logger=loggers[0] if loggers else None, loggers=loggers)
    writer.setup(trainer=trainer, pl_module=None, stage="predict")
    writer.on_predict_batch_end(
        trainer=trainer,
        pl_module=None,
        outputs=_prediction_output(),
        batch=None,
        batch_idx=0,
    )
    writer.on_predict_end(trainer=trainer, pl_module=None)


def test_wandb_artifact_writer_is_independent_batch_only_callback(tmp_path):
    writer = WandbArtifactWriter(
        output_dir=str(tmp_path),
        artifact_name="semantic-id",
        artifact_type="semantic_id",
        role="semantic_id",
        task_name="rqvae_inference",
    )

    assert isinstance(writer, Callback)
    assert not isinstance(writer, BasePredictionWriter)
    assert not hasattr(writer, "write_on_batch_end")
    assert not hasattr(writer, "write_on_epoch_end")
    forbidden_params = {"source_path", *FORBIDDEN_WRITER_RUN_FIELDS}
    assert forbidden_params.isdisjoint(inspect.signature(WandbArtifactWriter).parameters)


def test_wandb_checkpoint_writer_exposes_no_run_lifecycle_parameters():
    assert FORBIDDEN_WRITER_RUN_FIELDS.isdisjoint(inspect.signature(WandbCheckpointWriter).parameters)


def test_wandb_artifact_writer_flushes_merges_and_publishes_bundle(monkeypatch, tmp_path):
    monkeypatch.setattr(wandb_artifact_writer, "sync_file", lambda _: None)
    monkeypatch.setattr(wandb_artifact_writer, "distributed_barrier", lambda: None)
    active_run = _Run()
    fake_wandb = _WandbModule(run=None)
    monkeypatch.setitem(__import__("sys").modules, "wandb", fake_wandb)

    writer = WandbArtifactWriter(
        output_dir=str(tmp_path),
        flush_frequency=2,
        artifact_name="semantic-id",
        artifact_type="semantic_id",
        role="semantic_id",
        task_name="rqvae_inference",
        aliases=["v0"],
    )

    trainer = SimpleNamespace(global_rank=0, logger=WandbLogger(active_run), loggers=[WandbLogger(active_run)])
    writer.setup(trainer=trainer, pl_module=None, stage="predict")
    writer.on_predict_batch_end(
        trainer=trainer,
        pl_module=None,
        outputs=_prediction_output(),
        batch=None,
        batch_idx=0,
    )

    assert writer.buffer == []
    assert len(list(tmp_path.glob("wandb_predictions_*.pkl"))) == 1

    writer.on_predict_end(trainer=trainer, pl_module=None)

    bundle_path = tmp_path / "merged_predictions_tensor.pt"
    bundle = torch.load(bundle_path)
    assert torch.equal(bundle["keys"], torch.tensor([10, 20]))
    assert torch.equal(bundle["predictions"], torch.tensor([[1, 2], [3, 4]]))
    assert not list(tmp_path.glob("wandb_predictions_*.pkl"))
    artifact, aliases = active_run.logged[0]
    assert aliases == ["v0"]
    assert artifact.name == "semantic-id"
    assert artifact.type == "semantic_id"
    assert artifact.metadata["role"] == "semantic_id"
    assert artifact.metadata["task_name"] == "rqvae_inference"
    assert artifact.metadata["local_output_path"] == str(bundle_path)
    assert artifact.metadata["bundle_file"] == "merged_predictions_tensor.pt"
    assert artifact.files == [(str(bundle_path), "merged_predictions_tensor.pt")]
    assert fake_wandb.init_calls == []
    assert active_run.finished is False


def test_wandb_artifact_writer_publishes_through_wandb_logger_run(monkeypatch, tmp_path):
    monkeypatch.setattr(wandb_artifact_writer, "sync_file", lambda _: None)
    monkeypatch.setattr(wandb_artifact_writer, "distributed_barrier", lambda: None)
    active_run = _Run()
    fake_wandb = _WandbModule(run=None)
    monkeypatch.setitem(__import__("sys").modules, "wandb", fake_wandb)

    writer = WandbArtifactWriter(
        output_dir=str(tmp_path),
        artifact_name="recommendations",
        artifact_type="recommendation_output",
        role="recommendation_output",
        task_name="tiger_inference",
    )

    _run_wandb_writer_prediction(writer, run=active_run)

    assert len(active_run.logged) == 1
    assert fake_wandb.init_calls == []
    assert active_run.finished is False


def test_wandb_artifact_writer_fails_when_logger_run_is_missing(monkeypatch, tmp_path):
    fake_wandb = _WandbModule(run=None)
    monkeypatch.setitem(__import__("sys").modules, "wandb", fake_wandb)
    source = tmp_path / "merged_predictions_tensor.pt"
    source.write_bytes(b"bundle")
    writer = WandbArtifactWriter(
        output_dir=str(tmp_path),
        artifact_name="semantic-id",
        artifact_type="semantic_id",
        role="semantic_id",
        task_name="rqvae_inference",
    )

    with pytest.raises(RuntimeError, match="requires a configured Lightning WandbLogger"):
        writer._publish_file(
            trainer=SimpleNamespace(logger=None, loggers=[]),
            file_path=str(source),
            metadata={},
        )

    assert fake_wandb.init_calls == []


def test_wandb_artifact_writer_propagates_publish_failure(monkeypatch, tmp_path):
    monkeypatch.setattr(wandb_artifact_writer, "sync_file", lambda _: None)
    monkeypatch.setattr(wandb_artifact_writer, "distributed_barrier", lambda: None)
    active_run = _Run()

    def fail_log_artifact(artifact, aliases=None):
        raise RuntimeError("upload failed")

    active_run.log_artifact = fail_log_artifact
    fake_wandb = SimpleNamespace(
        run=None,
        Artifact=_Artifact,
        init=lambda **kwargs: (_ for _ in ()).throw(AssertionError("wandb.init must not be called")),
    )
    monkeypatch.setitem(__import__("sys").modules, "wandb", fake_wandb)
    source = tmp_path / "merged_predictions_tensor.pt"
    source.write_bytes(b"bundle")
    writer = WandbArtifactWriter(
        output_dir=str(tmp_path),
        artifact_name="semantic-id",
        artifact_type="semantic_id",
        role="semantic_id",
        task_name="rqvae_inference",
    )

    with pytest.raises(RuntimeError, match="upload failed"):
        writer._publish_file(
            trainer=SimpleNamespace(logger=WandbLogger(active_run), loggers=[WandbLogger(active_run)]),
            file_path=str(source),
            metadata={},
        )


def test_wandb_checkpoint_writer_publishes_model_checkpoint_best_path(monkeypatch, tmp_path):
    source = tmp_path / "best.ckpt"
    source.write_bytes(b"checkpoint")
    active_run = _Run()
    fake_wandb = _WandbModule(run=None)
    monkeypatch.setitem(__import__("sys").modules, "wandb", fake_wandb)
    checkpoint = ModelCheckpoint(monitor="val/recall@10", mode="max")
    checkpoint.best_model_path = str(source)
    checkpoint.best_model_score = torch.tensor(0.5)
    trainer = SimpleNamespace(global_rank=0, callbacks=[checkpoint], logger=WandbLogger(active_run), loggers=[WandbLogger(active_run)])
    writer = WandbCheckpointWriter(artifact_name="tiger-checkpoint", task_name="tiger_train", aliases=["best"])

    writer.on_train_end(trainer=trainer, pl_module=None)

    artifact, aliases = active_run.logged[0]
    assert aliases == ["best"]
    assert artifact.type == "checkpoint"
    assert artifact.metadata["role"] == "checkpoint"
    assert artifact.metadata["task_name"] == "tiger_train"
    assert artifact.metadata["best_model_path"] == str(source)
    assert artifact.metadata["monitor"] == "val/recall@10"
    assert artifact.metadata["mode"] == "max"
    assert artifact.metadata["best_model_score"] == pytest.approx(0.5)
    assert fake_wandb.init_calls == []
    assert active_run.finished is False


def test_wandb_checkpoint_writer_fails_when_logger_run_is_missing(monkeypatch, tmp_path):
    source = tmp_path / "best.ckpt"
    source.write_bytes(b"checkpoint")
    fake_wandb = _WandbModule(run=None)
    monkeypatch.setitem(__import__("sys").modules, "wandb", fake_wandb)
    checkpoint = ModelCheckpoint(monitor="train/loss", mode="min")
    checkpoint.best_model_path = str(source)
    trainer = SimpleNamespace(global_rank=0, callbacks=[checkpoint])
    writer = WandbCheckpointWriter(artifact_name="rqvae-checkpoint", task_name="rqvae_train")

    with pytest.raises(RuntimeError, match="requires a configured Lightning WandbLogger"):
        writer.on_train_end(trainer=trainer, pl_module=None)

    assert fake_wandb.init_calls == []


def test_wandb_checkpoint_writer_propagates_publish_failure(monkeypatch, tmp_path):
    source = tmp_path / "best.ckpt"
    source.write_bytes(b"checkpoint")
    active_run = _Run()

    def fail_log_artifact(artifact, aliases=None):
        raise RuntimeError("upload failed")

    active_run.log_artifact = fail_log_artifact
    fake_wandb = _WandbModule(run=None)
    monkeypatch.setitem(__import__("sys").modules, "wandb", fake_wandb)
    checkpoint = ModelCheckpoint()
    checkpoint.best_model_path = str(source)
    trainer = SimpleNamespace(global_rank=0, callbacks=[checkpoint], logger=WandbLogger(active_run), loggers=[WandbLogger(active_run)])
    writer = WandbCheckpointWriter(artifact_name="checkpoint", task_name="rqvae_train")

    with pytest.raises(RuntimeError, match="upload failed"):
        writer.on_train_end(trainer=trainer, pl_module=None)

    assert fake_wandb.init_calls == []


def test_wandb_checkpoint_writer_rejects_ambiguous_model_checkpoints(tmp_path):
    source = tmp_path / "best.ckpt"
    source.write_bytes(b"checkpoint")
    checkpoint_a = ModelCheckpoint()
    checkpoint_b = ModelCheckpoint()
    checkpoint_a.best_model_path = str(source)
    checkpoint_b.best_model_path = str(source)
    trainer = SimpleNamespace(global_rank=0, callbacks=[checkpoint_a, checkpoint_b])
    writer = WandbCheckpointWriter(artifact_name="checkpoint", task_name="rqvae_train")

    with pytest.raises(ValueError, match="Expected exactly one ModelCheckpoint"):
        writer.on_train_end(trainer=trainer, pl_module=None)


def test_wandb_checkpoint_writer_skips_missing_checkpoint_when_configured():
    trainer = SimpleNamespace(global_rank=0, callbacks=[])
    writer = WandbCheckpointWriter(artifact_name="checkpoint", task_name="rqvae_train", missing_checkpoint="skip")

    writer.on_train_end(trainer=trainer, pl_module=None)


def test_wandb_artifact_writer_runs_post_processing_before_publish(monkeypatch, tmp_path):
    monkeypatch.setattr(wandb_artifact_writer, "sync_file", lambda _: None)
    monkeypatch.setattr(wandb_artifact_writer, "distributed_barrier", lambda: None)
    active_run = _Run()
    fake_wandb = _WandbModule(run=None)
    monkeypatch.setitem(__import__("sys").modules, "wandb", fake_wandb)
    processed_paths = []

    def process(file_path: str):
        processed_paths.append(file_path)
        bundle = torch.load(file_path)
        bundle["predictions"] = bundle["predictions"] + 1
        torch.save(bundle, file_path)

    writer = WandbArtifactWriter(
        output_dir=str(tmp_path),
        artifact_name="semantic-id",
        artifact_type="semantic_id",
        role="semantic_id",
        task_name="rqvae_inference",
        post_processing_functions=[process],
    )

    _run_wandb_writer_prediction(writer, run=active_run)

    bundle_path = tmp_path / "merged_predictions_tensor.pt"
    bundle = torch.load(bundle_path)
    assert processed_paths == [str(bundle_path)]
    assert torch.equal(bundle["predictions"], torch.tensor([[2, 3], [4, 5]]))


def test_wandb_artifact_writer_can_coexist_with_local_pickle_writer(monkeypatch, tmp_path):
    monkeypatch.setattr(local_pickle_writer, "sync_file", lambda _: None)
    monkeypatch.setattr(local_pickle_writer, "distributed_barrier", lambda: None)
    monkeypatch.setattr(wandb_artifact_writer, "sync_file", lambda _: None)
    monkeypatch.setattr(wandb_artifact_writer, "distributed_barrier", lambda: None)
    active_run = _Run()
    fake_wandb = _WandbModule(run=None)
    monkeypatch.setitem(__import__("sys").modules, "wandb", fake_wandb)
    trainer = SimpleNamespace(global_rank=0, logger=WandbLogger(active_run), loggers=[WandbLogger(active_run)])
    local_writer = LocalPickleWriter(output_dir=str(tmp_path / "pickle"), flush_frequency=10)
    wandb_writer = WandbArtifactWriter(
        output_dir=str(tmp_path / "wandb_artifact"),
        artifact_name="missing",
        artifact_type="semantic_id",
        role="semantic_id",
        task_name="rqvae_inference",
    )
    output = _prediction_output()

    local_writer.setup(trainer=trainer, pl_module=None, stage="predict")
    wandb_writer.setup(trainer=trainer, pl_module=None, stage="predict")
    local_writer.on_predict_batch_end(trainer=trainer, pl_module=None, outputs=output, batch=None, batch_idx=0)
    wandb_writer.on_predict_batch_end(trainer=trainer, pl_module=None, outputs=output, batch=None, batch_idx=0)
    local_writer.on_predict_end(trainer=trainer, pl_module=None)
    wandb_writer.on_predict_end(trainer=trainer, pl_module=None)

    assert (tmp_path / "pickle" / "merged_predictions_tensor.pt").is_file()
    assert (tmp_path / "wandb_artifact" / "merged_predictions_tensor.pt").is_file()
    assert not list((tmp_path / "pickle").glob("wandb_predictions_*.pkl"))
    assert not list((tmp_path / "wandb_artifact").glob("predictions_*.pkl"))


def test_checkpoint_writer_does_not_import_artifact_writer():
    import src.common.writers.wandb_checkpoint_writer as checkpoint_writer

    source = inspect.getsource(checkpoint_writer)
    assert "wandb_artifact_writer" not in source
    assert "publish_wandb_artifact" not in source


def test_official_wandb_writer_configs_exclude_run_lifecycle_fields():
    callback_names = [
        "rkmeans_inference",
        "rqvae_inference",
        "rvq_inference",
        "sem_embeds_inference",
        "tiger_inference",
        "rkmeans_train",
        "rqvae_train",
        "rvq_train",
        "tiger_train",
    ]

    for name in callback_names:
        cfg = OmegaConf.load(PROJECT_ROOT / "configs" / "callbacks" / f"{name}.yaml")
        for writer_name in ("wandb_artifact_writer", "wandb_checkpoint_writer"):
            writer_cfg = cfg.get(writer_name)
            if writer_cfg:
                assert FORBIDDEN_WRITER_RUN_FIELDS.isdisjoint(writer_cfg.keys()), name


def test_official_wandb_logger_configs_keep_run_identity_fields():
    logger_names = [
        "rkmeans_inference",
        "rqvae_inference",
        "rvq_inference",
        "sem_embeds_inference",
        "tiger_inference",
        "rkmeans_train",
        "rqvae_train",
        "rvq_train",
        "tiger_train",
        "tail_sid_diagnosis",
    ]

    for name in logger_names:
        cfg = OmegaConf.load(PROJECT_ROOT / "configs" / "logger" / f"{name}.yaml")
        wandb_cfg = OmegaConf.to_container(cfg.wandb, resolve=False)
        assert wandb_cfg["_target_"] == "lightning.pytorch.loggers.wandb.WandbLogger"
        assert wandb_cfg["project"] == "${project}"
        assert wandb_cfg["group"] == "${group}"
        assert wandb_cfg["job_type"] in {"train", "inference", "analysis"}
        assert "notes" in wandb_cfg
