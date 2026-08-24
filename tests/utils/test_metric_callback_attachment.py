from omegaconf import OmegaConf
from torchmetrics import MeanMetric

from src.common.metrics import MetricCallback
from src.utils.launcher import apply_dry_run_overrides, attach_metric_callback, log_hyperparameters


class CapturingLogger:
    def __init__(self):
        self.logged_hyperparams = []

    def log_hyperparams(self, params, *args, **kwargs):
        self.logged_hyperparams.append(params)


def test_attach_metric_callback_appends_callback_when_model_metrics_exist():
    cfg = OmegaConf.create(
        {
            "model": {
                "metrics": {
                    "_target_": "src.common.metrics.MetricEngine",
                    "stages": {
                        "train": {
                            "loss": {
                                "metric": {
                                    "_target_": "torchmetrics.MeanMetric",
                                },
                                "spec": "loss",
                            }
                        }
                    },
                }
            }
        }
    )

    callbacks = attach_metric_callback([], cfg)

    assert len(callbacks) == 1
    assert isinstance(callbacks[0], MetricCallback)
    assert isinstance(callbacks[0].engine.metrics["stage_train"]["0_loss_loss"], MeanMetric)


def test_attach_metric_callback_passes_model_callback_options():
    cfg = OmegaConf.create(
        {
            "model": {
                "metric_callback": {
                    "logging_modes": {
                        "test": "summary",
                    }
                },
                "metrics": {
                    "_target_": "src.common.metrics.MetricEngine",
                    "stages": {
                        "test": {
                            "loss": {
                                "metric": {
                                    "_target_": "torchmetrics.MeanMetric",
                                },
                                "spec": "loss",
                            }
                        }
                    },
                },
            }
        }
    )

    callbacks = attach_metric_callback([], cfg)

    assert callbacks[0].logging_modes == {"test": "summary"}


def test_attach_metric_callback_leaves_callbacks_unchanged_without_model_metrics():
    existing_callback = object()
    cfg = OmegaConf.create({"model": {}})

    callbacks = attach_metric_callback([existing_callback], cfg)

    assert callbacks == [existing_callback]


def test_log_hyperparameters_logs_resolved_hydra_config_to_all_loggers():
    logger_a = CapturingLogger()
    logger_b = CapturingLogger()
    cfg = OmegaConf.create(
        {
            "data_dir": "data/beauty",
            "paths": {
                "data_dir": "${data_dir}",
            },
        }
    )

    log_hyperparameters([logger_a, logger_b], cfg)

    expected = {
        "data_dir": "data/beauty",
        "paths": {
            "data_dir": "data/beauty",
        },
    }
    assert logger_a.logged_hyperparams == [expected]
    assert logger_b.logged_hyperparams == [expected]


def test_log_hyperparameters_ignores_empty_logger_list():
    cfg = OmegaConf.create({"data_dir": "data/beauty"})

    log_hyperparameters([], cfg)


def test_dry_run_disables_wandb_artifact_writers():
    cfg = OmegaConf.create(
        {
            "dry_run": True,
            "run_mode": "inference",
            "callbacks": {
                "wandb_artifact_writer": {
                    "_target_": "src.common.writers.wandb_artifact_writer.WandbArtifactWriter",
                    "output_dir": "wandb_artifact",
                    "artifact_name": "predictions",
                    "artifact_type": "semantic_id",
                    "role": "semantic_id",
                    "task_name": "rqvae_inference",
                },
                "wandb_checkpoint_writer": {
                    "_target_": "src.common.writers.wandb_checkpoint_writer.WandbCheckpointWriter",
                    "artifact_name": "checkpoint",
                    "task_name": "rqvae_train",
                },
            },
            "logger": {},
            "trainer": {"root": {}},
        }
    )

    updated = apply_dry_run_overrides(cfg)

    assert updated.callbacks.wandb_artifact_writer is None
    assert updated.callbacks.wandb_checkpoint_writer is None
