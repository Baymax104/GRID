import pytest
import torch
from torchmetrics import MeanMetric

from src.common.metrics import MetricCallback, MetricEngine


class LoggingModule:
    device = torch.device("cpu")

    def __init__(self):
        self.logged = []

    def log_dict(self, metrics, **kwargs):
        self.logged.append((metrics, kwargs))


class SummaryLogger:
    def __init__(self):
        self.experiment = type("Experiment", (), {"summary": {}})()


class UnsupportedLogger:
    experiment = object()


def test_metric_callback_routes_train_batch_output_and_logs_on_step():
    engine = MetricEngine(
        stages={
            "train": {
                "loss": {
                    "metric": MeanMetric(),
                    "spec": "loss",
                }
            }
        }
    )
    callback = MetricCallback(engine=engine)
    module = LoggingModule()

    callback.setup(trainer=None, pl_module=module, stage="fit")
    callback.on_train_batch_end(
        trainer=None,
        pl_module=module,
        outputs={"loss": torch.tensor(2.0)},
        batch=None,
        batch_idx=0,
    )

    assert torch.equal(module.logged[0][0]["train/loss"], torch.tensor(2.0))
    assert module.logged[0][1]["on_step"]
    assert not module.logged[0][1]["on_epoch"]


def test_metric_callback_resets_train_metrics_after_each_logged_batch():
    engine = MetricEngine(
        stages={
            "train": {
                "loss": {
                    "metric": MeanMetric(),
                    "spec": "loss",
                }
            }
        }
    )
    callback = MetricCallback(engine=engine)
    module = LoggingModule()

    callback.setup(trainer=None, pl_module=module, stage="fit")
    callback.on_train_batch_end(
        trainer=None,
        pl_module=module,
        outputs={"loss": torch.tensor(2.0)},
        batch=None,
        batch_idx=0,
    )
    callback.on_train_batch_end(
        trainer=None,
        pl_module=module,
        outputs={"loss": torch.tensor(4.0)},
        batch=None,
        batch_idx=1,
    )

    assert torch.equal(module.logged[0][0]["train/loss"], torch.tensor(2.0))
    assert torch.equal(module.logged[1][0]["train/loss"], torch.tensor(4.0))


def test_metric_callback_logs_and_resets_validation_epoch_metrics():
    engine = MetricEngine(
        stages={
            "val": {
                "loss": {
                    "metric": MeanMetric(),
                    "spec": "loss",
                }
            }
        }
    )
    callback = MetricCallback(engine=engine)
    module = LoggingModule()

    callback.on_validation_start(trainer=None, pl_module=module)
    callback.on_validation_batch_end(
        trainer=None,
        pl_module=module,
        outputs={"loss": torch.tensor(2.0)},
        batch=None,
        batch_idx=0,
    )
    callback.on_validation_batch_end(
        trainer=None,
        pl_module=module,
        outputs={"loss": torch.tensor(4.0)},
        batch=None,
        batch_idx=1,
    )
    callback.on_validation_epoch_end(trainer=None, pl_module=module)

    assert torch.equal(module.logged[0][0]["val/loss"], torch.tensor(3.0))
    assert not module.logged[0][1]["on_step"]
    assert module.logged[0][1]["on_epoch"]
    engine.update("val", {"loss": torch.tensor(10.0)})
    assert torch.equal(engine.compute("val")["loss"], torch.tensor(10.0))


def test_metric_callback_summary_mode_writes_run_summary_without_log_dict():
    engine = MetricEngine(
        stages={
            "test": {
                "loss": {
                    "metric": MeanMetric(),
                    "spec": "loss",
                }
            }
        }
    )
    callback = MetricCallback(engine=engine, logging_modes={"test": "summary"})
    module = LoggingModule()
    summary_logger = SummaryLogger()
    trainer = type("Trainer", (), {"loggers": [summary_logger]})()

    callback.on_test_start(trainer=trainer, pl_module=module)
    callback.on_test_batch_end(
        trainer=trainer,
        pl_module=module,
        outputs={"loss": torch.tensor(2.0)},
        batch=None,
        batch_idx=0,
    )
    callback.on_test_epoch_end(trainer=trainer, pl_module=module)

    assert module.logged == []
    assert summary_logger.experiment.summary["test/loss"] == 2.0
    engine.update("test", {"loss": torch.tensor(10.0)})
    assert torch.equal(engine.compute("test")["loss"], torch.tensor(10.0))


def test_metric_callback_rejects_unsupported_logging_mode():
    with pytest.raises(ValueError, match="Unsupported metric logging mode"):
        MetricCallback(engine=MetricEngine(), logging_modes={"test": "table"})


def test_metric_callback_summary_mode_warns_for_unsupported_logger(monkeypatch):
    warnings = []
    engine = MetricEngine(
        stages={
            "test": {
                "loss": {
                    "metric": MeanMetric(),
                    "spec": "loss",
                }
            }
        }
    )
    callback = MetricCallback(engine=engine, logging_modes={"test": "summary"})
    module = LoggingModule()
    trainer = type("Trainer", (), {"loggers": [UnsupportedLogger()]})()
    monkeypatch.setattr("src.common.metrics.callback.logger.warning", warnings.append)

    callback.on_test_batch_end(
        trainer=trainer,
        pl_module=module,
        outputs={"loss": torch.tensor(2.0)},
        batch=None,
        batch_idx=0,
    )
    callback.on_test_epoch_end(trainer=trainer, pl_module=module)

    assert module.logged == []
    assert any("unsupported logger" in warning for warning in warnings)


def test_metric_callback_summary_mode_skips_non_scalar_metrics(monkeypatch):
    warnings = []
    engine = MetricEngine(stages={})
    callback = MetricCallback(engine=engine, logging_modes={"test": "summary"})
    module = LoggingModule()
    summary_logger = SummaryLogger()
    trainer = type("Trainer", (), {"loggers": [summary_logger]})()
    monkeypatch.setattr("src.common.metrics.callback.logger.warning", warnings.append)
    monkeypatch.setattr(
        engine,
        "compute_prefixed",
        lambda stage, only_updated=False: {"test/vector": torch.tensor([1.0, 2.0])},
    )

    callback.on_test_epoch_end(trainer=trainer, pl_module=module)

    assert summary_logger.experiment.summary == {}
    assert any("Skipping non-scalar metric" in warning for warning in warnings)


def test_metric_callback_ignores_unconfigured_stage():
    engine = MetricEngine(stages={})
    callback = MetricCallback(engine=engine)
    module = LoggingModule()

    callback.on_test_batch_end(
        trainer=None,
        pl_module=module,
        outputs={"loss": torch.tensor(1.0)},
        batch=None,
        batch_idx=0,
    )
    callback.on_test_epoch_end(trainer=None, pl_module=module)

    assert module.logged == []
