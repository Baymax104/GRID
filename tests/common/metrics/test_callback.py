import torch
from torchmetrics import MeanMetric

from src.common.metrics import MetricCallback, MetricEngine


class LoggingModule:
    device = torch.device("cpu")

    def __init__(self):
        self.logged = []

    def log_dict(self, metrics, **kwargs):
        self.logged.append((metrics, kwargs))


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
