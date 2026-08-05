import torch
from torchmetrics import MeanMetric

from src.common.metrics import MetricEngine


class LoggingModule:
    def __init__(self):
        self.logged = []

    def log_dict(self, metrics, **kwargs):
        self.logged.append((metrics, kwargs))


def test_metric_engine_updates_computes_resets_and_logs_scalar_metric():
    engine = MetricEngine(
        stages={
            "train": {
                "loss": {
                    "metric": MeanMetric(),
                    "input": {"key": "loss"},
                }
            }
        }
    )

    engine.update("train", {"loss": torch.tensor(2.0)})
    engine.update("train", {"loss": torch.tensor(4.0)})

    computed = engine.compute("train")
    assert torch.equal(computed["loss"], torch.tensor(3.0))

    module = LoggingModule()
    logged = engine.log(module, "train", on_step=True, logger=True)

    assert torch.equal(logged["train/loss"], torch.tensor(3.0))
    assert torch.equal(module.logged[0][0]["train/loss"], torch.tensor(3.0))
    assert module.logged[0][1] == {"on_step": True, "logger": True}

    engine.reset("train")
    engine.update("train", {"loss": torch.tensor(10.0)})
    assert torch.equal(engine.compute("train")["loss"], torch.tensor(10.0))


def test_metric_engine_skips_missing_stage():
    engine = MetricEngine(stages={})
    module = LoggingModule()

    engine.update("val", {"loss": torch.tensor(1.0)})
    logged = engine.log(module, "val")

    assert logged == {}
    assert module.logged == []


def test_metric_engine_expands_repeat_metrics_with_indexed_payload_values():
    engine = MetricEngine(
        stages={
            "train": {
                "layer_coverages": {
                    "repeat": {
                        "count": 3,
                        "index_name": "layer_idx",
                        "name_template": "layer_{layer_idx}/frac_layer_coverages",
                        "metric": MeanMetric,
                        "input": {
                            "key": "layer_coverages",
                            "index": "{layer_idx}",
                        },
                    }
                }
            }
        }
    )

    engine.update(
        "train",
        {
            "layer_coverages": [
                torch.tensor(0.25),
                torch.tensor(0.5),
                torch.tensor(0.75),
            ]
        },
    )

    computed = engine.compute("train")

    assert set(computed) == {
        "layer_0/frac_layer_coverages",
        "layer_1/frac_layer_coverages",
        "layer_2/frac_layer_coverages",
    }
    assert torch.equal(computed["layer_0/frac_layer_coverages"], torch.tensor(0.25))
    assert torch.equal(computed["layer_1/frac_layer_coverages"], torch.tensor(0.5))
    assert torch.equal(computed["layer_2/frac_layer_coverages"], torch.tensor(0.75))


def test_metric_engine_passes_kwargs_to_metric():
    class DifferenceMetric(MeanMetric):
        def update(self, preds, target):
            super().update(torch.mean(preds - target))

    engine = MetricEngine(
        stages={
            "val": {
                "difference": {
                    "metric": DifferenceMetric(),
                    "input": {
                        "kwargs": {
                            "preds": {"key": "preds"},
                            "target": {"key": "target"},
                        }
                    },
                }
            }
        }
    )

    engine.update(
        "val",
        {
            "preds": torch.tensor([3.0, 5.0]),
            "target": torch.tensor([1.0, 2.0]),
        },
    )

    assert torch.equal(engine.compute("val")["difference"], torch.tensor(2.5))


def test_metric_engine_moves_metrics_to_module_device():
    engine = MetricEngine(
        stages={
            "train": {
                "loss": {
                    "metric": MeanMetric(),
                    "input": "loss",
                }
            }
        }
    )

    engine.to(torch.device("cpu"))

    assert next(iter(engine.metrics["stage_train"].values())).device == torch.device("cpu")
