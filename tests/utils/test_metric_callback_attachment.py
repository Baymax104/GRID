from omegaconf import OmegaConf
from torchmetrics import MeanMetric

from src.common.metrics import MetricCallback
from src.utils.launcher import attach_metric_callback


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


def test_attach_metric_callback_leaves_callbacks_unchanged_without_model_metrics():
    existing_callback = object()
    cfg = OmegaConf.create({"model": {}})

    callbacks = attach_metric_callback([existing_callback], cfg)

    assert callbacks == [existing_callback]
