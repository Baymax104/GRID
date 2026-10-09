from types import SimpleNamespace

import pytest
import torch

from src.common.callbacks.tiger_prediction_metrics import TigerPredictionMetricsCallback
from src.data.components.data_models import ModelOutput, TigerLabelData, TigerModelInput


class RecordingLogger:
    def __init__(self) -> None:
        self.metrics = None

    def log_metrics(self, metrics) -> None:
        self.metrics = metrics


def _batch(labels: torch.Tensor):
    size, hierarchies = labels.shape
    return (
        TigerModelInput(
            input_ids=torch.zeros(size, hierarchies, dtype=torch.long),
            attention_mask=torch.ones(size, hierarchies, dtype=torch.long),
            output_keys=torch.arange(size),
        ),
        TigerLabelData(target_ids=labels),
    )


def test_tiger_prediction_metrics_logs_exact_sid_ranking_metrics():
    labels = torch.tensor([[1, 1], [2, 2], [3, 3]])
    predictions = torch.tensor(
        [
            [[1, 1], [9, 9], [8, 8], [7, 7], [6, 6], [5, 5], [4, 4], [0, 0], [2, 1], [3, 1]],
            [[9, 9], [8, 8], [7, 7], [6, 6], [5, 5], [2, 2], [4, 4], [0, 0], [2, 1], [3, 1]],
            [[9, 9], [8, 8], [7, 7], [6, 6], [5, 5], [4, 4], [0, 0], [2, 1], [3, 1], [1, 3]],
        ]
    )
    logger = RecordingLogger()
    trainer = SimpleNamespace(is_global_zero=True, loggers=[logger])
    module = SimpleNamespace(device=torch.device("cpu"))
    callback = TigerPredictionMetricsCallback()

    callback.on_predict_start(trainer, module)
    callback.on_predict_batch_end(
        trainer,
        module,
        ModelOutput(keys=torch.arange(3), predictions=predictions),
        _batch(labels),
        batch_idx=0,
    )
    callback.on_predict_end(trainer, module)

    assert logger.metrics["test/user_count"] == 3
    assert logger.metrics["test/recall@5"] == pytest.approx(1 / 3)
    assert logger.metrics["test/recall@10"] == pytest.approx(2 / 3)
    assert logger.metrics["test/ndcg@5"] == pytest.approx(1 / 3)
    assert logger.metrics["test/ndcg@10"] == pytest.approx((1 + 1 / torch.log2(torch.tensor(7.0)).item()) / 3)


def test_tiger_prediction_metrics_requires_labels():
    callback = TigerPredictionMetricsCallback()
    predictions = torch.zeros(1, 10, 2, dtype=torch.long)
    batch = (
        TigerModelInput(
            input_ids=torch.zeros(1, 2, dtype=torch.long),
            attention_mask=torch.ones(1, 2, dtype=torch.long),
            output_keys=torch.tensor([1]),
        ),
        None,
    )

    with pytest.raises(TypeError, match="TigerLabelData"):
        callback.on_predict_batch_end(
            SimpleNamespace(),
            SimpleNamespace(),
            ModelOutput(keys=torch.tensor([1]), predictions=predictions),
            batch,
            batch_idx=0,
        )
