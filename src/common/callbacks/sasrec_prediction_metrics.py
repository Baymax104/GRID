"""在统一 prediction 链路复用 MetricEngine 和 logging_modes。"""

import torch

from src.common.metrics.callback import HISTORY_LOGGING_MODE, MetricCallback, _write_summary_metrics
from src.data.components.data_models import ModelOutput, SASRecLabelData


class SASRecPredictionMetricsCallback(MetricCallback):
    def on_predict_start(self, trainer, pl_module) -> None:
        self.engine.to(pl_module.device)
        self.engine.reset("test")

    def on_predict_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0) -> None:
        if (
            not isinstance(outputs, ModelOutput)
            or not isinstance(batch, (tuple, list))
            or len(batch) != 2
            or not isinstance(batch[1], SASRecLabelData)
        ):
            raise TypeError("SASRec prediction metrics require ModelOutput and labeled SASRec batch.")
        labels = batch[1].target_ids
        predictions = outputs.predictions
        if labels.ndim != 1 or predictions.ndim != 2 or len(labels) != len(predictions):
            raise ValueError("SASRec prediction and label shapes disagree.")
        if not torch.equal(outputs.keys, batch[0].output_keys):
            raise ValueError("SASRec prediction user keys disagree with input keys.")
        # 预测已按固定 tie-break 排序，指标使用唯一 rank score 保留最终顺序。
        scores = torch.arange(predictions.shape[1], 0, -1, device=predictions.device).expand_as(predictions).float()
        self.engine.update(
            "test",
            {
                "generated_ids": predictions,
                "scores": scores,
                "labels": pl_module.item_keys[labels.long() - 1],
                "user_count": len(labels),
            },
        )

    def on_predict_end(self, trainer, pl_module) -> None:
        # 所有 rank 参与 metric compute/reduce，仅主 rank 发布。
        metrics = self.engine.compute_prefixed("test")
        if trainer.is_global_zero:
            if self.logging_modes.get("test", HISTORY_LOGGING_MODE) == HISTORY_LOGGING_MODE:
                for metric_logger in trainer.loggers:
                    metric_logger.log_metrics(metrics)
            else:
                _write_summary_metrics(trainer, metrics)
        self.engine.reset("test")
