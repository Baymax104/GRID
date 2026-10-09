"""LETTER的独立逐用户评价；由公共MetricCallback负责日志策略。"""

import torch
from torchmetrics import Metric

from src.common.metrics import MetricCallback


class LetterRankingMetric(Metric):
    def __init__(self, top_k=10, ndcg=False, **kwargs):
        super().__init__(**kwargs)
        self.top_k, self.ndcg = top_k, ndcg
        self.add_state("total", default=torch.tensor(0.0, dtype=torch.float64), dist_reduce_fx="sum")
        self.add_state("users", default=torch.tensor(0, dtype=torch.long), dist_reduce_fx="sum")

    def update(self, predictions, targets):
        if predictions.ndim != 2 or targets.shape != (len(predictions),) or predictions.shape[1] < self.top_k:
            raise ValueError("LETTER ranking requires one target and enough candidates per user.")
        hits = predictions[:, : self.top_k].eq(targets[:, None])
        found, rank = hits.max(-1)
        values = found.double()
        if self.ndcg:
            values /= torch.log2(rank.double() + 2)
        self.total += values.sum()
        self.users += len(targets)

    def compute(self):
        return self.total / self.users.clamp_min(1)


def letter_metric_inputs(payload):
    return {"predictions": payload["generated_ids"], "targets": payload["labels"]}


class LetterPredictionMetricsCallback(MetricCallback):
    def on_predict_start(self, trainer, pl_module):
        self.engine.reset("test")

    def on_predict_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        self.engine.update(
            "test",
            {
                "generated_ids": outputs.predictions.to(batch["target"].device),
                "labels": batch["target"],
                "user_count": len(batch["target"]),
            },
        )

    def on_predict_end(self, trainer, pl_module):
        self._log_stage(trainer, pl_module, "test", log_kwargs=self.test_log_kwargs)
        self.engine.reset("test")
