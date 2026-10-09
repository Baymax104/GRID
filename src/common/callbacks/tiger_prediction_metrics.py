"""Log retrieval metrics while writing TIGER prediction outputs."""

from typing import Any

import torch
import torch.distributed as dist
from lightning import LightningModule, Trainer
from lightning.pytorch.callbacks import Callback

from src.data.components.data_models import ModelOutput, TigerLabelData
from src.utils.distributed import is_distributed_initialized


class TigerPredictionMetricsCallback(Callback):
    """Accumulate exact-SID Recall/NDCG metrics during ``Trainer.predict``."""

    def __init__(self, top_ks: list[int] | tuple[int, ...] = (5, 10)) -> None:
        super().__init__()
        if not top_ks or any(top_k < 1 for top_k in top_ks):
            raise ValueError("top_ks must contain positive integers.")
        self.top_ks = tuple(sorted(set(top_ks)))
        self._reset()

    def _reset(self) -> None:
        self.user_count = 0
        self.recall_sums = {top_k: 0.0 for top_k in self.top_ks}
        self.ndcg_sums = {top_k: 0.0 for top_k in self.top_ks}

    def on_predict_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        self._reset()

    def on_predict_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: ModelOutput | None,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        if outputs is None:
            return
        if not isinstance(outputs, ModelOutput):
            raise TypeError("TIGER prediction metrics require ModelOutput predictions.")
        if not isinstance(batch, tuple) or len(batch) != 2 or not isinstance(batch[1], TigerLabelData):
            raise TypeError("TIGER prediction metrics require (TigerModelInput, TigerLabelData) batches.")

        predictions = outputs.predictions
        labels = batch[1].target_ids.to(predictions.device)
        if predictions.ndim != 3 or labels.ndim != 2:
            raise ValueError("TIGER predictions and labels must have shapes [B, C, H] and [B, H].")
        if predictions.shape[0] != labels.shape[0] or predictions.shape[2] != labels.shape[1]:
            raise ValueError("TIGER prediction and label shapes are incompatible.")
        if predictions.shape[1] < max(self.top_ks):
            raise ValueError(
                f"TIGER predictions contain {predictions.shape[1]} candidates, "
                f"but metrics require at least {max(self.top_ks)}."
            )

        matches = torch.all(predictions == labels[:, None, :], dim=-1)
        first_rank = matches.to(torch.int64).argmax(dim=1) + 1
        first_rank = torch.where(matches.any(dim=1), first_rank, 0)
        self.user_count += labels.shape[0]

        for top_k in self.top_ks:
            hits = (first_rank > 0) & (first_rank <= top_k)
            self.recall_sums[top_k] += hits.double().sum().item()
            ndcg = torch.where(
                hits,
                1.0 / torch.log2(first_rank.double().clamp_min(1) + 1.0),
                0.0,
            )
            self.ndcg_sums[top_k] += ndcg.sum().item()

    def on_predict_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        device = pl_module.device
        totals = torch.tensor(
            [
                float(self.user_count),
                *(self.recall_sums[top_k] for top_k in self.top_ks),
                *(self.ndcg_sums[top_k] for top_k in self.top_ks),
            ],
            dtype=torch.float64,
            device=device,
        )
        if is_distributed_initialized():
            dist.all_reduce(totals, op=dist.ReduceOp.SUM)
        if not trainer.is_global_zero:
            return

        user_count = int(totals[0].item())
        if user_count == 0:
            raise ValueError("TIGER prediction metrics received no users.")
        metric_count = len(self.top_ks)
        metrics: dict[str, float | int] = {"test/user_count": user_count}
        for index, top_k in enumerate(self.top_ks):
            metrics[f"test/recall@{top_k}"] = totals[1 + index].item() / user_count
            metrics[f"test/ndcg@{top_k}"] = totals[1 + metric_count + index].item() / user_count
        for logger in trainer.loggers:
            logger.log_metrics(metrics)
