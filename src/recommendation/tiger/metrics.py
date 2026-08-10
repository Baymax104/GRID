from collections.abc import Mapping
from typing import Any

import torch
from torchmetrics import Metric


def sid_retrieval_inputs(payload: Mapping[str, Any]) -> dict[str, torch.Tensor]:
    """Convert TIGER generated SID payloads to retrieval metric inputs."""
    marginal_probs = payload["marginal_probs"]
    generated_ids = payload["generated_ids"]
    labels = payload["labels"].to(generated_ids.device)

    batch_size, num_candidates, num_hierarchies = generated_ids.shape
    labels = labels.reshape(batch_size, 1, num_hierarchies)
    preds = marginal_probs.reshape(-1)

    matched_id_coord = torch.all(generated_ids == labels, dim=2).nonzero()
    target = torch.zeros(batch_size, num_candidates, dtype=torch.bool, device=preds.device)
    target[matched_id_coord[:, 0], matched_id_coord[:, 1]] = True
    target = target.reshape(-1)
    indexes = torch.arange(batch_size, device=preds.device).unsqueeze(-1).expand(batch_size, num_candidates).reshape(-1)

    return {
        "preds": preds,
        "target": target,
        "indexes": indexes,
    }


def _reshape_retrieval_inputs(
    preds: torch.Tensor,
    target: torch.Tensor,
    indexes: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    batch_size = indexes.unique().numel()
    if batch_size == 0:
        raise ValueError("Retrieval metric indexes must contain at least one batch index.")
    if preds.numel() % batch_size != 0:
        raise ValueError(
            f"Retrieval metric input length ({preds.numel()}) must be divisible by batch size ({batch_size})."
        )
    return preds.reshape(batch_size, -1), target.reshape(batch_size, -1).int()


class NDCG(Metric):
    """TIGER SID retrieval Normalized Discounted Cumulative Gain@K."""

    def __init__(self, top_k: int, **kwargs: Any):
        super().__init__(**kwargs)
        self.top_k = top_k
        self.add_state("metric_values", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total_values", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, preds: torch.Tensor, target: torch.Tensor, indexes: torch.Tensor, **kwargs: Any) -> None:
        preds, target = _reshape_retrieval_inputs(preds, target, indexes)
        topk_indices = torch.topk(preds, self.top_k)[1]
        topk_true = target.gather(1, topk_indices)

        discounts = torch.log2(torch.arange(2, self.top_k + 2, device=target.device).unsqueeze(0))
        dcg = torch.sum(topk_true / discounts, dim=1)

        ideal_indices = torch.topk(target, self.top_k)[1]
        ideal_dcg = torch.sum(target.gather(1, ideal_indices) / discounts, dim=1)
        ndcg = dcg / torch.where(ideal_dcg == 0, torch.ones_like(ideal_dcg), ideal_dcg)

        self.metric_values += ndcg.sum()
        self.total_values += preds.size(0)

    def compute(self) -> torch.Tensor:
        if self.total_values == 0:
            return torch.tensor(0.0, device=self.metric_values.device)
        return self.metric_values / self.total_values


class Recall(Metric):
    """TIGER SID retrieval Recall@K."""

    def __init__(self, top_k: int, **kwargs: Any):
        super().__init__(**kwargs)
        self.top_k = top_k
        self.add_state("metric_values", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total_values", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, preds: torch.Tensor, target: torch.Tensor, indexes: torch.Tensor, **kwargs: Any) -> None:
        preds, target = _reshape_retrieval_inputs(preds, target, indexes)
        topk_indices = torch.topk(preds, self.top_k)[1]
        topk_true = target.gather(1, topk_indices)

        true_positives = topk_true.sum(dim=1)
        total_relevant = target.sum(dim=1)
        recall = true_positives / total_relevant.minimum(torch.tensor(self.top_k, device=self.device)).clamp(min=1)

        self.metric_values += recall.sum()
        self.total_values += preds.size(0)

    def compute(self) -> torch.Tensor:
        if self.total_values == 0:
            return torch.tensor(0.0, device=self.metric_values.device)
        return self.metric_values / self.total_values
