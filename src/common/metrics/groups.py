from collections.abc import Callable, Mapping
from functools import partial
from typing import Any

import hydra
import torch
from omegaconf import DictConfig, OmegaConf
from torch import nn
from torchmetrics.metric import Metric


class SIDRetrievalMetricGroup(nn.Module):
    """Metric group for TIGER semantic-ID retrieval outputs."""

    def __init__(
        self,
        metrics: Mapping[str, Callable[..., Metric] | Metric | type[Metric] | dict[str, Any] | DictConfig],
        top_k_list: list[int],
    ):
        super().__init__()
        self.metrics = nn.ModuleDict()
        plain_metrics = _to_plain_container(metrics)
        for metric_name, metric_factory in plain_metrics.items():
            for top_k in top_k_list:
                output_name = f"{metric_name}@{top_k}"
                self.metrics[_safe_module_key(output_name)] = _instantiate_top_k_metric(metric_factory, top_k=top_k)
                self.metrics[_safe_module_key(output_name)].output_name = output_name

    def update_from_payload(self, payload: Mapping[str, Any]) -> None:
        marginal_probs = payload["marginal_probs"]
        generated_ids = payload["generated_ids"]
        labels = payload["labels"]

        batch_size, num_candidates, num_hierarchies = generated_ids.shape
        labels = labels.reshape(batch_size, 1, num_hierarchies)
        preds = marginal_probs.reshape(-1)

        matched_id_coord = torch.all((generated_ids == labels), dim=2).nonzero()
        target = torch.zeros(batch_size, num_candidates).bool()
        target[matched_id_coord[:, 0], matched_id_coord[:, 1]] = True
        target = target.reshape(-1)
        expanded_indexes = torch.arange(batch_size).unsqueeze(-1).expand(batch_size, num_candidates).reshape(-1)

        for metric in self.metrics.values():
            metric.update(
                preds,
                target.to(preds.device),
                indexes=expanded_indexes.to(preds.device),
            )

    def compute(self) -> dict[str, torch.Tensor]:
        return {metric.output_name: metric.compute() for metric in self.metrics.values()}

    def reset(self) -> None:
        for metric in self.metrics.values():
            metric.reset()


def _instantiate_top_k_metric(
    metric_factory: Callable[..., Metric] | Metric | type[Metric] | dict[str, Any],
    top_k: int,
) -> Metric:
    metric_factory = _to_plain_container(metric_factory)
    if isinstance(metric_factory, Metric):
        return metric_factory.__class__(top_k=top_k, sync_on_compute=False, compute_with_cache=False)
    if isinstance(metric_factory, type):
        return metric_factory(top_k=top_k, sync_on_compute=False, compute_with_cache=False)
    if isinstance(metric_factory, dict) and "_target_" in metric_factory:
        metric_factory = hydra.utils.instantiate(metric_factory)
    if isinstance(metric_factory, partial):
        return metric_factory(top_k=top_k, sync_on_compute=False, compute_with_cache=False)
    if callable(metric_factory):
        return metric_factory(top_k=top_k, sync_on_compute=False, compute_with_cache=False)
    raise TypeError(f"Unsupported retrieval metric factory: {metric_factory!r}")


def _to_plain_container(value: Any) -> Any:
    if isinstance(value, DictConfig):
        return OmegaConf.to_container(value, resolve=True)
    return value


def _safe_module_key(name: str) -> str:
    return name.replace(".", "_dot_").replace("/", "_slash_")
