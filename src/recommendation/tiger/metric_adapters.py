from collections.abc import Mapping
from typing import Any

import torch


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
