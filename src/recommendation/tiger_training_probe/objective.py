"""按正确父前缀条件概率加权，保持全词表 CE。"""

import math

import torch
from torch import nn
from torch.nn import functional as F


class ConditionalBranchObjective(nn.Module):
    def __init__(self, semantic_ids, expected_counts, codebook_size, layers=(2, 3), alpha=0.25, cap=2.0):
        super().__init__()
        sids = torch.as_tensor(semantic_ids).cpu()
        counts = torch.as_tensor(expected_counts, dtype=torch.float64).cpu()
        if sids.ndim != 2 or sids.is_floating_point() or counts.shape != (sids.size(0),):
            raise ValueError("Invalid SID/count shapes or dtype.")
        if codebook_size < 2 or (sids < 0).any() or (sids >= codebook_size).any():
            raise ValueError("Invalid vocabulary or SID token.")
        if not torch.isfinite(counts).all() or (counts < 0).any() or counts.sum() <= 0:
            raise ValueError("Expected counts must be finite, non-negative and have positive mass.")
        if not math.isfinite(alpha) or alpha < 0 or not math.isfinite(cap) or cap < 1:
            raise ValueError("alpha must be non-negative and cap must be at least one.")
        layers = tuple(layers)
        if (
            not layers
            or len(set(layers)) != len(layers)
            or any(type(x) is not int or x < 1 or x > sids.size(1) for x in layers)
        ):
            raise ValueError("layers must contain unique one-based hierarchy indices.")
        if codebook_size ** sids.size(1) > torch.iinfo(torch.int64).max:
            raise ValueError("Encoded SID exceeds int64 range.")
        self.codebook_size, self.num_hierarchies = codebook_size, sids.size(1)
        self.settings = {"layers": list(layers), "alpha": alpha, "cap": cap}
        encoded = torch.zeros(sids.size(0), dtype=torch.long)
        for depth in range(self.num_hierarchies):
            encoded = encoded * codebook_size + sids[:, depth]
            keys, inverse = torch.unique(encoded, sorted=True, return_inverse=True)
            mass = torch.zeros(keys.numel(), dtype=torch.float64).scatter_add_(0, inverse, counts)
            _, parent_inverse = torch.unique(keys // codebook_size, sorted=True, return_inverse=True)
            parent_mass = torch.zeros(int(parent_inverse.max()) + 1, dtype=torch.float64).scatter_add_(
                0, parent_inverse, mass
            )
            weights = torch.ones_like(mass)
            positive = mass > 0
            if depth + 1 in layers:
                probability = mass[positive] / parent_mass[parent_inverse[positive]]
                weights[positive] = torch.exp((-alpha * probability.log()).clamp(max=math.log(cap)))
                normalizer = (weights * mass).sum() / mass.sum()
                weights[positive] /= normalizer
            self.register_buffer(f"keys_{depth}", keys, persistent=False)
            self.register_buffer(f"weights_{depth}", weights.float(), persistent=False)

    def weights_for(self, targets):
        if targets.ndim != 2 or targets.size(1) != self.num_hierarchies or targets.is_floating_point():
            raise ValueError("Invalid target SID shape/dtype.")
        if ((targets < 0) | (targets >= self.codebook_size)).any():
            raise ValueError("Invalid target SID token.")
        encoded = torch.zeros(targets.size(0), dtype=torch.long, device=targets.device)
        result = []
        for depth in range(self.num_hierarchies):
            encoded = encoded * self.codebook_size + targets[:, depth]
            keys = getattr(self, f"keys_{depth}")
            indices = torch.searchsorted(keys, encoded)
            safe = indices.clamp(max=keys.numel() - 1)
            if ((indices >= keys.numel()) | (keys[safe] != encoded)).any():
                raise ValueError("Target SID absent from statistics catalog.")
            result.append(getattr(self, f"weights_{depth}")[safe])
        return torch.stack(result, dim=1)

    def forward(self, logits, targets):
        weights = self.weights_for(targets)
        losses = []
        for depth in range(self.num_hierarchies):
            loss = F.cross_entropy(
                logits[:, depth], targets[:, depth].long() + depth * self.codebook_size, reduction="none"
            )
            losses.append((loss * weights[:, depth]).mean())
        return torch.stack(losses).mean()

    def audit(self):
        return {
            "settings": self.settings,
            "lookup": {
                str(depth + 1): {
                    "prefix_keys": getattr(self, f"keys_{depth}").cpu(),
                    "weights": getattr(self, f"weights_{depth}").cpu(),
                }
                for depth in range(self.num_hierarchies)
            },
        }
