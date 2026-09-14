"""固定内容投影和稀疏前缀原型；全部 tensor 随 checkpoint 持久化。"""

import hashlib
import json

import numpy as np
import torch
from torch import nn


def encode_prefix(sids: torch.Tensor, radix: int) -> torch.Tensor:
    code = torch.zeros(sids.shape[:-1], dtype=torch.long, device=sids.device)
    for column in sids.unbind(-1):
        code = code * radix + column
    return code


def _cluster(points: np.ndarray, maximum: int, iterations: int):
    count = len(points)
    if count <= maximum:
        return points.copy(), np.ones(count, dtype=np.int64), np.zeros(count, dtype=np.float32)
    chosen = [int(np.argmax(np.square(points - points.mean(0)).sum(1)))]
    distance = np.square(points - points[chosen[0]]).sum(1)
    for _ in range(1, maximum):
        index = int(np.argmax(distance))
        if distance[index] <= 1e-12:
            break
        chosen.append(index)
        distance = np.minimum(distance, np.square(points - points[index]).sum(1))
    centers = points[chosen].copy()
    for _ in range(iterations):
        labels = np.square(points[:, None] - centers[None]).sum(-1).argmin(1)
        centers = np.stack([points[labels == j].mean(0) for j in range(len(centers)) if (labels == j).any()])
    labels = np.square(points[:, None] - centers[None]).sum(-1).argmin(1)
    groups = [points[labels == j] for j in range(len(centers)) if (labels == j).any()]
    means = np.stack([group.mean(0) for group in groups])
    counts = np.array([len(group) for group in groups], dtype=np.int64)
    radii = np.array([np.linalg.norm(group - mean, axis=1).max() for group, mean in zip(groups, means, strict=True)])
    return means, counts, radii


class PrefixCatalog(nn.Module):
    """每层保存实际存在的前缀，原型只为非空 cluster 分配空间。"""

    VERSION = 1

    def __init__(
        self,
        catalog: dict[str, torch.Tensor],
        codebook_size: int,
        num_hierarchies: int,
        projection_dim: int = 128,
        prototypes: int = 4,
        cluster_iterations: int = 5,
        shuffle_seed: int | None = None,
    ):
        super().__init__()
        keys = catalog["keys"].detach().cpu().long().reshape(-1)
        raw_sids = catalog["semantic_ids"].detach().cpu()
        raw = catalog["embeddings"].detach().cpu().float()
        if projection_dim < 1 or prototypes < 1 or cluster_iterations < 1:
            raise ValueError("Projection dimension, prototypes and cluster iterations must be positive.")
        if codebook_size < 2 or num_hierarchies < 1 or codebook_size**num_hierarchies > 2**63 - 1:
            raise ValueError("Invalid catalog radix or hierarchy count.")
        if raw_sids.ndim != 2 or raw_sids.shape != (keys.numel(), num_hierarchies):
            raise ValueError("Catalog must provide the complete SID with exactly num_hierarchies columns.")
        sids = raw_sids.long()
        if not torch.equal(raw_sids, sids) or (sids < 0).any() or (sids >= codebook_size).any():
            raise ValueError("Catalog SID tokens must be integers in the codebook range.")
        if keys.numel() < 2 or keys.unique().numel() != keys.numel() or (keys < 0).any():
            raise ValueError("Catalog requires at least two unique nonnegative item keys.")
        if sids.unique(dim=0).shape[0] != len(sids):
            raise ValueError("Complete catalog SIDs must be unique, including the deduplication layer.")
        if raw.ndim != 2 or raw.shape[0] != keys.numel() or raw.shape[1] < 1 or not raw.isfinite().all():
            raise ValueError("Catalog embeddings must be a finite item-by-feature matrix.")
        if not torch.all(keys[1:] > keys[:-1]):
            raise ValueError("Catalog keys must be sorted before bank construction.")
        self.radix = codebook_size
        self.depth = num_hierarchies
        self.max_prototypes = prototypes
        settings = dict(
            version=self.VERSION,
            radix=codebook_size,
            depth=num_hierarchies,
            projection_dim=projection_dim,
            prototypes=prototypes,
            cluster_iterations=cluster_iterations,
            shuffle_seed=shuffle_seed,
        )
        digest = hashlib.sha256(json.dumps(settings, sort_keys=True).encode())
        for tensor in (keys, sids, raw):
            digest.update(str(tuple(tensor.shape)).encode())
            digest.update(tensor.contiguous().numpy().tobytes())
        self.fingerprint = digest.hexdigest()
        values = raw.numpy().astype(np.float64)
        mean = values.mean(0)
        centered = values - mean
        dimension = min(projection_dim, raw.shape[1], len(keys) - 1)
        eigenvalues, basis = np.linalg.eigh(centered.T @ centered)
        projection = basis[:, np.argsort(eigenvalues)[::-1][:dimension]].copy()
        # 固定特征向量符号，避免等价正负号导致跨 arm 初始化差异。
        signs = np.sign(projection[np.abs(projection).argmax(0), np.arange(dimension)])
        projection *= np.where(signs == 0, 1, signs)
        features = centered @ projection
        norms = np.linalg.norm(features, axis=1, keepdims=True)
        if (norms < 1e-12).any():
            raise ValueError("PCA produced a zero item vector; inspect degenerate catalog content.")
        features = (features / norms).astype(np.float32)
        permutation = torch.arange(len(keys))
        if shuffle_seed is not None:
            permutation = torch.randperm(len(keys), generator=torch.Generator().manual_seed(shuffle_seed))
            features = features[permutation.numpy()]
        for name, tensor in dict(
            keys=keys,
            sids=sids,
            features=torch.from_numpy(features),
            pca_mean=torch.from_numpy(mean).float(),
            pca_projection=torch.from_numpy(projection).float(),
            permutation=permutation,
        ).items():
            self.register_buffer(name, tensor)
        for level in range(num_hierarchies):
            codes = encode_prefix(sids[:, : level + 1], codebook_size)
            unique, inverse = torch.unique(codes, sorted=True, return_inverse=True)
            groups = np.argsort(inverse.numpy(), kind="stable")
            splits = np.split(groups, np.cumsum(np.bincount(inverse.numpy()))[:-1])
            slots = torch.full((len(unique), prototypes), -1, dtype=torch.long)
            means, counts, radii = [], [], []
            offset = 0
            for index, members in enumerate(splits):
                mu, n, radius = _cluster(features[members], prototypes, cluster_iterations)
                slots[index, : len(mu)] = torch.arange(offset, offset + len(mu))
                means.append(mu)
                counts.append(n)
                radii.append(radius)
                offset += len(mu)
            for name, tensor in dict(
                codes=unique,
                slots=slots,
                means=torch.from_numpy(np.concatenate(means)).float(),
                counts=torch.from_numpy(np.concatenate(counts)).long(),
                radii=torch.from_numpy(np.concatenate(radii)).float(),
            ).items():
                self.register_buffer(f"{name}_{level}", tensor)
        full_codes, order = encode_prefix(sids, self.radix).sort()
        self.register_buffer("full_codes", full_codes)
        self.register_buffer("item_order", order)

    def item_indices(self, sids: torch.Tensor) -> torch.Tensor:
        codes = encode_prefix(sids.long(), self.radix)
        positions = torch.searchsorted(self.full_codes, codes).clamp(max=len(self.full_codes) - 1)
        if not torch.equal(self.full_codes[positions], codes):
            raise ValueError("Target or recommendation SID is absent from the catalog.")
        return self.item_order[positions]

    def lookup_children(self, prefixes: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        level = prefixes.shape[-1]
        codes = getattr(self, f"codes_{level}")
        candidates = encode_prefix(prefixes, self.radix)[:, None] * self.radix
        candidates = candidates + torch.arange(self.radix, device=prefixes.device)
        positions = torch.searchsorted(codes, candidates).clamp(max=len(codes) - 1)
        return positions, codes[positions] == candidates

    def log_mass(self, query: torch.Tensor, prefixes: torch.Tensor, temperature: float) -> torch.Tensor:
        positions, legal = self.lookup_children(prefixes)
        rows, tokens = legal.nonzero(as_tuple=True)
        level = prefixes.shape[-1]
        slots = getattr(self, f"slots_{level}")[positions[rows, tokens]]
        safe = slots.clamp(min=0)
        means = getattr(self, f"means_{level}")[safe]
        counts = getattr(self, f"counts_{level}")[safe]
        logits = (query[rows, None].float() * means).sum(-1) / temperature + counts.float().log()
        mass = logits.masked_fill(slots < 0, -torch.inf).logsumexp(-1)
        result = query.new_full(legal.shape, -torch.inf, dtype=torch.float32)
        result[rows, tokens] = mass
        return result
