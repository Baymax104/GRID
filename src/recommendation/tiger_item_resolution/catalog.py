"""只由固定 item 目录构造的前缀索引与内容特征。"""

import hashlib
import json

import numpy as np
import torch
from torch import nn


def encode(sids, radix):
    result = torch.zeros(sids.shape[:-1], dtype=torch.long, device=sids.device)
    for token in sids.unbind(-1):
        result = result * radix + token
    return result


class ResolutionCatalog(nn.Module):
    def __init__(self, catalog, radix, depth, projection_dim, max_bucket):
        super().__init__()
        keys = catalog["keys"].detach().cpu().reshape(-1)
        raw_sids = catalog["semantic_ids"].detach().cpu()
        raw = catalog["embeddings"].detach().cpu().float()
        if depth < 2 or radix < 2 or radix**depth > 2**63 - 1 or max_bucket < 1:
            raise ValueError("Invalid resolution depth, radix or bucket capacity.")
        if (
            keys.numel() < 2
            or (keys < 0).any()
            or not torch.equal(keys, keys.long())
            or not (keys[1:] > keys[:-1]).all()
        ):
            raise ValueError("Catalog requires sorted unique integer item keys.")
        keys = keys.long()
        if raw_sids.shape != (len(keys), depth) or not torch.equal(raw_sids, raw_sids.long()):
            raise ValueError("Catalog requires complete integer SIDs.")
        sids = raw_sids.long()
        if (sids < 0).any() or (sids >= radix).any() or sids.unique(dim=0).shape[0] != len(keys):
            raise ValueError("Complete SIDs must be unique and in range.")
        if raw.ndim != 2 or raw.shape[0] != len(keys) or not raw.isfinite().all() or projection_dim < 1:
            raise ValueError("Catalog content must be a finite item matrix.")
        self.radix, self.depth, self.max_bucket = radix, depth, max_bucket
        digest = hashlib.sha256(json.dumps(dict(radix=radix, depth=depth, projection_dim=projection_dim)).encode())
        for tensor in (keys, sids, raw):
            digest.update(str(tuple(tensor.shape)).encode())
            digest.update(tensor.contiguous().numpy().tobytes())
        self.fingerprint = digest.hexdigest()
        centered = raw.numpy().astype(np.float64)
        mean = centered.mean(0)
        centered -= mean
        eigenvalues, basis = np.linalg.eigh(centered.T @ centered)
        dimension = min(projection_dim, raw.shape[1], len(keys) - 1)
        projection = basis[:, np.argsort(eigenvalues)[::-1][:dimension]].copy()
        signs = np.sign(projection[np.abs(projection).argmax(0), np.arange(dimension)])
        projection *= np.where(signs == 0, 1, signs)
        features = centered @ projection
        norms = np.linalg.norm(features, axis=1, keepdims=True)
        if (norms < 1e-12).any():
            raise ValueError("Degenerate zero catalog content after PCA.")
        features = (features / norms).astype(np.float32)
        for name, tensor in dict(
            keys=keys,
            sids=sids,
            features=torch.from_numpy(features),
            pca_mean=torch.from_numpy(mean).float(),
            pca_projection=torch.from_numpy(projection).float(),
        ).items():
            self.register_buffer(name, tensor)

        prefixes = [torch.full((depth,), -1, dtype=torch.long)]
        depths, counts, members, offsets = [0], [len(keys)], [torch.arange(len(keys))], [0, len(keys)]
        self.level_offsets = [0]
        self.child_nodes = {}
        for level in range(1, depth + 1):
            self.level_offsets.append(len(prefixes))
            codes, inverse = encode(sids[:, :level], radix).unique(sorted=True, return_inverse=True)
            self.register_buffer(f"codes_{level}", codes)
            order = inverse.argsort(stable=True)
            groups = order.split(torch.bincount(inverse).tolist())
            for group in groups:
                row = torch.full((depth,), -1, dtype=torch.long)
                row[:level] = sids[group[0], :level]
                prefixes.append(row)
                depths.append(level)
                counts.append(len(group))
                members.append(group)
                offsets.append(offsets[-1] + len(group))
        self.register_buffer("prefixes", torch.stack(prefixes))
        self.register_buffer("depths", torch.tensor(depths))
        self.register_buffer("counts", torch.tensor(counts))
        self.register_buffer("members_flat", torch.cat(members))
        self.register_buffer("offsets", torch.tensor(offsets))
        # 静态树的调度元数据不从 CUDA 逐批下载；派生索引不进入旧 checkpoint。
        self.node_counts = tuple(counts)
        self.node_depths = tuple(depths)
        ancestors = torch.zeros(len(keys), depth + 1, dtype=torch.long)
        for node, group in enumerate(members):
            ancestors[group, depths[node]] = node
        self.register_buffer("item_ancestors", ancestors, persistent=False)
        for node in range(1, len(prefixes)):
            level = depths[node]
            parent = int(self.nodes(self.prefixes[node : node + 1, : level - 1])[0])
            self.child_nodes.setdefault(parent, []).append((int(prefixes[node][level - 1]), node))
        if self.counts[self.depths == depth - 1].max() > max_bucket:
            raise ValueError("Terminal semantic bucket exceeds max_bucket; increase capacity or semantic depth.")
        codes, order = encode(sids, radix).sort()
        self.register_buffer("full_codes", codes)
        self.register_buffer("item_order", order)

    def nodes(self, prefixes):
        level = prefixes.shape[-1]
        if level == 0:
            return torch.zeros(prefixes.shape[:-1], dtype=torch.long, device=prefixes.device)
        if level > self.depth or (prefixes < 0).any() or (prefixes >= self.radix).any():
            raise ValueError("Invalid catalog prefix.")
        codes = getattr(self, f"codes_{level}")
        query = encode(prefixes, self.radix)
        pos = torch.searchsorted(codes, query).clamp(max=len(codes) - 1)
        if not torch.equal(codes[pos], query):
            raise ValueError("Prefix absent from catalog.")
        return pos + self.level_offsets[level]

    def item_indices(self, sids):
        if sids.shape[-1] != self.depth or (sids < 0).any() or (sids >= self.radix).any():
            raise ValueError("Invalid full item SID.")
        query = encode(sids, self.radix)
        pos = torch.searchsorted(self.full_codes, query).clamp(max=len(self.full_codes) - 1)
        if not torch.equal(self.full_codes[pos], query):
            raise ValueError("Item SID absent from catalog.")
        return self.item_order[pos]

    def members(self, nodes):
        lengths = self.counts[nodes]
        slots = torch.arange(int(lengths.max()), device=nodes.device)
        valid = slots[None] < lengths[:, None]
        positions = self.offsets[nodes, None] + slots[None]
        positions = positions.minimum(self.offsets[nodes + 1, None] - 1)
        return self.members_flat[positions], valid

    def legal(self, prefixes):
        level = prefixes.shape[-1] + 1
        codes = getattr(self, f"codes_{level}")
        query = encode(prefixes, self.radix)[:, None] * self.radix
        query = query + torch.arange(self.radix, device=prefixes.device)
        pos = torch.searchsorted(codes, query).clamp(max=len(codes) - 1)
        return codes[pos] == query
