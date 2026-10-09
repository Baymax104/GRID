"""校验逐层真实 beam frontier 及目标路径观测。"""

import torch


def validate_liger_path_trace(bundle):
    if bundle.get("schema_version") != "liger_paths_v1":
        raise ValueError("Unknown LIGER path schema.")
    keys, labels = bundle["keys"].cpu(), bundle["labels"].cpu()
    t = {k: v.cpu() for k, v in bundle["trace"].items()}
    m = bundle["metadata"]
    n, h, beams, codes = len(keys), m["num_hierarchies"], m["generation_candidates"], m["codebook_size"]
    floats = {"target_generation_log_prob", "target_content_log_prob", "target_mixed_log_prob"}
    bools = {"target_prefix_survived", "target_parent_present"}
    integers = {"target_branch_rank", "legal_child_count", "first_failure_depth", "beam_prefixes"}
    if set(t) != floats | bools | integers:
        raise ValueError("Incomplete path fields.")
    if not n or keys.ndim != 1 or keys.unique().numel() != n or labels.shape != (n, h):
        raise ValueError("Invalid path keys/labels.")
    if labels.dtype != torch.long or ((labels < 0) | (labels >= codes)).any():
        raise ValueError("Invalid target SID.")
    for name, value in t.items():
        shape = (n, h, beams, h) if name == "beam_prefixes" else (n,) if name == "first_failure_depth" else (n, h)
        if value.shape != shape:
            raise ValueError("Invalid path dimensions: " + name)
        if (name in bools and value.dtype != torch.bool) or (name in integers and value.dtype != torch.long) or (name in floats and not value.is_floating_point()):
            raise ValueError("Invalid path dtype: " + name)
    prefixes = t["beam_prefixes"]
    observed = []
    for d in range(h):
        active = prefixes[:, d, :, :d + 1]
        if ((active < 0) | (active >= codes)).any() or (prefixes[:, d, :, d + 1:] != -1).any():
            raise ValueError("Invalid frontier SID/padding.")
        if d and not (active[:, :, None, :d] == prefixes[:, d - 1, None, :, :d]).all(-1).any(-1).all():
            raise ValueError("Frontier has an absent ancestor.")
        observed.append((active == labels[:, None, :d + 1]).all(-1).any(-1))
    survived = torch.stack(observed, 1)
    present = torch.cat([torch.ones(n, 1, dtype=torch.bool), survived[:, :-1]], 1)
    first = torch.where((~survived).any(-1), (~survived).long().argmax(-1) + 1, -1)
    if not torch.equal(survived, t["target_prefix_survived"]) or not torch.equal(present, t["target_parent_present"]) or not torch.equal(first, t["first_failure_depth"]):
        raise ValueError("Target path observation mismatch.")
    for name in floats:
        value = t[name]
        if not torch.isfinite(value[present]).all() or (value[present] > 1e-5).any() or not torch.isnan(value[~present]).all():
            raise ValueError("Invalid conditional probability: " + name)
    rank, count = t["target_branch_rank"], t["legal_child_count"]
    if ((rank[present] < 1) | (rank[present] > count[present]) | (count[present] > codes)).any() or (rank[~present] != 0).any() or (count[~present] != 0).any():
        raise ValueError("Invalid legal branch rank/count.")
