"""物品解析 trace 与训练集熵标定的数据契约。"""

import torch


def _base(bundle, schema):
    if bundle.get("schema_version") != schema:
        raise ValueError("Unsupported item resolution schema.")
    for key in ("keys", "labels"):
        bundle[key] = torch.as_tensor(bundle[key])
    keys, labels = bundle["keys"], bundle["labels"]
    if keys.ndim != 1 or keys.numel() == 0 or keys.unique().numel() != keys.numel():
        raise ValueError("Resolution trace requires nonempty unique keys.")
    if labels.ndim != 2 or len(labels) != len(keys):
        raise ValueError("Resolution labels must align with user keys.")
    if not torch.equal(keys, keys.long()) or not torch.equal(labels, labels.long()) or (labels < 0).any():
        raise ValueError("Resolution keys and labels must be integer identities.")
    if not isinstance(bundle.get("metadata"), dict) or not isinstance(bundle.get("trace"), dict):
        raise ValueError("Missing resolution metadata or tensors.")
    for name, value in bundle["trace"].items():
        value = torch.as_tensor(value)
        if value.ndim < 1 or len(value) != len(keys) or (value.is_floating_point() and not value.isfinite().all()):
            raise ValueError(f"Invalid trace tensor {name}.")
        bundle["trace"][name] = value


def validate_resolution_trace(bundle):
    _base(bundle, "item_resolution_v1")
    trace, metadata = bundle["trace"], bundle["metadata"]
    required = {
        "expanded_node",
        "expanded_depth",
        "arrival_mass",
        "stop_gate",
        "resolved_mass",
        "resolved_count",
        "states_evaluated",
        "items_scored",
        "remaining_mass",
        "resolved_total_mass",
        "topk_certified",
        "probability_bound_valid",
        "topk_scores",
        "topk_item_keys",
        "topk_depth_contribution",
        "decoder_batch_seconds",
        "batch_size",
    }
    if not required <= trace.keys():
        raise ValueError(f"Missing resolution trace fields: {required - trace.keys()}.")
    if metadata.get("data_split") not in ("evaluation", "testing") or not metadata.get("contract"):
        raise ValueError("Resolution trace needs an explicit evaluation/testing split and model contract.")
    batch = len(bundle["keys"])
    k, horizon = metadata["beam_width"], metadata["event_capacity"]
    if trace["expanded_node"].shape != (batch, horizon) or trace["topk_scores"].shape != (batch, k):
        raise ValueError("Resolution trace budget/top-k shape mismatch.")
    for field in ("expanded_depth", "arrival_mass", "stop_gate", "resolved_mass", "resolved_count"):
        if trace[field].shape != (batch, horizon):
            raise ValueError(f"Resolution event shape mismatch: {field}.")
    for field in (
        "states_evaluated",
        "items_scored",
        "remaining_mass",
        "resolved_total_mass",
        "topk_certified",
        "probability_bound_valid",
    ):
        if trace[field].shape != (batch,):
            raise ValueError(f"Resolution scalar shape mismatch: {field}.")
    if trace["topk_depth_contribution"].shape != (batch, k, metadata["num_hierarchies"] - 1):
        raise ValueError("Resolution depth contribution shape mismatch.")
    if trace["topk_item_keys"].shape != (batch, k) or not (trace["topk_scores"] > 0).all():
        raise ValueError("Resolution trace must contain K positive item scores.")
    if (trace["topk_item_keys"].sort(-1).values[:, 1:] == trace["topk_item_keys"].sort(-1).values[:, :-1]).any():
        raise ValueError("Duplicate item in resolution top K.")
    valid = trace["probability_bound_valid"].bool()
    if valid.any():
        total = trace["remaining_mass"][valid] + trace["resolved_total_mass"][valid]
        if not torch.allclose(total, torch.ones_like(total), atol=2e-5, rtol=2e-5):
            raise ValueError("Resolution probability mass is not conserved.")
        if (trace["remaining_mass"][valid] < -1e-6).any() or (trace["resolved_total_mass"][valid] < 0).any():
            raise ValueError("Resolution probability mass cannot be negative.")
        if not torch.allclose(
            trace["topk_depth_contribution"][valid].sum(-1), trace["topk_scores"][valid], atol=2e-6, rtol=2e-5
        ):
            raise ValueError("Resolution exit contributions do not sum to item scores.")
        if (trace["states_evaluated"][valid] > metadata["max_states"]).any() or (
            trace["items_scored"][valid] > metadata["max_item_scores"]
        ).any():
            raise ValueError("Resolution inference exceeded its budget.")
        responsibility = trace.get("target_depth_responsibility")
        if responsibility is None or responsibility.shape != (batch, metadata["num_hierarchies"] - 1):
            raise ValueError("Missing target resolution responsibility.")
        if not torch.allclose(responsibility.sum(1), torch.ones(batch, device=responsibility.device), atol=2e-5):
            raise ValueError("Invalid target resolution responsibility.")
    if ((trace["stop_gate"] < 0) | (trace["stop_gate"] > 1)).any():
        raise ValueError("Invalid resolution gate.")


def validate_resolution_calibration(bundle):
    _base(bundle, "item_resolution_calibration_v1")
    metadata = bundle["metadata"]
    if metadata.get("source_split") != "training":
        raise ValueError("WIDE calibration must originate from training split.")
    if not metadata.get("checkpoint_fingerprint") or not metadata.get("catalog_fingerprint"):
        raise ValueError("Calibration needs checkpoint and catalog fingerprints.")
    entropy = bundle["trace"].get("entropy")
    if entropy is None or entropy.shape != bundle["labels"].shape or (entropy < 0).any():
        raise ValueError("Calibration entropy must align with target SID layers.")
