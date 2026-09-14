"""固定用户抽样和评分/搜索审计产物契约。"""

import hashlib
import heapq

import torch


def select_audit_rows(rows, count, seed):
    """以 key 的 hash 抽样；不读取 label，遍历顺序不影响样本。"""
    if not 1 <= count <= 512 or seed < 0:
        raise ValueError("Audit requires 1..512 users and a nonnegative sampling seed.")
    heap, seen = [], set()
    for row in rows:
        value = torch.as_tensor(row["user_id"]).reshape(-1)
        if len(value) != 1 or not torch.equal(value, value.long()) or value[0] < 0:
            raise ValueError("Audit requires one nonnegative integer user key per row.")
        key = int(value[0])
        if key in seen:
            raise ValueError("Duplicate audit user key.")
        seen.add(key)
        score = int.from_bytes(hashlib.sha256(f"{seed}:{key}".encode()).digest(), "big")
        entry = (-score, -key, row)
        if len(heap) < count:
            heapq.heappush(heap, entry)
        elif entry[:2] > heap[0][:2]:
            heapq.heapreplace(heap, entry)
    if len(heap) != count:
        raise ValueError("Evaluation dataset has fewer users than the audit sample size.")
    return [row for _, key, row in sorted(heap, key=lambda entry: -entry[1])]


def input_fingerprint(inputs):
    digest = hashlib.sha256()
    for value in (inputs.input_ids, inputs.attention_mask):
        tensor = value.detach().cpu().contiguous()
        digest.update(str((tuple(tensor.shape), tensor.dtype)).encode())
        digest.update(tensor.numpy().tobytes())
    return torch.tensor(list(digest.digest()), dtype=torch.uint8)[None]


def validate_score_search_audit(bundle):
    if bundle.get("schema_version") != "item_resolution_score_search_v1":
        raise ValueError("Invalid score/search audit schema.")
    keys, labels, trace, meta = (bundle[name] for name in ("keys", "labels", "trace", "metadata"))
    if keys.ndim != 1 or not torch.equal(keys, keys.long()) or len(keys.unique()) != len(keys) or (keys < 0).any():
        raise ValueError("Audit user keys must be unique nonnegative integers.")
    n, items = len(keys), len(meta["item_keys"])
    depth, budgets, k = meta["semantic_depth"], len(meta["state_budgets"]), meta["top_k"]
    if meta["data_split"] != "evaluation" or not meta["checkpoint_fingerprint"] or not meta["catalog_fingerprint"]:
        raise ValueError("Audit requires evaluation and checkpoint/catalog identities.")
    if meta["item_keys"] != sorted(set(meta["item_keys"])) or not 0 < k <= items:
        raise ValueError("Invalid audit item catalog.")
    if labels.shape != (n, depth + 1) or not torch.equal(labels, labels.long()):
        raise ValueError("Invalid audit labels.")
    shapes = dict(
        input_sha256=(n, 32),
        exact_log_probability=(n, items),
        exact_target_rank=(n,),
        exact_target_log_probability=(n,),
        exact_topk_keys=(n, k),
        target_depth_responsibility=(n, depth),
        global_depth_mass=(n, depth),
        exact_seconds=(n,),
        encoder_seconds=(n,),
        search_target_rank=(n, budgets),
        search_ndcg=(n, budgets),
        search_seconds=(n, budgets),
        search_remaining_mass=(n, budgets),
        search_topk_certified=(n, budgets),
        search_states=(n, budgets),
        search_item_scores=(n, budgets),
        search_topk_keys=(n, budgets, k),
        search_topk_scores=(n, budgets, k),
        search_topk_probability_capture=(n, budgets, k),
        exact_topk_overlap=(n, budgets),
    )
    for name, shape in shapes.items():
        if name not in trace or tuple(trace[name].shape) != shape or not trace[name].isfinite().all():
            raise ValueError(f"Invalid audit trace field: {name}.")
    for name in [
        "exact_target_rank",
        "search_target_rank",
        "search_states",
        "search_item_scores",
        "exact_topk_keys",
        "search_topk_keys",
    ]:
        if not torch.equal(trace[name], trace[name].long()):
            raise ValueError(f"Audit field must contain integers: {name}.")
    for name in [
        "exact_seconds",
        "encoder_seconds",
        "search_seconds",
        "search_states",
        "search_item_scores",
        "search_topk_scores",
        "search_topk_probability_capture",
    ]:
        if (trace[name] < 0).any():
            raise ValueError(f"Negative audit field: {name}.")
    if trace["input_sha256"].dtype != torch.uint8 or trace["search_topk_certified"].dtype != torch.bool:
        raise ValueError("Invalid audit hash/certificate types.")
    for name in ["exact_topk_keys", "search_topk_keys"]:
        ordered = trace[name].sort(-1).values
        if (ordered[..., 1:] == ordered[..., :-1]).any():
            raise ValueError("Duplicate audit Top-K items.")
    if not torch.allclose(trace["exact_log_probability"].exp().sum(1), torch.ones(n), atol=1e-4, rtol=1e-4):
        raise ValueError("Exact catalog probabilities must sum to one.")
    for name in ["target_depth_responsibility", "global_depth_mass"]:
        if (trace[name] < 0).any() or not torch.allclose(trace[name].sum(1), torch.ones(n), atol=1e-4, rtol=1e-4):
            raise ValueError(f"Invalid depth mass: {name}.")
    if ((trace["exact_target_rank"] < 1) | (trace["exact_target_rank"] > items)).any():
        raise ValueError("Invalid exact target rank.")
    if (
        (trace["search_target_rank"] != -1) & ((trace["search_target_rank"] < 1) | (trace["search_target_rank"] > k))
    ).any():
        raise ValueError("Invalid search target rank.")
    for name in ["search_remaining_mass", "search_ndcg", "exact_topk_overlap"]:
        if ((trace[name] < 0) | (trace[name] > 1 + 1e-5)).any():
            raise ValueError(f"Invalid audit ratio: {name}.")
    if (trace["search_states"] > torch.tensor(meta["state_budgets"])[None]).any() or (
        trace["search_item_scores"] > meta["item_score_budget"]
    ).any():
        raise ValueError("Audit search exceeded declared budget.")
    for name in ["exact_topk_keys", "search_topk_keys"]:
        if not torch.isin(trace[name], torch.tensor(meta["item_keys"])).all():
            raise ValueError("Audit recommendation absent from catalog.")
