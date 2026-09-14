"""WIDE 风格控制：熵触发 wildcard 后继续生成，最终融合评分。"""

import torch
from torch.nn import functional as F

from .search import _empty_trace


def wide_search(model, encoded, mask):
    if model.calibration is None or model.loaded_checkpoint_fingerprint is None:
        raise ValueError("WIDE requires a verified checkpoint and training calibration.")
    catalog, device, batch = model.catalog, encoded.device, len(encoded)
    thresholds = torch.as_tensor(model.calibration["thresholds"], device=device)
    if thresholds.shape != (model.num_hierarchies,):
        raise ValueError("WIDE threshold depth mismatch.")
    # 条目为 node、无 wildcard 的 log 分数、可靠位置概率之和、wildcard 数。
    active = [[(0, 0.0, 0.0, 0)] for _ in range(batch)]
    states_used = [0] * batch
    pruned = [0] * batch
    for depth in range(model.num_hierarchies):
        entries = []
        for user in range(batch):
            # 为余下层保留状态容量；截断是显式近似，绝不标记为概率证书。
            capacity = max(0, (model.max_states - states_used[user]) // (model.num_hierarchies - depth))
            retained = sorted(active[user], key=lambda entry: entry[1], reverse=True)[:capacity]
            pruned[user] += len(active[user]) - len(retained)
            entries.extend((user, *entry) for entry in retained)
            states_used[user] += len(retained)
        if not entries:
            raise RuntimeError("WIDE state budget exhausted before complete item candidates.")
        users = torch.tensor([entry[0] for entry in entries], device=device)
        nodes = torch.tensor([entry[1] for entry in entries], device=device)
        prefixes = catalog.prefixes[nodes, :depth]
        # 分块避免 wildcard 扩张造成一次性 decoder 激活峰值。
        parts = []
        for start in range(0, len(entries), 128):
            end = start + 128
            hidden = model.states(encoded[users[start:end]], mask[users[start:end]], prefixes[start:end])[:, -1]
            parts.append(model.route_log(hidden, prefixes[start:end]))
        lp = torch.cat(parts)
        entropy = -(lp.exp() * lp.masked_fill(~lp.isfinite(), 0.0)).sum(-1)
        wildcard = (entropy > thresholds[depth]).cpu().tolist()
        lp = lp.cpu()
        next_active = [[] for _ in range(batch)]
        for index, (user, node, score, probability_sum, wild_count) in enumerate(entries):
            children = catalog.child_nodes[node]
            if wildcard[index]:
                next_active[user].extend((child, score, probability_sum, wild_count + 1) for _, child in children)
            else:
                ordered = sorted(children, key=lambda entry: float(lp[index, entry[0]]), reverse=True)[
                    : model.top_k_for_generation
                ]
                next_active[user].extend(
                    (
                        child,
                        score + float(lp[index, token]),
                        probability_sum + float(lp[index, token].exp()),
                        wild_count,
                    )
                    for token, child in ordered
                )
        active = next_active
    root = torch.empty(batch, 0, dtype=torch.long, device=device)
    hidden = model.states(encoded, mask, root)[:, -1]
    query = model.query(hidden, 0)
    vectors = model.item_vectors()
    chosen, scores, item_counts = [], [], []
    for user, entries in enumerate(active):
        entries = sorted(entries, key=lambda entry: entry[1], reverse=True)
        pruned[user] += max(0, len(entries) - model.max_item_scores)
        entries = entries[: model.max_item_scores]
        if len(entries) < model.top_k_for_generation:
            raise RuntimeError("WIDE budget produced fewer than K complete candidates.")
        nodes = torch.tensor([entry[0] for entry in entries], device=device)
        ids = catalog.item_indices(catalog.prefixes[nodes])
        wild = torch.tensor([entry[3] for entry in entries], device=device)
        discrete = torch.tensor([entry[2] for entry in entries], device=device) / (
            model.num_hierarchies - wild
        ).clamp_min(1)
        alpha = wild.float() / model.num_hierarchies
        similarity = (F.cosine_similarity(query[user : user + 1], vectors[ids]) + 1) / 2
        fused = (1 - alpha) * discrete + alpha * similarity
        values, positions = fused.topk(model.top_k_for_generation)
        chosen.append(ids[positions])
        scores.append(values.clamp_min(torch.finfo(values.dtype).tiny))
        item_counts.append(len(entries))
    trace = _empty_trace(model, batch, device)
    trace["states_evaluated"] = torch.tensor(states_used, device=device) + 1
    trace["items_scored"] = torch.tensor(item_counts, device=device)
    trace["wildcard_pruned_paths"] = torch.tensor(pruned, device=device)
    return catalog.sids[torch.stack(chosen)], torch.stack(scores), trace
