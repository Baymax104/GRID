"""预算前沿搜索与对照解码；训练和推断共用条件分布。"""

import math
import time

import torch
from torch.nn import functional as F

from .module import RESOLUTION_ARMS


def _empty_trace(model, batch, device):
    shape = (batch, model.max_states if model.arm in RESOLUTION_ARMS else 0)
    return {
        "expanded_node": torch.full(shape, -1, dtype=torch.long, device=device),
        "expanded_depth": torch.full(shape, -1, dtype=torch.long, device=device),
        "arrival_mass": torch.zeros(shape, device=device),
        "stop_gate": torch.zeros(shape, device=device),
        "resolved_mass": torch.zeros(shape, device=device),
        "resolved_count": torch.zeros(shape, dtype=torch.long, device=device),
        "states_evaluated": torch.zeros(batch, dtype=torch.long, device=device),
        "items_scored": torch.zeros(batch, dtype=torch.long, device=device),
        "remaining_mass": torch.zeros(batch, device=device),
        "resolved_total_mass": torch.zeros(batch, device=device),
        "topk_certified": torch.zeros(batch, dtype=torch.bool, device=device),
        "probability_bound_valid": torch.zeros(batch, dtype=torch.bool, device=device),
        "topk_depth_contribution": torch.zeros(batch, model.top_k_for_generation, model.semantic_depth, device=device),
    }


def _frontier_upper(catalog, lower, frontier):
    """前沿是 antichain，每个 item 至多对应其中一个节点；一次批量汇总上界。"""
    rows, nodes, masses = [], [], []
    for user, queue in enumerate(frontier):
        for node, mass in queue.items():
            rows.append(user)
            nodes.append(node)
            masses.append(mass)
    node_mass = lower.new_zeros((len(lower), len(catalog.node_counts)))
    if nodes:
        node_mass[torch.tensor(rows, device=lower.device), torch.tensor(nodes, device=lower.device)] = lower.new_tensor(
            masses
        )
    return lower + node_mass[:, catalog.item_ancestors].sum(-1)


def _resolution(model, encoded, mask, *, collect_trace=True):
    batch, device = len(encoded), encoded.device
    catalog = model.catalog
    trace = _empty_trace(model, batch, device) if collect_trace else {}
    contributions = torch.zeros(batch, len(catalog.keys), model.semantic_depth, device=device)
    frontier = [{0: 1.0} for _ in range(batch)]
    state_counts, score_counts = [0] * batch, [0] * batch
    counts, depths = catalog.node_counts, catalog.node_depths
    vectors = model.item_vectors()

    def resolve_cost(node):
        depth, count = depths[node], counts[node]
        if depth == 0 or count > model.max_bucket:
            return 0
        if model.arm == "depth2" and depth < min(2, model.semantic_depth) and count > 1:
            return 0
        return count

    while True:
        selected = []
        positive_counts = (contributions.sum(-1) > 0).sum(-1).tolist()
        for user, queue in enumerate(frontier):
            used = 0
            # 平坦首层下纯质量优先会一直展开浅层；先在原预算内形成 K 个完整候选。
            completing = positive_counts[user] < model.top_k_for_generation
            if completing:
                ordered = sorted(queue.items(), key=lambda pair: (depths[pair[0]], pair[1]), reverse=True)
            else:
                ordered = sorted(queue.items(), key=lambda pair: pair[1], reverse=True)
            for node, mass in ordered:
                cost = resolve_cost(node)
                if state_counts[user] >= model.max_states or used >= model.expansion_batch_size:
                    break
                if score_counts[user] + cost > model.max_item_scores:
                    continue
                selected.append((user, node, mass, state_counts[user], cost))
                state_counts[user] += 1
                score_counts[user] += cost
                used += 1
        if not selected:
            break
        for depth in sorted({depths[node] for _, node, _, _, _ in selected}):
            entries = [entry for entry in selected if depths[entry[1]] == depth]
            users = torch.tensor([entry[0] for entry in entries], device=device)
            nodes = torch.tensor([entry[1] for entry in entries], device=device)
            masses = torch.tensor([entry[2] for entry in entries], device=device)
            prefixes = catalog.prefixes[nodes, :depth]
            hidden = model.states(encoded[users], mask[users], prefixes)[:, -1]
            gates = model.gates(hidden, nodes, depth)
            if collect_trace:
                slots = torch.tensor([entry[3] for entry in entries], device=device)
                trace["expanded_node"][users, slots] = nodes
                trace["expanded_depth"][users, slots] = depth
                trace["arrival_mass"][users, slots] = masses
                trace["stop_gate"][users, slots] = gates
                trace["resolved_mass"][users, slots] = masses * gates
            if depth:
                resolving = gates > 0
                if resolving.any():
                    lp, item_ids, valid = model.resolve_log(hidden[resolving], nodes[resolving], depth, vectors)
                    values = masses[resolving, None] * gates[resolving, None] * lp.exp()
                    row_ids = users[resolving, None].expand_as(item_ids)
                    contributions[:, :, depth - 1].index_put_(
                        (row_ids[valid], item_ids[valid]), values[valid], accumulate=True
                    )
                    if collect_trace:
                        trace["resolved_count"][users[resolving], slots[resolving]] = valid.sum(1)
            continuing = gates < 1
            route_probs = {}
            if continuing.any():
                probs = model.route_log(hidden[continuing], prefixes[continuing]).exp().detach().cpu()
                route_probs = dict(zip(continuing.nonzero().flatten().tolist(), probs, strict=True))
            cpu_gates = gates.detach().cpu().tolist()
            for index, (user, node, mass, _, _) in enumerate(entries):
                del frontier[user][node]
                gate = cpu_gates[index]
                if gate < 1:
                    for token, child in catalog.child_nodes[node]:
                        child_mass = mass * (1 - gate) * float(route_probs[index][token])
                        if child_mass > 0:
                            frontier[user][child] = child_mass
    lower = contributions.sum(-1)
    scores, chosen = lower.topk(model.top_k_for_generation, dim=-1)
    if not (scores > 0).all():
        raise RuntimeError(
            "Resolution budget produced fewer than K positive unique items; increase max_states/max_item_scores."
        )
    if not collect_trace:
        return catalog.sids[chosen], scores, {}
    upper = _frontier_upper(catalog, lower, frontier)
    outside = upper.scatter(1, chosen, -torch.inf).max(-1).values
    trace.update(
        states_evaluated=torch.tensor(state_counts, device=device),
        items_scored=torch.tensor(score_counts, device=device),
        remaining_mass=torch.tensor([sum(queue.values()) for queue in frontier], device=device),
        resolved_total_mass=lower.sum(-1),
        topk_certified=scores[:, -1] > outside,
        probability_bound_valid=torch.ones(batch, dtype=torch.bool, device=device),
        topk_depth_contribution=contributions.gather(1, chosen[:, :, None].expand(-1, -1, model.semantic_depth)),
    )
    return catalog.sids[chosen], scores, trace


def _beam(model, encoded, mask, depth):
    batch, device = len(encoded), encoded.device
    width, k = model.num_embeddings_per_hierarchy, model.top_k_for_generation
    prefixes = torch.empty(batch, 1, 0, dtype=torch.long, device=device)
    scores = encoded.new_zeros((batch, 1), dtype=torch.float32)
    evaluated = 0
    for level in range(depth):
        beams = prefixes.shape[1]
        evaluated += beams
        flat = prefixes.reshape(batch * beams, level)
        hidden = model.states(encoded.repeat_interleave(beams, 0), mask.repeat_interleave(beams, 0), flat)[:, -1]
        local = model.route_log(hidden, flat).reshape(batch, beams, width)
        candidates = (scores[:, :, None] + local).flatten(1)
        scores, positions = candidates.topk(min(k, candidates.shape[1]), -1)
        parent = positions // width
        prefixes = torch.cat(
            (prefixes.gather(1, parent[:, :, None].expand(-1, -1, level)), (positions % width)[:, :, None]), -1
        )
        prefixes = torch.where(scores.isfinite()[:, :, None], prefixes, prefixes[:, :1])
    return prefixes, scores, evaluated


def _baseline(model, encoded, mask):
    batch, device = len(encoded), encoded.device
    trace = _empty_trace(model, batch, device)
    catalog, k = model.catalog, model.top_k_for_generation
    root = torch.empty(batch, 0, dtype=torch.long, device=device)
    if model.arm == "dense":
        hidden = model.states(encoded, mask, root)[:, -1]
        dense_log = F.log_softmax(model.query(hidden, 0) @ model.item_vectors().T / model.temperature, -1)
        scores, chosen = dense_log.topk(k, -1)
        trace["states_evaluated"][:] = 1
        trace["items_scored"][:] = len(catalog.keys)
        return catalog.sids[chosen], scores.exp(), trace
    depth = model.semantic_depth if model.arm == "cobra" else model.num_hierarchies
    prefixes, log_scores, states = _beam(model, encoded, mask, depth)
    trace["states_evaluated"][:] = states
    if model.arm in ("mask_ce", "token_content_init"):
        if not log_scores.isfinite().all():
            raise RuntimeError("Beam cannot produce K items.")
        return prefixes, log_scores.exp(), trace
    vectors = model.item_vectors()
    if model.arm == "cobra":
        # COBRA 适配：只对 sparse beam 下的 item 做 dense 解析和 BeamFusion。
        beams = prefixes.shape[1]
        flat = prefixes.flatten(0, 1)
        nodes = catalog.nodes(flat)
        hidden = model.states(encoded.repeat_interleave(beams, 0), mask.repeat_interleave(beams, 0), flat)[:, -1]
        local, item_ids, valid = model.resolve_log(hidden, nodes, depth, vectors)
        beam_probability = log_scores.softmax(-1)
        values = beam_probability.flatten()[:, None] * local.exp()
        all_scores = encoded.new_zeros((batch, len(catalog.keys)), dtype=torch.float32)
        rows = torch.arange(batch, device=device).repeat_interleave(beams)[:, None].expand_as(item_ids)
        finite = valid & log_scores.flatten().isfinite()[:, None]
        all_scores.index_put_((rows[finite], item_ids[finite]), values[finite], accumulate=True)
        scores, chosen = all_scores.topk(k, -1)
        if not (scores > 0).all():
            raise RuntimeError("COBRA sparse beam has fewer than K distinct items; increase beam_width.")
        trace["items_scored"] = valid.reshape(batch, beams, -1).sum((1, 2))
        trace["states_evaluated"] += beams
        return catalog.sids[chosen], scores, trace
    hidden = model.states(encoded, mask, root)[:, -1]
    dense_log = F.log_softmax(model.query(hidden, 0) @ vectors.T / model.temperature, -1)
    candidates = torch.cat((catalog.item_indices(prefixes), dense_log.topk(k, -1).indices), 1)
    count = candidates.shape[1]
    targets = catalog.sids[candidates].flatten(0, 1)
    hidden = model.states(encoded.repeat_interleave(count, 0), mask.repeat_interleave(count, 0), targets[:, :-1])
    likelihood = (
        torch.stack(
            [
                model.route_log(hidden[:, level], targets[:, :level])
                .gather(1, targets[:, level : level + 1])
                .squeeze(1)
                for level in range(model.num_hierarchies)
            ],
            1,
        )
        .sum(1)
        .reshape(batch, count)
    )
    mixture = torch.logaddexp(
        likelihood + math.log1p(-model.hybrid_weight), dense_log.gather(1, candidates) + math.log(model.hybrid_weight)
    )
    duplicates = (candidates[:, :, None] == candidates[:, None, :]) & torch.ones(
        count, count, dtype=torch.bool, device=device
    ).tril(-1)
    mixture.masked_fill_(duplicates.any(-1), -torch.inf)
    scores, positions = mixture.topk(k, -1)
    trace["items_scored"][:] = len(catalog.keys)
    trace["states_evaluated"] += 1 + count * model.num_hierarchies
    return catalog.sids[candidates.gather(1, positions)], scores.exp(), trace


@torch.no_grad()
def generate(model, encoded, mask, *, collect_trace=True):
    if collect_trace and encoded.is_cuda:
        torch.cuda.synchronize(encoded.device)
    started = time.perf_counter()
    if model.inference_policy == "wide":
        from .wide import wide_search

        result = wide_search(model, encoded, mask)
    elif model.arm in RESOLUTION_ARMS:
        result = _resolution(model, encoded, mask, collect_trace=collect_trace)
    else:
        result = _baseline(model, encoded, mask)
    if collect_trace and encoded.is_cuda:
        torch.cuda.synchronize(encoded.device)
    elapsed = time.perf_counter() - started
    if collect_trace:
        trace = result[2]
        trace["decoder_batch_seconds"] = torch.full((len(encoded),), elapsed, device=encoded.device)
        trace["batch_size"] = torch.full((len(encoded),), len(encoded), device=encoded.device, dtype=torch.long)
    model.catalog.item_indices(result[0])
    return result if collect_trace else (result[0], result[1], {})
