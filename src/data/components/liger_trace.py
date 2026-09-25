"""LIGER 候选记录的读取校验与已有输出汇总。"""

import torch


def validate_liger_trace(bundle):
    if bundle.get("schema_version") != "liger_candidates_v1":
        raise ValueError("Unknown LIGER trace schema.")
    keys, labels, t, m = (bundle[k] for k in ("keys", "labels", "trace", "metadata"))
    n = len(keys)
    if keys.ndim != 1 or keys.unique().numel() != n or not n or labels.ndim != 2 or len(labels) != n:
        raise ValueError("Invalid or duplicate trace keys/labels.")
    fields = {
        "generated_rows",
        "generated_unique_count",
        "invalid_generated_count",
        "candidate_count",
        "target_row",
        "target_cold",
        "target_generated",
        "target_covered",
        "dense_rank",
        "hybrid_rank",
        "dense_topk_rows",
        "hybrid_topk_sids",
    }
    if set(t) != fields or any(not isinstance(v, torch.Tensor) or len(v) != n for v in t.values()):
        raise ValueError("Incomplete trace fields.")
    if t["generated_rows"].shape != (n, m["generation_candidates"]) or t["dense_topk_rows"].shape != (n, m["top_k"]):
        raise ValueError("Invalid trace candidate dimensions.")
    if t["hybrid_topk_sids"].shape != (n, m["top_k"], labels.shape[1]):
        raise ValueError("Invalid trace SID dimensions.")
    if ((t["dense_rank"] < 1) | (t["dense_rank"] > m["catalog_size"])).any():
        raise ValueError("Invalid dense rank.")
    if not torch.equal(t["target_covered"], t["hybrid_rank"] > 0):
        raise ValueError("Coverage/rank mismatch.")
    if not torch.equal(t["target_covered"], t["target_cold"] | t["target_generated"]):
        raise ValueError("Hybrid union coverage mismatch.")
    if ((t["hybrid_rank"] < 0) | (t["hybrid_rank"] > t["candidate_count"])).any():
        raise ValueError("Invalid candidate rank.")
    if (t["hybrid_rank"][t["target_covered"]] > t["dense_rank"][t["target_covered"]]).any():
        raise ValueError("Subset rank cannot exceed full-catalog rank.")
    generated = t["generated_rows"]
    if not torch.equal(t["target_generated"], (generated == t["target_row"][:, None]).any(-1)):
        raise ValueError("Generated target membership mismatch.")
    if not torch.equal(t["invalid_generated_count"], (generated < 0).sum(-1)):
        raise ValueError("Invalid generation count mismatch.")
    for i, row in enumerate(generated):
        if int(t["generated_unique_count"][i]) != row[row >= 0].unique().numel():
            raise ValueError("Unique generated count mismatch.")
    hit = (t["hybrid_topk_sids"] == labels[:, None]).all(-1)
    expected = (t["hybrid_rank"] > 0) & (t["hybrid_rank"] <= m["top_k"])
    if not torch.equal(hit.any(-1), expected):
        raise ValueError("Hybrid output/rank mismatch.")
    if expected.any() and not torch.equal(hit.long().argmax(-1)[expected] + 1, t["hybrid_rank"][expected]):
        raise ValueError("Hybrid output position mismatch.")
    if ((t["generated_rows"] < -1) | (t["generated_rows"] >= m["catalog_size"])).any():
        raise ValueError("Invalid generated row.")


def summarize_liger_trace(bundle):
    validate_liger_trace(bundle)
    t = bundle["trace"]
    d = t["dense_rank"] <= 10
    h = (t["hybrid_rank"] > 0) & (t["hybrid_rank"] <= 10)
    result = {
        "users": len(bundle["keys"]),
        "target_coverage": t["target_covered"].double().mean().item(),
        "dense_only10": int((d & ~h).sum()),
        "hybrid_only10": int((h & ~d).sum()),
        "dense_only10_target_missing": int((d & ~h & ~t["target_covered"]).sum()),
        "mean_candidates": t["candidate_count"].double().mean().item(),
        "mean_valid_generated_unique": t["generated_unique_count"].double().mean().item(),
        "mean_invalid_generated": t["invalid_generated_count"].double().mean().item(),
    }
    for name in ("dense", "hybrid"):
        rank = t[name + "_rank"]
        for k in (5, 10):
            hit = (rank > 0) & (rank <= k)
            result[f"{name}/recall@{k}"] = hit.double().mean().item()
            result[f"{name}/ndcg@{k}"] = (
                torch.where(hit, 1 / torch.log2(rank.double().clamp_min(1) + 1), 0).mean().item()
            )
    return result


def compare_liger_traces(reference, candidate, *, bootstrap_samples=1000, seed=42):
    """按用户配对比较已完成产物；区间只反映当前开发集用户采样不确定性。"""
    for bundle in (reference, candidate):
        validate_liger_trace(bundle)
    for field in ("catalog_sha256", "catalog_size", "cold_count", "top_k", "generation_candidates"):
        if reference["metadata"][field] != candidate["metadata"][field]:
            raise ValueError(f"Mismatched comparison metadata: {field}")
    ri, ci = reference["keys"].argsort(), candidate["keys"].argsort()
    if not torch.equal(reference["keys"][ri], candidate["keys"][ci]):
        raise ValueError("Comparison user sets differ.")
    if not torch.equal(reference["labels"][ri], candidate["labels"][ci]):
        raise ValueError("Comparison target labels differ.")
    r = {k: v[ri].cpu() for k, v in reference["trace"].items()}
    c = {k: v[ci].cpu() for k, v in candidate["trace"].items()}
    for field in ("dense_rank", "dense_topk_rows", "target_row", "target_cold"):
        if not torch.equal(r[field], c[field]):
            raise ValueError(f"Shared dense/reference inputs changed: {field}")
    rh = (r["hybrid_rank"] > 0) & (r["hybrid_rank"] <= 10)
    ch = (c["hybrid_rank"] > 0) & (c["hybrid_rank"] <= 10)
    dh = r["dense_rank"] <= 10
    out = {
        "users": len(ri),
        "new_hits10": int((ch & ~rh).sum()),
        "lost_hits10": int((rh & ~ch).sum()),
        "net_hits10": int(ch.sum() - rh.sum()),
        "reference_dense_only10_count": int((dh & ~rh).sum()),
        "reference_dense_only10_recovered": int((dh & ~rh & ch).sum()),
        "reference_hybrid_only10_count": int((rh & ~dh).sum()),
        "reference_hybrid_only10_retained": int((rh & ~dh & ch).sum()),
        "target_coverage_delta": float(c["target_covered"].double().mean() - r["target_covered"].double().mean()),
    }
    deltas = []
    names = []
    for k in (5, 10):
        for metric in ("recall", "ndcg"):

            def values(rank, k=k, metric=metric):
                hit = (rank > 0) & (rank <= k)
                return (
                    hit.double()
                    if metric == "recall"
                    else torch.where(hit, 1 / torch.log2(rank.double().clamp_min(1) + 1), 0)
                )

            names.append(f"{metric}@{k}")
            deltas.append(values(c["hybrid_rank"]) - values(r["hybrid_rank"]))
    delta = torch.stack(deltas, dim=1)
    if bootstrap_samples < 1:
        raise ValueError("Positive bootstrap_samples required.")
    generator = torch.Generator().manual_seed(seed)
    samples = []
    for start in range(0, bootstrap_samples, 32):
        indices = torch.randint(len(ri), (min(32, bootstrap_samples - start), len(ri)), generator=generator)
        samples.append(delta[indices].mean(1))
    bounds = torch.quantile(torch.cat(samples), torch.tensor([0.025, 0.975], dtype=torch.double), dim=0)
    out["paired_metrics"] = {
        name: {"delta": float(delta[:, j].mean()), "ci95": bounds[:, j].tolist()} for j, name in enumerate(names)
    }
    out["bootstrap_samples"] = bootstrap_samples
    out["bootstrap_seed"] = seed
    return out
