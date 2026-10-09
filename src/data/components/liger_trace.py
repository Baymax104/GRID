"""LIGER 候选记录的读取校验与已有输出汇总。"""

import math

import torch


def validate_liger_trace(bundle):
    if bundle.get("schema_version") != "liger_candidates_v1":
        raise ValueError("Unknown LIGER trace schema.")
    keys, labels, t, m = (bundle[k] for k in ("keys", "labels", "trace", "metadata"))
    n = len(keys)
    final_score = m.get("final_score", "content")
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
    if final_score == "content_plus_scaled_decoder_relevance":
        fields |= {"content_scale", "hybrid_content_rank", "hybrid_topk_content_scores", "hybrid_topk_relevance_scores"}
    if final_score == "mixed_full_sid_log_probability":
        fields |= {
            "candidate_rows", "candidate_sids", "candidate_content_scores", "candidate_mixed_scores",
            "hybrid_content_rank", "content_topk_sids",
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
    if final_score == "content_plus_candidate_conditioned_relevance":
        version = m.get("copmrec_version")
        contract = m.get("copmrec_relevance", {})
        versions = {"v1": "copmrec-relevance-v1", "v1.1": "copmrec-relevance-v1.1"}
        if version not in versions or contract.get("version") != versions[version]:
            raise ValueError("Relevance trace requires a matching CoPMRec relevance contract.")
        if version == "v1.1":
            chunk = contract.get("candidate_chunk_size")
            if (
                contract.get("scoring_execution") != "batched-user-candidates-v1"
                or isinstance(chunk, bool)
                or not isinstance(chunk, int)
                or chunk < 1
            ):
                raise ValueError("Batched relevance trace requires a valid scoring contract.")
    elif final_score == "content_plus_scaled_decoder_relevance":
        _validate_scaled_relevance_trace(t, m, n)
    elif final_score == "mixed_full_sid_log_probability":
        _validate_v4_mixture_trace(t, m, labels)
    elif final_score != "content":
        raise ValueError("Unknown candidate final score.")
    # 只有同content分数排序才有子集排名上界；v1最终相关性与dense诊断分数不同。
    if (
        final_score == "content"
        and (t["hybrid_rank"][t["target_covered"]] > t["dense_rank"][t["target_covered"]]).any()
    ):
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


def _validate_v4_mixture_trace(trace, metadata, labels):
    required = dict(
        version="copmrec-v4-full-sid-mixture-ranking-v1",
        mode="mixed",
        score_scope="full_catalog_legal_conditionals_fixed_length_sid",
        aggregation="mass",
        gate_source="checkpoint",
        rank_ties="stable_catalog_row",
    )
    contract = metadata.get("copmrec_final_ranking")
    if (
        not isinstance(contract, dict)
        or set(contract) != set(required) | {"chunk_size", "cold_rows"}
        or any(contract.get(key) != value for key, value in required.items())
        or metadata.get("copmrec_version") != "v4"
        or metadata.get("joint_mixture_protocol") != "liger-joint-mixture-v1"
        or metadata.get("mechanism_control") != "learned_mass"
        or metadata.get("prefix_aggregation") != "mass"
        or metadata.get("mixture_alpha_source") != "checkpoint"
        or metadata.get("fixed_mixture_alpha") is not None
        or metadata.get("candidate_strategy") != "probability_mixture"
        or metadata.get("candidate_normalization") != "legal_conditional"
    ):
        raise ValueError("Mixed final ranking requires a matching v4 contract.")
    chunk = contract["chunk_size"]
    alpha = metadata.get("content_mixture_alpha")
    if (
        isinstance(chunk, bool) or not isinstance(chunk, int) or chunk < 1
        or isinstance(alpha, bool) or not isinstance(alpha, (int, float))
        or not math.isfinite(alpha) or not 0 <= alpha <= 1
    ):
        raise ValueError("Invalid v4 mixture scoring contract.")
    cold = contract["cold_rows"]
    if (
        not isinstance(cold, list)
        or any(isinstance(row, bool) or not isinstance(row, int) or not 0 <= row < metadata["catalog_size"] for row in cold)
        or cold != sorted(set(cold)) or len(cold) != metadata["cold_count"]
    ):
        raise ValueError("Invalid v4 cold catalog metadata.")
    users, depth = labels.shape
    width = metadata["generation_candidates"] + metadata["cold_count"]
    rows, sids = trace["candidate_rows"], trace["candidate_sids"]
    if (
        rows.shape != (users, width) or sids.shape != (users, width, depth)
        or rows.dtype != torch.long or sids.dtype != torch.long
        or ((rows < -1) | (rows >= metadata["catalog_size"])).any()
    ):
        raise ValueError("Invalid v4 candidate dimensions or rows.")
    for field in ("target_row", "target_cold", "candidate_count", "target_covered", "hybrid_rank", "hybrid_content_rank"):
        if trace[field].shape != (users,):
            raise ValueError("Invalid v4 target/rank dimensions.")
    if ((trace["target_row"] < 0) | (trace["target_row"] >= metadata["catalog_size"])).any():
        raise ValueError("Invalid v4 target row.")
    valid = rows >= 0
    if not (sids[~valid] == -1).all() or (sids[valid] < 0).any():
        raise ValueError("Invalid v4 candidate SID padding.")
    covered = rows == trace["target_row"][:, None]
    if not torch.equal(covered.any(1), trace["target_covered"]):
        raise ValueError("V4 candidate coverage mismatch.")
    if not torch.equal((sids == labels[:, None]).all(-1), covered):
        raise ValueError("V4 candidate target SID mismatch.")
    if not torch.equal(valid.sum(1), trace["candidate_count"]):
        raise ValueError("V4 candidate count mismatch.")
    for user in range(users):
        values = rows[user, valid[user]].tolist()
        generated = trace["generated_rows"][user].tolist()
        expected = sorted(set(row for row in generated if row >= 0) | set(cold))
        if values != expected or not valid[user, : len(values)].all():
            raise ValueError("V4 candidate union or tie ordering changed.")
        if sids[user, valid[user]].unique(dim=0).shape[0] != len(values):
            raise ValueError("Duplicate v4 candidate SID.")
        if bool(trace["target_cold"][user]) != (int(trace["target_row"][user]) in cold):
            raise ValueError("V4 target cold metadata mismatch.")
    for scores_name, rank_name, top_name in (
        ("candidate_content_scores", "hybrid_content_rank", "content_topk_sids"),
        ("candidate_mixed_scores", "hybrid_rank", "hybrid_topk_sids"),
    ):
        scores, rank, top = (trace[name] for name in (scores_name, rank_name, top_name))
        if (
            scores.shape != rows.shape or not scores.is_floating_point()
            or top.shape != (users, metadata["top_k"], depth)
            or not torch.isfinite(scores[valid]).all() or not torch.isneginf(scores[~valid]).all()
        ):
            raise ValueError("Invalid v4 ranking scores or padding.")
        if scores_name == "candidate_mixed_scores" and (scores[valid] > 1e-6).any():
            raise ValueError("V4 complete SID log probability must be nonpositive.")
        order = scores.argsort(dim=1, descending=True, stable=True)
        hits = rows.gather(1, order) == trace["target_row"][:, None]
        expected_rank = torch.where(hits.any(1), hits.long().argmax(1) + 1, 0)
        if not torch.equal(rank, expected_rank):
            raise ValueError("V4 ranking target rank mismatch.")
        expected_top = sids.gather(1, order[..., None].expand_as(sids))[:, : metadata["top_k"]]
        if expected_top.shape[1] < metadata["top_k"]:
            expected_top = torch.nn.functional.pad(
                expected_top, (0, 0, 0, metadata["top_k"] - expected_top.shape[1]), value=-1
            )
        if not torch.equal(top, expected_top):
            raise ValueError("V4 ranking TopK does not match scores.")
    if (trace["hybrid_content_rank"][trace["target_covered"]] > trace["dense_rank"][trace["target_covered"]]).any():
        raise ValueError("V4 content subset rank cannot exceed full-catalog rank.")


def _validate_scaled_relevance_trace(trace, metadata, users):
    contract = metadata.get("copmrec_relevance", {})
    required = dict(
        version="copmrec-relevance-v2",
        head_input="decoder_after_complete_sid",
        head_normalization="layer_norm",
        score_rule="content_plus_natural_scale_tanh",
        scale_scope="beam_plus_all_cold",
        scale_variance="population",
        scoring_execution="batched-user-candidates-v1",
    )
    if (
        metadata.get("copmrec_version") != "v2"
        or any(contract.get(key) != value for key, value in required.items())
        or contract.get("scale_stop_gradient") is not True
    ):
        raise ValueError("Scaled relevance trace requires a matching v2 contract.")
    for key in ("hidden_dim", "candidate_chunk_size", "ranking_examples_per_microbatch", "training_content_candidates"):
        value = contract.get(key)
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError("Invalid v2 relevance dimension contract.")
    for key in ("scale_epsilon", "content_temperature", "beta", "loss_weight"):
        value = contract.get(key)
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            or value < 0
            or (key != "beta" and value == 0)
        ):
            raise ValueError("Invalid v2 relevance scale/loss contract.")
    scale = trace["content_scale"]
    content, relevance = trace["hybrid_topk_content_scores"], trace["hybrid_topk_relevance_scores"]
    if scale.shape != (users,) or content.shape != (users, metadata["top_k"]) or relevance.shape != content.shape:
        raise ValueError("Invalid v2 relevance score dimensions.")
    if (
        not torch.isfinite(scale).all()
        or not torch.isfinite(content).all()
        or not torch.isfinite(relevance).all()
        or (scale < contract["scale_epsilon"] * (1 - 1e-5)).any()
        or (relevance.abs() > contract["beta"] * scale[:, None] + 1e-6).any()
    ):
        raise ValueError("Invalid or unbounded v2 relevance scores.")
    rank = trace["hybrid_content_rank"]
    if (
        rank.shape != (users,)
        or ((rank < 0) | (rank > trace["candidate_count"])).any()
        or not torch.equal(rank > 0, trace["target_covered"])
    ):
        raise ValueError("Invalid v2 candidate content ranks.")


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
