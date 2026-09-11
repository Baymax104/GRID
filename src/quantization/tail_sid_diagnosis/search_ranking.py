"""Search/ranking decomposition and static-risk standardized prefix evidence."""

from __future__ import annotations

import math
import random
from collections import defaultdict
from dataclasses import dataclass
from typing import Any

import torch

from src.data.components.data_models import DiagnosisBatch, PrefixTraceBundle, RecommendationOutcomeInput

ANALYSIS_SCHEMA_VERSION = "tail_search_ranking_v1"
ESTIMATOR_VERSION = "static_risk_overlap_v1"
GROUPS = ("All", "Head", "Mid", "Tail", "Tail-Cold", "Tail+Tail-Cold")
TRANSITIONS = (
    "hit_to_top10",
    "hit_to_below10",
    "hit_to_absent",
    "miss_to_top10",
    "miss_to_below10",
    "miss_to_absent",
)


@dataclass(frozen=True)
class SearchRankingEvidence:
    summary: dict[str, Any]
    scalar_section: dict[str, float]
    tables: dict[str, list[dict[str, Any]]]


def build_search_ranking_evidence(
    batch: DiagnosisBatch,
    item_rows: list[dict[str, Any]],
    *,
    search_ranking_enabled: bool,
    risk_standardization_enabled: bool,
    bin_count: int,
    bin_count_sensitivity: list[int] | tuple[int, ...],
    min_items_per_group_per_bin: int,
    min_item_retention: float,
    max_abs_raw_damage_smd: float,
    bootstrap_samples: int,
    bootstrap_confidence: float,
    min_bootstrap_valid_fraction: float,
    seed: int,
) -> SearchRankingEvidence:
    if not search_ranking_enabled and not risk_standardization_enabled:
        return SearchRankingEvidence(
            summary={"available": False, "reason": "analysis_disabled"},
            scalar_section={"available": 0.0},
            tables={},
        )
    values = (
        batch.recommendation,
        batch.widened_recommendation,
        batch.fixed_prefix_trace,
        batch.widened_prefix_trace,
    )
    if any(value is None for value in values):
        raise ValueError("Search-ranking analysis requires fixed/widened recommendation and Prefix Trace inputs.")
    fixed, widened, fixed_trace, widened_trace = values
    assert isinstance(fixed, RecommendationOutcomeInput)
    assert isinstance(widened, RecommendationOutcomeInput)
    assert isinstance(fixed_trace, PrefixTraceBundle)
    assert isinstance(widened_trace, PrefixTraceBundle)

    users, identity = _paired_user_rows(batch, item_rows, fixed, widened, fixed_trace, widened_trace)
    tables: dict[str, list[dict[str, Any]]] = {}
    summary: dict[str, Any] = {
        "available": True,
        "analysis_schema_version": ANALYSIS_SCHEMA_VERSION,
        "estimator_version": ESTIMATOR_VERSION,
        "data_unit": "evaluation_user",
        "exploratory": True,
        "identity": identity,
        "legacy_prefix_statistics": {
            "status": "legacy",
            "note": "Legacy matched estimate and cluster interval use different estimands.",
        },
    }
    group_rows: list[dict[str, Any]] = []
    if search_ranking_enabled:
        group_rows = _group_search_rows(users)
        tables.update(
            {
                "search_ranking_by_user.csv": users,
                "search_ranking_by_group.csv": group_rows,
                "prefix_attrition_by_layer.csv": _attrition_rows(users, fixed_trace),
                "frequency_attrition_descriptives.csv": _frequency_rows(users, fixed_trace),
            }
        )
        summary["search_ranking"] = {
            "available": True,
            "fixed_width": int(fixed.generated_sids.size(1)),
            "widened_width": int(widened.generated_sids.size(1)),
            "transition_counts_conserved": all(
                sum(int(row[name]) for name in TRANSITIONS) == int(row["support"])
                for row in group_rows
            ),
        }
    if risk_standardization_enabled:
        static_tables, static_summary = _static_risk_evidence(
            users,
            fixed_trace,
            bin_count=bin_count,
            sensitivities=bin_count_sensitivity,
            min_support=min_items_per_group_per_bin,
            min_retention=min_item_retention,
            max_abs_smd=max_abs_raw_damage_smd,
            bootstrap_samples=bootstrap_samples,
            confidence=bootstrap_confidence,
            min_valid_fraction=min_bootstrap_valid_fraction,
            seed=seed,
        )
        tables.update(static_tables)
        summary["risk_standardization"] = static_summary

    all_row = next((row for row in group_rows if row["group"] == "All"), None)
    tail_row = next((row for row in group_rows if row["group"] == "Tail"), None)
    return SearchRankingEvidence(
        summary=summary,
        scalar_section={
            "available": 1.0,
            "overall_widened_hit10": float(all_row["widened_hit10"]) if all_row else 0.0,
            "tail_widened_hit10": float(tail_row["widened_hit10"]) if tail_row else 0.0,
            "tail_oracle_headroom": float(tail_row["oracle_headroom"]) if tail_row else 0.0,
        },
        tables=tables,
    )


def _wandb_run_id(reference: Any) -> str | None:
    if not isinstance(reference, str) or not reference.startswith("wandb://"):
        return None
    return reference.split("?", 1)[0].rstrip("/").rsplit("/", 1)[-1]


def _rank(candidates: torch.Tensor, label: torch.Tensor) -> int | None:
    matches = torch.all(candidates == label, dim=1).nonzero().reshape(-1)
    return int(matches[0].item()) + 1 if matches.numel() else None


def _paired_user_rows(
    batch: DiagnosisBatch,
    item_rows: list[dict[str, Any]],
    fixed: RecommendationOutcomeInput,
    widened: RecommendationOutcomeInput,
    fixed_trace: PrefixTraceBundle,
    widened_trace: PrefixTraceBundle,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if int(widened.generated_sids.size(1)) <= int(fixed.generated_sids.size(1)):
        raise ValueError("Widened recommendation width must exceed fixed recommendation width.")
    if fixed.generated_sids.size(2) != batch.sid_views.model_sid.size(1):
        raise ValueError("Fixed recommendation SID width does not match the model SID catalog.")
    if widened.generated_sids.size(2) != batch.sid_views.model_sid.size(1):
        raise ValueError("Widened recommendation SID width does not match the model SID catalog.")

    catalog = {
        tuple(int(value) for value in sid): row
        for sid, row in zip(batch.sid_views.model_sid.tolist(), item_rows, strict=True)
    }
    if len(catalog) != len(item_rows):
        raise ValueError("Model semantic IDs must uniquely identify items for paired analysis.")
    catalog_sids = set(catalog)

    orders = [tensor.argsort() for tensor in (fixed.user_ids, widened.user_ids, fixed_trace.keys, widened_trace.keys)]
    keys = [tensor[index] for tensor, index in zip(
        (fixed.user_ids, widened.user_ids, fixed_trace.keys, widened_trace.keys), orders, strict=True
    )]
    if not all(torch.equal(keys[0], other) for other in keys[1:]):
        raise ValueError("Paired recommendation and Prefix Trace user keys must match exactly.")
    fixed_labels = fixed_trace.labels[orders[2]]
    widened_labels = widened_trace.labels[orders[3]]
    if not torch.equal(fixed_labels, widened_labels):
        raise ValueError("Fixed and widened Prefix Trace labels must match exactly.")
    if not torch.equal(fixed.label_item_ids[orders[0]], widened.label_item_ids[orders[1]]):
        raise ValueError("Fixed and widened recommendation label item IDs must match exactly.")

    fixed_candidates = fixed.generated_sids[orders[0]]
    widened_candidates = widened.generated_sids[orders[1]]
    invalid = {
        tuple(int(value) for value in sid)
        for candidates in (fixed_candidates, widened_candidates)
        for sid in candidates.reshape(-1, candidates.size(-1)).tolist()
        if tuple(int(value) for value in sid) not in catalog_sids
    }
    if invalid:
        raise ValueError(f"Recommendation output contains SIDs absent from the model catalog: {list(invalid)[:3]}.")

    fixed_survival = fixed_trace.trace["target_prefix_survived"][orders[2]]
    widened_survival = widened_trace.trace["target_prefix_survived"][orders[3]]
    fixed_beam_rank = fixed_trace.trace["target_beam_rank"][orders[2], -1]
    widened_beam_rank = widened_trace.trace["target_beam_rank"][orders[3], -1]
    rows = []
    for index, user_id in enumerate(keys[0].tolist()):
        label = tuple(int(value) for value in fixed_labels[index].tolist())
        if label not in catalog:
            raise KeyError(f"Prefix Trace target SID is absent from the model catalog: {label}.")
        item = catalog[label]
        label_item = int(fixed.label_item_ids[orders[0][index]].item())
        if label_item != int(item["item_id"]):
            raise ValueError("Recommendation label item does not match Prefix Trace target SID.")
        fixed_rank = _rank(fixed_candidates[index], fixed_labels[index])
        widened_rank = _rank(widened_candidates[index], fixed_labels[index])
        expected_fixed_rank = int(fixed_beam_rank[index].item())
        expected_widened_rank = int(widened_beam_rank[index].item())
        if (fixed_rank or -1) != expected_fixed_rank or (widened_rank or -1) != expected_widened_rank:
            raise ValueError("Recommendation target rank does not match Prefix Trace final beam rank.")
        if (fixed_rank is not None) != bool(fixed_survival[index, -1]) or (
            (widened_rank is not None) != bool(widened_survival[index, -1])
        ):
            raise ValueError("Recommendation target membership does not match Prefix Trace survival.")
        fixed_hit = fixed_rank is not None and fixed_rank <= 10
        widened_state = "absent" if widened_rank is None else ("top10" if widened_rank <= 10 else "below10")
        transition = f"{'hit' if fixed_hit else 'miss'}_to_{widened_state}"
        rows.append(
            {
                "user_id": int(user_id),
                "item_id": int(item["item_id"]),
                "group": str(item["group"]),
                "frequency": int(item["freq_train"]),
                "raw_damage": float(item["raw_damage"]),
                "model_sid": " ".join(str(value) for value in label),
                "fixed_rank": fixed_rank,
                "widened_rank": widened_rank,
                "fixed_hit10": int(fixed_hit),
                "widened_hit10": int(widened_rank is not None and widened_rank <= 10),
                "widened_access": int(widened_rank is not None),
                "transition": transition,
                "identity_verified": False,
                "trace_index": int(orders[2][index].item()),
            }
        )

    refs = batch.input_metadata
    artifact_identity = refs.get("resolved_artifact_identity", {})
    fixed_run = _wandb_run_id(refs.get("recommendation_output_reference"))
    widened_run = _wandb_run_id(refs.get("widened_recommendation_output_reference"))
    fixed_trace_run = _wandb_run_id(refs.get("fixed_prefix_trace_path"))
    widened_trace_run = _wandb_run_id(refs.get("widened_prefix_trace_path"))
    source_runs_verified = bool(
        fixed_run and widened_run and fixed_run == fixed_trace_run and widened_run == widened_trace_run
    )
    expected_fields = {
        "semantic_id_path",
        "recommendation_output_path",
        "widened_recommendation_output_path",
        "fixed_prefix_trace_path",
        "widened_prefix_trace_path",
    }
    artifact_identity_verified = bool(
        isinstance(artifact_identity, dict)
        and expected_fields.issubset(artifact_identity)
        and all(
            artifact_identity[field].get("artifact_name")
            and artifact_identity[field].get("artifact_version")
            and artifact_identity[field].get("resolved_path")
            for field in expected_fields
        )
    )
    identity_verified = source_runs_verified and artifact_identity_verified
    if all((fixed_run, fixed_trace_run)) and fixed_run != fixed_trace_run:
        raise ValueError("Fixed recommendation and Prefix Trace lineage conflict.")
    if all((widened_run, widened_trace_run)) and widened_run != widened_trace_run:
        raise ValueError("Widened recommendation and Prefix Trace lineage conflict.")
    for row in rows:
        row["identity_verified"] = identity_verified
    return rows, {
        "identity_verified": identity_verified,
        "fixed_source_run": fixed_run,
        "widened_source_run": widened_run,
        "fixed_trace_source_run": fixed_trace_run,
        "widened_trace_source_run": widened_trace_run,
        "source_runs_verified": source_runs_verified,
        "artifact_identity_verified": artifact_identity_verified,
        "artifacts": artifact_identity,
        "reason": None if identity_verified else "source_identity_unverified",
    }


def _select_group(users: list[dict[str, Any]], group: str) -> list[dict[str, Any]]:
    if group == "All":
        return users
    if group == "Tail+Tail-Cold":
        return [row for row in users if row["group"] in {"Tail", "Tail-Cold"}]
    return [row for row in users if row["group"] == group]


def _ratio(numerator: int | float, denominator: int | float) -> float | None:
    return float(numerator) / float(denominator) if denominator else None


def _ndcg(rank: int | None, cutoff: int = 10) -> float:
    return 1.0 / math.log2(rank + 1) if rank is not None and rank <= cutoff else 0.0


def _group_search_rows(users: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result = []
    for group in GROUPS:
        selected = _select_group(users, group)
        support = len(selected)
        counts = {name: sum(row["transition"] == name for row in selected) for name in TRANSITIONS}
        fixed_hits = sum(int(row["fixed_hit10"]) for row in selected)
        widened_hits = sum(int(row["widened_hit10"]) for row in selected)
        widened_access = sum(int(row["widened_access"]) for row in selected)
        fixed_misses = support - fixed_hits
        recovered = counts["miss_to_top10"] + counts["miss_to_below10"]
        row = {
            "group": group,
            "support": support,
            "fixed_hit10_count": fixed_hits,
            "fixed_hit10": _ratio(fixed_hits, support),
            "fixed_ndcg10": _ratio(sum(_ndcg(row["fixed_rank"]) for row in selected), support),
            "widened_hit10_count": widened_hits,
            "widened_hit10": _ratio(widened_hits, support),
            "widened_ndcg10": _ratio(sum(_ndcg(row["widened_rank"]) for row in selected), support),
            "widened_access_count": widened_access,
            "widened_access_rate": _ratio(widened_access, support),
            "oracle_hit10_ceiling": _ratio(widened_access, support),
            "oracle_headroom": _ratio(widened_access - widened_hits, support),
            "fixed_miss_count": fixed_misses,
            "k10_miss_recovered_count": recovered,
            "k10_miss_recovery_rate": _ratio(recovered, fixed_misses),
            "top10_added": counts["miss_to_top10"],
            "top10_lost": counts["hit_to_below10"] + counts["hit_to_absent"],
            "top10_net_change": widened_hits - fixed_hits,
        }
        row.update(counts)
        result.append(row)
    return result


def _finite_mean(values: torch.Tensor) -> float | None:
    finite = values[torch.isfinite(values)]
    return float(finite.float().mean().item()) if finite.numel() else None


def _attrition_rows(
    users: list[dict[str, Any]], fixed_trace: PrefixTraceBundle
) -> list[dict[str, Any]]:
    result = []
    trace = fixed_trace.trace
    depth_count = fixed_trace.labels.size(1)
    for group in GROUPS:
        selected = _select_group(users, group)
        indexes = torch.tensor([int(row["trace_index"]) for row in selected], dtype=torch.long)
        for layer in range(depth_count):
            support = len(selected)
            if support:
                survived = trace["target_prefix_survived"][indexes, layer]
                at_risk = (
                    torch.ones(support, dtype=torch.bool)
                    if layer == 0
                    else trace["target_prefix_survived"][indexes, layer - 1]
                )
                failed_here = at_risk & ~survived
                first_failure = trace["first_failure_depth"][indexes] == layer + 1
                failure_indexes = indexes[first_failure]
                teacher_ranks = trace["teacher_legal_rank"][failure_indexes, layer].float()
                parent_ranks = trace["target_parent_beam_rank"][failure_indexes, layer].float()
                margins = trace["cutoff_margin"][failure_indexes, layer]
                branching = trace["legal_candidate_count"][indexes, layer].float()
            else:
                survived = at_risk = failed_here = torch.empty(0, dtype=torch.bool)
                teacher_ranks = parent_ranks = margins = branching = torch.empty(0)
            valid_teacher_ranks = teacher_ranks[teacher_ranks >= 1]
            valid_parent_ranks = parent_ranks[parent_ranks >= 1]
            valid_margins = margins[torch.isfinite(margins)]
            result.append(
                {
                    "group": group,
                    "layer": layer + 1,
                    "group_support": support,
                    "survived_count": int(survived.sum().item()),
                    "cumulative_survival_rate": _ratio(int(survived.sum().item()), support),
                    "at_risk_count": int(at_risk.sum().item()),
                    "failed_at_layer_count": int(failed_here.sum().item()),
                    "conditional_failure_rate": _ratio(
                        int(failed_here.sum().item()), int(at_risk.sum().item())
                    ),
                    "first_failure_count": int(teacher_ranks.numel()),
                    "first_failure_teacher_legal_rank_mean": _finite_mean(valid_teacher_ranks),
                    "first_failure_teacher_rank_valid_count": int(valid_teacher_ranks.numel()),
                    "first_failure_teacher_rank_missing_count": int(
                        teacher_ranks.numel() - valid_teacher_ranks.numel()
                    ),
                    "first_failure_parent_beam_rank_mean": _finite_mean(valid_parent_ranks),
                    "first_failure_parent_rank_valid_count": int(valid_parent_ranks.numel()),
                    "first_failure_parent_rank_missing_count": int(
                        parent_ranks.numel() - valid_parent_ranks.numel()
                    ),
                    "first_failure_cutoff_margin_mean": _finite_mean(valid_margins),
                    "first_failure_margin_valid_count": int(valid_margins.numel()),
                    "first_failure_margin_missing_count": int(margins.numel() - valid_margins.numel()),
                    "catalog_branching_proxy_mean": _finite_mean(branching),
                }
            )
        full_survival = sum(
            int(fixed_trace.trace["first_failure_depth"][int(row["trace_index"])].item()) == -1
            for row in selected
        )
        result.append(
            {
                "group": group,
                "layer": "full_survival",
                "group_support": len(selected),
                "survived_count": full_survival,
                "cumulative_survival_rate": _ratio(full_survival, len(selected)),
                "at_risk_count": None,
                "failed_at_layer_count": None,
                "conditional_failure_rate": None,
                "first_failure_count": 0,
                "first_failure_teacher_legal_rank_mean": None,
                "first_failure_teacher_rank_valid_count": 0,
                "first_failure_teacher_rank_missing_count": 0,
                "first_failure_parent_beam_rank_mean": None,
                "first_failure_parent_rank_valid_count": 0,
                "first_failure_parent_rank_missing_count": 0,
                "first_failure_cutoff_margin_mean": None,
                "first_failure_margin_valid_count": 0,
                "first_failure_margin_missing_count": 0,
                "catalog_branching_proxy_mean": None,
            }
        )
    return result


def _frequency_rows(
    users: list[dict[str, Any]], fixed_trace: PrefixTraceBundle, bin_count: int = 5
) -> list[dict[str, Any]]:
    if not users:
        return []
    values = torch.tensor([math.log1p(int(row["frequency"])) for row in users])
    edges = torch.unique(torch.quantile(values, torch.linspace(0, 1, bin_count + 1)))
    bins = torch.bucketize(values, edges[1:-1], right=True)
    result = []
    for group in GROUPS:
        for frequency_bin in range(max(1, len(edges) - 1)):
            selected = []
            for index, row in enumerate(users):
                group_matches = (
                    group == "All"
                    or row["group"] == group
                    or (group == "Tail+Tail-Cold" and row["group"] in {"Tail", "Tail-Cold"})
                )
                if int(bins[index].item()) == frequency_bin and group_matches:
                    selected.append(row)
            indexes = torch.tensor([int(row["trace_index"]) for row in selected], dtype=torch.long)
            for layer in range(fixed_trace.labels.size(1)):
                if len(selected):
                    survival = fixed_trace.trace["target_prefix_survived"][indexes, layer]
                    at_risk = (
                        torch.ones(len(selected), dtype=torch.bool)
                        if layer == 0
                        else fixed_trace.trace["target_prefix_survived"][indexes, layer - 1]
                    )
                    failed = at_risk & ~survival
                    branching = fixed_trace.trace["legal_candidate_count"][indexes, layer].float()
                else:
                    survival = at_risk = failed = torch.empty(0, dtype=torch.bool)
                    branching = torch.empty(0)
                result.append(
                    {
                        "group": group,
                        "frequency_bin": frequency_bin,
                        "log_frequency_lower": float(edges[frequency_bin].item()),
                        "log_frequency_upper": float(edges[min(frequency_bin + 1, len(edges) - 1)].item()),
                        "layer": layer + 1,
                        "support": len(selected),
                        "survival_rate": _ratio(int(survival.sum()), len(selected)),
                        "at_risk_count": int(at_risk.sum()),
                        "conditional_failure_rate": _ratio(int(failed.sum()), int(at_risk.sum())),
                        "catalog_branching_proxy_mean": _finite_mean(branching),
                        "catalog_branching_proxy_valid_count": int(branching.numel()),
                    }
                )
    return result


def _aggregate_items(
    users: list[dict[str, Any]], fixed_trace: PrefixTraceBundle
) -> list[dict[str, Any]]:
    accumulators: dict[int, dict[str, Any]] = {}
    survival = fixed_trace.trace["target_prefix_survived"]
    for user in users:
        if user["group"] not in {"Head", "Tail"}:
            continue
        item_id = int(user["item_id"])
        if item_id not in accumulators:
            accumulators[item_id] = {
                "item_id": item_id,
                "group": user["group"],
                "raw_damage": float(user["raw_damage"]),
                "frequency": int(user["frequency"]),
                "model_sid": tuple(int(value) for value in str(user["model_sid"]).split()),
                "user_count": 0,
                "survival_sum": torch.zeros(fixed_trace.labels.size(1), dtype=torch.float64),
                "at_risk_sum": torch.zeros(fixed_trace.labels.size(1), dtype=torch.float64),
                "failure_sum": torch.zeros(fixed_trace.labels.size(1), dtype=torch.float64),
            }
        item = accumulators[item_id]
        index = int(user["trace_index"])
        observed = survival[index].double()
        at_risk = torch.cat([torch.ones(1, dtype=torch.float64), observed[:-1]])
        item["user_count"] += 1
        item["survival_sum"] += observed
        item["at_risk_sum"] += at_risk
        item["failure_sum"] += at_risk * (1.0 - observed)
    result = []
    for item in accumulators.values():
        users_for_item = float(item["user_count"])
        item["survival"] = item.pop("survival_sum") / users_for_item
        item["at_risk"] = item.pop("at_risk_sum") / users_for_item
        item["failure"] = item.pop("failure_sum") / users_for_item
        result.append(item)
    return sorted(result, key=lambda row: int(row["item_id"]))


def _risk_edges(items: list[dict[str, Any]], bin_count: int) -> list[float]:
    if bin_count <= 0:
        raise ValueError("Static-risk bin_count must be positive.")
    values = torch.tensor([float(row["raw_damage"]) for row in items], dtype=torch.float64)
    if not values.numel():
        return []
    edges = torch.unique(torch.quantile(values, torch.linspace(0, 1, bin_count + 1, dtype=torch.float64)))
    return [float(value) for value in edges.tolist()]


def _assign_bin(value: float, edges: list[float]) -> int:
    return int(torch.bucketize(torch.tensor(value), torch.tensor(edges[1:-1]), right=True).item())


def _weighted_mean_variance(values: list[float], weights: list[float]) -> tuple[float, float]:
    total = sum(weights)
    mean = sum(value * weight for value, weight in zip(values, weights, strict=True)) / total
    variance = sum(weight * (value - mean) ** 2 for value, weight in zip(values, weights, strict=True)) / total
    return mean, variance


def _overlap_estimate(
    items: list[dict[str, Any]],
    *,
    edges: list[float],
    min_support: int,
    min_retention: float,
    max_abs_smd: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    if len(edges) < 2:
        return [], [], {"available": False, "quality_status": "insufficient_overlap"}
    by_bin: dict[int, dict[str, list[dict[str, Any]]]] = defaultdict(lambda: {"Head": [], "Tail": []})
    for item in items:
        by_bin[_assign_bin(float(item["raw_damage"]), edges)][str(item["group"])].append(item)
    valid = [
        index
        for index, groups in by_bin.items()
        if len(groups["Head"]) >= min_support and len(groups["Tail"]) >= min_support
    ]
    overlap_rows = []
    common_total = sum(min(len(by_bin[index]["Head"]), len(by_bin[index]["Tail"])) for index in valid)
    for index in range(len(edges) - 1):
        heads = by_bin[index]["Head"]
        tails = by_bin[index]["Tail"]
        common = min(len(heads), len(tails)) if index in valid else 0
        overlap_rows.append(
            {
                "risk_bin": index,
                "raw_damage_lower": edges[index],
                "raw_damage_upper": edges[index + 1],
                "head_items": len(heads),
                "tail_items": len(tails),
                "valid_bin": int(index in valid),
                "common_item_weight": common,
                "normalized_common_weight": _ratio(common, common_total),
            }
        )
    total_by_group = {group: sum(row["group"] == group for row in items) for group in ("Head", "Tail")}
    retained_by_group = {
        group: sum(len(by_bin[index][group]) for index in valid) for group in ("Head", "Tail")
    }
    retention = {
        group: float(_ratio(retained_by_group[group], total_by_group[group]) or 0.0)
        for group in ("Head", "Tail")
    }
    if not valid or common_total == 0:
        return overlap_rows, [], {
            "available": False,
            "quality_status": "insufficient_overlap",
            "retention": retention,
            "raw_damage_smd": None,
        }

    item_weights: dict[str, tuple[list[float], list[float]]] = {
        "Head": ([], []),
        "Tail": ([], []),
    }
    for index in valid:
        common_weight = min(len(by_bin[index]["Head"]), len(by_bin[index]["Tail"])) / common_total
        for group in ("Head", "Tail"):
            rows = by_bin[index][group]
            for row in rows:
                item_weights[group][0].append(float(row["raw_damage"]))
                item_weights[group][1].append(common_weight / len(rows))
    head_mean, head_variance = _weighted_mean_variance(*item_weights["Head"])
    tail_mean, tail_variance = _weighted_mean_variance(*item_weights["Tail"])
    pooled_sd = math.sqrt((head_variance + tail_variance) / 2.0)
    smd = 0.0 if pooled_sd == 0 and head_mean == tail_mean else (
        None if pooled_sd == 0 else (tail_mean - head_mean) / pooled_sd
    )
    quality = "qualified"
    if min(retention.values()) < min_retention:
        quality = "insufficient_overlap"
    elif smd is None or abs(smd) > max_abs_smd:
        quality = "imbalanced"

    layer_rows = []
    depth_count = len(items[0]["survival"])
    for layer in range(depth_count):
        values: dict[str, dict[str, float]] = {}
        for group in ("Head", "Tail"):
            standardized_survival = 0.0
            weighted_failure = 0.0
            weighted_at_risk = 0.0
            raw_values = [float(row["survival"][layer]) for row in items if row["group"] == group]
            for index in valid:
                rows = by_bin[index][group]
                weight = min(len(by_bin[index]["Head"]), len(by_bin[index]["Tail"])) / common_total
                standardized_survival += weight * sum(float(row["survival"][layer]) for row in rows) / len(rows)
                weighted_failure += weight * sum(float(row["failure"][layer]) for row in rows) / len(rows)
                weighted_at_risk += weight * sum(float(row["at_risk"][layer]) for row in rows) / len(rows)
            values[group] = {
                "survival": standardized_survival,
                "failure": weighted_failure / weighted_at_risk if weighted_at_risk else math.nan,
                "raw_survival": sum(raw_values) / len(raw_values),
            }
        layer_rows.append(
            {
                "bin_count": len(edges) - 1,
                "layer": layer + 1,
                "estimand": "item_macro_static_risk_standardized_tail_minus_head",
                "standardized_survival_difference": values["Tail"]["survival"] - values["Head"]["survival"],
                "raw_survival_difference": values["Tail"]["raw_survival"] - values["Head"]["raw_survival"],
                "standardized_conditional_failure_difference": (
                    values["Tail"]["failure"] - values["Head"]["failure"]
                    if math.isfinite(values["Tail"]["failure"]) and math.isfinite(values["Head"]["failure"])
                    else None
                ),
                "quality_status": quality,
            }
        )
    return overlap_rows, layer_rows, {
        "available": True,
        "quality_status": quality,
        "retention": retention,
        "raw_damage_smd": smd,
        "valid_bins": len(valid),
        "effective_bin_count": len(edges) - 1,
    }


def _bootstrap_intervals(
    items: list[dict[str, Any]],
    *,
    edges: list[float],
    min_support: int,
    min_retention: float,
    max_abs_smd: float,
    bootstrap_samples: int,
    confidence: float,
    min_valid_fraction: float,
    seed: int,
) -> list[dict[str, Any]]:
    if bootstrap_samples <= 0:
        raise ValueError("Static-risk bootstrap_samples must be positive.")
    if not 0 < confidence < 1 or not 0 < min_valid_fraction <= 1:
        raise ValueError("Bootstrap confidence and minimum valid fraction must be within (0, 1].")
    clusters: dict[tuple[int, ...], list[dict[str, Any]]] = defaultdict(list)
    for item in items:
        clusters[tuple(item["model_sid"][: min(2, len(item["model_sid"]))])].append(item)
    cluster_values = list(clusters.values())
    rng = random.Random(seed)
    estimates: dict[tuple[int, str], list[float]] = defaultdict(list)
    invalid = 0
    for _ in range(bootstrap_samples):
        sampled = [row for _ in cluster_values for row in rng.choice(cluster_values)]
        _, layers, metadata = _overlap_estimate(
            sampled,
            edges=edges,
            min_support=min_support,
            min_retention=min_retention,
            max_abs_smd=max_abs_smd,
        )
        if not metadata.get("available") or not layers:
            invalid += 1
            continue
        for row in layers:
            layer = int(row["layer"])
            estimates[(layer, "standardized_survival")].append(
                float(row["standardized_survival_difference"])
            )
            estimates[(layer, "raw_survival")].append(float(row["raw_survival_difference"]))
            conditional = row["standardized_conditional_failure_difference"]
            if conditional is not None:
                estimates[(layer, "standardized_conditional_failure")].append(float(conditional))
    minimum_valid = math.ceil(bootstrap_samples * min_valid_fraction)
    alpha = (1.0 - confidence) / 2.0
    result = []
    for layer in range(1, len(items[0]["survival"]) + 1):
        row: dict[str, Any] = {
            "layer": layer,
            "confidence": confidence,
            "invalid_bootstrap_replicates": invalid,
            "requested_bootstrap_replicates": bootstrap_samples,
            "num_prefix_clusters": len(cluster_values),
            "interval_scope": "pointwise_exploratory",
        }
        for metric in (
            "standardized_survival",
            "raw_survival",
            "standardized_conditional_failure",
        ):
            values = sorted(estimates[(layer, metric)])
            available = len(values) >= minimum_valid
            row[f"{metric}_ci_available"] = available
            row[f"{metric}_ci_lower"] = (
                values[min(len(values) - 1, int(alpha * len(values)))] if available else None
            )
            row[f"{metric}_ci_upper"] = (
                values[min(len(values) - 1, int((1.0 - alpha) * len(values)))]
                if available
                else None
            )
            row[f"{metric}_valid_bootstrap_replicates"] = len(values)
        result.append(row)
    return result


def _static_risk_evidence(
    users: list[dict[str, Any]],
    fixed_trace: PrefixTraceBundle,
    *,
    bin_count: int,
    sensitivities: list[int] | tuple[int, ...],
    min_support: int,
    min_retention: float,
    max_abs_smd: float,
    bootstrap_samples: int,
    confidence: float,
    min_valid_fraction: float,
    seed: int,
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    items = _aggregate_items(users, fixed_trace)
    edges = _risk_edges(items, bin_count)
    overlap, primary_rows, metadata = _overlap_estimate(
        items,
        edges=edges,
        min_support=min_support,
        min_retention=min_retention,
        max_abs_smd=max_abs_smd,
    )
    intervals = (
        _bootstrap_intervals(
            items,
            edges=edges,
            min_support=min_support,
            min_retention=min_retention,
            max_abs_smd=max_abs_smd,
            bootstrap_samples=bootstrap_samples,
            confidence=confidence,
            min_valid_fraction=min_valid_fraction,
            seed=seed,
        )
        if primary_rows
        else []
    )
    interval_by_layer = {int(row["layer"]): row for row in intervals}
    for row in primary_rows:
        row.update(interval_by_layer.get(int(row["layer"]), {}))
        row["primary_analysis"] = True
        row["estimator_version"] = ESTIMATOR_VERSION

    sensitivity_rows = []
    for sensitivity in sorted(
        {int(value) for value in sensitivities if int(value) > 0 and int(value) != bin_count}
    ):
        sensitivity_edges = _risk_edges(items, sensitivity)
        _, rows, sensitivity_metadata = _overlap_estimate(
            items,
            edges=sensitivity_edges,
            min_support=min_support,
            min_retention=min_retention,
            max_abs_smd=max_abs_smd,
        )
        for row in rows:
            row.update(
                {
                    "primary_analysis": sensitivity == bin_count,
                    "estimator_version": ESTIMATOR_VERSION,
                    "sensitivity_requested_bin_count": sensitivity,
                    "sensitivity_quality_status": sensitivity_metadata.get("quality_status"),
                }
            )
            sensitivity_rows.append(row)
    return {
        "static_risk_overlap.csv": overlap,
        "static_risk_standardized_by_layer.csv": primary_rows + sensitivity_rows,
    }, {
        **metadata,
        "available": bool(primary_rows),
        "estimator_version": ESTIMATOR_VERSION,
        "primary_bin_count": bin_count,
        "sensitivity_bin_counts": sorted({int(value) for value in sensitivities}),
        "min_items_per_group_per_bin": min_support,
        "min_item_retention": min_retention,
        "max_abs_raw_damage_smd": max_abs_smd,
        "min_bootstrap_valid_fraction": min_valid_fraction,
        "bootstrap_samples": bootstrap_samples,
        "bootstrap_confidence": confidence,
        "item_count": len(items),
    }
