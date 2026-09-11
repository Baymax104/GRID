"""Prefix-survival mechanism evidence for Tail-SID diagnosis."""

from __future__ import annotations

import math
import random
from collections import defaultdict
from dataclasses import dataclass
from typing import Any

import torch

from src.data.components.data_models import DiagnosisBatch, PrefixTraceBundle

GROUPS = ("Head", "Mid", "Tail", "Tail-Cold")


@dataclass(frozen=True)
class PrefixMechanismEvidence:
    summary: dict[str, Any]
    scalar_section: dict[str, float]
    tables: dict[str, list[dict[str, Any]]]
    verdict: str


def build_prefix_mechanism_evidence(
    batch: DiagnosisBatch,
    item_rows: list[dict[str, Any]],
    *,
    bootstrap_samples: int,
    confidence: float,
    seed: int,
    include_widened_recovery: bool = True,
) -> PrefixMechanismEvidence:
    fixed = batch.fixed_prefix_trace
    if fixed is None:
        return PrefixMechanismEvidence(
            summary={"available": False, "recovery_available": False},
            scalar_section={"available": 0.0, "recovery_available": 0.0},
            tables={},
            verdict="unavailable",
        )

    widened = batch.widened_prefix_trace if include_widened_recovery else None
    _validate_trace_identity(batch, fixed, widened)
    users = _build_user_rows(batch, fixed, item_rows)
    layer_rows = _layer_group_rows(users, fixed)
    first_failure_rows = _first_failure_rows(users, fixed)
    matched = _risk_matched_survival(users)
    competition = _partial_competition_association(users)
    clustered = _prefix_cluster_bootstrap(
        users,
        bootstrap_samples=bootstrap_samples,
        confidence=confidence,
        seed=seed,
    )
    recovery_rows, recovery_summary = _recovery_evidence(users, fixed, widened)

    supported = (
        matched.get("available", False)
        and float(matched["tail_minus_head_survival"]) < 0
        and competition.get("available", False)
        and float(competition["partial_correlation_with_survival"]) < 0
    )
    verdict = "supported" if supported else "not_supported"
    summary = {
        "available": True,
        "schema_version": fixed.schema_version,
        "data_split": fixed.metadata["data_split"],
        "fixed_beam_width": int(fixed.metadata["beam_width"]),
        "widened_beam_width": int(widened.metadata["beam_width"]) if widened is not None else None,
        "num_users": len(users),
        "risk_matched_survival": matched,
        "competition_association": competition,
        "prefix_cluster_bootstrap": clustered,
        "recovery_available": widened is not None,
        "recovery": recovery_summary,
    }
    scalar_section = {
        "available": 1.0,
        "recovery_available": 1.0 if widened is not None else 0.0,
        "risk_matched_tail_minus_head_survival": float(matched.get("tail_minus_head_survival", 0.0)),
        "competition_partial_correlation": float(competition.get("partial_correlation_with_survival", 0.0)),
        "tail_recovery_rate": float(recovery_summary.get("tail_recovery_rate", 0.0)),
    }
    tables = {
        "prefix_survival_by_layer.csv": layer_rows,
        "prefix_first_failure.csv": first_failure_rows,
        "prefix_competition_association.csv": [competition],
        "prefix_cluster_bootstrap.csv": [clustered],
    }
    if recovery_rows:
        tables["prefix_widened_recovery.csv"] = recovery_rows
    return PrefixMechanismEvidence(summary, scalar_section, tables, verdict)


def _validate_trace_identity(
    batch: DiagnosisBatch,
    fixed: PrefixTraceBundle,
    widened: PrefixTraceBundle | None,
) -> None:
    expected_sid = batch.input_metadata.get("semantic_id_reference")
    actual_sid = fixed.metadata.get("semantic_id_reference")
    if expected_sid is not None and actual_sid != expected_sid:
        raise ValueError(
            "Prefix Trace semantic-ID identity does not match diagnosis input: "
            f"trace={actual_sid!r}, diagnosis={expected_sid!r}."
        )
    if widened is None:
        return
    for field_name in ("data_split", "checkpoint_reference", "semantic_id_reference", "num_hierarchies", "codebook_size"):
        if fixed.metadata.get(field_name) != widened.metadata.get(field_name):
            raise ValueError(f"Fixed and widened Prefix Trace metadata differ for {field_name!r}.")
    if not torch.equal(fixed.keys, widened.keys):
        raise ValueError("Fixed and widened Prefix Trace user keys must match exactly.")
    if not torch.equal(fixed.labels, widened.labels):
        raise ValueError("Fixed and widened Prefix Trace target labels must match exactly.")
    if int(widened.metadata["beam_width"]) <= int(fixed.metadata["beam_width"]):
        raise ValueError("Widened Prefix Trace beam width must exceed the fixed beam width.")


def _build_user_rows(
    batch: DiagnosisBatch,
    fixed: PrefixTraceBundle,
    item_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    by_sid: dict[tuple[int, ...], dict[str, Any]] = {}
    for row, sid in zip(item_rows, batch.sid_views.model_sid.tolist(), strict=True):
        sid_key = tuple(int(value) for value in sid)
        if sid_key in by_sid:
            raise ValueError(f"Model semantic IDs are not unique for trace label {sid_key}.")
        by_sid[sid_key] = row

    if batch.recommendation is not None:
        if not torch.equal(fixed.keys, batch.recommendation.user_ids):
            raise ValueError("Prefix Trace and recommendation user keys must match exactly.")
        item_id_by_sid = {sid: int(row["item_id"]) for sid, row in by_sid.items()}
        traced_items = torch.tensor(
            [item_id_by_sid.get(tuple(int(value) for value in label), -1) for label in fixed.labels.tolist()]
        )
        if not torch.equal(traced_items, batch.recommendation.label_item_ids.cpu()):
            raise ValueError("Prefix Trace target labels do not match recommendation labels.")

    users = []
    for index, (user_id, label) in enumerate(zip(fixed.keys.tolist(), fixed.labels.tolist(), strict=True)):
        sid_key = tuple(int(value) for value in label)
        if sid_key not in by_sid:
            raise KeyError(f"Prefix Trace target SID {sid_key} is absent from semantic ID input.")
        item = by_sid[sid_key]
        users.append(
            {
                "index": index,
                "user_id": int(user_id),
                "label": sid_key,
                "item_id": int(item["item_id"]),
                "group": str(item["group"]),
                "frequency": float(item["freq_train"]),
                "raw_damage": float(item["raw_damage"]),
                "mean_survival": float(fixed.trace["target_prefix_survived"][index].float().mean().item()),
                "final_survival": float(fixed.trace["target_prefix_survived"][index, -1].item()),
                "mean_legal_candidate_count": float(
                    fixed.trace["legal_candidate_count"][index].float().mean().item()
                ),
            }
        )
    return users


def _finite_mean(values: torch.Tensor) -> float | None:
    finite = values[torch.isfinite(values)]
    return float(finite.float().mean().item()) if finite.numel() else None


def _layer_group_rows(users: list[dict[str, Any]], fixed: PrefixTraceBundle) -> list[dict[str, Any]]:
    rows = []
    for hierarchy in range(fixed.labels.size(1)):
        for group in GROUPS:
            indexes = [int(user["index"]) for user in users if user["group"] == group]
            if not indexes:
                continue
            index = torch.tensor(indexes, dtype=torch.long)
            rows.append(
                {
                    "layer": hierarchy + 1,
                    "group": group,
                    "support": len(indexes),
                    "teacher_target_probability": _finite_mean(fixed.trace["teacher_target_probability"][index, hierarchy]),
                    "teacher_legal_rank": _finite_mean(fixed.trace["teacher_legal_rank"][index, hierarchy].float()),
                    "teacher_target_vs_best_legal_margin": _finite_mean(
                        fixed.trace["teacher_target_vs_best_legal_margin"][index, hierarchy]
                    ),
                    "prefix_survival_rate": _finite_mean(fixed.trace["target_prefix_survived"][index, hierarchy].float()),
                    "avg_cutoff_margin": _finite_mean(fixed.trace["cutoff_margin"][index, hierarchy]),
                    "avg_legal_candidate_count": _finite_mean(fixed.trace["legal_candidate_count"][index, hierarchy].float()),
                }
            )
    return rows


def _first_failure_rows(users: list[dict[str, Any]], fixed: PrefixTraceBundle) -> list[dict[str, Any]]:
    rows = []
    for group in GROUPS:
        indexes = [int(user["index"]) for user in users if user["group"] == group]
        if not indexes:
            continue
        values = fixed.trace["first_failure_depth"][indexes]
        failures = values[values != -1]
        rows.append(
            {
                "group": group,
                "support": len(indexes),
                "failure_rate": float((values != -1).float().mean().item()),
                "mean_first_failure_depth": _finite_mean(failures.float()),
                "full_survival_rate": float((values == -1).float().mean().item()),
            }
        )
    return rows


def _risk_matched_survival(users: list[dict[str, Any]]) -> dict[str, Any]:
    heads = [user for user in users if user["group"] == "Head"]
    tails = [user for user in users if user["group"] in {"Tail", "Tail-Cold"}]
    if not heads or not tails:
        return {"available": False, "support": 0}
    damage_scale = max(max(abs(user["raw_damage"]) for user in users), 1.0)
    frequency_scale = max(max(math.log1p(user["frequency"]) for user in users), 1.0)
    differences = []
    for tail in tails:
        head = min(
            heads,
            key=lambda row: abs(row["raw_damage"] - tail["raw_damage"]) / damage_scale
            + abs(math.log1p(row["frequency"]) - math.log1p(tail["frequency"])) / frequency_scale,
        )
        differences.append(tail["mean_survival"] - head["mean_survival"])
    return {
        "available": True,
        "support": len(differences),
        "tail_minus_head_survival": sum(differences) / len(differences),
    }


def _partial_competition_association(users: list[dict[str, Any]]) -> dict[str, Any]:
    if len(users) < 4:
        return {"available": False, "support": len(users)}
    controls = torch.tensor(
        [[1.0, math.log1p(user["frequency"]), user["raw_damage"]] for user in users],
        dtype=torch.float64,
    )
    competition = torch.tensor([user["mean_legal_candidate_count"] for user in users], dtype=torch.float64)
    survival = torch.tensor([user["mean_survival"] for user in users], dtype=torch.float64)
    competition_residual = competition - controls @ torch.linalg.lstsq(controls, competition).solution
    survival_residual = survival - controls @ torch.linalg.lstsq(controls, survival).solution
    denominator = torch.linalg.vector_norm(competition_residual) * torch.linalg.vector_norm(survival_residual)
    if float(denominator) == 0.0:
        return {"available": False, "support": len(users)}
    correlation = float(torch.dot(competition_residual, survival_residual) / denominator)
    return {
        "available": True,
        "support": len(users),
        "partial_correlation_with_survival": correlation,
        "controls": "log1p_frequency,raw_damage",
    }


def _prefix_cluster_bootstrap(
    users: list[dict[str, Any]], *, bootstrap_samples: int, confidence: float, seed: int
) -> dict[str, Any]:
    clusters: dict[tuple[int, ...], list[dict[str, Any]]] = defaultdict(list)
    for user in users:
        clusters[user["label"][: min(2, len(user["label"]))]].append(user)
    cluster_values = list(clusters.values())
    if not cluster_values or bootstrap_samples <= 0:
        return {"available": False, "num_clusters": len(cluster_values)}
    rng = random.Random(seed)
    estimates = []
    for _ in range(bootstrap_samples):
        sampled = [row for _ in cluster_values for row in rng.choice(cluster_values)]
        heads = [row["mean_survival"] for row in sampled if row["group"] == "Head"]
        tails = [row["mean_survival"] for row in sampled if row["group"] in {"Tail", "Tail-Cold"}]
        if heads and tails:
            estimates.append(sum(tails) / len(tails) - sum(heads) / len(heads))
    if not estimates:
        return {"available": False, "num_clusters": len(cluster_values)}
    ordered = sorted(estimates)
    alpha = (1.0 - confidence) / 2.0
    lower = ordered[min(len(ordered) - 1, int(alpha * len(ordered)))]
    upper = ordered[min(len(ordered) - 1, int((1.0 - alpha) * len(ordered)))]
    return {
        "available": True,
        "num_clusters": len(cluster_values),
        "bootstrap_samples": len(estimates),
        "ci_lower": lower,
        "ci_upper": upper,
    }


def _recovery_evidence(
    users: list[dict[str, Any]],
    fixed: PrefixTraceBundle,
    widened: PrefixTraceBundle | None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if widened is None:
        return [], {"available": False}
    rows = []
    tail_recovery_rate = 0.0
    for group in GROUPS:
        indexes = [int(user["index"]) for user in users if user["group"] == group]
        if not indexes:
            continue
        fixed_survival = fixed.trace["target_prefix_survived"][indexes]
        widened_survival = widened.trace["target_prefix_survived"][indexes]
        fixed_miss = ~fixed_survival[:, -1]
        recovered = fixed_miss & widened_survival[:, -1]
        recovery_rate = float(recovered.float().sum().item() / fixed_miss.float().sum().item()) if fixed_miss.any() else 0.0
        if group in {"Tail", "Tail-Cold"}:
            tail_recovery_rate = max(tail_recovery_rate, recovery_rate)
        fixed_depth = torch.where(
            fixed.trace["first_failure_depth"][indexes] == -1,
            torch.full_like(fixed.trace["first_failure_depth"][indexes], fixed.labels.size(1) + 1),
            fixed.trace["first_failure_depth"][indexes],
        )
        widened_depth = torch.where(
            widened.trace["first_failure_depth"][indexes] == -1,
            torch.full_like(widened.trace["first_failure_depth"][indexes], widened.labels.size(1) + 1),
            widened.trace["first_failure_depth"][indexes],
        )
        for hierarchy in range(fixed.labels.size(1)):
            rows.append(
                {
                    "group": group,
                    "layer": hierarchy + 1,
                    "support": len(indexes),
                    "fixed_miss_support": int(fixed_miss.sum().item()),
                    "final_recovery_rate": recovery_rate,
                    "failure_depth_shift": float((widened_depth - fixed_depth).float().mean().item()),
                    "coverage_recovery": float(
                        widened_survival[:, hierarchy].float().mean().item()
                        - fixed_survival[:, hierarchy].float().mean().item()
                    ),
                }
            )
    return rows, {"available": True, "tail_recovery_rate": tail_recovery_rate}
