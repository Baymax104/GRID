"""Matched evidence for the prefix-balanced candidate allocation probe."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import torch

from src.data.components.data_models import DiagnosisBatch, PrefixTraceBundle, RecommendationOutcomeInput

ANALYSIS_SCHEMA_VERSION = "tail_candidate_allocation_probe_v1"
GROUPS = ("All", "Head", "Mid", "Tail", "Tail-Cold", "Tail+Tail-Cold")
TOP10_TRANSITIONS = ("hit_to_hit", "hit_to_miss", "miss_to_hit", "miss_to_miss")
ACCESS_TRANSITIONS = ("present_to_present", "present_to_absent", "absent_to_present", "absent_to_absent")


@dataclass(frozen=True)
class CandidateAllocationEvidence:
    summary: dict[str, Any]
    scalar_section: dict[str, float]
    tables: dict[str, list[dict[str, Any]]]


def build_candidate_allocation_evidence(
    batch: DiagnosisBatch,
    item_rows: list[dict[str, Any]],
    *,
    enabled: bool,
    overall_hit10_loss_guardrail: float = 0.002,
    head_hit10_loss_guardrail: float = 0.005,
) -> CandidateAllocationEvidence:
    if not enabled:
        return CandidateAllocationEvidence(
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
        raise ValueError(
            "Candidate allocation analysis requires baseline/intervention recommendation "
            "and Prefix Trace inputs."
        )
    baseline, intervention, baseline_trace, intervention_trace = values
    assert isinstance(baseline, RecommendationOutcomeInput)
    assert isinstance(intervention, RecommendationOutcomeInput)
    assert isinstance(baseline_trace, PrefixTraceBundle)
    assert isinstance(intervention_trace, PrefixTraceBundle)

    users, identity = _paired_rows(
        batch,
        item_rows,
        baseline,
        intervention,
        baseline_trace,
        intervention_trace,
    )
    groups = _group_rows(users)
    layers = _layer_rows(users, baseline_trace, intervention_trace)
    setting_gate = evaluate_setting_gate(
        groups,
        overall_hit10_loss_guardrail=overall_hit10_loss_guardrail,
        head_hit10_loss_guardrail=head_hit10_loss_guardrail,
    )
    tail = next(row for row in groups if row["group"] == "Tail+Tail-Cold")
    overall = next(row for row in groups if row["group"] == "All")
    return CandidateAllocationEvidence(
        summary={
            "available": True,
            "analysis_schema_version": ANALYSIS_SCHEMA_VERSION,
            "data_unit": "evaluation_user",
            "identity": identity,
            "setting_gate": setting_gate,
            "cross_setting_verdict": cross_setting_verdict([setting_gate]),
        },
        scalar_section={
            "available": 1.0,
            "overall_hit10_delta": float(overall["hit10_delta"]),
            "tail_candidate_access_delta": float(tail["candidate_access_delta"]),
            "tail_top10_net_change": float(tail["top10_net_change"]),
        },
        tables={
            "candidate_allocation_by_user.csv": users,
            "candidate_allocation_by_group.csv": groups,
            "candidate_allocation_by_layer.csv": layers,
        },
    )


def _allocation_metadata(trace: PrefixTraceBundle, role: str) -> dict[str, Any]:
    allocation = trace.metadata.get("prefix_allocation")
    if not isinstance(allocation, dict):
        raise ValueError(f"{role} Prefix Trace is missing prefix_allocation metadata.")
    required = {"enabled", "reserved_slots", "pool_multiplier", "source_split", "strategy"}
    missing = sorted(required - set(allocation))
    if missing:
        raise ValueError(f"{role} Prefix Trace allocation metadata is missing fields: {missing}.")
    return allocation


def _audit_trace_identity(
    baseline: PrefixTraceBundle,
    intervention: PrefixTraceBundle,
) -> dict[str, Any]:
    required_equal = (
        "data_split",
        "checkpoint_reference",
        "semantic_id_reference",
        "beam_width",
        "num_hierarchies",
        "codebook_size",
        "seed",
    )
    for field in required_equal:
        if field not in baseline.metadata or field not in intervention.metadata:
            raise ValueError(f"Prefix Trace identity is missing required field {field!r}.")
        if baseline.metadata[field] != intervention.metadata[field]:
            raise ValueError(f"Baseline/intervention Prefix Trace {field} mismatch.")
    baseline_allocation = _allocation_metadata(baseline, "Baseline")
    intervention_allocation = _allocation_metadata(intervention, "Intervention")
    if bool(baseline_allocation["enabled"]):
        raise ValueError("Baseline Prefix Trace must have prefix allocation disabled.")
    if not bool(intervention_allocation["enabled"]):
        raise ValueError("Intervention Prefix Trace must have prefix allocation enabled.")
    if intervention_allocation["source_split"] != "training":
        raise ValueError("Intervention prefix allocation must use the training source split.")
    if baseline.metadata["data_split"] not in {"evaluation", "testing"}:
        raise ValueError("Probe Prefix Trace data_split must be evaluation or testing.")
    return {
        "data_split": baseline.metadata["data_split"],
        "checkpoint_reference": baseline.metadata["checkpoint_reference"],
        "semantic_id_reference": baseline.metadata["semantic_id_reference"],
        "beam_width": int(baseline.metadata["beam_width"]),
        "num_hierarchies": int(baseline.metadata["num_hierarchies"]),
        "codebook_size": int(baseline.metadata["codebook_size"]),
        "seed": int(baseline.metadata["seed"]),
        "baseline_allocation": dict(baseline_allocation),
        "intervention_allocation": dict(intervention_allocation),
    }


def _wandb_run_id(reference: Any) -> str | None:
    if not isinstance(reference, str) or not reference.startswith("wandb://"):
        return None
    return reference.split("?", 1)[0].rstrip("/").rsplit("/", 1)[-1]


def _rank(candidates: torch.Tensor, label: torch.Tensor) -> int | None:
    matches = torch.all(candidates == label, dim=1).nonzero().reshape(-1)
    return int(matches[0].item()) + 1 if matches.numel() else None


def _paired_rows(
    batch: DiagnosisBatch,
    item_rows: list[dict[str, Any]],
    baseline: RecommendationOutcomeInput,
    intervention: RecommendationOutcomeInput,
    baseline_trace: PrefixTraceBundle,
    intervention_trace: PrefixTraceBundle,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    trace_identity = _audit_trace_identity(baseline_trace, intervention_trace)
    beam_width = trace_identity["beam_width"]
    if baseline.generated_sids.size(1) != intervention.generated_sids.size(1):
        raise ValueError("Baseline/intervention recommendation beam width mismatch.")
    if baseline.generated_sids.size(1) != beam_width:
        raise ValueError("Recommendation width does not match Prefix Trace beam width.")
    sid_width = batch.sid_views.model_sid.size(1)
    if baseline.generated_sids.size(2) != sid_width or intervention.generated_sids.size(2) != sid_width:
        raise ValueError("Probe recommendation SID width does not match the model SID catalog.")

    catalog = {
        tuple(int(value) for value in sid): row
        for sid, row in zip(batch.sid_views.model_sid.tolist(), item_rows, strict=True)
    }
    if len(catalog) != len(item_rows):
        raise ValueError("Model semantic IDs must uniquely identify items for allocation analysis.")
    catalog_sids = set(catalog)
    tensors = (
        baseline.user_ids,
        intervention.user_ids,
        baseline_trace.keys,
        intervention_trace.keys,
    )
    orders = [tensor.argsort() for tensor in tensors]
    keys = [tensor[order] for tensor, order in zip(tensors, orders, strict=True)]
    if not all(torch.equal(keys[0], other) for other in keys[1:]):
        raise ValueError("Baseline/intervention recommendation and Prefix Trace user keys must match exactly.")
    baseline_labels = baseline_trace.labels[orders[2]]
    intervention_labels = intervention_trace.labels[orders[3]]
    if not torch.equal(baseline_labels, intervention_labels):
        raise ValueError("Baseline/intervention Prefix Trace labels must match exactly.")
    if not torch.equal(baseline.label_item_ids[orders[0]], intervention.label_item_ids[orders[1]]):
        raise ValueError("Baseline/intervention recommendation label item IDs must match exactly.")

    baseline_candidates = baseline.generated_sids[orders[0]]
    intervention_candidates = intervention.generated_sids[orders[1]]
    invalid = {
        tuple(int(value) for value in sid)
        for candidates in (baseline_candidates, intervention_candidates)
        for sid in candidates.reshape(-1, sid_width).tolist()
        if tuple(int(value) for value in sid) not in catalog_sids
    }
    if invalid:
        raise ValueError(f"Probe recommendation contains SIDs absent from the model catalog: {list(invalid)[:3]}.")

    required_trace_fields = {
        "target_prefix_training_mass",
        "target_allocation_shortlisted",
        "target_selected_by_reserve",
        "allocation_reserved_count",
    }
    for role, trace in (("Baseline", baseline_trace), ("Intervention", intervention_trace)):
        missing = sorted(required_trace_fields - set(trace.trace))
        if missing:
            raise ValueError(f"{role} Prefix Trace is missing allocation tensors: {missing}.")

    baseline_final_rank = baseline_trace.trace["target_beam_rank"][orders[2], -1]
    intervention_final_rank = intervention_trace.trace["target_beam_rank"][orders[3], -1]
    rows = []
    for index, user_id in enumerate(keys[0].tolist()):
        label = tuple(int(value) for value in baseline_labels[index].tolist())
        if label not in catalog:
            raise KeyError(f"Prefix Trace target SID is absent from the model catalog: {label}.")
        item = catalog[label]
        label_item = int(baseline.label_item_ids[orders[0][index]].item())
        if label_item != int(item["item_id"]):
            raise ValueError("Recommendation label item does not match Prefix Trace target SID.")
        baseline_rank = _rank(baseline_candidates[index], baseline_labels[index])
        intervention_rank = _rank(intervention_candidates[index], baseline_labels[index])
        if (baseline_rank or -1) != int(baseline_final_rank[index].item()) or (
            (intervention_rank or -1) != int(intervention_final_rank[index].item())
        ):
            raise ValueError("Probe recommendation target rank does not match Prefix Trace final beam rank.")
        baseline_survived = bool(
            baseline_trace.trace["target_prefix_survived"][orders[2][index], -1]
        )
        intervention_survived = bool(
            intervention_trace.trace["target_prefix_survived"][orders[3][index], -1]
        )
        if (baseline_rank is not None) != baseline_survived or (
            (intervention_rank is not None) != intervention_survived
        ):
            raise ValueError("Probe recommendation target membership does not match Prefix Trace survival.")
        baseline_hit = baseline_rank is not None and baseline_rank <= 10
        intervention_hit = intervention_rank is not None and intervention_rank <= 10
        rows.append(
            {
                "user_id": int(user_id),
                "item_id": int(item["item_id"]),
                "group": str(item["group"]),
                "frequency": int(item["freq_train"]),
                "model_sid": " ".join(str(value) for value in label),
                "baseline_rank": baseline_rank,
                "intervention_rank": intervention_rank,
                "baseline_candidate_access": int(baseline_rank is not None),
                "intervention_candidate_access": int(intervention_rank is not None),
                "baseline_hit10": int(baseline_hit),
                "intervention_hit10": int(intervention_hit),
                "access_transition": (
                    f"{'present' if baseline_rank is not None else 'absent'}_to_"
                    f"{'present' if intervention_rank is not None else 'absent'}"
                ),
                "top10_transition": (
                    f"{'hit' if baseline_hit else 'miss'}_to_"
                    f"{'hit' if intervention_hit else 'miss'}"
                ),
                "baseline_trace_index": int(orders[2][index].item()),
                "intervention_trace_index": int(orders[3][index].item()),
            }
        )

    refs = batch.input_metadata
    reference_names = {
        "baseline_recommendation": "baseline_recommendation_output_reference",
        "intervention_recommendation": "intervention_recommendation_output_reference",
        "baseline_trace": "baseline_prefix_trace_reference",
        "intervention_trace": "intervention_prefix_trace_reference",
    }
    fallbacks = {
        "baseline_recommendation": "recommendation_output_reference",
        "intervention_recommendation": "widened_recommendation_output_reference",
        "baseline_trace": "fixed_prefix_trace_path",
        "intervention_trace": "widened_prefix_trace_path",
    }
    references = {
        role: refs.get(name) or refs.get(fallbacks[role])
        for role, name in reference_names.items()
    }
    run_ids = {role: _wandb_run_id(reference) for role, reference in references.items()}
    if all((run_ids["baseline_recommendation"], run_ids["baseline_trace"])) and (
        run_ids["baseline_recommendation"] != run_ids["baseline_trace"]
    ):
        raise ValueError("Baseline recommendation and Prefix Trace lineage conflict.")
    if all((run_ids["intervention_recommendation"], run_ids["intervention_trace"])) and (
        run_ids["intervention_recommendation"] != run_ids["intervention_trace"]
    ):
        raise ValueError("Intervention recommendation and Prefix Trace lineage conflict.")

    artifact_identity = refs.get("resolved_artifact_identity", {})
    expected_fields = {
        "semantic_id_path",
        "baseline_recommendation_output_path",
        "intervention_recommendation_output_path",
        "baseline_prefix_trace_path",
        "intervention_prefix_trace_path",
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
    source_runs_verified = bool(
        run_ids["baseline_recommendation"]
        and run_ids["intervention_recommendation"]
        and run_ids["baseline_recommendation"] == run_ids["baseline_trace"]
        and run_ids["intervention_recommendation"] == run_ids["intervention_trace"]
    )
    if not source_runs_verified or not artifact_identity_verified:
        raise ValueError(
            "Candidate allocation pair identity is unverified; complete W&B source lineage "
            "and resolved Artifact identity are required."
        )
    return rows, {
        **trace_identity,
        "identity_verified": True,
        "source_runs_verified": source_runs_verified,
        "artifact_identity_verified": artifact_identity_verified,
        "source_runs": run_ids,
        "references": references,
        "artifacts": artifact_identity,
    }


def _select_group(users: list[dict[str, Any]], group: str) -> list[dict[str, Any]]:
    if group == "All":
        return users
    if group == "Tail+Tail-Cold":
        return [row for row in users if row["group"] in {"Tail", "Tail-Cold"}]
    return [row for row in users if row["group"] == group]


def _ratio(numerator: int | float, denominator: int | float) -> float:
    return float(numerator) / float(denominator) if denominator else 0.0


def _ndcg(rank: int | None) -> float:
    return 1.0 / math.log2(rank + 1) if rank is not None and rank <= 10 else 0.0


def _group_rows(users: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for group in GROUPS:
        selected = _select_group(users, group)
        support = len(selected)
        baseline_access = sum(row["baseline_candidate_access"] for row in selected)
        intervention_access = sum(row["intervention_candidate_access"] for row in selected)
        baseline_hits = sum(row["baseline_hit10"] for row in selected)
        intervention_hits = sum(row["intervention_hit10"] for row in selected)
        top10_counts = {name: sum(row["top10_transition"] == name for row in selected) for name in TOP10_TRANSITIONS}
        access_counts = {name: sum(row["access_transition"] == name for row in selected) for name in ACCESS_TRANSITIONS}
        row = {
            "group": group,
            "support": support,
            "baseline_candidate_access_count": baseline_access,
            "baseline_candidate_access_rate": _ratio(baseline_access, support),
            "intervention_candidate_access_count": intervention_access,
            "intervention_candidate_access_rate": _ratio(intervention_access, support),
            "candidate_access_delta": _ratio(intervention_access - baseline_access, support),
            "candidate_added": access_counts["absent_to_present"],
            "candidate_lost": access_counts["present_to_absent"],
            "baseline_hit10_count": baseline_hits,
            "baseline_hit10": _ratio(baseline_hits, support),
            "baseline_ndcg10": _ratio(sum(_ndcg(row["baseline_rank"]) for row in selected), support),
            "intervention_hit10_count": intervention_hits,
            "intervention_hit10": _ratio(intervention_hits, support),
            "intervention_ndcg10": _ratio(
                sum(_ndcg(row["intervention_rank"]) for row in selected), support
            ),
            "hit10_delta": _ratio(intervention_hits - baseline_hits, support),
            "top10_added": top10_counts["miss_to_hit"],
            "top10_lost": top10_counts["hit_to_miss"],
            "top10_net_change": intervention_hits - baseline_hits,
        }
        row.update(top10_counts)
        row.update(access_counts)
        rows.append(row)
    return rows


def _mean(values: torch.Tensor) -> float | None:
    finite = values[torch.isfinite(values)]
    return float(finite.float().mean().item()) if finite.numel() else None


def _layer_rows(
    users: list[dict[str, Any]],
    baseline_trace: PrefixTraceBundle,
    intervention_trace: PrefixTraceBundle,
) -> list[dict[str, Any]]:
    rows = []
    for group in GROUPS:
        selected = _select_group(users, group)
        baseline_indexes = torch.tensor(
            [row["baseline_trace_index"] for row in selected], dtype=torch.long
        )
        intervention_indexes = torch.tensor(
            [row["intervention_trace_index"] for row in selected], dtype=torch.long
        )
        for layer in range(baseline_trace.labels.size(1)):
            baseline_survival = baseline_trace.trace["target_prefix_survived"][baseline_indexes, layer]
            intervention_survival = intervention_trace.trace["target_prefix_survived"][intervention_indexes, layer]
            shortlisted = intervention_trace.trace["target_allocation_shortlisted"][intervention_indexes, layer]
            reserved = intervention_trace.trace["target_selected_by_reserve"][intervention_indexes, layer]
            masses = intervention_trace.trace["target_prefix_training_mass"][intervention_indexes, layer].float()
            reserve_counts = intervention_trace.trace["allocation_reserved_count"][intervention_indexes, layer].float()
            support = len(selected)
            rows.append(
                {
                    "group": group,
                    "layer": layer + 1,
                    "support": support,
                    "baseline_survival_rate": _ratio(int(baseline_survival.sum().item()), support),
                    "intervention_survival_rate": _ratio(int(intervention_survival.sum().item()), support),
                    "survival_delta": _ratio(
                        int(intervention_survival.sum().item()) - int(baseline_survival.sum().item()),
                        support,
                    ),
                    "intervention_shortlist_rate": _ratio(int(shortlisted.sum().item()), support),
                    "intervention_reserve_retention_rate": _ratio(int(reserved.sum().item()), support),
                    "target_prefix_training_mass_mean": _mean(masses),
                    "allocation_reserved_count_mean": _mean(reserve_counts),
                }
            )
    return rows


def evaluate_setting_gate(
    group_rows: list[dict[str, Any]],
    *,
    overall_hit10_loss_guardrail: float,
    head_hit10_loss_guardrail: float,
) -> dict[str, Any]:
    by_group = {row["group"]: row for row in group_rows}
    for required in ("All", "Head", "Tail+Tail-Cold"):
        if required not in by_group:
            raise ValueError(f"Candidate allocation group evidence is missing {required!r}.")
    overall = by_group["All"]
    head = by_group["Head"]
    tail = by_group["Tail+Tail-Cold"]
    tail_access_improved = float(tail["candidate_access_delta"]) > 0.0
    tail_top10_added = int(tail["top10_added"])
    overall_guardrail_passed = float(overall["hit10_delta"]) >= -overall_hit10_loss_guardrail
    head_guardrail_passed = float(head["hit10_delta"]) >= -head_hit10_loss_guardrail
    return {
        "tail_candidate_access_improved": tail_access_improved,
        "tail_candidate_access_delta": float(tail["candidate_access_delta"]),
        "tail_top10_added": tail_top10_added,
        "tail_top10_net_change": int(tail["top10_net_change"]),
        "overall_hit10_delta": float(overall["hit10_delta"]),
        "head_hit10_delta": float(head["hit10_delta"]),
        "overall_guardrail_passed": overall_guardrail_passed,
        "head_guardrail_passed": head_guardrail_passed,
        "setting_gate_passed": bool(
            tail_access_improved
            and tail_top10_added > 0
            and overall_guardrail_passed
            and head_guardrail_passed
        ),
    }


def cross_setting_verdict(settings: list[dict[str, Any]]) -> dict[str, Any]:
    if len(settings) != 4:
        return {
            "verdict": "inconclusive",
            "settings_observed": len(settings),
            "settings_required": 4,
            "reason": "incomplete_registered_settings",
        }
    access_improved = sum(bool(setting["tail_candidate_access_improved"]) for setting in settings)
    settings_with_tail_additions = sum(int(setting["tail_top10_added"]) > 0 for setting in settings)
    cost_guardrails_passed = all(
        bool(setting["overall_guardrail_passed"] and setting["head_guardrail_passed"])
        for setting in settings
    )
    advance = access_improved >= 3 and settings_with_tail_additions >= 2 and cost_guardrails_passed
    return {
        "verdict": "advance" if advance else "stop",
        "settings_observed": 4,
        "tail_access_improved_settings": access_improved,
        "tail_top10_added_settings": settings_with_tail_additions,
        "cost_guardrails_passed": cost_guardrails_passed,
        "reason": None if advance else "predeclared_gate_not_met",
    }
