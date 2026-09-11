"""Structured evidence assembly for Tail-SID diagnosis."""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, TypedDict

import torch
import torch.nn.functional as F

from src.common.writers.structured_analysis import StructuredAnalysisOutput
from src.data.components.data_models import DiagnosisBatch
from src.quantization.tail_sid_diagnosis.candidate_allocation import (
    build_candidate_allocation_evidence,
)
from src.quantization.tail_sid_diagnosis.metrics import (
    SCORE_CONTRIBUTION_MAX,
    SCORE_CONTRIBUTION_MIN,
    SCORE_IQR_STABILITY_THRESHOLD,
    SCORE_NORMALIZATION_VERSION,
    SemanticMismatchValues,
    _compute_damage_scores,
    _compute_prefix_rows,
    _compute_structural_values,
    _mean,
    build_diagnosis_context,
    stable_robust_risk_scores,
)
from src.quantization.tail_sid_diagnosis.prefix_trace import build_prefix_mechanism_evidence
from src.quantization.tail_sid_diagnosis.search_ranking import build_search_ranking_evidence

GROUPS = ("Head", "Mid", "Tail", "Tail-Cold")
EVIDENCE_SCHEMA_VERSION = "tail_sid_diagnosis_evidence_v1"


@dataclass(frozen=True)
class AsymmetricEvidenceValues:
    near_partner_counts: list[dict[str, int]]
    full_partner_counts: list[dict[str, int]]
    head_dominance: list[float]
    tail_isolation_deficit: list[float]
    tail_head_pressure: list[float]


@dataclass(frozen=True)
class SemanticEvidenceValues:
    primary: SemanticMismatchValues
    bucket_relative_outlier: list[float]
    harmful_partner_counts: list[dict[str, int]]
    pair_rows: list[dict[str, Any]]
    metadata: dict[str, Any]


class EvidenceVerdict(TypedDict):
    tail_structural_asymmetry: str
    equal_risk_tail_vulnerability: str
    generation_risk_validity: str
    cross_setting_stability: str
    prefix_survival_mechanism: str


@dataclass(frozen=True)
class DiagnosisEvidence:
    summary: dict[str, Any]
    scalar_sections: dict[str, dict[str, float]]
    group_rows: list[dict[str, Any]]
    item_rows: list[dict[str, Any]]
    prefix_rows: list[dict[str, Any]]
    harmful_pair_rows: list[dict[str, Any]] = field(default_factory=list)
    sensitivity_rows: list[dict[str, Any]] = field(default_factory=list)
    recommendation_rows: list[dict[str, Any]] = field(default_factory=list)
    mechanism_tables: dict[str, list[dict[str, Any]]] = field(default_factory=dict)
    analysis_tables: dict[str, list[dict[str, Any]]] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_structured_output(self) -> StructuredAnalysisOutput:
        """Adapt domain evidence to the common writer's domain-neutral payload."""
        tables = {
            "group_metrics.csv": self.group_rows,
            "item_damage_scores.csv": self.item_rows,
            "prefix_risk_scores.csv": self.prefix_rows,
            "harmful_overlap_pairs.csv": self.harmful_pair_rows,
        }
        if self.sensitivity_rows:
            tables["sensitivity_metrics.csv"] = self.sensitivity_rows
        if self.recommendation_rows:
            tables["recommendation_metrics.csv"] = self.recommendation_rows
        tables.update(self.mechanism_tables)
        tables.update(self.analysis_tables)
        return StructuredAnalysisOutput(
            documents={"summary.json": self.summary},
            tables=tables,
            metadata=self.metadata,
        )


def build_diagnosis_evidence(
    batch: DiagnosisBatch,
    *,
    max_neighbors_per_bucket: int = 512,
    priority_multipliers: dict[str, float] | None = None,
    semantic_reference_quantile: float = 0.75,
    semantic_bucket_quantile: float = 0.25,
    semantic_reference_pairs: int = 100000,
    semantic_seed: int = 42,
    max_harmful_pairs: int = 100000,
    max_harmful_pairs_per_item: int = 20,
    hit_ks: list[int] | tuple[int, ...] = (5, 10),
    bootstrap_samples: int = 1000,
    bootstrap_confidence: float = 0.95,
    frequency_bin_count: int = 5,
    min_frequency_bin_support: int = 20,
    tail_ratio_sensitivity: list[float] | tuple[float, ...] = (0.1, 0.2, 0.3),
    semantic_quantile_sensitivity: list[float] | tuple[float, ...] = (0.5, 0.75, 0.9),
    damage_component_sensitivity: list[str] | tuple[str, ...] = (
        "all",
        "structural_only",
        "semantic_only",
    ),
    head_ratio: float = 0.2,
    tail_ratio: float = 0.2,
    search_ranking_enabled: bool = False,
    risk_standardization_enabled: bool = False,
    risk_standardization_bin_count: int = 5,
    risk_standardization_bin_count_sensitivity: list[int] | tuple[int, ...] = (3, 5, 10),
    risk_standardization_min_items_per_group_per_bin: int = 20,
    risk_standardization_min_item_retention: float = 0.5,
    risk_standardization_max_abs_raw_damage_smd: float = 0.1,
    risk_standardization_min_bootstrap_valid_fraction: float = 0.9,
    candidate_allocation_probe_enabled: bool = False,
    candidate_allocation_overall_hit10_loss_guardrail: float = 0.002,
    candidate_allocation_head_hit10_loss_guardrail: float = 0.005,
) -> DiagnosisEvidence:
    """Compute one complete, internally consistent evidence result."""
    priority_multipliers = {
        str(group): float(multiplier)
        for group, multiplier in (
            priority_multipliers
            or {
                "Head": 1.0,
                "Mid": 1.1,
                "Tail": 1.25,
                "Tail-Cold": 1.35,
            }
        ).items()
    }
    context = build_diagnosis_context(
        sid_views=batch.sid_views,
        frequencies=batch.frequencies,
        groups_by_item=batch.groups_by_item,
        embeddings=batch.embeddings,
    )
    structural = _compute_structural_values(context)
    asymmetric = _compute_asymmetric_values(context)
    semantic_evidence = _compute_semantic_evidence(
        context,
        max_neighbors_per_bucket=max_neighbors_per_bucket,
        reference_quantile=semantic_reference_quantile,
        bucket_quantile=semantic_bucket_quantile,
        reference_pair_count=semantic_reference_pairs,
        seed=semantic_seed,
        max_pairs=max_harmful_pairs,
        max_pairs_per_item=max_harmful_pairs_per_item,
    )
    semantic = semantic_evidence.primary
    damage_scores = _compute_damage_scores(
        context.groups_by_index,
        structural,
        semantic,
        priority_multipliers=priority_multipliers,
    )
    prefix_rows = _compute_prefix_rows(context, damage_scores, semantic)

    recommendation_by_item, recommendation_rows, recommendation_metadata = _compute_recommendation_evidence(
        batch, hit_ks
    )
    item_rows: list[dict[str, Any]] = []
    for idx, item_id in enumerate(context.item_ids):
        raw_damage = damage_scores.raw_damage[idx]
        row = {
                "item_id": item_id,
                "group": context.groups_by_index[idx],
                "freq_train": context.frequencies.get(item_id, 0),
                "raw_sid": " ".join(str(value) for value in context.sid_views.raw_sid[idx].tolist()),
                "model_sid": " ".join(str(value) for value in context.sid_views.model_sid[idx].tolist()),
                "dedup_digit": int(context.sid_views.dedup_digit[idx].item()),
                "full_collision_flag": int(structural.full_collision_size[idx] > 1),
                "full_collision_size": structural.full_collision_size[idx],
                "mpod_raw": structural.mpod[idx],
                "near_collision_count_strict": structural.near_count[idx],
                "local_density": structural.local_density[idx],
                "suffix_weakness": structural.suffix_weakness[idx],
                "last_step_burden": structural.last_step_burden[idx],
                "semantic_mismatch": semantic.semantic_mismatch[idx],
                "harmful_overlap_count": semantic.harmful_count[idx],
                "bucket_relative_semantic_outlier": semantic_evidence.bucket_relative_outlier[idx],
                "near_overlap_head_count": asymmetric.near_partner_counts[idx]["Head"],
                "near_overlap_mid_count": asymmetric.near_partner_counts[idx]["Mid"],
                "near_overlap_tail_count": asymmetric.near_partner_counts[idx]["Tail"],
                "near_overlap_tail_cold_count": asymmetric.near_partner_counts[idx]["Tail-Cold"],
                "full_collision_head_count": asymmetric.full_partner_counts[idx]["Head"],
                "full_collision_mid_count": asymmetric.full_partner_counts[idx]["Mid"],
                "full_collision_tail_count": asymmetric.full_partner_counts[idx]["Tail"],
                "full_collision_tail_cold_count": asymmetric.full_partner_counts[idx]["Tail-Cold"],
                "head_dominance": asymmetric.head_dominance[idx],
                "tail_isolation_deficit": asymmetric.tail_isolation_deficit[idx],
                "tail_head_near_collision_pressure": asymmetric.tail_head_pressure[idx],
                "harmful_overlap_head_count": semantic_evidence.harmful_partner_counts[idx]["Head"],
                "harmful_overlap_mid_count": semantic_evidence.harmful_partner_counts[idx]["Mid"],
                "harmful_overlap_tail_count": semantic_evidence.harmful_partner_counts[idx]["Tail"],
                "harmful_overlap_tail_cold_count": semantic_evidence.harmful_partner_counts[idx]["Tail-Cold"],
                "raw_damage": raw_damage,
                "priority_score": damage_scores.priority_score[idx],
            }
        row.update(recommendation_by_item.get(item_id, {}))
        item_rows.append(row)

    group_rows = [_group_row(group, item_rows) for group in GROUPS]
    group_by_name = {row["group"]: row for row in group_rows}
    tail = group_by_name["Tail"]
    head = group_by_name["Head"]
    verdict: EvidenceVerdict = {
        "tail_structural_asymmetry": "unavailable",
        "equal_risk_tail_vulnerability": "unavailable",
        "generation_risk_validity": "unavailable",
        "cross_setting_stability": "unavailable",
        "prefix_survival_mechanism": "unavailable",
    }
    structural_section = {
        "num_items": float(len(context.item_ids)),
        "raw_num_hierarchies": float(context.sid_views.raw_sid.size(1)),
        "sid_length": float(context.sid_views.model_sid.size(1)),
        "dedup_digit_nonzero_rate": _mean(
            [1.0 if int(value.item()) != 0 else 0.0 for value in context.sid_views.dedup_digit]
        ),
        "full_collision_rate_tail": _scalar(tail["full_collision_rate"]),
        "near_collision_rate_tail_strict": _scalar(tail["near_collision_rate_strict"]),
        "avg_local_density_tail": _scalar(tail["avg_local_density"]),
        "avg_suffix_weakness_tail": _scalar(tail["avg_suffix_weakness"]),
        "avg_last_step_burden_tail": _scalar(tail["avg_last_step_burden"]),
    }
    semantic_section = {
        "avg_semantic_mismatch_tail": _scalar(tail["avg_semantic_mismatch"]),
        "avg_harmful_overlap_count_tail": _scalar(tail["avg_harmful_overlap_count"]),
    }
    damage_section = {
        "avg_damage_head": _scalar(head["avg_raw_damage"]),
        "avg_damage_mid": _scalar(group_by_name["Mid"]["avg_raw_damage"]),
        "avg_damage_tail": _scalar(tail["avg_raw_damage"]),
        "avg_tail_damage_tail": _scalar(tail["avg_priority_score"]),
        "avg_raw_damage_head": _scalar(head["avg_raw_damage"]),
        "avg_raw_damage_mid": _scalar(group_by_name["Mid"]["avg_raw_damage"]),
        "avg_raw_damage_tail": _scalar(tail["avg_raw_damage"]),
        "avg_priority_score_tail": _scalar(tail["avg_priority_score"]),
        "score_iqr_stability_threshold": SCORE_IQR_STABILITY_THRESHOLD,
        "score_component_clamp_min": SCORE_CONTRIBUTION_MIN,
        "score_component_clamp_max": SCORE_CONTRIBUTION_MAX,
        "score_degenerate_component_count": float(
            sum(1 for row in damage_scores.component_metadata if int(row["is_degenerate"]) == 1)
        ),
    }
    prefix_section = {
        "top_prefix_risk": float(prefix_rows[0]["prefix_risk"]) if prefix_rows else 0.0,
        "avg_prefix_risk": _mean([float(row["prefix_risk"]) for row in prefix_rows]),
        "max_prefix_bucket_size": max([float(row["bucket_size"]) for row in prefix_rows], default=0.0),
    }
    statistics = _compute_statistical_evidence(
        item_rows,
        bootstrap_samples=bootstrap_samples,
        confidence=bootstrap_confidence,
        seed=semantic_seed,
        frequency_bin_count=frequency_bin_count,
        min_bin_support=min_frequency_bin_support,
        hit_ks=hit_ks,
    )
    sensitivity_rows = _compute_sensitivity_rows(
        context,
        structural,
        semantic_evidence,
        item_rows,
        tail_ratios=tail_ratio_sensitivity,
        semantic_quantiles=semantic_quantile_sensitivity,
        component_settings=damage_component_sensitivity,
        primary_head_ratio=head_ratio,
        primary_tail_ratio=tail_ratio,
        max_neighbors_per_bucket=max_neighbors_per_bucket,
        reference_pair_count=semantic_reference_pairs,
        semantic_seed=semantic_seed,
        bucket_quantile=semantic_bucket_quantile,
    )
    verdict.update(_build_verdict(statistics, sensitivity_rows))
    mechanism = build_prefix_mechanism_evidence(
        batch,
        item_rows,
        bootstrap_samples=bootstrap_samples,
        confidence=bootstrap_confidence,
        seed=semantic_seed,
        include_widened_recovery=not candidate_allocation_probe_enabled,
    )
    search_ranking = build_search_ranking_evidence(
        batch,
        item_rows,
        search_ranking_enabled=search_ranking_enabled,
        risk_standardization_enabled=risk_standardization_enabled,
        bin_count=risk_standardization_bin_count,
        bin_count_sensitivity=risk_standardization_bin_count_sensitivity,
        min_items_per_group_per_bin=risk_standardization_min_items_per_group_per_bin,
        min_item_retention=risk_standardization_min_item_retention,
        max_abs_raw_damage_smd=risk_standardization_max_abs_raw_damage_smd,
        bootstrap_samples=bootstrap_samples,
        bootstrap_confidence=bootstrap_confidence,
        min_bootstrap_valid_fraction=risk_standardization_min_bootstrap_valid_fraction,
        seed=semantic_seed,
    )
    candidate_allocation = build_candidate_allocation_evidence(
        batch,
        item_rows,
        enabled=candidate_allocation_probe_enabled,
        overall_hit10_loss_guardrail=candidate_allocation_overall_hit10_loss_guardrail,
        head_hit10_loss_guardrail=candidate_allocation_head_hit10_loss_guardrail,
    )
    verdict["prefix_survival_mechanism"] = mechanism.verdict
    metadata = {
        "schema_version": EVIDENCE_SCHEMA_VERSION,
        "normalization_version": SCORE_NORMALIZATION_VERSION,
        "priority_multipliers": priority_multipliers,
        "component_metadata": damage_scores.component_metadata,
        "semantic_evidence_available": batch.embeddings is not None,
        "recommendation_evidence_available": batch.recommendation is not None,
        "recommendation": recommendation_metadata,
        "prefix_trace": mechanism.summary,
        "search_ranking_analysis": search_ranking.summary,
        "candidate_allocation_probe": candidate_allocation.summary,
        "resolved_inputs": batch.input_metadata,
        "semantic": semantic_evidence.metadata,
        "statistical_settings": {
            "bootstrap_samples": bootstrap_samples,
            "bootstrap_confidence": bootstrap_confidence,
            "frequency_bin_count": frequency_bin_count,
            "min_frequency_bin_support": min_frequency_bin_support,
        },
        "primary_settings": {
            "head_ratio": head_ratio,
            "tail_ratio": tail_ratio,
            "semantic_reference_quantile": semantic_reference_quantile,
            "semantic_bucket_quantile": semantic_bucket_quantile,
            "damage_components": "all",
        },
    }
    summary = {
        "schema_version": EVIDENCE_SCHEMA_VERSION,
        "verdict": verdict,
        "metadata": metadata,
        "structural": structural_section,
        "semantic": semantic_section,
        "damage": damage_section,
        "prefix_risk": prefix_section,
        "prefix_mechanism": mechanism.summary,
        "search_ranking": search_ranking.summary,
        "candidate_allocation": candidate_allocation.summary,
        "statistics": statistics,
    }
    return DiagnosisEvidence(
        summary=summary,
        scalar_sections={
            "structural": structural_section,
            "semantic": semantic_section,
            "damage": damage_section,
            "prefix_risk": prefix_section,
            "prefix_mechanism": mechanism.scalar_section,
            "search_ranking": search_ranking.scalar_section,
            "candidate_allocation": candidate_allocation.scalar_section,
        },
        group_rows=group_rows,
        item_rows=item_rows,
        prefix_rows=prefix_rows,
        harmful_pair_rows=semantic_evidence.pair_rows,
        sensitivity_rows=sensitivity_rows,
        recommendation_rows=recommendation_rows,
        mechanism_tables=mechanism.tables,
        analysis_tables={**search_ranking.tables, **candidate_allocation.tables},
        metadata=metadata,
    )


def _compute_statistical_evidence(
    item_rows: list[dict[str, Any]],
    *,
    bootstrap_samples: int,
    confidence: float,
    seed: int,
    frequency_bin_count: int,
    min_bin_support: int,
    hit_ks: list[int] | tuple[int, ...],
) -> dict[str, Any]:
    if bootstrap_samples <= 0:
        raise ValueError("bootstrap_samples must be positive.")
    if not 0.0 < confidence < 1.0:
        raise ValueError("bootstrap_confidence must be within (0, 1).")
    if frequency_bin_count <= 0 or min_bin_support <= 0:
        raise ValueError("Frequency bin count and minimum support must be positive.")
    primary_components = (
        "raw_damage",
        "near_collision_count_strict",
        "local_density",
        "suffix_weakness",
        "last_step_burden",
        "semantic_mismatch",
        "tail_head_near_collision_pressure",
    )
    generator = torch.Generator(device="cpu").manual_seed(seed)
    bootstrap = {}
    for component in primary_components:
        head = [float(row[component]) for row in item_rows if row["group"] == "Head"]
        tail = [float(row[component]) for row in item_rows if row["group"] == "Tail"]
        bootstrap[component] = _bootstrap_difference(
            head, tail, samples=bootstrap_samples, confidence=confidence, generator=generator
        )

    normalized_ks = sorted({int(k) for k in hit_ks})
    outcome_field = f"hit@{normalized_ks[-1]}" if normalized_ks else None
    outcome_rows = [row for row in item_rows if outcome_field and outcome_field in row]
    associations = {
        "available": bool(outcome_rows),
        "outcome_field": outcome_field,
        "overall": _risk_outcome_association(outcome_rows, outcome_field),
        "tail": _risk_outcome_association(
            [row for row in outcome_rows if row["group"] in {"Tail", "Tail-Cold"}], outcome_field
        ),
    }
    frequency_matched = _frequency_matched_comparison(
        outcome_rows,
        outcome_field,
        bin_count=frequency_bin_count,
        min_support=min_bin_support,
    )
    equal_risk = _equal_risk_group_comparison(
        outcome_rows,
        outcome_field,
        bin_count=frequency_bin_count,
        min_support=min_bin_support,
    )
    return {
        "bootstrap_tail_minus_head": bootstrap,
        "risk_outcome_association": associations,
        "frequency_matched_low_minus_high_damage": frequency_matched,
        "equal_risk_tail_minus_head_outcome": equal_risk,
    }


def _bootstrap_difference(
    head: list[float],
    tail: list[float],
    *,
    samples: int,
    confidence: float,
    generator: torch.Generator,
) -> dict[str, Any]:
    if not head or not tail:
        return {"available": False, "head_support": len(head), "tail_support": len(tail)}
    head_tensor = torch.tensor(head, dtype=torch.float64)
    tail_tensor = torch.tensor(tail, dtype=torch.float64)
    differences = []
    for _ in range(samples):
        head_sample = head_tensor[torch.randint(len(head), (len(head),), generator=generator)]
        tail_sample = tail_tensor[torch.randint(len(tail), (len(tail),), generator=generator)]
        differences.append(float(tail_sample.mean() - head_sample.mean()))
    alpha = (1.0 - confidence) / 2.0
    head_mean = _mean(head)
    tail_mean = _mean(tail)
    return {
        "available": True,
        "head_support": len(head),
        "tail_support": len(tail),
        "head_mean": head_mean,
        "tail_mean": tail_mean,
        "absolute_difference": tail_mean - head_mean,
        "ratio": tail_mean / head_mean if head_mean != 0 else None,
        "confidence": confidence,
        "ci_lower": _quantile(differences, alpha),
        "ci_upper": _quantile(differences, 1.0 - alpha),
    }


def _risk_outcome_association(rows: list[dict[str, Any]], outcome_field: str | None) -> dict[str, Any]:
    if outcome_field is None or len(rows) < 2:
        return {"available": False, "support": len(rows)}
    risk = [float(row["raw_damage"]) for row in rows]
    outcome = [float(row[outcome_field]) for row in rows]
    coefficient = _spearman(risk, outcome)
    return {
        "available": coefficient is not None,
        "support": len(rows),
        "spearman": coefficient,
    }


def _rank_values(values: list[float]) -> list[float]:
    order = sorted(range(len(values)), key=lambda idx: (values[idx], idx))
    ranks = [0.0] * len(values)
    start = 0
    while start < len(order):
        end = start + 1
        while end < len(order) and values[order[end]] == values[order[start]]:
            end += 1
        average_rank = (start + end - 1) / 2.0
        for position in range(start, end):
            ranks[order[position]] = average_rank
        start = end
    return ranks


def _spearman(left: list[float], right: list[float]) -> float | None:
    if len(left) != len(right) or len(left) < 2:
        return None
    left_rank = _rank_values(left)
    right_rank = _rank_values(right)
    left_mean, right_mean = _mean(left_rank), _mean(right_rank)
    numerator = sum(
        (a - left_mean) * (b - right_mean)
        for a, b in zip(left_rank, right_rank, strict=True)
    )
    left_scale = math.sqrt(sum((a - left_mean) ** 2 for a in left_rank))
    right_scale = math.sqrt(sum((b - right_mean) ** 2 for b in right_rank))
    if left_scale == 0 or right_scale == 0:
        return None
    return numerator / (left_scale * right_scale)


def _frequency_matched_comparison(
    rows: list[dict[str, Any]],
    outcome_field: str | None,
    *,
    bin_count: int,
    min_support: int,
) -> dict[str, Any]:
    if outcome_field is None or not rows:
        return {"available": False, "eligible_bins": 0, "rows": []}
    ordered = sorted(rows, key=lambda row: (math.log1p(float(row["freq_train"])), int(row["item_id"])))
    bins: list[list[dict[str, Any]]] = [[] for _ in range(min(bin_count, len(ordered)))]
    for rank, row in enumerate(ordered):
        bins[min(len(bins) - 1, rank * len(bins) // len(ordered))].append(row)
    result_rows = []
    for bin_index, bucket in enumerate(bins):
        damage_median = _quantile([float(row["raw_damage"]) for row in bucket], 0.5)
        low = [row for row in bucket if float(row["raw_damage"]) <= float(damage_median)]
        high = [row for row in bucket if float(row["raw_damage"]) > float(damage_median)]
        eligible = len(low) >= min_support and len(high) >= min_support
        result_rows.append(
            {
                "bin": bin_index,
                "low_support": len(low),
                "high_support": len(high),
                "eligible": eligible,
                "outcome_difference": (
                    _mean([float(row[outcome_field]) for row in low])
                    - _mean([float(row[outcome_field]) for row in high])
                    if eligible
                    else None
                ),
            }
        )
    eligible_values = [float(row["outcome_difference"]) for row in result_rows if row["eligible"]]
    return {
        "available": bool(eligible_values),
        "eligible_bins": len(eligible_values),
        "mean_outcome_difference": _mean(eligible_values) if eligible_values else None,
        "rows": result_rows,
    }


def _equal_risk_group_comparison(
    rows: list[dict[str, Any]],
    outcome_field: str | None,
    *,
    bin_count: int,
    min_support: int,
) -> dict[str, Any]:
    if outcome_field is None or not rows:
        return {"available": False, "eligible_bins": 0}
    ordered = sorted(rows, key=lambda row: (float(row["raw_damage"]), int(row["item_id"])))
    bins: list[list[dict[str, Any]]] = [[] for _ in range(min(bin_count, len(ordered)))]
    for rank, row in enumerate(ordered):
        bins[min(len(bins) - 1, rank * len(bins) // len(ordered))].append(row)
    gaps = []
    for bucket in bins:
        head = [float(row[outcome_field]) for row in bucket if row["group"] == "Head"]
        tail = [float(row[outcome_field]) for row in bucket if row["group"] in {"Tail", "Tail-Cold"}]
        if len(head) >= min_support and len(tail) >= min_support:
            gaps.append(_mean(tail) - _mean(head))
    return {
        "available": bool(gaps),
        "eligible_bins": len(gaps),
        "mean_outcome_difference": _mean(gaps) if gaps else None,
    }


def _compute_sensitivity_rows(
    context,
    structural,
    semantic_evidence: SemanticEvidenceValues,
    item_rows: list[dict[str, Any]],
    *,
    tail_ratios: list[float] | tuple[float, ...],
    semantic_quantiles: list[float] | tuple[float, ...],
    component_settings: list[str] | tuple[str, ...],
    primary_head_ratio: float,
    primary_tail_ratio: float,
    max_neighbors_per_bucket: int,
    reference_pair_count: int,
    semantic_seed: int,
    bucket_quantile: float,
) -> list[dict[str, Any]]:
    rows = []
    for ratio in sorted({float(value) for value in tail_ratios}):
        groups = _groups_for_ratios(context.item_ids, context.frequencies, primary_head_ratio, ratio)
        row = _sensitivity_row("tail_ratio", ratio, groups, [row["raw_damage"] for row in item_rows])
        row["is_primary"] = math.isclose(ratio, primary_tail_ratio)
        rows.append(row)
    for quantile in sorted({float(value) for value in semantic_quantiles}):
        semantic = _compute_semantic_evidence(
            context,
            max_neighbors_per_bucket=max_neighbors_per_bucket,
            reference_quantile=quantile,
            bucket_quantile=bucket_quantile,
            reference_pair_count=reference_pair_count,
            seed=semantic_seed,
            max_pairs=0,
            max_pairs_per_item=0,
        ).primary
        damage = _compute_damage_scores(context.groups_by_index, structural, semantic).raw_damage
        row = _sensitivity_row("semantic_quantile", quantile, context.groups_by_index, damage)
        row["is_primary"] = math.isclose(
            quantile, float(semantic_evidence.metadata.get("reference_quantile", -1.0))
        )
        rows.append(row)
    component_values = {
        "full_collision": [max(0.0, float(value) - 1.0) for value in structural.full_collision_size],
        "near_collision": [float(value) for value in structural.near_count],
        "local_density": structural.local_density,
        "suffix_weakness": structural.suffix_weakness,
        "last_step_burden": structural.last_step_burden,
        "semantic_mismatch": semantic_evidence.primary.semantic_mismatch,
    }
    for setting in component_settings:
        names = {
            "all": tuple(component_values),
            "structural_only": tuple(name for name in component_values if name != "semantic_mismatch"),
            "semantic_only": ("semantic_mismatch",),
        }.get(str(setting))
        if names is None:
            raise ValueError(f"Unknown damage component sensitivity setting: {setting!r}.")
        normalized = [stable_robust_risk_scores(name, component_values[name])[0] for name in names]
        damage = [sum(values[idx] for values in normalized) for idx in range(len(context.item_ids))]
        row = _sensitivity_row("damage_components", str(setting), context.groups_by_index, damage)
        row["is_primary"] = str(setting) == "all"
        rows.append(row)
    return rows


def _groups_for_ratios(
    item_ids: list[int], frequencies: dict[int, int], head_ratio: float, tail_ratio: float
) -> list[str]:
    if head_ratio + tail_ratio > 1.0 or min(head_ratio, tail_ratio) < 0.0:
        raise ValueError("Sensitivity head/tail ratios must be valid proportions.")
    non_cold = sorted(
        (item for item in item_ids if frequencies.get(item, 0) > 0),
        key=lambda item: (-frequencies[item], item),
    )
    mapping = {}
    for rank, item in enumerate(non_cold):
        percentile = rank / len(non_cold) if non_cold else 0.0
        mapping[item] = "Head" if percentile < head_ratio else "Tail" if percentile >= 1 - tail_ratio else "Mid"
    return [mapping.get(item, "Tail-Cold") for item in item_ids]


def _sensitivity_row(setting_type: str, setting_value: Any, groups: list[str], damage: list[float]):
    head = [damage[idx] for idx, group in enumerate(groups) if group == "Head"]
    tail = [damage[idx] for idx, group in enumerate(groups) if group == "Tail"]
    difference = _mean(tail) - _mean(head) if head and tail else None
    return {
        "setting_type": setting_type,
        "setting_value": setting_value,
        "head_support": len(head),
        "tail_support": len(tail),
        "tail_minus_head_raw_damage": difference,
        "direction": 1 if difference is not None and difference > 0 else -1 if difference is not None and difference < 0 else 0,
    }


def _build_verdict(statistics: dict[str, Any], sensitivity_rows: list[dict[str, Any]]) -> EvidenceVerdict:
    result: EvidenceVerdict = {
        "tail_structural_asymmetry": "unavailable",
        "equal_risk_tail_vulnerability": "unavailable",
        "generation_risk_validity": "unavailable",
        "cross_setting_stability": "unavailable",
        "prefix_survival_mechanism": "unavailable",
    }
    raw_damage = statistics["bootstrap_tail_minus_head"]["raw_damage"]
    if raw_damage["available"]:
        result["tail_structural_asymmetry"] = (
            "supported" if float(raw_damage["ci_lower"]) > 0 else "not_supported"
        )
    equal_risk = statistics["equal_risk_tail_minus_head_outcome"]
    if equal_risk["available"]:
        result["equal_risk_tail_vulnerability"] = (
            "supported" if float(equal_risk["mean_outcome_difference"]) < 0 else "not_supported"
        )
    association = statistics["risk_outcome_association"]["overall"]
    matched = statistics["frequency_matched_low_minus_high_damage"]
    if association["available"] and matched["available"]:
        result["generation_risk_validity"] = (
            "supported"
            if float(association["spearman"]) < 0 and float(matched["mean_outcome_difference"]) > 0
            else "not_supported"
        )
    directions = [int(row["direction"]) for row in sensitivity_rows if int(row["direction"]) != 0]
    if directions:
        result["cross_setting_stability"] = (
            "supported" if len(set(directions)) == 1 else "not_supported"
        )
    return result


def _compute_recommendation_evidence(
    batch: DiagnosisBatch, hit_ks: list[int] | tuple[int, ...]
) -> tuple[dict[int, dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    recommendation = batch.recommendation
    if recommendation is None:
        return {}, [], {"available": False}
    normalized_ks = sorted({int(k) for k in hit_ks})
    if not normalized_ks or normalized_ks[0] <= 0:
        raise ValueError("hit_ks must contain positive integers.")
    if recommendation.generated_sids.size(2) != batch.sid_views.model_sid.size(1):
        raise ValueError(
            "Recommendation SID width must match model SID width: "
            f"generated={recommendation.generated_sids.size(2)}, "
            f"model={batch.sid_views.model_sid.size(1)}."
        )
    item_index = {int(item_id): idx for idx, item_id in enumerate(batch.sid_views.item_ids.tolist())}
    aggregates: dict[int, list[dict[str, Any]]] = defaultdict(list)
    user_rows = []
    num_candidates = recommendation.generated_sids.size(1)
    for row_idx, user_id in enumerate(recommendation.user_ids.tolist()):
        label_item_id = int(recommendation.label_item_ids[row_idx].item())
        if label_item_id not in item_index:
            raise KeyError(f"Testing label item {label_item_id} is absent from semantic ID input.")
        label_sid = batch.sid_views.model_sid[item_index[label_item_id]]
        matches = torch.all(recommendation.generated_sids[row_idx] == label_sid, dim=1).nonzero().reshape(-1)
        rank = int(matches[0].item()) + 1 if matches.numel() else num_candidates + 1
        row = {
            "user_id": int(user_id),
            "label_item_id": label_item_id,
            "rank": rank,
            "reciprocal_rank": 1.0 / rank if rank <= num_candidates else 0.0,
        }
        for k in normalized_ks:
            hit = 1.0 if rank <= min(k, num_candidates) else 0.0
            row[f"hit@{k}"] = hit
            row[f"ndcg@{k}"] = 1.0 / math.log2(rank + 1) if hit else 0.0
        user_rows.append(row)
        aggregates[label_item_id].append(row)

    by_item = {}
    for item_id, rows in aggregates.items():
        summary = {
            "label_support": len(rows),
            "avg_rank": _mean([float(row["rank"]) for row in rows]),
            "mean_reciprocal_rank": _mean([float(row["reciprocal_rank"]) for row in rows]),
        }
        for k in normalized_ks:
            summary[f"hit@{k}"] = _mean([float(row[f"hit@{k}"]) for row in rows])
            summary[f"ndcg@{k}"] = _mean([float(row[f"ndcg@{k}"]) for row in rows])
        by_item[item_id] = summary
    return by_item, user_rows, {
        "available": True,
        "num_users": len(user_rows),
        "num_label_items": len(by_item),
        "num_candidates": num_candidates,
        "hit_ks": normalized_ks,
        "missing_rank_value": num_candidates + 1,
    }


def _group_row(group: str, item_rows: list[dict[str, Any]]) -> dict[str, Any]:
    rows = [row for row in item_rows if row["group"] == group]
    raw_damage = [float(row["raw_damage"]) for row in rows]

    def avg(field: str) -> float | None:
        return _mean([float(row[field]) for row in rows]) if rows else None

    def rate(field: str) -> float | None:
        return _mean([1.0 if float(row[field]) > 0 else 0.0 for row in rows]) if rows else None

    return {
        "group": group,
        "num_items": len(rows),
        "avg_freq_train": avg("freq_train"),
        "full_collision_rate": rate("full_collision_flag"),
        "avg_full_collision_size": avg("full_collision_size"),
        "near_collision_rate_strict": rate("near_collision_count_strict"),
        "avg_near_collision_count_strict": avg("near_collision_count_strict"),
        "avg_mpod_raw": avg("mpod_raw"),
        "avg_local_density": avg("local_density"),
        "avg_suffix_weakness": avg("suffix_weakness"),
        "avg_last_step_burden": avg("last_step_burden"),
        "avg_semantic_mismatch": avg("semantic_mismatch"),
        "avg_harmful_overlap_count": avg("harmful_overlap_count"),
        "avg_bucket_relative_semantic_outlier": avg("bucket_relative_semantic_outlier"),
        "avg_near_overlap_head_count": avg("near_overlap_head_count"),
        "avg_near_overlap_mid_count": avg("near_overlap_mid_count"),
        "avg_near_overlap_tail_count": avg("near_overlap_tail_count"),
        "avg_near_overlap_tail_cold_count": avg("near_overlap_tail_cold_count"),
        "avg_full_collision_head_count": avg("full_collision_head_count"),
        "avg_full_collision_mid_count": avg("full_collision_mid_count"),
        "avg_full_collision_tail_count": avg("full_collision_tail_count"),
        "avg_full_collision_tail_cold_count": avg("full_collision_tail_cold_count"),
        "avg_head_dominance": avg("head_dominance"),
        "avg_tail_isolation_deficit": avg("tail_isolation_deficit"),
        "avg_tail_head_near_collision_pressure": avg("tail_head_near_collision_pressure"),
        "avg_raw_damage": avg("raw_damage"),
        "avg_priority_score": avg("priority_score"),
        "p50_raw_damage": _quantile(raw_damage, 0.5),
        "p90_raw_damage": _quantile(raw_damage, 0.9),
        "p95_raw_damage": _quantile(raw_damage, 0.95),
    }


def _quantile(values: list[float], quantile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, math.ceil(quantile * len(ordered)) - 1))
    return float(ordered[index])


def _scalar(value: Any) -> float:
    return float(value) if value is not None else 0.0


def _compute_asymmetric_values(context) -> AsymmetricEvidenceValues:
    raw_sid = context.sid_views.raw_sid
    near_partner_counts = []
    full_partner_counts = []
    head_dominance = []
    tail_isolation_deficit = []
    tail_head_pressure = []
    for idx, sid in enumerate(raw_sid.tolist()):
        full_bucket = context.buckets[context.sid_length][tuple(int(value) for value in sid)]
        strict_bucket = context.buckets[context.strict_depth][
            tuple(int(value) for value in sid[: context.strict_depth])
        ]
        full_neighbors = [other for other in full_bucket if other != idx]
        near_neighbors = [
            other
            for other in strict_bucket
            if other != idx and not torch.equal(raw_sid[idx], raw_sid[other])
        ]
        full_counts = _count_groups(full_neighbors, context.groups_by_index)
        near_counts = _count_groups(near_neighbors, context.groups_by_index)
        full_partner_counts.append(full_counts)
        near_partner_counts.append(near_counts)
        head_dominance.append(
            sum(1 for other in strict_bucket if context.groups_by_index[other] == "Head") / len(strict_bucket)
        )
        isolation = 0.0
        for depth in range(1, context.sid_length + 1):
            bucket = context.buckets[depth][tuple(int(value) for value in sid[:depth])]
            tail_share = (
                sum(1 for other in bucket if context.groups_by_index[other] in {"Tail", "Tail-Cold"}) / len(bucket)
            )
            isolation += (depth / context.sid_length) * (1.0 - tail_share) * math.log1p(len(bucket))
        is_tail = context.groups_by_index[idx] in {"Tail", "Tail-Cold"}
        tail_isolation_deficit.append(isolation if is_tail else 0.0)
        tail_head_pressure.append(
            math.log1p(near_counts["Head"] + full_counts["Head"]) if is_tail else 0.0
        )
    return AsymmetricEvidenceValues(
        near_partner_counts=near_partner_counts,
        full_partner_counts=full_partner_counts,
        head_dominance=head_dominance,
        tail_isolation_deficit=tail_isolation_deficit,
        tail_head_pressure=tail_head_pressure,
    )


def _count_groups(indexes: list[int], groups: list[str]) -> dict[str, int]:
    counts = {group: 0 for group in GROUPS}
    for idx in indexes:
        counts[groups[idx]] += 1
    return counts


def _compute_semantic_evidence(
    context,
    *,
    max_neighbors_per_bucket: int,
    reference_quantile: float,
    bucket_quantile: float,
    reference_pair_count: int,
    seed: int,
    max_pairs: int,
    max_pairs_per_item: int,
) -> SemanticEvidenceValues:
    num_items = len(context.item_ids)
    zeros = [0.0] * num_items
    empty_partner_counts = [{group: 0 for group in GROUPS} for _ in range(num_items)]
    if context.embeddings is None:
        return SemanticEvidenceValues(
            primary=SemanticMismatchValues(zeros, [0] * num_items),
            bucket_relative_outlier=zeros.copy(),
            harmful_partner_counts=empty_partner_counts,
            pair_rows=[],
            metadata={"available": False},
        )
    if num_items < 2:
        return SemanticEvidenceValues(
            primary=SemanticMismatchValues(zeros, [0] * num_items),
            bucket_relative_outlier=zeros.copy(),
            harmful_partner_counts=empty_partner_counts,
            pair_rows=[],
            metadata={
                "available": True,
                "reference_quantile": reference_quantile,
                "reference_threshold": 0.0,
                "reference_pair_count_requested": max(0, reference_pair_count),
                "reference_pair_count_observed": 0,
                "bucket_quantile": bucket_quantile,
                "seed": seed,
                "candidate_pair_count": 0,
                "retained_pair_count": 0,
                "truncated_pair_count": 0,
                "max_pairs": max_pairs,
                "max_pairs_per_item": max_pairs_per_item,
            },
        )
    if not 0.0 <= reference_quantile <= 1.0 or not 0.0 <= bucket_quantile <= 1.0:
        raise ValueError("Semantic quantiles must be within [0, 1].")

    normalized = F.normalize(context.embeddings, dim=1)
    generator = torch.Generator(device="cpu").manual_seed(seed)
    sample_target = max(0, reference_pair_count)
    first = torch.randint(num_items, (sample_target,), generator=generator)
    second = torch.randint(num_items, (sample_target,), generator=generator)
    valid = first != second
    first, second = first[valid], second[valid]
    reference_sims = (normalized[first] * normalized[second]).sum(dim=1)
    global_threshold = (
        float(torch.quantile(reference_sims, reference_quantile).item()) if reference_sims.numel() else 0.0
    )

    primary_sum = [0.0] * num_items
    primary_count = [0] * num_items
    bucket_sum = [0.0] * num_items
    bucket_count = [0] * num_items
    harmful_count = [0] * num_items
    partner_counts = [{group: 0 for group in GROUPS} for _ in range(num_items)]
    candidates = []
    raw_sid = context.sid_views.raw_sid
    for bucket_indexes in context.buckets[context.strict_depth].values():
        selected = list(bucket_indexes)
        if len(selected) > max_neighbors_per_bucket:
            tail = [idx for idx in selected if context.groups_by_index[idx] in {"Tail", "Tail-Cold"}]
            tail_set = set(tail)
            other = [idx for idx in selected if idx not in tail_set]
            selected = tail + other[: max(0, max_neighbors_per_bucket - len(tail))]
        if len(selected) < 2:
            continue
        sims = normalized[selected] @ normalized[selected].T
        off_diag = sims[~torch.eye(len(selected), dtype=torch.bool, device=sims.device)]
        bucket_threshold = float(torch.quantile(off_diag, bucket_quantile).item()) if off_diag.numel() else 0.0
        for local_i, item_i in enumerate(selected):
            for local_j in range(local_i + 1, len(selected)):
                item_j = selected[local_j]
                if torch.equal(raw_sid[item_i], raw_sid[item_j]):
                    continue
                similarity = float(sims[local_i, local_j].item())
                primary_mismatch = max(0.0, global_threshold - similarity)
                bucket_outlier = max(0.0, bucket_threshold - similarity)
                for source in (item_i, item_j):
                    primary_sum[source] += primary_mismatch
                    primary_count[source] += 1
                    bucket_sum[source] += bucket_outlier
                    bucket_count[source] += 1
                involves_tail = any(
                    context.groups_by_index[item] in {"Tail", "Tail-Cold"} for item in (item_i, item_j)
                )
                if primary_mismatch <= 0 or not involves_tail:
                    continue
                harmful_count[item_i] += 1
                harmful_count[item_j] += 1
                partner_counts[item_i][context.groups_by_index[item_j]] += 1
                partner_counts[item_j][context.groups_by_index[item_i]] += 1
                candidates.append(
                    {
                        "item_i": context.item_ids[item_i],
                        "item_j": context.item_ids[item_j],
                        "group_i": context.groups_by_index[item_i],
                        "group_j": context.groups_by_index[item_j],
                        "prefix_overlap_depth": context.strict_depth,
                        "cosine_similarity": similarity,
                        "semantic_mismatch_global": primary_mismatch,
                        "bucket_relative_semantic_outlier": bucket_outlier,
                        "freq_i": context.frequencies.get(context.item_ids[item_i], 0),
                        "freq_j": context.frequencies.get(context.item_ids[item_j], 0),
                    }
                )
    primary = [primary_sum[idx] / primary_count[idx] if primary_count[idx] else 0.0 for idx in range(num_items)]
    bucket_relative = [bucket_sum[idx] / bucket_count[idx] if bucket_count[idx] else 0.0 for idx in range(num_items)]
    retained = _limit_harmful_pairs(candidates, max_pairs, max_pairs_per_item)
    return SemanticEvidenceValues(
        primary=SemanticMismatchValues(primary, harmful_count),
        bucket_relative_outlier=bucket_relative,
        harmful_partner_counts=partner_counts,
        pair_rows=retained,
        metadata={
            "available": True,
            "reference_quantile": reference_quantile,
            "reference_threshold": global_threshold,
            "reference_pair_count_requested": sample_target,
            "reference_pair_count_observed": int(reference_sims.numel()),
            "bucket_quantile": bucket_quantile,
            "seed": seed,
            "candidate_pair_count": len(candidates),
            "retained_pair_count": len(retained),
            "truncated_pair_count": len(candidates) - len(retained),
            "max_pairs": max_pairs,
            "max_pairs_per_item": max_pairs_per_item,
        },
    )


def _limit_harmful_pairs(
    candidates: list[dict[str, Any]], max_pairs: int, max_pairs_per_item: int
) -> list[dict[str, Any]]:
    if max_pairs <= 0 or max_pairs_per_item <= 0:
        return []
    ordered = sorted(
        candidates,
        key=lambda row: (-float(row["semantic_mismatch_global"]), int(row["item_i"]), int(row["item_j"])),
    )
    counts: defaultdict[int, int] = defaultdict(int)
    retained = []
    for row in ordered:
        item_i, item_j = int(row["item_i"]), int(row["item_j"])
        if counts[item_i] >= max_pairs_per_item or counts[item_j] >= max_pairs_per_item:
            continue
        retained.append(row)
        counts[item_i] += 1
        counts[item_j] += 1
        if len(retained) >= max_pairs:
            break
    return retained
