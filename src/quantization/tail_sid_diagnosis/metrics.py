from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass
from statistics import median

import torch
import torch.nn.functional as F
from torchmetrics import Metric

from src.data.components.data_models import SIDViews

SCORE_NORMALIZATION_VERSION = "positive_robust_iqr_v1"
SCORE_IQR_STABILITY_THRESHOLD = 1e-6
SCORE_CONTRIBUTION_MIN = 0.0
SCORE_CONTRIBUTION_MAX = 5.0
SCORE_COMPONENT_NAMES = (
    "full_collision",
    "near_collision_strict",
    "local_density",
    "suffix_weakness",
    "semantic_mismatch",
    "last_step_burden",
)


@dataclass(frozen=True)
class DiagnosisContext:
    sid_views: SIDViews
    frequencies: dict[int, int]
    groups_by_item: dict[int, str]
    embeddings: torch.Tensor | None
    item_ids: list[int]
    groups_by_index: list[str]
    buckets: dict[int, dict[tuple[int, ...], list[int]]]
    strict_depth: int
    sid_length: int


@dataclass(frozen=True)
class StructuralMetricValues:
    full_collision_size: list[int]
    mpod: list[int]
    near_count: list[int]
    local_density: list[float]
    suffix_weakness: list[float]
    last_step_burden: list[float]
    density_by_depth: list[list[int]]


@dataclass(frozen=True)
class SemanticMismatchValues:
    semantic_mismatch: list[float]
    harmful_count: list[int]


@dataclass(frozen=True)
class DamageScoreValues:
    damage: list[float]
    tail_damage: list[float]
    component_metadata: list[dict[str, float | int | str]]


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def build_diagnosis_context(
    sid_views: SIDViews,
    frequencies: dict[int, int],
    groups_by_item: dict[int, str],
    embeddings: torch.Tensor | None,
) -> DiagnosisContext:
    raw_sid = sid_views.raw_sid
    sid_length = raw_sid.size(1)
    item_ids = [int(item_id) for item_id in sid_views.item_ids.tolist()]
    return DiagnosisContext(
        sid_views=sid_views,
        frequencies=frequencies,
        groups_by_item=groups_by_item,
        embeddings=embeddings,
        item_ids=item_ids,
        groups_by_index=[groups_by_item[item_id] for item_id in item_ids],
        buckets=build_prefix_buckets(raw_sid),
        strict_depth=max(1, sid_length - 1),
        sid_length=sid_length,
    )


def build_prefix_buckets(raw_sid: torch.Tensor) -> dict[int, dict[tuple[int, ...], list[int]]]:
    buckets: dict[int, dict[tuple[int, ...], list[int]]] = {}
    for depth in range(1, raw_sid.size(1) + 1):
        depth_buckets: dict[tuple[int, ...], list[int]] = defaultdict(list)
        for row_idx, sid in enumerate(raw_sid.tolist()):
            depth_buckets[tuple(int(value) for value in sid[:depth])].append(row_idx)
        buckets[depth] = dict(depth_buckets)
    return buckets


class StructuralSIDMetric(Metric):
    """Computes loggable structural SID collision, density, and suffix-burden metrics."""

    full_state_update = False

    def __init__(self):
        super().__init__()
        self.add_state("contexts", default=[], dist_reduce_fx=None)

    def update(
        self,
        sid_views: SIDViews,
        item_ids: list[int],
        groups_by_index: list[str],
        buckets: dict[int, dict[tuple[int, ...], list[int]]],
        strict_depth: int,
        sid_length: int,
    ) -> None:
        self.contexts.append(
            _context_from_pre_state(
                sid_views=sid_views,
                frequencies={},
                groups_by_item={},
                embeddings=None,
                item_ids=item_ids,
                groups_by_index=groups_by_index,
                buckets=buckets,
                strict_depth=strict_depth,
                sid_length=sid_length,
            )
        )

    def compute(self) -> dict[str, float]:
        context = _single_context(self.contexts, self.__class__.__name__)
        structural = _compute_structural_values(context)
        tail_indexes = _indexes_for_group(context.groups_by_index, "Tail")
        return {
            "num_items": float(len(context.item_ids)),
            "raw_num_hierarchies": float(context.sid_views.raw_sid.size(1)),
            "sid_length": float(context.sid_views.model_sid.size(1)),
            "dedup_digit_nonzero_rate": _mean(
                [1.0 if int(value.item()) != 0 else 0.0 for value in context.sid_views.dedup_digit]
            ),
            "full_collision_rate_tail": _mean(
                [1.0 if structural.full_collision_size[idx] > 1 else 0.0 for idx in tail_indexes]
            ),
            "near_collision_rate_tail_strict": _mean(
                [1.0 if structural.near_count[idx] > 0 else 0.0 for idx in tail_indexes]
            ),
            "avg_local_density_tail": _mean([structural.local_density[idx] for idx in tail_indexes]),
            "avg_suffix_weakness_tail": _mean([structural.suffix_weakness[idx] for idx in tail_indexes]),
            "avg_last_step_burden_tail": _mean([structural.last_step_burden[idx] for idx in tail_indexes]),
        }


class SemanticMismatchMetric(Metric):
    """Computes embedding-based semantic mismatch for strict SID-overlap neighbors."""

    full_state_update = False

    def __init__(self, max_neighbors_per_bucket: int = 512):
        super().__init__()
        self.max_neighbors_per_bucket = max_neighbors_per_bucket
        self.add_state("contexts", default=[], dist_reduce_fx=None)

    def update(
        self,
        sid_views: SIDViews,
        groups_by_index: list[str],
        buckets: dict[int, dict[tuple[int, ...], list[int]]],
        strict_depth: int,
        embeddings: torch.Tensor | None,
    ) -> None:
        self.contexts.append(
            _context_from_pre_state(
                sid_views=sid_views,
                frequencies={},
                groups_by_item={},
                embeddings=embeddings,
                item_ids=[int(item_id) for item_id in sid_views.item_ids.tolist()],
                groups_by_index=groups_by_index,
                buckets=buckets,
                strict_depth=strict_depth,
                sid_length=sid_views.raw_sid.size(1),
            )
        )

    def compute(self) -> dict[str, float]:
        context = _single_context(self.contexts, self.__class__.__name__)
        semantic = _compute_semantic_values(context, self.max_neighbors_per_bucket)
        tail_indexes = _indexes_for_group(context.groups_by_index, "Tail")
        return {
            "avg_semantic_mismatch_tail": _mean([semantic.semantic_mismatch[idx] for idx in tail_indexes]),
            "avg_harmful_overlap_count_tail": _mean([float(semantic.harmful_count[idx]) for idx in tail_indexes]),
        }


class DamageScoreMetric(Metric):
    """Computes normalized composite damage and tail-weighted damage."""

    full_state_update = False

    def __init__(self, max_neighbors_per_bucket: int = 512):
        super().__init__()
        self.max_neighbors_per_bucket = max_neighbors_per_bucket
        self.add_state("inputs", default=[], dist_reduce_fx=None)

    def update(
        self,
        sid_views: SIDViews,
        groups_by_index: list[str],
        buckets: dict[int, dict[tuple[int, ...], list[int]]],
        strict_depth: int,
        sid_length: int,
        embeddings: torch.Tensor | None,
    ) -> None:
        context = _context_from_pre_state(
            sid_views=sid_views,
            frequencies={},
            groups_by_item={},
            embeddings=embeddings,
            item_ids=[int(item_id) for item_id in sid_views.item_ids.tolist()],
            groups_by_index=groups_by_index,
            buckets=buckets,
            strict_depth=strict_depth,
            sid_length=sid_length,
        )
        self.inputs.append(context)

    def compute(self) -> dict[str, float]:
        if len(self.inputs) != 1:
            raise ValueError(f"{self.__class__.__name__} expects exactly one complete input.")
        context = self.inputs[0]
        structural = _compute_structural_values(context)
        semantic = _compute_semantic_values(context, self.max_neighbors_per_bucket)
        damage_scores = _compute_damage_scores(context.groups_by_index, structural, semantic)
        by_group = _values_by_group(context.groups_by_index, damage_scores.damage)
        tail_by_group = _values_by_group(context.groups_by_index, damage_scores.tail_damage)
        return {
            "avg_damage_head": _mean(by_group["Head"]),
            "avg_damage_mid": _mean(by_group["Mid"]),
            "avg_damage_tail": _mean(by_group["Tail"]),
            "avg_tail_damage_tail": _mean(tail_by_group["Tail"]),
            "score_iqr_stability_threshold": SCORE_IQR_STABILITY_THRESHOLD,
            "score_component_clamp_min": SCORE_CONTRIBUTION_MIN,
            "score_component_clamp_max": SCORE_CONTRIBUTION_MAX,
            "score_degenerate_component_count": float(
                sum(1 for row in damage_scores.component_metadata if int(row["is_degenerate"]) == 1)
            ),
        }


class PrefixRiskMetric(Metric):
    """Computes prefix-level risk rows from shared buckets and item scores."""

    full_state_update = False

    def __init__(self, max_neighbors_per_bucket: int = 512):
        super().__init__()
        self.max_neighbors_per_bucket = max_neighbors_per_bucket
        self.add_state("inputs", default=[], dist_reduce_fx=None)

    def update(
        self,
        sid_views: SIDViews,
        frequencies: dict[int, int],
        groups_by_item: dict[int, str],
        embeddings: torch.Tensor | None,
        item_ids: list[int],
        groups_by_index: list[str],
        buckets: dict[int, dict[tuple[int, ...], list[int]]],
        strict_depth: int,
        sid_length: int,
    ) -> None:
        self.inputs.append(
            _context_from_pre_state(
                sid_views=sid_views,
                frequencies=frequencies,
                groups_by_item=groups_by_item,
                embeddings=embeddings,
                item_ids=item_ids,
                groups_by_index=groups_by_index,
                buckets=buckets,
                strict_depth=strict_depth,
                sid_length=sid_length,
            )
        )

    def compute(self) -> dict[str, float]:
        if len(self.inputs) != 1:
            raise ValueError(f"{self.__class__.__name__} expects exactly one complete input.")
        context = self.inputs[0]
        structural = _compute_structural_values(context)
        semantic = _compute_semantic_values(context, self.max_neighbors_per_bucket)
        damage_scores = _compute_damage_scores(context.groups_by_index, structural, semantic)
        rows = _compute_prefix_rows(context, damage_scores, semantic)
        return {
            "top_prefix_risk": float(rows[0]["prefix_risk"]) if rows else 0.0,
            "avg_prefix_risk": _mean([float(row["prefix_risk"]) for row in rows]),
            "max_prefix_bucket_size": max([float(row["bucket_size"]) for row in rows], default=0.0),
        }


def stable_robust_risk_scores(
    component_name: str,
    values: list[float],
) -> tuple[list[float], dict[str, float | int | str]]:
    if not values:
        return [], {
            "component": component_name,
            "median": 0.0,
            "iqr": 0.0,
            "is_degenerate": 1,
        }
    sorted_values = sorted(values)
    med = median(sorted_values)
    q1 = sorted_values[len(sorted_values) // 4]
    q3 = sorted_values[(len(sorted_values) * 3) // 4]
    iqr = q3 - q1
    metadata = {
        "component": component_name,
        "median": float(med),
        "iqr": float(iqr),
        "is_degenerate": int(iqr < SCORE_IQR_STABILITY_THRESHOLD),
    }
    if iqr < SCORE_IQR_STABILITY_THRESHOLD:
        return [0.0 for _ in values], metadata

    scores = []
    for value in values:
        robust_score = (value - med) / iqr
        scores.append(min(max(robust_score, SCORE_CONTRIBUTION_MIN), SCORE_CONTRIBUTION_MAX))
    return scores, metadata


def _context_from_pre_state(
    sid_views: SIDViews,
    frequencies: dict[int, int],
    groups_by_item: dict[int, str],
    embeddings: torch.Tensor | None,
    item_ids: list[int],
    groups_by_index: list[str],
    buckets: dict[int, dict[tuple[int, ...], list[int]]],
    strict_depth: int,
    sid_length: int,
) -> DiagnosisContext:
    return DiagnosisContext(
        sid_views=sid_views,
        frequencies=frequencies,
        groups_by_item=groups_by_item,
        embeddings=embeddings,
        item_ids=item_ids,
        groups_by_index=groups_by_index,
        buckets=buckets,
        strict_depth=strict_depth,
        sid_length=sid_length,
    )


def _single_context(contexts: list[DiagnosisContext], metric_name: str) -> DiagnosisContext:
    if len(contexts) != 1:
        raise ValueError(f"{metric_name} expects exactly one complete diagnosis context.")
    return contexts[0]


def _indexes_for_group(groups_by_index: list[str], group: str) -> list[int]:
    return [idx for idx, value in enumerate(groups_by_index) if value == group]


def _values_by_group(groups_by_index: list[str], values: list[float]) -> dict[str, list[float]]:
    grouped = {group: [] for group in ["Head", "Mid", "Tail", "Tail-Cold"]}
    for idx, group in enumerate(groups_by_index):
        grouped[group].append(values[idx])
    return grouped


def _compute_structural_values(context: DiagnosisContext) -> StructuralMetricValues:
    raw_sid = context.sid_views.raw_sid
    full_collision_size: list[int] = []
    mpod: list[int] = []
    near_count: list[int] = []
    local_density: list[float] = []
    suffix_weakness: list[float] = []
    last_step_burden: list[float] = []
    density_by_depth: list[list[int]] = [[] for _ in range(len(context.item_ids))]

    for idx, sid in enumerate(raw_sid.tolist()):
        sid_tuple = tuple(int(value) for value in sid)
        full_bucket = context.buckets[context.sid_length][sid_tuple]
        full_size = len(full_bucket)
        full_collision_size.append(full_size)

        item_mpod = 0
        item_local_density = 0.0
        for depth in range(1, context.sid_length + 1):
            prefix = tuple(int(value) for value in sid[:depth])
            density = len(context.buckets[depth][prefix])
            density_by_depth[idx].append(density)
            if density > 1:
                item_mpod = depth
            item_local_density += (depth / context.sid_length) * math.log1p(density)
        mpod.append(item_mpod)
        local_density.append(item_local_density)

        strict_prefix = tuple(int(value) for value in sid[: context.strict_depth])
        strict_bucket = context.buckets[context.strict_depth][strict_prefix]
        near_count.append(max(0, len(strict_bucket) - full_size))
        last_step_burden.append(math.log1p(len(strict_bucket)) if len(strict_bucket) > 1 and full_size == 1 else 0.0)

        valid_depths = range(max(1, math.ceil(context.sid_length / 2)), context.sid_length)
        uniqueness_values = []
        for depth in valid_depths:
            prefix = tuple(int(value) for value in sid[:depth])
            bucket = context.buckets[depth][prefix]
            suffixes = {tuple(int(value) for value in raw_sid[item_idx, depth:].tolist()) for item_idx in bucket}
            uniqueness_values.append(len(suffixes) / len(bucket) if bucket else 1.0)
        suffix_weakness.append(1.0 - min(uniqueness_values) if uniqueness_values else 0.0)

    return StructuralMetricValues(
        full_collision_size=full_collision_size,
        mpod=mpod,
        near_count=near_count,
        local_density=local_density,
        suffix_weakness=suffix_weakness,
        last_step_burden=last_step_burden,
        density_by_depth=density_by_depth,
    )


def _compute_semantic_values(context: DiagnosisContext, max_neighbors_per_bucket: int) -> SemanticMismatchValues:
    raw_sid = context.sid_views.raw_sid
    num_items = raw_sid.size(0)
    mismatch_sum = [0.0 for _ in range(num_items)]
    mismatch_count = [0 for _ in range(num_items)]
    harmful_count = [0 for _ in range(num_items)]
    if context.embeddings is None:
        return SemanticMismatchValues(mismatch_sum, harmful_count)

    normalized = F.normalize(context.embeddings, dim=1)
    for item_indexes in context.buckets[context.strict_depth].values():
        if len(item_indexes) < 2:
            continue
        selected = item_indexes
        if len(selected) > max_neighbors_per_bucket:
            tail_indexes = [idx for idx in selected if context.groups_by_index[idx] in {"Tail", "Tail-Cold"}]
            remaining = [idx for idx in selected if idx not in set(tail_indexes)]
            selected = tail_indexes + remaining[: max(0, max_neighbors_per_bucket - len(tail_indexes))]
        if len(selected) < 2:
            continue

        sims = normalized[selected] @ normalized[selected].T
        off_diag = sims[~torch.eye(len(selected), dtype=torch.bool, device=sims.device)]
        tau = float(torch.quantile(off_diag, 0.25).item()) if off_diag.numel() > 0 else 0.0
        for local_i, item_i in enumerate(selected):
            values: list[float] = []
            harmful = 0
            for local_j, item_j in enumerate(selected):
                if item_i == item_j:
                    continue
                if torch.equal(raw_sid[item_i], raw_sid[item_j]):
                    continue
                sim = float(sims[local_i, local_j].item())
                mismatch = max(0.0, tau - sim)
                values.append(mismatch)
                if mismatch > 0 and (
                    context.groups_by_index[item_i] in {"Tail", "Tail-Cold"}
                    or context.groups_by_index[item_j] in {"Tail", "Tail-Cold"}
                ):
                    harmful += 1
            if values:
                mismatch_sum[item_i] += sum(values)
                mismatch_count[item_i] += len(values)
                harmful_count[item_i] += harmful

    mismatch = [mismatch_sum[idx] / mismatch_count[idx] if mismatch_count[idx] else 0.0 for idx in range(num_items)]
    return SemanticMismatchValues(mismatch, harmful_count)


def _compute_damage_scores(
    groups_by_index: list[str],
    structural: StructuralMetricValues,
    semantic: SemanticMismatchValues,
) -> DamageScoreValues:
    z_inputs = [
        ("full_collision", [1.0 if size > 1 else 0.0 for size in structural.full_collision_size]),
        ("near_collision_strict", [float(value) for value in structural.near_count]),
        ("local_density", structural.local_density),
        ("suffix_weakness", structural.suffix_weakness),
        ("semantic_mismatch", semantic.semantic_mismatch),
        ("last_step_burden", structural.last_step_burden),
    ]
    z_components = []
    component_metadata = []
    for component_name, values in z_inputs:
        scores, metadata = stable_robust_risk_scores(component_name, values)
        z_components.append(scores)
        component_metadata.append(metadata)
    damage = [sum(component[idx] for component in z_components) for idx in range(len(groups_by_index))]
    gate = {"Head": 1.0, "Mid": 1.1, "Tail": 1.25, "Tail-Cold": 1.35}
    tail_damage = [damage[idx] * gate[groups_by_index[idx]] for idx in range(len(groups_by_index))]
    return DamageScoreValues(damage=damage, tail_damage=tail_damage, component_metadata=component_metadata)


def _compute_prefix_rows(
    context: DiagnosisContext,
    damage_scores: DamageScoreValues,
    semantic: SemanticMismatchValues,
) -> list[dict[str, float | int | str]]:
    rows: list[dict[str, float | int | str]] = []
    raw_sid = context.sid_views.raw_sid
    for depth, depth_buckets in context.buckets.items():
        for prefix, indexes in depth_buckets.items():
            group_counts = {
                group: sum(1 for idx in indexes if context.groups_by_index[idx] == group)
                for group in ["Head", "Mid", "Tail", "Tail-Cold"]
            }
            tail_count = group_counts["Tail"] + group_counts["Tail-Cold"]
            suffix_uniqueness = 1.0
            if depth < raw_sid.size(1):
                suffixes = {tuple(int(value) for value in raw_sid[idx, depth:].tolist()) for idx in indexes}
                suffix_uniqueness = len(suffixes) / len(indexes)
            avg_tail_damage = _mean([damage_scores.tail_damage[idx] for idx in indexes])
            mismatch_rate = _mean([1.0 if semantic.semantic_mismatch[idx] > 0 else 0.0 for idx in indexes])
            tail_ratio = tail_count / len(indexes)
            rows.append(
                {
                    "prefix_depth": depth,
                    "prefix": " ".join(str(value) for value in prefix),
                    "bucket_size": len(indexes),
                    "head_count": group_counts["Head"],
                    "mid_count": group_counts["Mid"],
                    "tail_count": group_counts["Tail"],
                    "tail_cold_count": group_counts["Tail-Cold"],
                    "tail_ratio": tail_ratio,
                    "avg_tail_damage": avg_tail_damage,
                    "mismatch_rate": mismatch_rate,
                    "suffix_uniqueness": suffix_uniqueness,
                    "prefix_risk": avg_tail_damage
                    * math.log1p(len(indexes))
                    * (1.0 + tail_ratio)
                    * (1.0 + mismatch_rate),
                }
            )
    rows.sort(key=lambda row: float(row["prefix_risk"]), reverse=True)
    return rows

