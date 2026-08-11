from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass
from statistics import median

import torch
import torch.nn.functional as F
from torchmetrics import Metric

from src.quantization.tail_sid_diagnosis.data import SIDViews

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
class DiagnosisResult:
    summary: dict[str, float | int | str | None]
    group_rows: list[dict[str, float | int | str]]
    item_rows: list[dict[str, float | int | str]]
    prefix_rows: list[dict[str, float | int | str]]


def assign_frequency_groups(
    item_ids: torch.Tensor,
    frequencies: dict[int, int],
    head_ratio: float,
    tail_ratio: float,
) -> dict[int, str]:
    if not 0 <= head_ratio <= 1 or not 0 <= tail_ratio <= 1 or head_ratio + tail_ratio > 1:
        raise ValueError("head_ratio and tail_ratio must be within [0, 1] and sum to at most 1.")

    groups: dict[int, str] = {}
    non_cold = [int(item_id) for item_id in item_ids.tolist() if frequencies.get(int(item_id), 0) > 0]
    non_cold.sort(key=lambda item_id: (-frequencies[item_id], item_id))
    total = len(non_cold)

    for rank, item_id in enumerate(non_cold):
        percentile = rank / total if total else 0.0
        if percentile < head_ratio:
            groups[item_id] = "Head"
        elif percentile >= 1.0 - tail_ratio:
            groups[item_id] = "Tail"
        else:
            groups[item_id] = "Mid"

    for item_id in item_ids.tolist():
        item_id = int(item_id)
        if frequencies.get(item_id, 0) == 0:
            groups[item_id] = "Tail-Cold"
    return groups


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


class TailSIDDiagnosisMetric(Metric):
    """Structured metric entrypoint for offline Tail-SID diagnosis."""

    full_state_update = False

    def __init__(self, max_neighbors_per_bucket: int = 512):
        super().__init__()
        self.max_neighbors_per_bucket = max_neighbors_per_bucket
        self.add_state("sid_views", default=[], dist_reduce_fx=None)
        self.add_state("frequencies", default=[], dist_reduce_fx=None)
        self.add_state("groups_by_item", default=[], dist_reduce_fx=None)
        self.add_state("embeddings", default=[], dist_reduce_fx=None)

    def update(
        self,
        sid_views: SIDViews,
        frequencies: dict[int, int],
        groups_by_item: dict[int, str],
        embeddings: torch.Tensor | None,
    ) -> None:
        self.sid_views.append(sid_views)
        self.frequencies.append(frequencies)
        self.groups_by_item.append(groups_by_item)
        self.embeddings.append(embeddings)

    def compute(self) -> DiagnosisResult:
        if len(self.sid_views) != 1:
            raise ValueError("TailSIDDiagnosisMetric expects exactly one complete diagnosis input.")

        sid_views: SIDViews = self.sid_views[0]
        frequencies: dict[int, int] = self.frequencies[0]
        groups_by_item: dict[int, str] = self.groups_by_item[0]
        embeddings: torch.Tensor | None = self.embeddings[0]

        raw_sid = sid_views.raw_sid
        sid_length = raw_sid.size(1)
        item_ids = [int(item_id) for item_id in sid_views.item_ids.tolist()]
        groups_by_index = [groups_by_item[item_id] for item_id in item_ids]
        buckets = self._build_prefix_buckets(raw_sid)
        strict_depth = max(1, sid_length - 1)

        item_metrics = self._compute_item_metric_values(
            sid_views=sid_views,
            frequencies=frequencies,
            groups_by_index=groups_by_index,
            buckets=buckets,
            strict_depth=strict_depth,
            embeddings=embeddings,
        )
        item_rows = self._compute_item_rows(
            sid_views=sid_views,
            frequencies=frequencies,
            groups_by_index=groups_by_index,
            item_metrics=item_metrics,
        )
        prefix_rows = self._compute_prefix_rows(
            raw_sid=raw_sid,
            buckets=buckets,
            groups_by_index=groups_by_index,
            tail_damage=item_metrics.tail_damage,
            semantic_mismatch=item_metrics.semantic_mismatch,
        )
        group_rows = self._compute_group_rows(item_rows)
        summary = self._compute_summary(
            item_rows=item_rows,
            group_rows=group_rows,
            sid_views=sid_views,
            component_metadata=item_metrics.component_metadata,
        )
        return DiagnosisResult(summary=summary, group_rows=group_rows, item_rows=item_rows, prefix_rows=prefix_rows)

    @staticmethod
    def _build_prefix_buckets(raw_sid: torch.Tensor) -> dict[int, dict[tuple[int, ...], list[int]]]:
        buckets: dict[int, dict[tuple[int, ...], list[int]]] = {}
        for depth in range(1, raw_sid.size(1) + 1):
            depth_buckets: dict[tuple[int, ...], list[int]] = defaultdict(list)
            for row_idx, sid in enumerate(raw_sid.tolist()):
                depth_buckets[tuple(int(value) for value in sid[:depth])].append(row_idx)
            buckets[depth] = dict(depth_buckets)
        return buckets

    def _compute_item_metric_values(
        self,
        sid_views: SIDViews,
        frequencies: dict[int, int],
        groups_by_index: list[str],
        buckets: dict[int, dict[tuple[int, ...], list[int]]],
        strict_depth: int,
        embeddings: torch.Tensor | None,
    ) -> _ItemMetricValues:
        raw_sid = sid_views.raw_sid
        sid_length = raw_sid.size(1)
        item_ids = [int(item_id) for item_id in sid_views.item_ids.tolist()]

        full_collision_size: list[int] = []
        mpod: list[int] = []
        near_count: list[int] = []
        local_density: list[float] = []
        suffix_weakness: list[float] = []
        last_step_burden: list[float] = []
        density_by_depth: list[list[int]] = [[] for _ in range(len(item_ids))]

        for idx, sid in enumerate(raw_sid.tolist()):
            sid_tuple = tuple(int(value) for value in sid)
            full_bucket = buckets[sid_length][sid_tuple]
            full_size = len(full_bucket)
            full_collision_size.append(full_size)

            item_mpod = 0
            item_local_density = 0.0
            for depth in range(1, sid_length + 1):
                prefix = tuple(int(value) for value in sid[:depth])
                density = len(buckets[depth][prefix])
                density_by_depth[idx].append(density)
                if density > 1:
                    item_mpod = depth
                item_local_density += (depth / sid_length) * math.log1p(density)
            mpod.append(item_mpod)
            local_density.append(item_local_density)

            strict_prefix = tuple(int(value) for value in sid[:strict_depth])
            strict_bucket = buckets[strict_depth][strict_prefix]
            near_count.append(max(0, len(strict_bucket) - full_size))
            last_step_burden.append(math.log1p(len(strict_bucket)) if len(strict_bucket) > 1 and full_size == 1 else 0.0)

            valid_depths = range(max(1, math.ceil(sid_length / 2)), sid_length)
            uniqueness_values = []
            for depth in valid_depths:
                prefix = tuple(int(value) for value in sid[:depth])
                bucket = buckets[depth][prefix]
                suffixes = {tuple(int(value) for value in raw_sid[item_idx, depth:].tolist()) for item_idx in bucket}
                uniqueness_values.append(len(suffixes) / len(bucket) if bucket else 1.0)
            suffix_weakness.append(1.0 - min(uniqueness_values) if uniqueness_values else 0.0)

        semantic_mismatch, harmful_count = self._compute_semantic_mismatch(
            raw_sid=raw_sid,
            groups_by_index=groups_by_index,
            embeddings=embeddings,
            strict_depth=strict_depth,
            buckets=buckets,
        )
        damage, tail_damage, component_metadata = self._compute_damage_scores(
            groups_by_index=groups_by_index,
            full_collision_size=full_collision_size,
            near_count=near_count,
            local_density=local_density,
            suffix_weakness=suffix_weakness,
            semantic_mismatch=semantic_mismatch,
            last_step_burden=last_step_burden,
        )
        return _ItemMetricValues(
            full_collision_size=full_collision_size,
            mpod=mpod,
            near_count=near_count,
            local_density=local_density,
            suffix_weakness=suffix_weakness,
            last_step_burden=last_step_burden,
            density_by_depth=density_by_depth,
            semantic_mismatch=semantic_mismatch,
            harmful_count=harmful_count,
            damage=damage,
            tail_damage=tail_damage,
            component_metadata=component_metadata,
        )

    def _compute_semantic_mismatch(
        self,
        raw_sid: torch.Tensor,
        groups_by_index: list[str],
        embeddings: torch.Tensor | None,
        strict_depth: int,
        buckets: dict[int, dict[tuple[int, ...], list[int]]],
    ) -> tuple[list[float], list[int]]:
        num_items = raw_sid.size(0)
        mismatch_sum = [0.0 for _ in range(num_items)]
        mismatch_count = [0 for _ in range(num_items)]
        harmful_count = [0 for _ in range(num_items)]
        if embeddings is None:
            return mismatch_sum, harmful_count

        normalized = F.normalize(embeddings, dim=1)
        for item_indexes in buckets[strict_depth].values():
            if len(item_indexes) < 2:
                continue
            selected = item_indexes
            if len(selected) > self.max_neighbors_per_bucket:
                tail_indexes = [idx for idx in selected if groups_by_index[idx] in {"Tail", "Tail-Cold"}]
                remaining = [idx for idx in selected if idx not in set(tail_indexes)]
                selected = tail_indexes + remaining[: max(0, self.max_neighbors_per_bucket - len(tail_indexes))]
            if len(selected) < 2:
                continue

            sims = normalized[selected] @ normalized[selected].T
            off_diag = sims[~torch.eye(len(selected), dtype=torch.bool)]
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
                        groups_by_index[item_i] in {"Tail", "Tail-Cold"}
                        or groups_by_index[item_j] in {"Tail", "Tail-Cold"}
                    ):
                        harmful += 1
                if values:
                    mismatch_sum[item_i] += sum(values)
                    mismatch_count[item_i] += len(values)
                    harmful_count[item_i] += harmful

        mismatch = [mismatch_sum[idx] / mismatch_count[idx] if mismatch_count[idx] else 0.0 for idx in range(num_items)]
        return mismatch, harmful_count

    def _compute_damage_scores(
        self,
        groups_by_index: list[str],
        full_collision_size: list[int],
        near_count: list[int],
        local_density: list[float],
        suffix_weakness: list[float],
        semantic_mismatch: list[float],
        last_step_burden: list[float],
    ) -> tuple[list[float], list[float], list[dict[str, float | int | str]]]:
        z_inputs = [
            ("full_collision", [1.0 if size > 1 else 0.0 for size in full_collision_size]),
            ("near_collision_strict", [float(value) for value in near_count]),
            ("local_density", local_density),
            ("suffix_weakness", suffix_weakness),
            ("semantic_mismatch", semantic_mismatch),
            ("last_step_burden", last_step_burden),
        ]
        z_components = []
        component_metadata = []
        for component_name, values in z_inputs:
            scores, metadata = self._stable_robust_risk_scores(component_name, values)
            z_components.append(scores)
            component_metadata.append(metadata)
        damage = [sum(component[idx] for component in z_components) for idx in range(len(groups_by_index))]
        gate = {"Head": 1.0, "Mid": 1.1, "Tail": 1.25, "Tail-Cold": 1.35}
        tail_damage = [damage[idx] * gate[groups_by_index[idx]] for idx in range(len(groups_by_index))]
        return damage, tail_damage, component_metadata

    @staticmethod
    def _stable_robust_risk_scores(
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

    @staticmethod
    def _compute_item_rows(
        sid_views: SIDViews,
        frequencies: dict[int, int],
        groups_by_index: list[str],
        item_metrics: _ItemMetricValues,
    ) -> list[dict[str, float | int | str]]:
        item_rows: list[dict[str, float | int | str]] = []
        item_ids = [int(item_id) for item_id in sid_views.item_ids.tolist()]
        for idx, item_id in enumerate(item_ids):
            row: dict[str, float | int | str] = {
                "item_id": item_id,
                "group": groups_by_index[idx],
                "freq_train": frequencies.get(item_id, 0),
                "raw_sid": " ".join(str(value) for value in sid_views.raw_sid[idx].tolist()),
                "model_sid": " ".join(str(value) for value in sid_views.model_sid[idx].tolist()),
                "dedup_digit": int(sid_views.dedup_digit[idx].item()),
                "full_collision_flag": int(item_metrics.full_collision_size[idx] > 1),
                "full_collision_size": item_metrics.full_collision_size[idx],
                "mpod_raw": item_metrics.mpod[idx],
                "near_collision_count_strict": item_metrics.near_count[idx],
                "local_density": item_metrics.local_density[idx],
                "suffix_weakness": item_metrics.suffix_weakness[idx],
                "last_step_burden": item_metrics.last_step_burden[idx],
                "semantic_mismatch": item_metrics.semantic_mismatch[idx],
                "qualified_harmful_overlap_count": item_metrics.harmful_count[idx],
                "damage": item_metrics.damage[idx],
                "tail_damage": item_metrics.tail_damage[idx],
            }
            for depth, density in enumerate(item_metrics.density_by_depth[idx], start=1):
                row[f"density_{depth}"] = density
            item_rows.append(row)
        return item_rows

    @staticmethod
    def _compute_prefix_rows(
        raw_sid: torch.Tensor,
        buckets: dict[int, dict[tuple[int, ...], list[int]]],
        groups_by_index: list[str],
        tail_damage: list[float],
        semantic_mismatch: list[float],
    ) -> list[dict[str, float | int | str]]:
        rows: list[dict[str, float | int | str]] = []
        for depth, depth_buckets in buckets.items():
            for prefix, indexes in depth_buckets.items():
                group_counts = {
                    group: sum(1 for idx in indexes if groups_by_index[idx] == group)
                    for group in ["Head", "Mid", "Tail", "Tail-Cold"]
                }
                tail_count = group_counts["Tail"] + group_counts["Tail-Cold"]
                suffix_uniqueness = 1.0
                if depth < raw_sid.size(1):
                    suffixes = {tuple(int(value) for value in raw_sid[idx, depth:].tolist()) for idx in indexes}
                    suffix_uniqueness = len(suffixes) / len(indexes)
                avg_tail_damage = _mean([tail_damage[idx] for idx in indexes])
                mismatch_rate = _mean([1.0 if semantic_mismatch[idx] > 0 else 0.0 for idx in indexes])
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

    @staticmethod
    def _compute_group_rows(item_rows: list[dict[str, float | int | str]]) -> list[dict[str, float | int | str]]:
        rows = []
        for group in ["Head", "Mid", "Tail", "Tail-Cold"]:
            group_items = [row for row in item_rows if row["group"] == group]
            if not group_items:
                continue
            rows.append(
                {
                    "group": group,
                    "num_items": len(group_items),
                    "avg_freq_train": _mean([float(row["freq_train"]) for row in group_items]),
                    "full_collision_rate": _mean([float(row["full_collision_flag"]) for row in group_items]),
                    "near_collision_rate_strict": _mean(
                        [1.0 if float(row["near_collision_count_strict"]) > 0 else 0.0 for row in group_items]
                    ),
                    "avg_mpod": _mean([float(row["mpod_raw"]) for row in group_items]),
                    "avg_local_density": _mean([float(row["local_density"]) for row in group_items]),
                    "avg_suffix_weakness": _mean([float(row["suffix_weakness"]) for row in group_items]),
                    "avg_last_step_burden": _mean([float(row["last_step_burden"]) for row in group_items]),
                    "avg_semantic_mismatch": _mean([float(row["semantic_mismatch"]) for row in group_items]),
                    "avg_damage": _mean([float(row["damage"]) for row in group_items]),
                    "avg_tail_damage": _mean([float(row["tail_damage"]) for row in group_items]),
                }
            )
        return rows

    @staticmethod
    def _compute_summary(
        item_rows: list[dict[str, float | int | str]],
        group_rows: list[dict[str, float | int | str]],
        sid_views: SIDViews,
        component_metadata: list[dict[str, float | int | str]],
    ) -> dict[str, float | int | str | None]:
        by_group = {str(row["group"]): row for row in group_rows}
        tail_row = by_group.get("Tail", {})
        degenerate_components = [
            str(row["component"]) for row in component_metadata if int(row["is_degenerate"]) == 1
        ]
        return {
            "num_items": len(item_rows),
            "raw_num_hierarchies": int(sid_views.raw_sid.size(1)),
            "sid_length": int(sid_views.model_sid.size(1)),
            "dedup_digit_nonzero_rate": _mean(
                [1.0 if int(value.item()) != 0 else 0.0 for value in sid_views.dedup_digit]
            ),
            "full_collision_rate_tail": tail_row.get("full_collision_rate", 0.0),
            "near_collision_rate_tail_strict": tail_row.get("near_collision_rate_strict", 0.0),
            "avg_local_density_tail": tail_row.get("avg_local_density", 0.0),
            "avg_damage_head": by_group.get("Head", {}).get("avg_damage", 0.0),
            "avg_damage_mid": by_group.get("Mid", {}).get("avg_damage", 0.0),
            "avg_damage_tail": tail_row.get("avg_damage", 0.0),
            "score_normalization_version": SCORE_NORMALIZATION_VERSION,
            "score_normalization_method": "positive robust IQR score; degenerate components contribute zero",
            "score_iqr_stability_threshold": SCORE_IQR_STABILITY_THRESHOLD,
            "score_component_clamp_min": SCORE_CONTRIBUTION_MIN,
            "score_component_clamp_max": SCORE_CONTRIBUTION_MAX,
            "score_components": ", ".join(SCORE_COMPONENT_NAMES),
            "score_degenerate_component_count": len(degenerate_components),
            "score_degenerate_components": ", ".join(degenerate_components) if degenerate_components else "none",
        }


@dataclass(frozen=True)
class _ItemMetricValues:
    full_collision_size: list[int]
    mpod: list[int]
    near_count: list[int]
    local_density: list[float]
    suffix_weakness: list[float]
    last_step_burden: list[float]
    density_by_depth: list[list[int]]
    semantic_mismatch: list[float]
    harmful_count: list[int]
    damage: list[float]
    tail_damage: list[float]
    component_metadata: list[dict[str, float | int | str]]
