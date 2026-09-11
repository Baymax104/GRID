from __future__ import annotations

from typing import Any

from lightning import LightningModule

from src.data.components.data_models import DiagnosisBatch
from src.quantization.tail_sid_diagnosis.evidence import build_diagnosis_evidence


class TailSIDDiagnosisModule(LightningModule):
    """Test-only LightningModule for Tail-SID Resolution Damage diagnosis."""

    def __init__(
        self,
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
    ):
        super().__init__()
        self.max_neighbors_per_bucket = max_neighbors_per_bucket
        self.priority_multipliers = priority_multipliers
        self.semantic_reference_quantile = semantic_reference_quantile
        self.semantic_bucket_quantile = semantic_bucket_quantile
        self.semantic_reference_pairs = semantic_reference_pairs
        self.semantic_seed = semantic_seed
        self.max_harmful_pairs = max_harmful_pairs
        self.max_harmful_pairs_per_item = max_harmful_pairs_per_item
        self.hit_ks = hit_ks
        self.bootstrap_samples = bootstrap_samples
        self.bootstrap_confidence = bootstrap_confidence
        self.frequency_bin_count = frequency_bin_count
        self.min_frequency_bin_support = min_frequency_bin_support
        self.tail_ratio_sensitivity = tail_ratio_sensitivity
        self.semantic_quantile_sensitivity = semantic_quantile_sensitivity
        self.damage_component_sensitivity = damage_component_sensitivity
        self.head_ratio = head_ratio
        self.tail_ratio = tail_ratio
        self.search_ranking_enabled = search_ranking_enabled
        self.risk_standardization_enabled = risk_standardization_enabled
        self.risk_standardization_bin_count = risk_standardization_bin_count
        self.risk_standardization_bin_count_sensitivity = risk_standardization_bin_count_sensitivity
        self.risk_standardization_min_items_per_group_per_bin = (
            risk_standardization_min_items_per_group_per_bin
        )
        self.risk_standardization_min_item_retention = risk_standardization_min_item_retention
        self.risk_standardization_max_abs_raw_damage_smd = risk_standardization_max_abs_raw_damage_smd
        self.risk_standardization_min_bootstrap_valid_fraction = (
            risk_standardization_min_bootstrap_valid_fraction
        )
        self.candidate_allocation_probe_enabled = candidate_allocation_probe_enabled
        self.candidate_allocation_overall_hit10_loss_guardrail = (
            candidate_allocation_overall_hit10_loss_guardrail
        )
        self.candidate_allocation_head_hit10_loss_guardrail = (
            candidate_allocation_head_hit10_loss_guardrail
        )

    def test_step(self, batch: DiagnosisBatch, batch_idx: int) -> dict[str, Any]:
        evidence = build_diagnosis_evidence(
            batch,
            max_neighbors_per_bucket=self.max_neighbors_per_bucket,
            priority_multipliers=self.priority_multipliers,
            semantic_reference_quantile=self.semantic_reference_quantile,
            semantic_bucket_quantile=self.semantic_bucket_quantile,
            semantic_reference_pairs=self.semantic_reference_pairs,
            semantic_seed=self.semantic_seed,
            max_harmful_pairs=self.max_harmful_pairs,
            max_harmful_pairs_per_item=self.max_harmful_pairs_per_item,
            hit_ks=self.hit_ks,
            bootstrap_samples=self.bootstrap_samples,
            bootstrap_confidence=self.bootstrap_confidence,
            frequency_bin_count=self.frequency_bin_count,
            min_frequency_bin_support=self.min_frequency_bin_support,
            tail_ratio_sensitivity=self.tail_ratio_sensitivity,
            semantic_quantile_sensitivity=self.semantic_quantile_sensitivity,
            damage_component_sensitivity=self.damage_component_sensitivity,
            head_ratio=self.head_ratio,
            tail_ratio=self.tail_ratio,
            search_ranking_enabled=self.search_ranking_enabled,
            risk_standardization_enabled=self.risk_standardization_enabled,
            risk_standardization_bin_count=self.risk_standardization_bin_count,
            risk_standardization_bin_count_sensitivity=self.risk_standardization_bin_count_sensitivity,
            risk_standardization_min_items_per_group_per_bin=(
                self.risk_standardization_min_items_per_group_per_bin
            ),
            risk_standardization_min_item_retention=self.risk_standardization_min_item_retention,
            risk_standardization_max_abs_raw_damage_smd=self.risk_standardization_max_abs_raw_damage_smd,
            risk_standardization_min_bootstrap_valid_fraction=(
                self.risk_standardization_min_bootstrap_valid_fraction
            ),
            candidate_allocation_probe_enabled=self.candidate_allocation_probe_enabled,
            candidate_allocation_overall_hit10_loss_guardrail=(
                self.candidate_allocation_overall_hit10_loss_guardrail
            ),
            candidate_allocation_head_hit10_loss_guardrail=(
                self.candidate_allocation_head_hit10_loss_guardrail
            ),
        )
        return {
            "evidence": evidence,
            "structured_analysis": evidence.to_structured_output(),
        }
