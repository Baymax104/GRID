from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir
from omegaconf.errors import MissingMandatoryValue

from src.recommendation.tiger.prefix_allocation import PrefixAllocationConfig
from src.utils.launcher import apply_dry_run_overrides

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _compose(*extra: str):
    with initialize_config_dir(config_dir=str(PROJECT_ROOT / "configs"), version_base="1.3"):
        return compose(
            config_name="main",
            overrides=[
                "experiment=tiger_prefix_allocation_probe",
                "data_dir=data/beauty",
                "data_split=evaluation",
                "beam_width=10",
                "devices=[0]",
                "group=rkmeans",
                "ckpt_path=model.ckpt",
                "semantic_id_path=semantic.pt",
                "prefix_allocation.enabled=true",
                "prefix_allocation.reserved_slots=1",
                "prefix_allocation.pool_multiplier=2",
                *extra,
            ],
        )


def test_prefix_allocation_probe_composes_training_only_prior_and_trace_identity():
    cfg = _compose()

    assert cfg.task_name == "tiger_prefix_allocation_probe"
    assert cfg.model.root.prefix_allocation.enabled is True
    assert cfg.model.root.prefix_allocation.reserved_slots == 1
    assert cfg.model.root.item_frequencies.training_data_dir == "data/beauty/training"
    assert cfg.model.root.item_frequencies.source_split == "training"
    assert cfg.model.root.prefix_trace_metadata.seed == 42
    assert cfg.model.root.prefix_trace_metadata.checkpoint_reference is None
    assert cfg.callbacks.prefix_trace_writer.role == "prefix_trace"


def test_prefix_allocation_probe_requires_explicit_strategy_values():
    with initialize_config_dir(config_dir=str(PROJECT_ROOT / "configs"), version_base="1.3"):
        cfg = compose(
            config_name="main",
            overrides=[
                "experiment=tiger_prefix_allocation_probe",
                "data_dir=data/beauty",
                "data_split=evaluation",
                "beam_width=10",
                "devices=[0]",
                "group=rkmeans",
                "ckpt_path=model.ckpt",
                "semantic_id_path=semantic.pt",
            ],
        )

    with pytest.raises(MissingMandatoryValue):
        _ = cfg.prefix_allocation.enabled


def test_prefix_allocation_probe_rejects_non_training_source_at_component_boundary():
    cfg = _compose("prefix_allocation.source_split=testing")
    component = cfg.model.root.prefix_allocation

    with pytest.raises(ValueError, match="source_split"):
        PrefixAllocationConfig(
            enabled=component.enabled,
            reserved_slots=component.reserved_slots,
            pool_multiplier=component.pool_multiplier,
            source_split=component.source_split,
            strategy=component.strategy,
        ).validate(cfg.beam_width)


def test_prefix_allocation_probe_dry_run_disables_business_writers():
    cfg = _compose()
    cfg.dry_run = True

    updated = apply_dry_run_overrides(cfg)

    assert updated.callbacks.prefix_trace_writer is None
    assert updated.callbacks.recommendation_artifact_writer is None


def test_tail_diagnosis_allocation_mode_composes_without_changing_search_defaults():
    with initialize_config_dir(config_dir=str(PROJECT_ROOT / "configs"), version_base="1.3"):
        cfg = compose(
            config_name="main",
            overrides=[
                "experiment=tail_sid_diagnosis",
                "data_dir=data/beauty",
                "semantic_id_path=semantic.pt",
                "raw_num_hierarchies=3",
                "group=rkmeans",
                "candidate_allocation_probe.enabled=true",
                "baseline_recommendation_output_path=baseline.pt",
                "intervention_recommendation_output_path=intervention.pt",
                "baseline_prefix_trace_path=baseline-trace.pt",
                "intervention_prefix_trace_path=intervention-trace.pt",
            ],
        )

    assert cfg.search_ranking.enabled is False
    assert cfg.risk_standardization.enabled is False
    assert cfg.model.root.candidate_allocation_probe_enabled is True
    assert cfg.data.test_dataloader.baseline_recommendation_output_path == "baseline.pt"
    assert cfg.data.test_dataloader.intervention_prefix_trace_path == "intervention-trace.pt"
    assert "candidate_allocation" in cfg.model.metrics.stages.test
