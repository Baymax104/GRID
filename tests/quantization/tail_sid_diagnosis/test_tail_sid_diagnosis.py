import json
import math
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from hydra import compose, initialize_config_dir
from lightning.fabric.utilities.apply_func import move_data_to_device
from omegaconf import OmegaConf

from src.common.callbacks.wandb_artifact_lineage import WandbArtifactLineageCallback
from src.common.metrics import MetricCallback, MetricEngine
from src.data.components.artifacts import (
    ResolvedArtifactReference,
    get_resolved_artifact_registry,
)
from src.data.components.data_models import (
    DiagnosisBatch,
    ModelOutput,
    RecommendationOutcomeInput,
    SIDViews,
)
from src.data.components.preprocessing import assign_frequency_groups
from src.data.datamodule import DiagnosisDataModule
from src.data.datasets import DiagnosisDataset
from src.quantization.tail_sid_diagnosis.evidence import (
    _compute_statistical_evidence,
    build_diagnosis_evidence,
)
from src.quantization.tail_sid_diagnosis.metrics import (
    DamageScoreMetric,
    EvidenceSectionMetric,
    PrefixRiskMetric,
    SemanticMismatchMetric,
    StructuralSIDMetric,
    build_diagnosis_context,
)
from src.quantization.tail_sid_diagnosis.module import TailSIDDiagnosisModule

PROJECT_ROOT = Path(__file__).resolve().parents[3]


def _evidence_batch(*, embeddings: torch.Tensor | None = None) -> DiagnosisBatch:
    return DiagnosisBatch(
        sid_views=load_views_from_model_output(
            ModelOutput(
                keys=torch.tensor([10, 20, 30, 40]),
                predictions=torch.tensor(
                    [[1, 1, 0], [1, 1, 1], [1, 2, 0], [9, 9, 0]]
                ),
            ),
            raw_num_hierarchies=2,
        ),
        frequencies={10: 10, 20: 2, 30: 1, 40: 0},
        groups_by_item={10: "Head", 20: "Mid", 30: "Tail", 40: "Tail-Cold"},
        embeddings=embeddings,
    )


def _diagnosis_dataset(
    data_folder: str = "data/beauty",
    semantic_id_path: str = "semantic.pt",
    raw_num_hierarchies: int = 3,
    embedding_path: str | None = None,
) -> DiagnosisDataset:
    return DiagnosisDataset(
        dataset_config=SimpleNamespace(preprocessing_functions=[]),
        data_folder=data_folder,
        semantic_id_path=semantic_id_path,
        raw_num_hierarchies=raw_num_hierarchies,
        embedding_path=embedding_path,
    )


def _diagnosis_dataloader_config(
    dataset_class,
    *,
    semantic_id_path: str = "semantic.pt",
    embedding_path: str | None = None,
    recommendation_output_path: str | None = None,
    wandb_entity: str = "baymaxam",
    wandb_project: str = "GRID",
):
    return SimpleNamespace(
        dataset_class=dataset_class,
        dataset_config=SimpleNamespace(preprocessing_functions=[]),
        data_folder="data/beauty",
        semantic_id_path=semantic_id_path,
        raw_num_hierarchies=1,
        embedding_path=embedding_path,
        recommendation_output_path=recommendation_output_path,
        wandb_entity=wandb_entity,
        wandb_project=wandb_project,
        batch_size_per_device=1,
        num_workers=0,
        pin_memory=False,
        persistent_workers=False,
        drop_last=False,
        collate_fn=lambda rows: rows[0],
        timeout=0,
    )


def test_load_sid_views_splits_raw_model_and_dedup_digit(tmp_path):
    file_path = tmp_path / "semantic_ids.pt"
    torch.save(
        {
            "keys": torch.tensor([30, 10]),
            "predictions": torch.tensor([[3, 4, 5, 1], [1, 2, 3, 0]]),
        },
        file_path,
    )

    views = _diagnosis_dataset(semantic_id_path=str(file_path), raw_num_hierarchies=3)._load_sid_views()

    assert torch.equal(views.item_ids, torch.tensor([10, 30]))
    assert torch.equal(views.raw_sid, torch.tensor([[1, 2, 3], [3, 4, 5]]))
    assert torch.equal(views.model_sid, torch.tensor([[1, 2, 3, 0], [3, 4, 5, 1]]))
    assert torch.equal(views.dedup_digit, torch.tensor([0, 1]))


def test_frequency_grouping_uses_training_counts_and_tail_cold():
    item_ids = torch.tensor([10, 20, 30, 40, 50])
    rows = [
        {"sequence_data": torch.tensor([10, 10, 20, 30])},
        {"sequence_data": torch.tensor([10, 40])},
    ]

    dataset = _diagnosis_dataset()
    dataset._iter_training_rows = lambda: iter(rows)
    frequencies = dataset._compute_train_frequencies(item_ids)
    batch = DiagnosisBatch(
        sid_views=SIDViews(
            item_ids=item_ids,
            raw_sid=torch.empty(5, 1, dtype=torch.long),
            model_sid=torch.empty(5, 1, dtype=torch.long),
            dedup_digit=torch.zeros(5, dtype=torch.long),
        ),
        frequencies=frequencies,
        groups_by_item={},
        embeddings=None,
    )
    groups = assign_frequency_groups(batch, head_ratio=0.25, tail_ratio=0.25).groups_by_item

    assert frequencies == {10: 3, 20: 1, 30: 1, 40: 1, 50: 0}
    assert groups[10] == "Head"
    assert groups[50] == "Tail-Cold"
    assert "Tail" in set(groups.values())


def test_tail_sid_metric_emits_collision_density_and_suffix_burden():
    sid_views = ModelOutput(
        keys=torch.tensor([10, 20, 30, 40]),
        predictions=torch.tensor([[1, 1, 1, 0], [1, 1, 2, 0], [1, 1, 2, 1], [9, 9, 9, 0]]),
    )
    views = load_views_from_model_output(sid_views, raw_num_hierarchies=3)
    frequencies = {10: 10, 20: 1, 30: 1, 40: 0}
    groups = {10: "Head", 20: "Tail", 30: "Tail", 40: "Tail-Cold"}
    embeddings = torch.tensor(
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [0.0, 0.9],
            [1.0, 1.0],
        ]
    )

    context = build_diagnosis_context(
        sid_views=views,
        frequencies=frequencies,
        groups_by_item=groups,
        embeddings=embeddings,
    )
    structural_metric = StructuralSIDMetric()
    structural_metric.update(
        sid_views=context.sid_views,
        item_ids=context.item_ids,
        groups_by_index=context.groups_by_index,
        buckets=context.buckets,
        strict_depth=context.strict_depth,
        sid_length=context.sid_length,
    )
    structural_summary = structural_metric.compute()

    prefix_metric = PrefixRiskMetric()
    prefix_metric.update(
        sid_views=context.sid_views,
        frequencies=context.frequencies,
        groups_by_item=context.groups_by_item,
        embeddings=context.embeddings,
        item_ids=context.item_ids,
        groups_by_index=context.groups_by_index,
        buckets=context.buckets,
        strict_depth=context.strict_depth,
        sid_length=context.sid_length,
    )
    prefix_summary = prefix_metric.compute()

    assert structural_summary["full_collision_rate_tail"] == 1.0
    assert structural_summary["near_collision_rate_tail_strict"] == 1.0
    assert structural_summary["avg_local_density_tail"] > 0
    assert prefix_summary["top_prefix_risk"] >= prefix_summary["avg_prefix_risk"]


def test_tail_sid_metric_neutralizes_degenerate_score_components():
    sid_views = ModelOutput(
        keys=torch.tensor([1, 2, 3, 4]),
        predictions=torch.tensor([[1, 1, 0], [2, 2, 0], [3, 3, 0], [4, 4, 0]]),
    )
    views = load_views_from_model_output(sid_views, raw_num_hierarchies=2)

    context = build_diagnosis_context(
        sid_views=views,
        frequencies={1: 4, 2: 3, 3: 2, 4: 1},
        groups_by_item={1: "Head", 2: "Mid", 3: "Tail", 4: "Tail"},
        embeddings=None,
    )
    damage_metric = DamageScoreMetric()
    damage_metric.update(
        sid_views=context.sid_views,
        groups_by_index=context.groups_by_index,
        buckets=context.buckets,
        strict_depth=context.strict_depth,
        sid_length=context.sid_length,
        embeddings=context.embeddings,
    )
    damage_summary = damage_metric.compute()

    assert damage_summary["avg_damage_tail"] == 0.0
    assert damage_summary["avg_tail_damage_tail"] == 0.0
    assert damage_summary["score_degenerate_component_count"] == 6.0


def test_evidence_keeps_raw_damage_and_verdict_independent_of_priority_multipliers():
    batch = _evidence_batch()

    baseline = build_diagnosis_evidence(
        batch,
        priority_multipliers={group: 1.0 for group in ("Head", "Mid", "Tail", "Tail-Cold")},
    )
    prioritized = build_diagnosis_evidence(
        batch,
        priority_multipliers={"Head": 1.0, "Mid": 2.0, "Tail": 5.0, "Tail-Cold": 9.0},
    )

    assert [row["raw_damage"] for row in baseline.item_rows] == [
        row["raw_damage"] for row in prioritized.item_rows
    ]
    assert [row["priority_score"] for row in baseline.item_rows] != [
        row["priority_score"] for row in prioritized.item_rows
    ]
    assert baseline.summary["verdict"] == prioritized.summary["verdict"]
    assert [row["group"] for row in baseline.group_rows] == ["Head", "Mid", "Tail", "Tail-Cold"]
    assert len({tuple(row.keys()) for row in baseline.group_rows}) == 1


def test_evidence_normalizes_hydra_mapping_before_json_serialization():
    evidence = build_diagnosis_evidence(
        _evidence_batch(),
        priority_multipliers=OmegaConf.create(
            {"Head": 1.0, "Mid": 1.1, "Tail": 1.25, "Tail-Cold": 1.35}
        ),
    )

    assert isinstance(evidence.metadata["priority_multipliers"], dict)
    json.dumps(evidence.summary)


def test_evidence_types_asymmetric_partners_and_separates_full_from_near_overlap():
    evidence = build_diagnosis_evidence(_evidence_batch())
    by_item = {row["item_id"]: row for row in evidence.item_rows}

    assert by_item[20]["full_collision_head_count"] == 1
    assert by_item[30]["near_overlap_head_count"] == 1
    assert by_item[30]["full_collision_head_count"] == 0
    assert by_item[30]["head_dominance"] > 0
    assert by_item[30]["tail_isolation_deficit"] > 0
    assert by_item[30]["tail_head_near_collision_pressure"] > 0


def test_semantic_evidence_is_reproducible_and_pair_limits_do_not_change_aggregates():
    embeddings = torch.tensor([[1.0, 0.0], [-1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    batch = _evidence_batch(embeddings=embeddings)

    complete = build_diagnosis_evidence(batch, semantic_seed=7, max_harmful_pairs=100)
    truncated = build_diagnosis_evidence(batch, semantic_seed=7, max_harmful_pairs=0)

    assert complete.metadata["semantic"]["reference_threshold"] == truncated.metadata["semantic"][
        "reference_threshold"
    ]
    assert [row["harmful_overlap_count"] for row in complete.item_rows] == [
        row["harmful_overlap_count"] for row in truncated.item_rows
    ]
    assert truncated.harmful_pair_rows == []
    assert truncated.metadata["semantic"]["candidate_pair_count"] > 0
    assert truncated.metadata["semantic"]["truncated_pair_count"] == truncated.metadata["semantic"][
        "candidate_pair_count"
    ]
    assert "semantic_mismatch" in complete.item_rows[0]
    assert "bucket_relative_semantic_outlier" in complete.item_rows[0]


def test_semantic_evidence_records_unavailable_embeddings():
    evidence = build_diagnosis_evidence(_evidence_batch())

    assert evidence.metadata["semantic_evidence_available"] is False
    assert evidence.metadata["semantic"] == {"available": False}
    assert all(row["semantic_mismatch"] == 0.0 for row in evidence.item_rows)


def test_recommendation_input_joins_shuffled_users_and_aggregates_hit_rank_ndcg(tmp_path):
    recommendation_path = tmp_path / "recommendations.pt"
    torch.save(
        {
            "keys": torch.tensor([2, 1]),
            "predictions": torch.tensor(
                [
                    [[1, 1, 1], [1, 2, 0]],
                    [[1, 1, 0], [1, 2, 0]],
                ]
            ),
        },
        recommendation_path,
    )
    dataset = _diagnosis_dataset()
    dataset.recommendation_output_path = str(recommendation_path)
    dataset._iter_testing_rows = lambda: iter(
        [
            {"user_id": torch.tensor([1]), "sequence_data": torch.tensor([99, 10])},
            {"user_id": torch.tensor([2]), "sequence_data": torch.tensor([99, 30])},
        ]
    )

    recommendation = dataset._load_recommendation_input()
    batch = _evidence_batch()
    batch.recommendation = recommendation
    evidence = build_diagnosis_evidence(batch, hit_ks=[1, 2])
    by_item = {row["item_id"]: row for row in evidence.item_rows}

    assert recommendation.user_ids.tolist() == [1, 2]
    assert recommendation.label_item_ids.tolist() == [10, 30]
    assert by_item[10]["hit@1"] == 1.0
    assert by_item[30]["hit@1"] == 0.0
    assert by_item[30]["hit@2"] == 1.0
    assert by_item[30]["avg_rank"] == 2.0
    assert by_item[30]["ndcg@2"] == 1.0 / math.log2(3)
    assert evidence.metadata["recommendation_evidence_available"] is True


def test_recommendation_input_rejects_key_mismatch_and_invalid_shape(tmp_path):
    recommendation_path = tmp_path / "recommendations.pt"
    torch.save(
        {"keys": torch.tensor([1, 3]), "predictions": torch.ones(2, 2, 3, dtype=torch.long)},
        recommendation_path,
    )
    dataset = _diagnosis_dataset()
    dataset.recommendation_output_path = str(recommendation_path)
    dataset._iter_testing_rows = lambda: iter(
        [
            {"user_id": torch.tensor([1]), "sequence_data": torch.tensor([10])},
            {"user_id": torch.tensor([2]), "sequence_data": torch.tensor([20])},
        ]
    )

    with pytest.raises(ValueError, match="must match exactly"):
        dataset._load_recommendation_input()

    batch = _evidence_batch()
    batch.recommendation = RecommendationOutcomeInput(
        user_ids=torch.tensor([1]),
        label_item_ids=torch.tensor([10]),
        generated_sids=torch.ones(1, 2, 2, dtype=torch.long),
    )
    with pytest.raises(ValueError, match="SID width"):
        build_diagnosis_evidence(batch)


def test_statistical_evidence_is_seeded_and_controls_frequency_bins():
    rows = [
        {
            "item_id": index,
            "group": "Head" if index < 4 else "Tail",
            "freq_train": index % 2 + 1,
            "raw_damage": float(index),
            "near_collision_count_strict": float(index),
            "local_density": float(index),
            "suffix_weakness": float(index),
            "last_step_burden": float(index),
            "semantic_mismatch": float(index),
            "tail_head_near_collision_pressure": float(index),
            "hit@10": 1.0 if index < 4 else 0.0,
        }
        for index in range(8)
    ]
    kwargs = {
        "bootstrap_samples": 100,
        "confidence": 0.9,
        "seed": 11,
        "frequency_bin_count": 2,
        "min_bin_support": 1,
        "hit_ks": [10],
    }

    first = _compute_statistical_evidence(rows, **kwargs)
    second = _compute_statistical_evidence(rows, **kwargs)

    assert first == second
    raw = first["bootstrap_tail_minus_head"]["raw_damage"]
    assert raw["absolute_difference"] == 4.0
    assert raw["ci_lower"] > 0
    assert first["risk_outcome_association"]["overall"]["spearman"] < 0
    assert first["frequency_matched_low_minus_high_damage"]["available"] is True


def test_evidence_emits_declared_sensitivity_settings_and_machine_readable_verdict():
    evidence = build_diagnosis_evidence(
        _evidence_batch(),
        bootstrap_samples=50,
        tail_ratio_sensitivity=[0.25],
        semantic_quantile_sensitivity=[0.5],
        damage_component_sensitivity=["all", "structural_only", "semantic_only"],
    )

    assert {(row["setting_type"], str(row["setting_value"])) for row in evidence.sensitivity_rows} == {
        ("tail_ratio", "0.25"),
        ("semantic_quantile", "0.5"),
        ("damage_components", "all"),
        ("damage_components", "structural_only"),
        ("damage_components", "semantic_only"),
    }
    assert set(evidence.summary["verdict"].values()) <= {
        "supported",
        "not_supported",
        "unavailable",
    }
    assert evidence.summary["verdict"]["generation_risk_validity"] == "unavailable"


def test_split_tail_sid_metrics_reuse_shared_context():
    views = load_views_from_model_output(
        ModelOutput(
            keys=torch.tensor([10, 20, 30]),
            predictions=torch.tensor([[1, 1, 0], [1, 2, 0], [1, 2, 1]]),
        ),
        raw_num_hierarchies=2,
    )
    context = build_diagnosis_context(
        sid_views=views,
        frequencies={10: 3, 20: 1, 30: 1},
        groups_by_item={10: "Head", 20: "Tail", 30: "Tail"},
        embeddings=None,
    )

    structural_metric = StructuralSIDMetric()
    structural_metric.update(
        sid_views=context.sid_views,
        item_ids=context.item_ids,
        groups_by_index=context.groups_by_index,
        buckets=context.buckets,
        strict_depth=context.strict_depth,
        sid_length=context.sid_length,
    )
    structural_summary = structural_metric.compute()

    semantic_metric = SemanticMismatchMetric()
    semantic_metric.update(
        sid_views=context.sid_views,
        groups_by_index=context.groups_by_index,
        buckets=context.buckets,
        strict_depth=context.strict_depth,
        embeddings=context.embeddings,
    )
    semantic_summary = semantic_metric.compute()

    damage_metric = DamageScoreMetric()
    damage_metric.update(
        sid_views=context.sid_views,
        groups_by_index=context.groups_by_index,
        buckets=context.buckets,
        strict_depth=context.strict_depth,
        sid_length=context.sid_length,
        embeddings=context.embeddings,
    )
    damage_summary = damage_metric.compute()

    prefix_metric = PrefixRiskMetric()
    prefix_metric.update(
        sid_views=context.sid_views,
        frequencies=context.frequencies,
        groups_by_item=context.groups_by_item,
        embeddings=context.embeddings,
        item_ids=context.item_ids,
        groups_by_index=context.groups_by_index,
        buckets=context.buckets,
        strict_depth=context.strict_depth,
        sid_length=context.sid_length,
    )
    prefix_summary = prefix_metric.compute()

    assert structural_summary["num_items"] == 3.0
    assert structural_summary["near_collision_rate_tail_strict"] == 1.0
    assert semantic_summary["avg_semantic_mismatch_tail"] == 0.0
    assert "avg_damage_tail" in damage_summary
    assert prefix_summary["max_prefix_bucket_size"] >= 2.0


def test_load_embeddings_aligns_by_item_key(tmp_path):
    file_path = tmp_path / "embeddings.pt"
    torch.save(
        {
            "keys": torch.tensor([30, 10, 20]),
            "predictions": torch.tensor([[3.0], [1.0], [2.0]]),
        },
        file_path,
    )

    embeddings = _diagnosis_dataset(embedding_path=str(file_path))._load_embeddings_for_items(torch.tensor([20, 10]))

    assert torch.equal(embeddings, torch.tensor([[2.0], [1.0]]))


def test_diagnosis_dataset_loads_one_test_batch(monkeypatch):
    calls = []
    views = load_views_from_model_output(
        ModelOutput(
            keys=torch.tensor([10, 20]),
            predictions=torch.tensor([[1, 1, 0], [1, 2, 0]]),
        ),
        raw_num_hierarchies=2,
    )
    embeddings = torch.tensor([[1.0], [2.0]])
    frequencies = {10: 2, 20: 1}
    groups = {10: "Head", 20: "Tail"}

    dataset = DiagnosisDataset(
        dataset_config=SimpleNamespace(
            preprocessing_functions=[
                lambda batch: calls.append(("groups", batch.frequencies, 0.2, 0.2))
                or DiagnosisBatch(
                    sid_views=batch.sid_views,
                    frequencies=batch.frequencies,
                    groups_by_item=groups,
                    embeddings=batch.embeddings,
                )
            ],
        ),
        data_folder="data/beauty",
        semantic_id_path="semantic.pt",
        raw_num_hierarchies=3,
        embedding_path=None,
    )

    def compute_frequencies(self, item_ids):
        rows = self._iter_training_rows()
        calls.append(("frequencies", rows, item_ids))
        return frequencies

    monkeypatch.setattr(
        "src.data.datasets.DiagnosisDataset._load_sid_views",
        lambda self: calls.append(("load_sid", self.semantic_id_path, self.raw_num_hierarchies)) or views,
    )
    monkeypatch.setattr(
        "src.data.datasets.DiagnosisDataset._load_embeddings_for_items",
        lambda self, item_ids: calls.append(("load_embeddings", self.embedding_path, item_ids)) or embeddings,
    )
    monkeypatch.setattr(
        "src.data.datasets.DiagnosisDataset._iter_training_rows",
        lambda self: calls.append(("iter_rows", self.data_folder)) or ["rows"],
    )
    monkeypatch.setattr(
        "src.data.datasets.DiagnosisDataset._compute_train_frequencies",
        compute_frequencies,
    )

    batch = dataset[0]

    assert batch.sid_views == views
    assert batch.frequencies == frequencies
    assert batch.groups_by_item == groups
    assert torch.equal(batch.embeddings, embeddings)
    assert calls[0] == ("load_sid", "semantic.pt", 3)
    assert any(call == ("iter_rows", "data/beauty") for call in calls)
    assert any(call[0] == "frequencies" for call in calls)


def test_diagnosis_datamodule_builds_test_dataloader(monkeypatch):
    expected = DiagnosisBatch(
        sid_views=load_views_from_model_output(
            ModelOutput(
                keys=torch.tensor([1]),
                predictions=torch.tensor([[1, 0]]),
            ),
            raw_num_hierarchies=1,
        ),
        frequencies={1: 1},
        groups_by_item={1: "Head"},
        embeddings=None,
    )
    resolve_calls = []

    def track_local_reference(reference, field_name, **kwargs):
        resolve_calls.append((reference, field_name, kwargs))
        return reference

    monkeypatch.setattr("src.data.datamodule.diagnosis.resolve_reference", track_local_reference)
    datamodule = DiagnosisDataModule(
        test_dataloader_config=_diagnosis_dataloader_config(dataset_class=lambda **kwargs: [expected])
    )
    datamodule.trainer = SimpleNamespace()

    datamodule.setup("test")
    batch = next(iter(datamodule.test_dataloader()))

    assert batch == expected
    assert resolve_calls == [
        (
            "semantic.pt",
            "semantic_id_path",
            {"default_entity": "baymaxam", "default_project": "GRID"},
        )
    ]


def test_diagnosis_datamodule_resolves_wandb_inputs_with_field_roles(monkeypatch):
    registry = get_resolved_artifact_registry()
    registry.clear()
    dataset_kwargs = {}
    resolver_calls = []

    def dataset_class(**kwargs):
        dataset_kwargs.update(kwargs)
        return []

    def fake_resolve_wandb_artifact(**kwargs):
        resolver_calls.append(kwargs)
        return ResolvedArtifactReference(
            field_name=kwargs["field_name"],
            original_uri=kwargs["uri"].original_uri,
            resolved_path=f"resolved-{kwargs['role']}.pt",
            producer_run_id=kwargs["uri"].run_id,
            entity=kwargs["entity"],
            project=kwargs["project"],
            artifact_name=f"{kwargs['role']}:v0",
            artifact_version="v0",
            artifact_type=kwargs["role"],
            artifact_path=f"{kwargs['entity']}/{kwargs['project']}/{kwargs['role']}:v0",
            role=kwargs["role"],
            file=kwargs["target_file"],
        )

    monkeypatch.setattr("src.data.components.artifacts.resolve_wandb_artifact", fake_resolve_wandb_artifact)
    datamodule = DiagnosisDataModule(
        test_dataloader_config=_diagnosis_dataloader_config(
            dataset_class=dataset_class,
            semantic_id_path="wandb://semantic-run",
            embedding_path="wandb://other-user/other-project/embedding-run",
            recommendation_output_path="wandb://recommendation-run",
        )
    )
    datamodule.trainer = SimpleNamespace()

    datamodule.setup("test")
    datamodule.setup("test")

    assert dataset_kwargs["semantic_id_path"] == "resolved-semantic_id.pt"
    assert dataset_kwargs["embedding_path"] == "resolved-semantic_embedding.pt"
    assert dataset_kwargs["recommendation_output_path"] == "resolved-recommendation_output.pt"
    assert [
        (call["field_name"], call["role"], call["entity"], call["project"])
        for call in resolver_calls
    ] == [
        ("semantic_id_path", "semantic_id", "baymaxam", "GRID"),
        ("embedding_path", "semantic_embedding", "other-user", "other-project"),
        ("recommendation_output_path", "recommendation_output", "baymaxam", "GRID"),
    ]
    assert [(reference.field_name, reference.role) for reference in registry.records()] == [
        ("semantic_id_path", "semantic_id"),
        ("embedding_path", "semantic_embedding"),
        ("recommendation_output_path", "recommendation_output"),
    ]
    registry.clear()


def test_diagnosis_datamodule_registers_inputs_before_lineage_callback_setup(monkeypatch):
    registry = get_resolved_artifact_registry()
    registry.clear()

    def fake_resolve_wandb_artifact(**kwargs):
        return ResolvedArtifactReference(
            field_name=kwargs["field_name"],
            original_uri=kwargs["uri"].original_uri,
            resolved_path=f"resolved-{kwargs['role']}.pt",
            producer_run_id=kwargs["uri"].run_id,
            entity=kwargs["entity"],
            project=kwargs["project"],
            artifact_name=f"{kwargs['role']}:v0",
            artifact_version="v0",
            artifact_type=kwargs["role"],
            artifact_path=f"{kwargs['entity']}/{kwargs['project']}/{kwargs['role']}:v0",
            role=kwargs["role"],
            file=kwargs["target_file"],
        )

    monkeypatch.setattr("src.data.components.artifacts.resolve_wandb_artifact", fake_resolve_wandb_artifact)
    datamodule = DiagnosisDataModule(
        test_dataloader_config=_diagnosis_dataloader_config(
            dataset_class=lambda **kwargs: [],
            semantic_id_path="wandb://semantic-run",
            embedding_path="wandb://embedding-run",
        )
    )
    datamodule.trainer = SimpleNamespace()
    datamodule.setup("test")

    class _Run:
        def __init__(self):
            self.used = []

        def use_artifact(self, artifact_path):
            self.used.append(artifact_path)

    class WandbLogger:
        def __init__(self, run):
            self.experiment = run

    run = _Run()
    logger = WandbLogger(run)
    trainer = SimpleNamespace(logger=logger, loggers=[logger])
    callback = WandbArtifactLineageCallback(clear_after_recording=True)

    callback.setup(trainer=trainer, pl_module=None, stage="test")

    assert run.used == [
        "baymaxam/GRID/semantic_id:v0",
        "baymaxam/GRID/semantic_embedding:v0",
    ]
    assert registry.records() == ()


def test_tail_sid_diagnosis_module_returns_metric_pre_state():
    batch = DiagnosisBatch(
        sid_views=load_views_from_model_output(
            ModelOutput(
                keys=torch.tensor([1, 2]),
                predictions=torch.tensor([[1, 1, 0], [1, 2, 0]]),
            ),
            raw_num_hierarchies=2,
        ),
        frequencies={1: 2, 2: 1},
        groups_by_item={1: "Head", 2: "Tail"},
        embeddings=None,
    )

    module = TailSIDDiagnosisModule(max_neighbors_per_bucket=128)

    output = module.test_step(batch, batch_idx=0)

    evidence = output["evidence"]
    assert [row["item_id"] for row in evidence.item_rows] == [1, 2]
    assert [row["group"] for row in evidence.item_rows] == ["Head", "Tail"]
    assert evidence.summary["structural"]["raw_num_hierarchies"] == 2.0
    assert evidence.summary["metadata"]["recommendation_evidence_available"] is False


def test_diagnosis_batch_can_be_moved_by_lightning_transfer():
    batch = DiagnosisBatch(
        sid_views=load_views_from_model_output(
            ModelOutput(
                keys=torch.tensor([1, 2]),
                predictions=torch.tensor([[1, 1, 0], [1, 2, 0]]),
            ),
            raw_num_hierarchies=2,
        ),
        frequencies={1: 2, 2: 1},
        groups_by_item={1: "Head", 2: "Tail"},
        embeddings=None,
    )

    moved = move_data_to_device(batch, torch.device("cpu"))

    assert torch.equal(moved.sid_views.item_ids, batch.sid_views.item_ids)
    assert moved.frequencies == batch.frequencies


def test_tail_sid_standard_metric_callback_logs_diagnosis_summary():
    logged = []
    payload = diagnosis_payload(
        sid_views=load_views_from_model_output(
            ModelOutput(
                keys=torch.tensor([1, 2]),
                predictions=torch.tensor([[1, 1, 0], [1, 2, 0]]),
            ),
            raw_num_hierarchies=2,
        ),
        frequencies={1: 2, 2: 1},
        groups_by_item={1: "Head", 2: "Tail"},
        embeddings=None,
    )
    engine = diagnosis_metric_engine()
    callback = MetricCallback(engine=engine)
    module = SimpleNamespace(
        device=torch.device("cpu"),
        log_dict=lambda metrics, **kwargs: logged.append((metrics, kwargs)),
    )

    callback.setup(trainer=None, pl_module=module, stage="test")
    callback.on_test_start(trainer=None, pl_module=module)
    callback.on_test_batch_end(trainer=None, pl_module=module, outputs=payload, batch=None, batch_idx=0)
    callback.on_test_epoch_end(trainer=None, pl_module=module)

    assert logged[0][0]["test/structural/num_items"] == 2.0
    assert "test/semantic/avg_semantic_mismatch_tail" in logged[0][0]
    assert "test/damage/avg_damage_tail" in logged[0][0]
    assert "test/prefix_risk/top_prefix_risk" in logged[0][0]
    assert "test/damage/score_normalization_version" not in logged[0][0]
    assert logged[0][1]["logger"]


def test_tail_sid_diagnosis_hydra_config_composes():
    with initialize_config_dir(config_dir=str(PROJECT_ROOT / "configs"), version_base="1.3"):
        cfg = compose(
            config_name="main",
            overrides=[
                "experiment=tail_sid_diagnosis",
                "data_dir=data/beauty",
                "semantic_id_path=semantic.pt",
                "raw_num_hierarchies=3",
                "extras.print_config=false",
                "extras.print_config_warnings=false",
            ],
        )

    assert cfg.run_mode == "analysis"
    assert cfg.data.datamodule._target_ == "src.data.datamodule.DiagnosisDataModule"
    assert cfg.data.test_dataloader.dataset_class._target_ == "src.data.datasets.DiagnosisDataset"
    assert cfg.model.root._target_ == "src.quantization.tail_sid_diagnosis.module.TailSIDDiagnosisModule"
    assert cfg.model.metrics._target_ == "src.common.metrics.MetricEngine"
    assert cfg.model.metric_callback.logging_modes.test == "summary"
    for section in ["structural", "semantic", "damage", "prefix_risk"]:
        assert cfg.model.metrics.stages.test[section].metric._target_ == (
            "src.quantization.tail_sid_diagnosis.metrics.EvidenceSectionMetric"
        )
        assert cfg.model.metrics.stages.test[section].metric.section == section
    assert "adapter" not in cfg.model.metrics.stages.test.structural.spec
    assert cfg.callbacks.wandb_artifact_lineage._target_ == (
        "src.common.callbacks.wandb_artifact_lineage.WandbArtifactLineageCallback"
    )
    assert cfg.callbacks.wandb_artifact_lineage.fail_on_missing_run is False
    assert cfg.trainer.root._target_ == "lightning.pytorch.trainer.Trainer"
    assert cfg.logger.wandb._target_ == "lightning.pytorch.loggers.wandb.WandbLogger"
    assert cfg.logger.wandb.group == "tail_sid_diagnosis"


@pytest.mark.parametrize(
    ("embedding_path", "recommendation_output_path"),
    [
        ("null", "null"),
        ("wandb://embedding-run", "null"),
        ("null", "wandb://recommendation-run"),
        ("wandb://embedding-run", "wandb://recommendation-run"),
    ],
)
def test_tail_sid_diagnosis_optional_input_combinations_compose(
    embedding_path, recommendation_output_path
):
    with initialize_config_dir(config_dir=str(PROJECT_ROOT / "configs"), version_base="1.3"):
        cfg = compose(
            config_name="main",
            overrides=[
                "experiment=tail_sid_diagnosis",
                "data_dir=data/beauty",
                "semantic_id_path=semantic.pt",
                "raw_num_hierarchies=3",
                f"embedding_path={embedding_path}",
                f"recommendation_output_path={recommendation_output_path}",
            ],
        )

    assert cfg.data.test_dataloader.embedding_path == (
        None if embedding_path == "null" else embedding_path
    )
    assert cfg.data.test_dataloader.recommendation_output_path == (
        None if recommendation_output_path == "null" else recommendation_output_path
    )


def load_views_from_model_output(bundle: ModelOutput, raw_num_hierarchies: int):
    predictions = bundle.predictions.long()
    dedup_digit = predictions[:, -1] if predictions.size(1) > raw_num_hierarchies else torch.zeros(predictions.size(0))

    return SIDViews(
        item_ids=bundle.keys.long(),
        raw_sid=predictions[:, :raw_num_hierarchies],
        model_sid=predictions,
        dedup_digit=dedup_digit.long(),
    )


def diagnosis_payload(sid_views, frequencies, groups_by_item, embeddings):
    module = TailSIDDiagnosisModule()
    return module.test_step(
        DiagnosisBatch(
            sid_views=sid_views,
            frequencies=frequencies,
            groups_by_item=groups_by_item,
            embeddings=embeddings,
        ),
        batch_idx=0,
    )


def diagnosis_metric_engine(max_neighbors_per_bucket: int = 512):
    return MetricEngine(
        stages={
            "test": {
                "structural": {
                    "metric": EvidenceSectionMetric("structural"),
                    "spec": {
                        "kwargs": {
                            "evidence": {"key": "evidence"},
                        }
                    },
                },
                "semantic": {
                    "metric": EvidenceSectionMetric("semantic"),
                    "spec": {
                        "kwargs": {
                            "evidence": {"key": "evidence"},
                        }
                    },
                },
                "damage": {
                    "metric": EvidenceSectionMetric("damage"),
                    "spec": {
                        "kwargs": {
                            "evidence": {"key": "evidence"},
                        }
                    },
                },
                "prefix_risk": {
                    "metric": EvidenceSectionMetric("prefix_risk"),
                    "spec": {
                        "kwargs": {
                            "evidence": {"key": "evidence"},
                        }
                    },
                },
            }
        }
    )
