from pathlib import Path
from types import SimpleNamespace

import torch
from hydra import compose, initialize_config_dir

from src.common.metrics import MetricCallback, MetricEngine
from src.data.components.data_models import DiagnosisBatch, ModelOutput, SIDViews
from src.data.components.preprocessing import assign_frequency_groups
from src.data.datamodule import DiagnosisDataModule
from src.data.datasets import DiagnosisDataset
from src.quantization.tail_sid_diagnosis.metrics import (
    DamageScoreMetric,
    PrefixRiskMetric,
    SemanticMismatchMetric,
    StructuralSIDMetric,
    build_diagnosis_context,
)
from src.quantization.tail_sid_diagnosis.module import TailSIDDiagnosisModule

PROJECT_ROOT = Path(__file__).resolve().parents[3]


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


def test_diagnosis_datamodule_builds_test_dataloader():
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
    datamodule = DiagnosisDataModule(
        test_dataloader_config=SimpleNamespace(
            dataset_class=lambda **kwargs: [expected],
            dataset_config=SimpleNamespace(preprocessing_functions=[]),
            data_folder="data/beauty",
            semantic_id_path="semantic.pt",
            raw_num_hierarchies=1,
            embedding_path=None,
            batch_size_per_device=1,
            num_workers=0,
            pin_memory=False,
            persistent_workers=False,
            drop_last=False,
            collate_fn=lambda rows: rows[0],
            timeout=0,
        )
    )
    datamodule.trainer = SimpleNamespace()

    datamodule.setup("test")
    batch = next(iter(datamodule.test_dataloader()))

    assert batch == expected


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

    assert output["sid_views"] == batch.sid_views
    assert output["frequencies"] == batch.frequencies
    assert output["groups_by_item"] == batch.groups_by_item
    assert output["item_ids"] == [1, 2]
    assert output["groups_by_index"] == ["Head", "Tail"]
    assert output["strict_depth"] == 1
    assert output["sid_length"] == 2
    assert output["buckets"][1][(1,)] == [0, 1]


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
    assert cfg.model.metrics.stages.test.structural.metric._target_ == (
        "src.quantization.tail_sid_diagnosis.metrics.StructuralSIDMetric"
    )
    assert cfg.model.metrics.stages.test.semantic.metric._target_ == (
        "src.quantization.tail_sid_diagnosis.metrics.SemanticMismatchMetric"
    )
    assert cfg.model.metrics.stages.test.damage.metric._target_ == (
        "src.quantization.tail_sid_diagnosis.metrics.DamageScoreMetric"
    )
    assert cfg.model.metrics.stages.test.prefix_risk.metric._target_ == (
        "src.quantization.tail_sid_diagnosis.metrics.PrefixRiskMetric"
    )
    assert "adapter" not in cfg.model.metrics.stages.test.structural.spec
    assert "callbacks" not in cfg
    assert cfg.trainer.root._target_ == "lightning.pytorch.trainer.Trainer"
    assert cfg.logger.wandb._target_ == "lightning.pytorch.loggers.wandb.WandbLogger"
    assert cfg.logger.wandb.group == "tail_sid_diagnosis"


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
                    "metric": StructuralSIDMetric(),
                    "spec": {
                        "kwargs": {
                            "sid_views": {"key": "sid_views"},
                            "item_ids": {"key": "item_ids"},
                            "groups_by_index": {"key": "groups_by_index"},
                            "buckets": {"key": "buckets"},
                            "strict_depth": {"key": "strict_depth"},
                            "sid_length": {"key": "sid_length"},
                        }
                    },
                },
                "semantic": {
                    "metric": SemanticMismatchMetric(max_neighbors_per_bucket=max_neighbors_per_bucket),
                    "spec": {
                        "kwargs": {
                            "sid_views": {"key": "sid_views"},
                            "groups_by_index": {"key": "groups_by_index"},
                            "buckets": {"key": "buckets"},
                            "strict_depth": {"key": "strict_depth"},
                            "embeddings": {"key": "embeddings"},
                        }
                    },
                },
                "damage": {
                    "metric": DamageScoreMetric(max_neighbors_per_bucket=max_neighbors_per_bucket),
                    "spec": {
                        "kwargs": {
                            "sid_views": {"key": "sid_views"},
                            "groups_by_index": {"key": "groups_by_index"},
                            "buckets": {"key": "buckets"},
                            "strict_depth": {"key": "strict_depth"},
                            "sid_length": {"key": "sid_length"},
                            "embeddings": {"key": "embeddings"},
                        }
                    },
                },
                "prefix_risk": {
                    "metric": PrefixRiskMetric(max_neighbors_per_bucket=max_neighbors_per_bucket),
                    "spec": {
                        "kwargs": {
                            "sid_views": {"key": "sid_views"},
                            "frequencies": {"key": "frequencies"},
                            "groups_by_item": {"key": "groups_by_item"},
                            "embeddings": {"key": "embeddings"},
                            "item_ids": {"key": "item_ids"},
                            "groups_by_index": {"key": "groups_by_index"},
                            "buckets": {"key": "buckets"},
                            "strict_depth": {"key": "strict_depth"},
                            "sid_length": {"key": "sid_length"},
                        }
                    },
                },
            }
        }
    )
