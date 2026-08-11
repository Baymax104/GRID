import json
from pathlib import Path

import torch
from hydra import compose, initialize_config_dir

from src.data.components.data_models import ModelOutput
from src.quantization.tail_sid_diagnosis.io import compute_train_frequencies, load_embeddings_for_items, load_sid_views
from src.quantization.tail_sid_diagnosis.metrics import TailSIDDiagnosisMetric, assign_frequency_groups
from src.quantization.tail_sid_diagnosis.reporting import print_summary, write_outputs
from src.quantization.tail_sid_diagnosis.runner import TailSIDDiagnosisRunner

PROJECT_ROOT = Path(__file__).resolve().parents[3]


def test_load_sid_views_splits_raw_model_and_dedup_digit(tmp_path):
    file_path = tmp_path / "semantic_ids.pt"
    torch.save(
        {
            "keys": torch.tensor([30, 10]),
            "predictions": torch.tensor([[3, 4, 5, 1], [1, 2, 3, 0]]),
        },
        file_path,
    )

    views = load_sid_views(str(file_path), raw_num_hierarchies=3)

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

    frequencies = compute_train_frequencies(rows, item_ids)
    groups = assign_frequency_groups(item_ids, frequencies, head_ratio=0.25, tail_ratio=0.25)

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

    result = compute_metric_result(views, frequencies, groups, embeddings)
    rows_by_item = {row["item_id"]: row for row in result.item_rows}

    assert rows_by_item[20]["full_collision_flag"] == 1
    assert rows_by_item[20]["full_collision_size"] == 2
    assert rows_by_item[10]["near_collision_count_strict"] == 2
    assert rows_by_item[10]["last_step_burden"] > 0
    assert rows_by_item[20]["local_density"] > rows_by_item[40]["local_density"]
    assert result.prefix_rows[0]["prefix_risk"] >= result.prefix_rows[-1]["prefix_risk"]


def test_tail_sid_metric_neutralizes_degenerate_score_components():
    sid_views = ModelOutput(
        keys=torch.tensor([1, 2, 3, 4]),
        predictions=torch.tensor([[1, 1, 0], [2, 2, 0], [3, 3, 0], [4, 4, 0]]),
    )
    views = load_views_from_model_output(sid_views, raw_num_hierarchies=2)

    result = compute_metric_result(
        sid_views=views,
        frequencies={1: 4, 2: 3, 3: 2, 4: 1},
        groups_by_item={1: "Head", 2: "Mid", 3: "Tail", 4: "Tail"},
        embeddings=None,
    )

    assert max(float(row["damage"]) for row in result.item_rows) == 0.0
    assert max(float(row["tail_damage"]) for row in result.item_rows) == 0.0
    assert result.summary["score_degenerate_component_count"] == 6
    assert result.summary["score_degenerate_components"] != "none"


def test_load_embeddings_aligns_by_item_key(tmp_path):
    file_path = tmp_path / "embeddings.pt"
    torch.save(
        {
            "keys": torch.tensor([30, 10, 20]),
            "predictions": torch.tensor([[3.0], [1.0], [2.0]]),
        },
        file_path,
    )

    embeddings = load_embeddings_for_items(str(file_path), torch.tensor([20, 10]))

    assert torch.equal(embeddings, torch.tensor([[2.0], [1.0]]))


def test_write_outputs_creates_required_files(tmp_path):
    sid_views = load_views_from_model_output(
        ModelOutput(
            keys=torch.tensor([1, 2]),
            predictions=torch.tensor([[1, 1, 0], [1, 2, 0]]),
        ),
        raw_num_hierarchies=2,
    )
    result = compute_metric_result(
        sid_views=sid_views,
        frequencies={1: 2, 2: 1},
        groups_by_item={1: "Head", 2: "Tail"},
        embeddings=None,
    )

    write_outputs(result, str(tmp_path), top_k_report=1)

    expected = {"summary.json", "group_metrics.csv", "item_damage_scores.csv", "prefix_risk_scores.csv", "report.md"}
    assert expected.issubset({path.name for path in tmp_path.iterdir()})
    summary = json.loads((tmp_path / "summary.json").read_text(encoding="utf-8"))
    assert summary["num_items"] == 2
    assert "full_collision_rate_tail" in summary
    assert summary["score_normalization_version"] == "positive_robust_iqr_v1"
    report = (tmp_path / "report.md").read_text(encoding="utf-8")
    assert "Tail-SID Resolution Damage Report" in report
    assert "## Score Normalization" in report
    assert "## Top Risky Items" in report
    assert report.count("| item_id |") == 1


def test_print_summary_shows_report_and_top_risk_preview(tmp_path, capsys):
    sid_views = load_views_from_model_output(
        ModelOutput(
            keys=torch.tensor([1, 2]),
            predictions=torch.tensor([[1, 1, 0], [1, 2, 0]]),
        ),
        raw_num_hierarchies=2,
    )
    result = compute_metric_result(
        sid_views=sid_views,
        frequencies={1: 2, 2: 1},
        groups_by_item={1: "Head", 2: "Tail"},
        embeddings=None,
    )

    print_summary(result, str(tmp_path))

    output = capsys.readouterr().out
    assert "report.md" in output
    assert "Top risky item:" in output
    assert "Top risky prefix:" in output
    assert "+-------+" in output
    assert "| group |" in output
    assert "| prefix_depth |" in output


def test_tail_sid_diagnosis_runner_orchestrates_metric_and_reporting(monkeypatch, tmp_path):
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

    class _MetricStub:
        def __init__(self, max_neighbors_per_bucket):
            calls.append(("metric_init", max_neighbors_per_bucket))

        def update(self, sid_views, frequencies, groups_by_item, embeddings):
            calls.append(("metric_update", sid_views, frequencies, groups_by_item, embeddings))

        def compute(self):
            calls.append(("metric_compute",))
            return "result"

    runner = TailSIDDiagnosisRunner(
        data_dir="data/beauty",
        semantic_id_path="semantic.pt",
        raw_num_hierarchies=3,
        output_dir=str(tmp_path),
        embedding_path=None,
        max_neighbors_per_bucket=128,
        top_k_report=7,
    )

    monkeypatch.setattr(
        "src.quantization.tail_sid_diagnosis.runner.load_sid_views",
        lambda semantic_id_path, raw_num_hierarchies: calls.append(
            ("load_sid", semantic_id_path, raw_num_hierarchies)
        )
        or views,
    )
    monkeypatch.setattr(
        "src.quantization.tail_sid_diagnosis.runner.load_embeddings_for_items",
        lambda embedding_path, item_ids: calls.append(("load_embeddings", embedding_path, item_ids)) or embeddings,
    )
    monkeypatch.setattr(
        "src.quantization.tail_sid_diagnosis.runner.iter_training_rows",
        lambda data_dir: calls.append(("iter_rows", data_dir)) or ["rows"],
    )
    monkeypatch.setattr(
        "src.quantization.tail_sid_diagnosis.runner.compute_train_frequencies",
        lambda rows, item_ids: calls.append(("frequencies", rows, item_ids)) or frequencies,
    )
    monkeypatch.setattr(
        "src.quantization.tail_sid_diagnosis.runner.assign_frequency_groups",
        lambda item_ids, frequencies, head_ratio, tail_ratio: calls.append(
            ("groups", item_ids, frequencies, head_ratio, tail_ratio)
        )
        or groups,
    )
    monkeypatch.setattr("src.quantization.tail_sid_diagnosis.runner.TailSIDDiagnosisMetric", _MetricStub)
    monkeypatch.setattr(
        "src.quantization.tail_sid_diagnosis.runner.write_outputs",
        lambda result, output_dir, top_k_report: calls.append(("write", result, output_dir, top_k_report)),
    )
    monkeypatch.setattr(
        "src.quantization.tail_sid_diagnosis.runner.print_summary",
        lambda result, output_dir: calls.append(("print", result, output_dir)),
    )

    runner.run()

    assert calls[0] == ("load_sid", "semantic.pt", 3)
    assert calls[1][0] == "load_embeddings"
    assert calls[2] == ("iter_rows", "data/beauty")
    assert calls[3][0] == "frequencies"
    assert calls[4][0] == "groups"
    assert calls[5] == ("metric_init", 128)
    assert calls[6] == ("metric_update", views, frequencies, groups, embeddings)
    assert calls[7] == ("metric_compute",)
    assert calls[8] == ("write", "result", str(tmp_path), 7)
    assert calls[9] == ("print", "result", str(tmp_path))


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
    assert cfg.analysis.runner._target_ == "src.quantization.tail_sid_diagnosis.runner.TailSIDDiagnosisRunner"
    assert cfg.analysis.runner.semantic_id_path == "semantic.pt"


def load_views_from_model_output(bundle: ModelOutput, raw_num_hierarchies: int):
    predictions = bundle.predictions.long()
    dedup_digit = predictions[:, -1] if predictions.size(1) > raw_num_hierarchies else torch.zeros(predictions.size(0))
    from src.quantization.tail_sid_diagnosis.io import SIDViews

    return SIDViews(
        item_ids=bundle.keys.long(),
        raw_sid=predictions[:, :raw_num_hierarchies],
        model_sid=predictions,
        dedup_digit=dedup_digit.long(),
    )


def compute_metric_result(sid_views, frequencies, groups_by_item, embeddings):
    metric = TailSIDDiagnosisMetric()
    metric.update(
        sid_views=sid_views,
        frequencies=frequencies,
        groups_by_item=groups_by_item,
        embeddings=embeddings,
    )
    return metric.compute()
