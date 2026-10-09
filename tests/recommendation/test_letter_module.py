from types import SimpleNamespace

import pytest
import torch

from src.common.configs.model import TrainingModelConfig
from src.common.metrics import MetricEngine
from src.data.components.data_models import ModelOutput
from src.data.components.letter import LetterCatalog
from src.recommendation.letter.metrics import LetterPredictionMetricsCallback, LetterRankingMetric, letter_metric_inputs
from src.recommendation.letter.module import LetterRecommender


def test_metrics_first_match_and_user_weighting():
    metric = LetterRankingMetric(3, True)
    metric.update(torch.tensor([[1, 2, 3], [2, 1, 3]]), torch.tensor([1, 1]))
    metric.update(torch.tensor([[2, 3, 4]]), torch.tensor([1]))
    assert float(metric.compute()) == pytest.approx((1 + 1 / torch.log2(torch.tensor(3.0))) / 3)
    recall = LetterRankingMetric(3)
    recall.update(torch.tensor([[1, 1, 2], [3, 4, 5]]), torch.tensor([1, 1]))
    assert recall.compute() == 0.5


def test_checkpoint_identity_and_training():
    catalog = LetterCatalog(torch.arange(4), torch.tensor([[0, 0, 0, i] for i in range(4)]), 4, 8)
    model = LetterRecommender(
        catalog,
        TrainingModelConfig(optimizer=lambda params: torch.optim.AdamW(params, lr=0.001)),
        d_model=8,
        d_ff=16,
        d_kv=4,
        num_heads=2,
        num_layers=1,
        dropout=0.0,
        generation_candidates=2,
        top_k=1,
    )
    checkpoint = {}
    model.on_save_checkpoint(checkpoint)
    model.on_load_checkpoint(checkpoint)
    checkpoint["letter_identity"] = {"catalog_sha256": "wrong"}
    with pytest.raises(ValueError, match="identity"):
        model.on_load_checkpoint(checkpoint)
    batch = {
        "input_ids": torch.tensor([[8, 9, 10, 11, 1]]),
        "attention_mask": torch.ones(1, 5, dtype=torch.long),
        "labels": torch.cat((catalog.tokens[:1], torch.ones(1, 1, dtype=torch.long)), -1),
    }
    loss = model.training_step(batch, 0)["loss"]
    loss.backward()
    assert torch.isfinite(loss) and model.backbone.t5.shared.weight.grad.abs().sum() > 0
    groups = model.configure_optimizers().param_groups
    assert groups[1]["weight_decay"] == 0


def test_predict_metrics_use_public_summary_policy():
    engine = MetricEngine(
        {"test": {"ndcg@2": {"metric": LetterRankingMetric(2, True), "spec": {"adapter": letter_metric_inputs}}}}
    )
    callback = LetterPredictionMetricsCallback(engine, logging_modes={"test": "summary"})
    summary = {}
    trainer = SimpleNamespace(loggers=[SimpleNamespace(experiment=SimpleNamespace(summary=summary))])
    callback.on_predict_start(trainer, None)
    callback.on_predict_batch_end(
        trainer,
        None,
        ModelOutput(torch.tensor([7, 8]), torch.tensor([[1, 2], [2, 1]])),
        {"target": torch.tensor([1, 1])},
        0,
    )
    callback.on_predict_end(trainer, None)
    assert summary["test/ndcg@2"] == pytest.approx((1 + 1 / torch.log2(torch.tensor(3.0))) / 2)
