from functools import partial
from types import SimpleNamespace

import pytest
import torch
from torchmetrics import SumMetric

from src.common.callbacks.sasrec_prediction_metrics import SASRecPredictionMetricsCallback
from src.common.configs.model import TrainingModelConfig
from src.common.metrics import MetricEngine
from src.data.components.collate import collate_fn_sasrec
from src.data.components.sasrec import ItemCatalog, SASRecPreprocessor
from src.recommendation.sasrec import SASRec
from src.recommendation.sasrec.metrics import item_retrieval_inputs
from src.recommendation.tiger.metrics import NDCG, Recall


def model(**kwargs):
    options = dict(
        catalog=ItemCatalog(torch.arange(0, 24, 2)),
        max_history_items=3,
        hidden_size=8,
        dropout=0,
        top_k=10,
        catalog_chunk_size=4,
        training_model_config=TrainingModelConfig(optimizer=partial(torch.optim.Adam, lr=0.001, betas=(0.9, 0.98))),
    )
    options.update(kwargs)
    return SASRec(**options)


def evaluation_batch(m):
    preprocess = SASRecPreprocessor(ItemCatalog(m.item_keys), max_history_items=3)
    return collate_fn_sasrec(
        [
            preprocess({"sequence_data": torch.tensor([0, 2, 4]), "user_id": torch.tensor([0])}),
            preprocess({"sequence_data": torch.tensor([6, 8]), "user_id": torch.tensor([9])}),
        ]
    )


def test_training_loss_optimizer_and_parameter_update():
    m = model()
    preprocess = SASRecPreprocessor(ItemCatalog(m.item_keys), max_history_items=3, training=True)
    batch = collate_fn_sasrec([preprocess({"sequence_data": torch.tensor([0, 2, 4, 6])})])
    optimizer = m.configure_optimizers()
    assert optimizer.defaults["betas"] == (0.9, 0.98)
    before = m.backbone.item_embedding.weight.detach().clone()
    output = m.training_step(batch, 0)
    assert torch.isfinite(output["loss"])
    output["loss"].backward()
    optimizer.step()
    assert not torch.equal(before, m.backbone.item_embedding.weight)


@pytest.mark.parametrize("chunk_size", [1, 3, 5, 20])
@pytest.mark.parametrize("ties", [False, True])
def test_full_catalog_chunked_ranking_matches_direct_sort(chunk_size, ties):
    torch.manual_seed(42)
    m = model(catalog_chunk_size=chunk_size).eval()
    if ties:
        with torch.no_grad():
            m.backbone.item_embedding.weight[1:] = 0.1
    batch = evaluation_batch(m)
    ids, scores = m.retrieve(batch[0])
    query = m.backbone.encode(batch[0].input_ids)[:, -1]
    direct = (query[:, None] * m.backbone.item_embedding.weight[None, 1:]).sum(-1)
    expected = direct.argsort(descending=True, stable=True)[:, :10]
    torch.testing.assert_close(ids, m.item_keys[expected])
    torch.testing.assert_close(scores, direct.gather(1, expected))
    payload = m.validation_step(batch, 0)
    torch.testing.assert_close(payload["labels"], torch.tensor([4, 8]))
    assert payload["user_count"] == 2
    output = m.predict_step(batch)
    torch.testing.assert_close(output.keys, torch.tensor([0, 9]))
    torch.testing.assert_close(output.predictions, ids)


def test_metric_adapter_preserves_final_tie_order_and_all_users():
    payload = {
        "generated_ids": torch.tensor([[0, 2, 4], [0, 2, 4], [0, 2, 4]]),
        "scores": torch.zeros(3, 3),
        "labels": torch.tensor([0, 4, 8]),
    }
    inputs = item_retrieval_inputs(payload)
    recall = Recall(top_k=3)
    ndcg = NDCG(top_k=3)
    recall.update(**inputs)
    ndcg.update(**inputs)
    assert recall.compute().item() == pytest.approx(2 / 3)
    assert ndcg.compute().item() == pytest.approx((1 + 1 / 2) / 3)


def test_checkpoint_rejects_mapping_and_structure_changes():
    m = model()
    checkpoint = {"state_dict": m.state_dict()}
    m.on_save_checkpoint(checkpoint)
    m.on_load_checkpoint(checkpoint)
    model(catalog_chunk_size=1, top_k=5).on_load_checkpoint(checkpoint)
    with pytest.raises(ValueError, match="identity"):
        model(max_history_items=4).on_load_checkpoint(checkpoint)
    with pytest.raises(ValueError, match="identity"):
        model(catalog=ItemCatalog(torch.arange(1, 25, 2))).on_load_checkpoint(checkpoint)
    checkpoint["state_dict"]["item_keys"] = torch.arange(1, 25, 2)
    with pytest.raises(ValueError, match="mapping"):
        m.on_load_checkpoint(checkpoint)


def test_prediction_callback_uses_shared_metric_engine():
    m = model().eval()
    batch = evaluation_batch(m)
    definitions = {
        "recall@10": {"metric": Recall(10), "spec": {"adapter": item_retrieval_inputs}},
        "ndcg@10": {"metric": NDCG(10), "spec": {"adapter": item_retrieval_inputs}},
        "user_count": {"metric": SumMetric(), "spec": {"key": "user_count"}},
    }
    engine = MetricEngine(stages={"test": definitions})
    callback = SASRecPredictionMetricsCallback(engine)
    recorded = []
    trainer = SimpleNamespace(is_global_zero=True, loggers=[SimpleNamespace(log_metrics=recorded.append)])
    callback.on_predict_start(trainer, m)
    output = m.predict_step(batch)
    callback.on_predict_batch_end(trainer, m, output, list(batch), 0)
    callback.on_predict_end(trainer, m)
    assert recorded[0]["test/user_count"].item() == 2
    payload = m.evaluation_payload(batch)
    expected = Recall(10)
    expected.update(**item_retrieval_inputs(payload))
    assert recorded[0]["test/recall@10"].item() == pytest.approx(expected.compute().item())


def test_prediction_summary_logging_and_key_mismatch():
    m = model().eval()
    batch = evaluation_batch(m)
    engine = MetricEngine(stages={"test": {"user_count": {"metric": SumMetric(), "spec": {"key": "user_count"}}}})
    callback = SASRecPredictionMetricsCallback(engine, logging_modes={"test": "summary"})
    summary = {}
    trainer = SimpleNamespace(
        is_global_zero=True, loggers=[SimpleNamespace(experiment=SimpleNamespace(summary=summary))]
    )
    callback.on_predict_start(trainer, m)
    output = m.predict_step(batch)
    callback.on_predict_batch_end(trainer, m, output, batch, 0)
    callback.on_predict_end(trainer, m)
    assert summary["test/user_count"] == 2
    output.keys = output.keys.flip(0)
    with pytest.raises(ValueError, match="user keys"):
        callback.on_predict_batch_end(trainer, m, output, batch, 0)
