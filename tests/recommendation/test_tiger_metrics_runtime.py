import inspect
from pathlib import Path

import hydra
import torch
from omegaconf import OmegaConf
from transformers import T5Config, T5EncoderModel
from transformers.models.t5.modeling_t5 import T5Stack

from src.data.components.collate import collate_fn_sequence
from src.recommendation.tiger.tiger import Tiger

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def create_tiger() -> Tiger:
    encoder_config = T5Config(
        vocab_size=8,
        d_model=4,
        num_heads=2,
        d_ff=8,
        d_kv=2,
        num_layers=1,
    )
    decoder_config = T5Config(
        vocab_size=8,
        d_model=4,
        num_heads=2,
        d_ff=8,
        d_kv=2,
        num_layers=1,
        is_decoder=True,
        is_encoder_decoder=False,
    )
    return Tiger(
        encoder=T5EncoderModel(encoder_config),
        decoder=T5Stack(decoder_config),
        semantic_ids=torch.zeros(4, 2),
        num_hierarchies=2,
        codebook_size=4,
        embedding_dim=4,
        training_model_config=None,
    )


def test_tiger_constructor_no_longer_accepts_evaluator():
    parameters = inspect.signature(Tiger).parameters

    assert "evaluator" not in parameters


def test_tiger_does_not_call_save_hyperparameters():
    assert "save_hyperparameters" not in inspect.getsource(Tiger.__init__)


def test_tiger_does_not_create_metric_attributes():
    model = create_tiger()

    for attribute_name in [
        "evaluator",
        "train_loss",
        "val_loss",
        "test_loss",
    ]:
        assert not hasattr(model, attribute_name)

    assert not hasattr(model, "log_metrics")
    assert "on_validation_epoch_end" not in Tiger.__dict__
    assert "on_test_epoch_end" not in Tiger.__dict__


def test_tiger_predict_step_accepts_inference_collate_tuple(monkeypatch):
    model = create_tiger()
    rows = [
        {
            "sequence_data": torch.tensor([1, 2, 3, 4]),
            "attention_mask": torch.tensor([1, 1, 1, 1]),
            "user_id": torch.tensor([17]),
        },
        {
            "sequence_data": torch.tensor([4, 3, 2, 1]),
            "attention_mask": torch.tensor([1, 1, 1, 1]),
            "user_id": torch.tensor([23]),
        },
    ]
    batch = collate_fn_sequence(
        rows,
        input_field_name="sequence_data",
        attention_mask_field_name="attention_mask",
        target_field_name=None,
        output_key_field_name="user_id",
    )
    generated_sids = torch.tensor(
        [
            [[1, 1], [2, 2]],
            [[3, 3], [4, 4]],
        ]
    )
    monkeypatch.setattr(model, "generate", lambda **kwargs: (generated_sids, None))

    output = model.predict_step(batch)

    assert torch.equal(output.keys, torch.tensor([17, 23]))
    assert torch.equal(output.predictions, generated_sids)


def test_tiger_config_declares_retrieval_metrics_as_concrete_instances():
    config = OmegaConf.load(PROJECT_ROOT / "configs/model/tiger_train.yaml")

    for stage_name in ["val", "test"]:
        stage_metrics = config.metrics.stages[stage_name]
        assert "retrieval" not in stage_metrics
        assert set(stage_metrics) == {"loss", "ndcg@5", "ndcg@10", "recall@5", "recall@10"}
        assert stage_metrics["ndcg@5"].metric.top_k == 5
        assert stage_metrics["ndcg@10"].metric.top_k == 10
        assert stage_metrics["recall@5"].metric.top_k == 5
        assert stage_metrics["recall@10"].metric.top_k == 10
        assert stage_metrics["ndcg@5"].metric._target_ == "src.recommendation.tiger.metrics.NDCG"
        assert stage_metrics["recall@5"].metric._target_ == "src.recommendation.tiger.metrics.Recall"
        assert stage_metrics["ndcg@5"].spec.adapter._target_ == (
            "src.recommendation.tiger.metrics.sid_retrieval_inputs"
        )


def test_tiger_metric_config_instantiates_adapter_metrics():
    config = OmegaConf.load(PROJECT_ROOT / "configs/model/tiger_train.yaml")

    engine = hydra.utils.instantiate(config.metrics, _recursive_=False)
    engine.update(
        "val",
        {
            "loss": torch.tensor(1.0),
            "marginal_probs": torch.tensor([[0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0]]),
            "generated_ids": torch.tensor(
                [[[1, 1], [2, 2], [3, 3], [4, 4], [5, 5], [6, 6], [7, 7], [8, 8], [9, 9], [10, 10]]]
            ),
            "labels": torch.tensor([[1, 1]]),
        },
    )

    assert set(engine.compute("val")) == {"loss", "ndcg@5", "ndcg@10", "recall@5", "recall@10"}
