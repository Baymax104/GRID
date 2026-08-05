import inspect

import torch
from transformers import T5Config, T5EncoderModel
from transformers.models.t5.modeling_t5 import T5Stack

from src.recommendation.tiger.tiger import Tiger


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
