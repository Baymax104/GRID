from pathlib import Path

import pytest
import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
from omegaconf.errors import MissingMandatoryValue

from src.data.components.collate import collate_fn_sequence
from src.data.datamodule.tiger_trace import TigerTraceDataModule
from src.utils.launcher import apply_dry_run_overrides

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _compose(*extra: str):
    with initialize_config_dir(config_dir=str(PROJECT_ROOT / "configs"), version_base="1.3"):
        return compose(
            config_name="main",
            overrides=[
                "experiment=tiger_prefix_trace",
                "data_dir=data/beauty",
                "data_split=evaluation",
                "beam_width=10",
                "devices=[0]",
                "group=rkmeans",
                "ckpt_path=model.ckpt",
                "semantic_id_path=semantic.pt",
                *extra,
            ],
        )


@pytest.mark.parametrize(("data_split", "beam_width"), [("evaluation", 10), ("testing", 50)])
def test_trace_experiment_composes_split_and_runtime_beam(data_split, beam_width):
    cfg = _compose(f"data_split={data_split}", f"beam_width={beam_width}")

    assert cfg.run_mode == "inference"
    assert cfg.task_name == "tiger_prefix_trace"
    assert cfg.data.predict_dataloader.data_folder == f"data/beauty/{data_split}"
    assert cfg.data.collate.target_field_name == "target_ids"
    assert cfg.model.root.top_k_for_generation == beam_width
    assert cfg.model.root.trace_prefix_survival is True
    assert cfg.model.root.prefix_trace_metadata.data_split == data_split
    assert cfg.model.root.prefix_trace_metadata.beam_width == beam_width
    assert cfg.callbacks.prefix_trace_writer.role == "prefix_trace"


def test_trace_experiment_requires_explicit_split():
    with initialize_config_dir(config_dir=str(PROJECT_ROOT / "configs"), version_base="1.3"):
        cfg = compose(
            config_name="main",
            overrides=[
                "experiment=tiger_prefix_trace",
                "data_dir=data/beauty",
                "devices=[0]",
                "group=rkmeans",
                "ckpt_path=model.ckpt",
                "semantic_id_path=semantic.pt",
            ],
        )

    with pytest.raises(MissingMandatoryValue, match="data_split"):
        _ = cfg.data_split


def test_trace_datamodule_rejects_invalid_split_before_dataset_read():
    cfg = _compose("data_split=invalid")

    with pytest.raises(ValueError, match="evaluation.*testing"):
        TigerTraceDataModule("invalid", cfg.data.predict_dataloader)


def test_trace_collate_preserves_targets_and_keeps_user_id_out_of_features():
    model_input, labels = collate_fn_sequence(
        [
            {
                "input_ids": torch.tensor([1, 2, 3, 4]),
                "attention_mask": torch.tensor([1, 1, 1, 1]),
                "target_ids": torch.tensor([2, 3]),
                "user_id": torch.tensor([19]),
            }
        ],
        input_field_name="input_ids",
        attention_mask_field_name="attention_mask",
        target_field_name="target_ids",
        output_key_field_name="user_id",
    )

    assert model_input.output_keys.tolist() == [19]
    assert labels is not None and labels.target_ids.tolist() == [[2, 3]]
    assert model_input.input_ids.shape == (1, 4)


def test_sequence_collate_rejects_non_scalar_output_keys():
    with pytest.raises(ValueError, match="one scalar output key per row"):
        collate_fn_sequence(
            [
                {
                    "input_ids": torch.tensor([1, 2]),
                    "attention_mask": torch.tensor([1, 1]),
                    "target_ids": torch.tensor([2, 3]),
                    "user_id": torch.tensor([19, 20]),
                }
            ],
            input_field_name="input_ids",
            attention_mask_field_name="attention_mask",
            target_field_name="target_ids",
            output_key_field_name="user_id",
        )


def test_trace_dry_run_disables_business_writers():
    cfg = _compose()
    cfg.dry_run = True

    updated = apply_dry_run_overrides(cfg)

    assert updated.callbacks.prefix_trace_writer is None
    assert updated.callbacks.recommendation_artifact_writer is None


def test_ordinary_inference_remains_trace_disabled():
    config = OmegaConf.load(PROJECT_ROOT / "configs/model/tiger_inference.yaml")

    assert config.root.get("trace_prefix_survival", False) is False
