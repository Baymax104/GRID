from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest
from lightning.pytorch.trainer.states import TrainerFn
from omegaconf import OmegaConf

from src.data.datamodule import FileDataModule, StageDataModule
from src.utils.file import list_files


@dataclass
class _TrainerStub:
    world_size: int = 1
    global_rank: int = 0


class _ReaderStub:
    @staticmethod
    def get_file_suffix() -> str:
        return ".tfrecord"


def _collate(items: list[Any]) -> list[Any]:
    return items


class _StageDataModuleStub(StageDataModule):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.prepared_stages: list[TrainerFn] = []

    def setup_stage(self, stage: TrainerFn) -> None:
        if stage in self.prepared_stages:
            return
        self.prepared_stages.append(stage)

    def build_dataloader(self, stage: TrainerFn) -> str:
        if stage not in self.prepared_stages:
            raise AttributeError(f"Stage {stage} must be prepared before building dataloader.")
        return stage.value


def _stage_config(folder: Path):
    return OmegaConf.create(
        {
            "dataset_class": None,
            "dataset_config": {
                "data_reader": _ReaderStub,
            },
            "data_folder": str(folder),
            "assign_files_by_size": False,
            "batch_size_per_device": 1,
            "num_workers": 0,
            "pin_memory": False,
            "persistent_workers": False,
            "drop_last": False,
            "collate_fn": _collate,
            "timeout": 0,
        },
        flags={"allow_objects": True},
    )


def _write_stage_file(folder: Path, name: str = "part.tfrecord") -> str:
    folder.mkdir(exist_ok=True)
    file_path = folder / name
    file_path.write_text("x")
    return list_files(str(folder), "*.tfrecord")[0]


@pytest.fixture
def stage_dirs(tmp_path: Path) -> dict[TrainerFn, Path]:
    return {
        TrainerFn.FITTING: tmp_path / "training",
        TrainerFn.VALIDATING: tmp_path / "evaluation",
        TrainerFn.TESTING: tmp_path / "testing",
        TrainerFn.PREDICTING: tmp_path / "items",
    }


def _datamodule(stage_dirs: dict[TrainerFn, Path]) -> FileDataModule:
    datamodule = FileDataModule(
        train_dataloader_config=_stage_config(stage_dirs[TrainerFn.FITTING]),
        val_dataloader_config=_stage_config(stage_dirs[TrainerFn.VALIDATING]),
        test_dataloader_config=_stage_config(stage_dirs[TrainerFn.TESTING]),
        predict_dataloader_config=_stage_config(stage_dirs[TrainerFn.PREDICTING]),
    )
    datamodule.trainer = _TrainerStub()
    return datamodule


def test_setup_fit_prepares_train_and_validation_only(stage_dirs: dict[TrainerFn, Path]) -> None:
    train_file = _write_stage_file(stage_dirs[TrainerFn.FITTING])
    val_file = _write_stage_file(stage_dirs[TrainerFn.VALIDATING])
    datamodule = _datamodule(stage_dirs)

    datamodule.setup("fit")

    assert datamodule.stage_to_file_map == {
        TrainerFn.FITTING: {0: [str(train_file)]},
        TrainerFn.VALIDATING: {0: [str(val_file)]},
    }


def test_stage_base_setup_fit_prepares_train_and_validation() -> None:
    datamodule = _StageDataModuleStub(
        train_dataloader_config=OmegaConf.create({}),
        val_dataloader_config=OmegaConf.create({}),
        test_dataloader_config=OmegaConf.create({}),
        predict_dataloader_config=OmegaConf.create({}),
    )
    datamodule.trainer = _TrainerStub()

    datamodule.setup("fit")

    assert datamodule.prepared_stages == [TrainerFn.FITTING, TrainerFn.VALIDATING]
    assert datamodule.train_dataloader() == "fit"
    assert datamodule.val_dataloader() == "validate"


def test_stage_base_skips_none_config_stages() -> None:
    datamodule = _StageDataModuleStub(
        train_dataloader_config=None,
        val_dataloader_config=OmegaConf.create({}),
        test_dataloader_config=None,
        predict_dataloader_config=None,
    )
    datamodule.trainer = _TrainerStub()

    datamodule.setup(None)

    assert datamodule.prepared_stages == [TrainerFn.VALIDATING]
    with pytest.raises(AttributeError, match="has no dataloader config"):
        datamodule.train_dataloader()


def test_stage_base_requires_setup_before_dataloader() -> None:
    datamodule = _StageDataModuleStub(test_dataloader_config=OmegaConf.create({}))
    datamodule.trainer = _TrainerStub()

    with pytest.raises(AttributeError, match="must be prepared"):
        datamodule.test_dataloader()


def test_setup_test_prepares_test_only(stage_dirs: dict[TrainerFn, Path]) -> None:
    test_file = _write_stage_file(stage_dirs[TrainerFn.TESTING])
    datamodule = _datamodule(stage_dirs)

    datamodule.setup("test")

    assert datamodule.stage_to_file_map == {TrainerFn.TESTING: {0: [str(test_file)]}}


def test_setup_predict_prepares_predict_only(stage_dirs: dict[TrainerFn, Path]) -> None:
    predict_file = _write_stage_file(stage_dirs[TrainerFn.PREDICTING])
    datamodule = _datamodule(stage_dirs)

    datamodule.setup(TrainerFn.PREDICTING)

    assert datamodule.stage_to_file_map == {TrainerFn.PREDICTING: {0: [str(predict_file)]}}


def test_setup_none_prepares_all_configured_stages(stage_dirs: dict[TrainerFn, Path]) -> None:
    expected = {}
    for stage, folder in stage_dirs.items():
        expected[stage] = {0: [str(_write_stage_file(folder))]}
    datamodule = _datamodule(stage_dirs)

    datamodule.setup(None)

    assert datamodule.stage_to_file_map == expected


def test_setup_is_idempotent_for_prepared_stage(stage_dirs: dict[TrainerFn, Path]) -> None:
    first_file = _write_stage_file(stage_dirs[TrainerFn.TESTING], "first.tfrecord")
    datamodule = _datamodule(stage_dirs)
    datamodule.setup("test")
    _write_stage_file(stage_dirs[TrainerFn.TESTING], "second.tfrecord")

    datamodule.setup("test")

    assert datamodule.stage_to_file_map == {TrainerFn.TESTING: {0: [str(first_file)]}}
