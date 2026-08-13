"""Shared LightningDataModule stage lifecycle utilities."""

from abc import ABC, abstractmethod
from typing import Any

from lightning import LightningDataModule
from lightning.pytorch.trainer.states import TrainerFn
from omegaconf import DictConfig


class StageDataModule(LightningDataModule, ABC):
    """Shared setup-stage lifecycle for project datamodules."""

    def __init__(
        self,
        train_dataloader_config: DictConfig | None = None,
        val_dataloader_config: DictConfig | None = None,
        test_dataloader_config: DictConfig | None = None,
        predict_dataloader_config: DictConfig | None = None,
    ):
        super().__init__()
        self.stage_to_config = {
            TrainerFn.FITTING: train_dataloader_config,
            TrainerFn.VALIDATING: val_dataloader_config,
            TrainerFn.TESTING: test_dataloader_config,
            TrainerFn.PREDICTING: predict_dataloader_config,
        }

    def _resolve_setup_stages(self, stage: str | TrainerFn | None) -> tuple[TrainerFn, ...]:
        if stage is None:
            return tuple(self.stage_to_config)

        stage_value = stage.value if isinstance(stage, TrainerFn) else stage
        match stage_value:
            case TrainerFn.FITTING.value:
                return TrainerFn.FITTING, TrainerFn.VALIDATING
            case TrainerFn.VALIDATING.value:
                return (TrainerFn.VALIDATING,)
            case TrainerFn.TESTING.value:
                return (TrainerFn.TESTING,)
            case TrainerFn.PREDICTING.value:
                return (TrainerFn.PREDICTING,)
            case _:
                raise ValueError(f"Unsupported datamodule setup stage: {stage!r}.")

    def setup(self, stage: str | TrainerFn | None):
        if not hasattr(self, "trainer") or self.trainer is None:
            raise AttributeError("self.trainer must be initialized before call to setup().")

        for trainer_stage in self._resolve_setup_stages(stage):
            if self.stage_to_config[trainer_stage] is None:
                continue

            self.setup_stage(trainer_stage)

    def get_stage_config(self, stage: TrainerFn) -> DictConfig:
        if not hasattr(self, "trainer"):
            raise AttributeError("self.trainer must be initialized before call to get_dataloader().")
        config = self.stage_to_config[stage]
        if config is None:
            raise AttributeError(f"Stage {stage} has no dataloader config.")
        return config

    @abstractmethod
    def setup_stage(self, stage: TrainerFn) -> None:
        """Set up resources required to build the dataloader for a stage."""

    @abstractmethod
    def build_dataloader(self, stage: TrainerFn) -> Any:
        """Build the dataloader for a prepared stage."""

    def get_dataloader(self, stage: TrainerFn):
        self.get_stage_config(stage)
        return self.build_dataloader(stage)

    def train_dataloader(self):
        return self.get_dataloader(stage=TrainerFn.FITTING)

    def val_dataloader(self):
        return self.get_dataloader(stage=TrainerFn.VALIDATING)

    def test_dataloader(self):
        return self.get_dataloader(stage=TrainerFn.TESTING)

    def predict_dataloader(self):
        return self.get_dataloader(stage=TrainerFn.PREDICTING)
