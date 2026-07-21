"""Shared LightningDataModule for file-backed iterable datasets."""

from typing import Any

from lightning import LightningDataModule
from lightning.pytorch.trainer.states import TrainerFn
from omegaconf import DictConfig

from src.data.dataloaders import DataloaderWithIterationRetry
from src.data.utils import assign_files_to_workers
from src.utils.file_utils import list_files
from src.utils.pylogger import RankedLogger

logger = RankedLogger(__name__, rank_zero_only=True)


class BaseDataModule(LightningDataModule):
    """Shared file-assignment and dataloader assembly for loading pipelines."""

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
        self.stage_to_file_map: dict[TrainerFn, dict[int, list[str]]] = {}

    @staticmethod
    def _get_shuffle_files(config: DictConfig) -> bool:
        return getattr(config.dataset_config, "shuffle_files", False)

    def get_file_suffix_from_config(self, config: DictConfig) -> str:
        file_format: str | None = getattr(config.dataset_config, "file_format", None)
        if file_format:
            return file_format
        data_reader_factory = config.dataset_config.data_reader_factory
        data_reader_target = getattr(data_reader_factory, "func", data_reader_factory)
        return data_reader_target.get_file_suffix()  # noqa

    def setup(self, stage: str):
        if not hasattr(self, "trainer") or self.trainer is None:
            raise AttributeError("self.trainer must be initialized before call to setup().")

        for trainer_stage, config in self.stage_to_config.items():
            if config is None:
                self.stage_to_file_map[trainer_stage] = {}
                continue

            if trainer_stage in self.stage_to_file_map:
                continue

            list_of_files = list_files(
                folder_path=config.data_folder,
                suffix=f"*{self.get_file_suffix_from_config(config)}",
            )
            if hasattr(config, "limit_files") and config.limit_files:
                list_of_files = list_of_files[: config.limit_files]

            self.stage_to_file_map[trainer_stage], _ = assign_files_to_workers(
                list_of_files=list_of_files,
                total_workers=self.trainer.world_size,
                assign_by_size=config.assign_files_by_size,
                shuffle_files=self._get_shuffle_files(config),
            )

    def _get_stage_config(self, stage: TrainerFn) -> DictConfig | None:
        if not hasattr(self, "trainer"):
            raise AttributeError("self.trainer must be initialized before call to get_dataloader().")
        if not self.stage_to_file_map[stage]:
            raise AttributeError(f"Stage {stage} must initialize file map.")
        return self.stage_to_config[stage]

    def _build_dataset(self, stage: TrainerFn, curr_config: DictConfig):
        assert self.trainer is not None
        device_file_list = self.stage_to_file_map[stage].get(self.trainer.global_rank, [])
        dataset = curr_config.dataset_class(
            dataset_config=curr_config.dataset_config,
            data_folder=curr_config.data_folder,
            list_of_file_paths=device_file_list,
            global_rank=self.trainer.global_rank,
            is_for_training=stage == TrainerFn.FITTING,
        )

        return dataset

    def _resolve_persistent_workers(self, curr_config: DictConfig) -> bool:
        if curr_config.num_workers == 0:
            logger.warning(
                "num_workers is set to 0, persistent_workers will be set to False as persistent workers require num_workers > 0"
            )
            return False
        return curr_config.persistent_workers

    def _build_collate_fn(self, curr_config: DictConfig):
        return curr_config.collate_fn

    def get_dataloader(self, stage: TrainerFn):
        curr_config = self.stage_to_config[stage]
        assert curr_config is not None

        dataset = self._build_dataset(stage, curr_config)
        collate_fn = self._build_collate_fn(curr_config)
        persistent_workers = self._resolve_persistent_workers(curr_config)

        return DataloaderWithIterationRetry(
            dataset=dataset,
            batch_size=curr_config.batch_size_per_device,
            num_workers=curr_config.num_workers,
            pin_memory=curr_config.pin_memory,
            persistent_workers=persistent_workers,
            drop_last=curr_config.drop_last,
            collate_fn=collate_fn,
            timeout=curr_config.timeout,
        )

    def train_dataloader(self):
        return self.get_dataloader(stage=TrainerFn.FITTING)

    def val_dataloader(self):
        return self.get_dataloader(stage=TrainerFn.VALIDATING)

    def test_dataloader(self):
        return self.get_dataloader(stage=TrainerFn.TESTING)

    def predict_dataloader(self):
        return self.get_dataloader(stage=TrainerFn.PREDICTING)

    def state_dict(self) -> dict[Any, Any]:
        return {}

    def load_state_dict(self, state_dict: dict[str, Any]):
        pass
