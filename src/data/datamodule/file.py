"""File-backed LightningDataModule for iterable datasets."""

from lightning.pytorch.trainer.states import TrainerFn
from omegaconf import DictConfig

from src.data.dataloaders import DataloaderWithIterationRetry
from src.data.datamodule.stage import StageDataModule
from src.data.utils import assign_files_to_workers
from src.utils.file import list_files
from src.utils.pylogger import RankedLogger

logger = RankedLogger(__name__, rank_zero_only=True)


class FileDataModule(StageDataModule):
    """File-assignment and dataloader assembly for loading pipelines."""

    def __init__(
        self,
        train_dataloader_config: DictConfig | None = None,
        val_dataloader_config: DictConfig | None = None,
        test_dataloader_config: DictConfig | None = None,
        predict_dataloader_config: DictConfig | None = None,
    ):
        super().__init__(
            train_dataloader_config=train_dataloader_config,
            val_dataloader_config=val_dataloader_config,
            test_dataloader_config=test_dataloader_config,
            predict_dataloader_config=predict_dataloader_config,
        )
        self.stage_to_file_map: dict[TrainerFn, dict[int, list[str]]] = {}

    def get_file_suffix_from_config(self, config: DictConfig) -> str:
        file_format: str | None = getattr(config.dataset_config, "file_format", None)
        if file_format:
            return file_format
        data_reader_factory = config.dataset_config.data_reader
        data_reader_target = getattr(data_reader_factory, "func", data_reader_factory)
        return data_reader_target.get_file_suffix()  # noqa

    def setup_stage(self, stage: TrainerFn) -> None:
        if stage in self.stage_to_file_map:
            return

        config = self.get_stage_config(stage)
        list_of_files = list_files(
            folder_path=config.data_folder,
            suffix=f"*{self.get_file_suffix_from_config(config)}",
        )
        if hasattr(config, "limit_files") and config.limit_files:
            list_of_files = list_of_files[: config.limit_files]

        self.stage_to_file_map[stage], _ = assign_files_to_workers(
            list_of_files=list_of_files,
            total_workers=self.trainer.world_size,
            assign_by_size=config.assign_files_by_size,
            shuffle_files=getattr(config.dataset_config, "shuffle_files", False),
        )

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

    def build_dataloader(self, stage: TrainerFn):
        if stage not in self.stage_to_file_map:
            raise AttributeError(f"Stage {stage} must initialize file map.")
        curr_config = self.get_stage_config(stage)

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
