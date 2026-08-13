"""Diagnosis LightningDataModule for artifact-backed analysis datasets."""

from lightning.pytorch.trainer.states import TrainerFn
from torch.utils.data import DataLoader

from src.data.datamodule.stage import StageDataModule


class DiagnosisDataModule(StageDataModule):
    """Test-only datamodule for diagnosis datasets."""

    def __init__(self, test_dataloader_config):
        super().__init__(test_dataloader_config=test_dataloader_config)
        self.stage_to_dataset = {}

    def setup_stage(self, stage: TrainerFn) -> None:
        if stage != TrainerFn.TESTING:
            raise ValueError(f"DiagnosisDataModule only supports test stage, got {stage}.")
        if stage in self.stage_to_dataset:
            return
        config = self.get_stage_config(stage)
        self.stage_to_dataset[stage] = config.dataset_class(
            dataset_config=config.dataset_config,
            data_folder=config.data_folder,
            semantic_id_path=config.semantic_id_path,
            raw_num_hierarchies=config.raw_num_hierarchies,
            embedding_path=getattr(config, "embedding_path", None),
        )

    def build_dataloader(self, stage: TrainerFn):
        if stage not in self.stage_to_dataset:
            raise AttributeError(f"Stage {stage} must initialize diagnosis dataset.")
        config = self.get_stage_config(stage)
        return DataLoader(
            dataset=self.stage_to_dataset[stage],
            batch_size=config.batch_size_per_device,
            num_workers=config.num_workers,
            pin_memory=config.pin_memory,
            persistent_workers=config.persistent_workers if config.num_workers > 0 else False,
            drop_last=config.drop_last,
            collate_fn=config.collate_fn,
            timeout=config.timeout,
        )
