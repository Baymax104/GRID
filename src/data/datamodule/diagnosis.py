"""Diagnosis LightningDataModule for artifact-backed analysis datasets."""

from lightning.pytorch.trainer.states import TrainerFn
from torch.utils.data import DataLoader

from src.data.components.artifacts import resolve_reference
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
        wandb_entity = getattr(config, "wandb_entity", None)
        wandb_project = getattr(config, "wandb_project", None)
        semantic_id_path = resolve_reference(
            config.semantic_id_path,
            field_name="semantic_id_path",
            default_entity=wandb_entity,
            default_project=wandb_project,
        )
        embedding_path = getattr(config, "embedding_path", None)
        if embedding_path is not None:
            embedding_path = resolve_reference(
                embedding_path,
                field_name="embedding_path",
                default_entity=wandb_entity,
                default_project=wandb_project,
            )
        recommendation_output_path = getattr(config, "recommendation_output_path", None)
        if recommendation_output_path is not None:
            recommendation_output_path = resolve_reference(
                recommendation_output_path,
                field_name="recommendation_output_path",
                default_entity=wandb_entity,
                default_project=wandb_project,
            )
        self.stage_to_dataset[stage] = config.dataset_class(
            dataset_config=config.dataset_config,
            data_folder=config.data_folder,
            semantic_id_path=semantic_id_path,
            raw_num_hierarchies=config.raw_num_hierarchies,
            embedding_path=embedding_path,
            recommendation_output_path=recommendation_output_path,
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
