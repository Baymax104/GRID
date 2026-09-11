"""Diagnosis LightningDataModule for artifact-backed analysis datasets."""

from lightning.pytorch.trainer.states import TrainerFn
from torch.utils.data import DataLoader

from src.data.components.artifacts import (
    get_resolved_artifact_registry,
    load_prefix_trace_artifact,
    resolve_reference,
)
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
        candidate_allocation_probe_enabled = bool(
            getattr(config, "candidate_allocation_probe_enabled", False)
        )
        if candidate_allocation_probe_enabled:
            recommendation_output_reference = getattr(
                config, "baseline_recommendation_output_path", None
            )
            recommendation_field_name = "baseline_recommendation_output_path"
            widened_recommendation_output_reference = getattr(
                config, "intervention_recommendation_output_path", None
            )
            widened_recommendation_field_name = "intervention_recommendation_output_path"
            fixed_prefix_trace_reference = getattr(config, "baseline_prefix_trace_path", None)
            fixed_prefix_trace_field_name = "baseline_prefix_trace_path"
            widened_prefix_trace_reference = getattr(
                config, "intervention_prefix_trace_path", None
            )
            widened_prefix_trace_field_name = "intervention_prefix_trace_path"
        else:
            recommendation_output_reference = getattr(config, "recommendation_output_path", None)
            recommendation_field_name = "recommendation_output_path"
            widened_recommendation_output_reference = getattr(
                config, "widened_recommendation_output_path", None
            )
            widened_recommendation_field_name = "widened_recommendation_output_path"
            fixed_prefix_trace_reference = getattr(config, "fixed_prefix_trace_path", None)
            fixed_prefix_trace_field_name = "fixed_prefix_trace_path"
            widened_prefix_trace_reference = getattr(config, "widened_prefix_trace_path", None)
            widened_prefix_trace_field_name = "widened_prefix_trace_path"

        recommendation_output_path = recommendation_output_reference
        if recommendation_output_path is not None:
            recommendation_output_path = resolve_reference(
                recommendation_output_path,
                field_name=recommendation_field_name,
                default_entity=wandb_entity,
                default_project=wandb_project,
            )
        widened_recommendation_output_path = widened_recommendation_output_reference
        if widened_recommendation_output_path is not None:
            widened_recommendation_output_path = resolve_reference(
                widened_recommendation_output_path,
                field_name=widened_recommendation_field_name,
                default_entity=wandb_entity,
                default_project=wandb_project,
            )
        fixed_prefix_trace = None
        if fixed_prefix_trace_reference is not None:
            fixed_prefix_trace = load_prefix_trace_artifact(
                fixed_prefix_trace_reference,
                field_name=fixed_prefix_trace_field_name,
                wandb_entity=wandb_entity,
                wandb_project=wandb_project,
            )
        widened_prefix_trace = None
        if widened_prefix_trace_reference is not None:
            widened_prefix_trace = load_prefix_trace_artifact(
                widened_prefix_trace_reference,
                field_name=widened_prefix_trace_field_name,
                wandb_entity=wandb_entity,
                wandb_project=wandb_project,
            )
        resolved_artifact_identity = {
            record.field_name: {
                "original_uri": record.original_uri,
                "resolved_path": record.resolved_path,
                "producer_run_id": record.producer_run_id,
                "artifact_name": record.artifact_name,
                "artifact_version": record.artifact_version,
                "artifact_type": record.artifact_type,
                "artifact_path": record.artifact_path,
                "role": record.role,
                "file": record.file,
            }
            for record in get_resolved_artifact_registry().records()
            if record.field_name
            in {
                "semantic_id_path",
                "embedding_path",
                "recommendation_output_path",
                "widened_recommendation_output_path",
                "fixed_prefix_trace_path",
                "widened_prefix_trace_path",
                "baseline_recommendation_output_path",
                "intervention_recommendation_output_path",
                "baseline_prefix_trace_path",
                "intervention_prefix_trace_path",
            }
        }
        self.stage_to_dataset[stage] = config.dataset_class(
            dataset_config=config.dataset_config,
            data_folder=config.data_folder,
            semantic_id_path=semantic_id_path,
            raw_num_hierarchies=config.raw_num_hierarchies,
            embedding_path=embedding_path,
            recommendation_output_path=recommendation_output_path,
            widened_recommendation_output_path=widened_recommendation_output_path,
            fixed_prefix_trace=fixed_prefix_trace,
            widened_prefix_trace=widened_prefix_trace,
            semantic_id_reference=config.semantic_id_path,
            fixed_prefix_trace_reference=fixed_prefix_trace_reference,
            widened_prefix_trace_reference=widened_prefix_trace_reference,
            recommendation_output_reference=recommendation_output_reference,
            widened_recommendation_output_reference=widened_recommendation_output_reference,
            calibration_statistics_ready=bool(getattr(config, "calibration_statistics_ready", False)),
            search_ranking_enabled=bool(getattr(config, "search_ranking_enabled", False)),
            risk_standardization_enabled=bool(getattr(config, "risk_standardization_enabled", False)),
            candidate_allocation_probe_enabled=candidate_allocation_probe_enabled,
            input_metadata_aliases=(
                {
                    "baseline_recommendation_output_reference": recommendation_output_reference,
                    "intervention_recommendation_output_reference": widened_recommendation_output_reference,
                    "baseline_prefix_trace_reference": fixed_prefix_trace_reference,
                    "intervention_prefix_trace_reference": widened_prefix_trace_reference,
                }
                if candidate_allocation_probe_enabled
                else None
            ),
            resolved_artifact_identity=resolved_artifact_identity,
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
