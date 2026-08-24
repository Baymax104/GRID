"""Callback for recording W&B artifact input lineage."""

from lightning import LightningModule, Trainer
from lightning.pytorch.callbacks import Callback

from src.data.components.artifacts import ResolvedArtifactRegistry, get_resolved_artifact_registry
from src.utils.pylogger import RankedLogger
from src.utils.wandb import require_wandb_logger_run

logger = RankedLogger(__name__, rank_zero_only=True)


class WandbArtifactLineageCallback(Callback):
    """Record resolved W&B input artifacts on the active W&B run."""

    def __init__(
        self,
        fail_on_missing_run: bool = False,
        clear_after_recording: bool = False,
        registry: ResolvedArtifactRegistry | None = None,
    ):
        super().__init__()
        self.fail_on_missing_run = fail_on_missing_run
        self.clear_after_recording = clear_after_recording
        self.registry = registry
        self._recorded_artifact_paths: set[str] = set()

    def setup(self, trainer: Trainer, pl_module: LightningModule, stage: str):
        self._record_lineage(trainer)

    def on_fit_start(self, trainer: Trainer, pl_module: LightningModule):
        self._record_lineage(trainer)

    def on_predict_start(self, trainer: Trainer, pl_module: LightningModule):
        self._record_lineage(trainer)

    def on_test_start(self, trainer: Trainer, pl_module: LightningModule):
        self._record_lineage(trainer)

    def _record_lineage(self, trainer: Trainer):
        registry = self.registry or get_resolved_artifact_registry()
        references = registry.records()
        if not references:
            return

        try:
            active_run = require_wandb_logger_run(trainer, purpose="W&B artifact lineage recording")
        except RuntimeError as exc:
            message = str(exc)
            if self.fail_on_missing_run:
                raise
            logger.warning(message)
            return

        for reference in references:
            if reference.artifact_path in self._recorded_artifact_paths:
                continue
            active_run.use_artifact(reference.artifact_path)
            self._recorded_artifact_paths.add(reference.artifact_path)
            logger.info(f"Recorded W&B artifact lineage: {reference.artifact_path}.")

        if self.clear_after_recording:
            registry.clear()
