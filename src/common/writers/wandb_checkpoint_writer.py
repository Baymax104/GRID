"""W&B Artifact writer for Lightning checkpoints."""

from pathlib import Path
from typing import Any

from lightning import LightningModule, Trainer
from lightning.pytorch.callbacks import Callback, ModelCheckpoint

from src.utils.pylogger import RankedLogger
from src.utils.wandb import require_wandb_logger_run

logger = RankedLogger(__name__, rank_zero_only=True)


class WandbCheckpointWriter(Callback):
    """Publish a checkpoint produced by Lightning ModelCheckpoint."""

    def __init__(
        self,
        artifact_name: str,
        task_name: str,
        artifact_type: str = "checkpoint",
        role: str = "checkpoint",
        source_path: str | None = None,
        selection: str = "best",
        aliases: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        missing_checkpoint: str = "fail",
    ):
        super().__init__()
        self.artifact_name = artifact_name
        self.task_name = task_name
        self.artifact_type = artifact_type
        self.role = role
        self.source_path = source_path
        self.selection = selection
        self.aliases = aliases or ["latest"]
        self.metadata = metadata or {}
        self.missing_checkpoint = missing_checkpoint

    def on_train_end(self, trainer: Trainer, pl_module: LightningModule):
        if getattr(trainer, "global_rank", 0) not in (None, 0):
            return
        selected = self._select_checkpoint(trainer)
        if selected is None:
            return
        checkpoint_path, checkpoint_metadata = selected
        self._publish_file(
            trainer=trainer,
            source_path=checkpoint_path,
            metadata={
                **self.metadata,
                **checkpoint_metadata,
                "role": self.role,
                "task_name": self.task_name,
                "local_output_path": checkpoint_path,
                "bundle_file": Path(checkpoint_path).name,
            },
        )

    def _select_checkpoint(self, trainer: Trainer) -> tuple[str, dict[str, Any]] | None:
        if self.source_path:
            return self.source_path, {"selection": "explicit"}

        checkpoints = [callback for callback in trainer.callbacks if isinstance(callback, ModelCheckpoint)]
        if len(checkpoints) != 1:
            return self._handle_missing_or_ambiguous(
                f"Expected exactly one ModelCheckpoint callback, found {len(checkpoints)}."
            )

        checkpoint = checkpoints[0]
        if self.selection == "best":
            checkpoint_path = getattr(checkpoint, "best_model_path", None)
        elif self.selection == "last":
            checkpoint_path = getattr(checkpoint, "last_model_path", None)
        else:
            raise ValueError("selection must be one of: best, last.")

        if not checkpoint_path:
            return self._handle_missing_or_ambiguous(f"ModelCheckpoint has no {self.selection} checkpoint path.")
        if not Path(checkpoint_path).is_file():
            return self._handle_missing_or_ambiguous(f"Checkpoint file does not exist: {checkpoint_path}.")

        return checkpoint_path, {
            "selection": self.selection,
            "best_model_path": getattr(checkpoint, "best_model_path", None),
            "last_model_path": getattr(checkpoint, "last_model_path", None),
            "monitor": getattr(checkpoint, "monitor", None),
            "mode": getattr(checkpoint, "mode", None),
            "best_model_score": _stringify_metric(getattr(checkpoint, "best_model_score", None)),
        }

    def _handle_missing_or_ambiguous(self, message: str) -> tuple[str, dict[str, Any]] | None:
        if self.missing_checkpoint == "skip":
            logger.warning(f"Skipping W&B checkpoint artifact publish: {message}")
            return None
        raise ValueError(message)

    def _publish_file(self, trainer: Trainer, source_path: str, metadata: dict[str, Any]):
        path = Path(source_path)
        if not path.is_file():
            raise FileNotFoundError(f"Artifact source file does not exist: {source_path}.")

        import wandb

        run = require_wandb_logger_run(trainer, purpose="W&B checkpoint artifact publishing")
        artifact = wandb.Artifact(name=self.artifact_name, type=self.artifact_type, metadata=metadata)
        artifact.add_file(str(path), name=path.name)
        run.log_artifact(artifact, aliases=self.aliases)
        logger.info(f"Published W&B checkpoint artifact {self.artifact_name} from {source_path}.")


def _stringify_metric(value) -> str | float | int | None:
    if value is None:
        return None
    if hasattr(value, "item"):
        return value.item()
    return value
