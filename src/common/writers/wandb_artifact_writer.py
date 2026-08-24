"""W&B Artifact prediction writer."""

import datetime
import os
import pickle
from collections.abc import Callable
from pathlib import Path
from typing import Any

import torch
from lightning import LightningModule, Trainer

from src.common.writers.base import BaseBufferedWriter
from src.data.components.data_models import ModelOutput
from src.utils.decorators import retry
from src.utils.distributed import distributed_barrier
from src.utils.file import sync_file
from src.utils.pylogger import RankedLogger
from src.utils.wandb import require_wandb_logger_run

logger = RankedLogger(__name__, rank_zero_only=True)


class WandbArtifactWriter(BaseBufferedWriter):
    """Write prediction outputs and publish the merged bundle as a W&B Artifact."""

    def __init__(
        self,
        output_dir: str,
        artifact_name: str,
        artifact_type: str,
        role: str,
        task_name: str,
        flush_frequency: int = 1000,
        output_filename: str = "merged_predictions_tensor.pt",
        post_processing_functions: list[Callable[[str], Any]] | None = None,
        aliases: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
    ):
        super().__init__(flush_frequency=flush_frequency)
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        self.output_filename = output_filename
        self.post_processing_functions = post_processing_functions or []
        self.artifact_name = artifact_name
        self.artifact_type = artifact_type
        self.role = role
        self.task_name = task_name
        self.aliases = aliases or ["latest"]
        self.metadata = metadata or {}

    def _local_file_path(self, file_path: str) -> str:
        return os.path.join(self.output_dir, file_path)

    @property
    def merged_output_path(self) -> str:
        return os.path.join(self.output_dir, self.output_filename)

    @retry()
    def _flush_buffer(self):
        file_path = f"wandb_predictions_{self.global_rank}_{datetime.datetime.now(datetime.UTC).strftime('%Y%m%dT%H%M%S%f')[:-3]}.pkl"
        with open(self._local_file_path(file_path), "wb") as f:
            pickle.dump(self.buffer, f)

        logger.info(
            f"Global Rank: {self.global_rank} wrote {self._buffer_sample_count()} "
            f"samples to {self._local_file_path(file_path=file_path)}."
        )

    @retry()
    def on_predict_end(self, trainer: Trainer, pl_module: LightningModule):
        assert trainer.global_rank is not None, "Global rank was not provided."

        super().on_predict_end(trainer, pl_module)

        distributed_barrier()
        if self.global_rank != 0:
            logger.info(f"Rank {self.global_rank} exits on predict end.")
            return

        logger.info("Merging W&B artifact prediction shards on main process.")
        output_path = self._merge_files()

        for process_func in self.post_processing_functions:
            process_func(output_path)

        self._publish_file(
            trainer=trainer,
            file_path=output_path,
            metadata={
                **self.metadata,
                "role": self.role,
                "task_name": self.task_name,
                "local_output_path": output_path,
                "bundle_file": Path(output_path).name,
            },
        )

    def _merge_files(self) -> str:
        """Merge this writer's pickle shards into one keyed prediction bundle."""
        sync_file(self.output_dir)
        all_files = [f for f in os.listdir(self.output_dir) if f.startswith("wandb_predictions_") and f.endswith(".pkl")]
        all_outputs: list[ModelOutput] = []
        for file in all_files:
            file_path = os.path.join(self.output_dir, file)
            with open(file_path, "rb") as f:
                all_outputs.extend(pickle.load(f))
            os.remove(file_path)

        keys = torch.tensor([int(k) for output in all_outputs for k in output.keys], dtype=torch.long)
        predictions = torch.cat([torch.as_tensor(output.predictions) for output in all_outputs], dim=0)
        cpu_bundle = {"keys": keys.cpu(), "predictions": predictions.cpu()}
        output_path = self.merged_output_path
        torch.save(cpu_bundle, output_path)
        logger.info(f"Merged {len(cpu_bundle['keys'])} keyed rows into W&B artifact model output bundle.")
        logger.info(f"W&B artifact model output bundle saved to {output_path}.")
        return output_path

    def _publish_file(self, trainer: Trainer, file_path: str, metadata: dict[str, Any]):
        path = Path(file_path)
        if not path.is_file():
            raise FileNotFoundError(f"Artifact file does not exist: {file_path}.")

        import wandb

        run = require_wandb_logger_run(trainer, purpose="W&B artifact publishing")
        artifact = wandb.Artifact(name=self.artifact_name, type=self.artifact_type, metadata=metadata)
        artifact.add_file(str(path), name=path.name)
        run.log_artifact(artifact, aliases=self.aliases)
        logger.info(f"Published W&B artifact {self.artifact_name} from {file_path}.")
