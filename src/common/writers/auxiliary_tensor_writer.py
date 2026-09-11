"""Domain-neutral writer for named keyed tensor payloads."""

from __future__ import annotations

import datetime
import os
import pickle
from pathlib import Path
from typing import Any

import torch
from lightning import LightningModule, Trainer

from src.common.writers.base import BaseBufferedWriter
from src.data.components.data_models import ModelOutput
from src.data.components.prefix_trace import validate_prefix_trace_bundle
from src.utils.decorators import retry
from src.utils.distributed import distributed_barrier
from src.utils.file import sync_file
from src.utils.pylogger import RankedLogger
from src.utils.wandb import require_wandb_logger_run

logger = RankedLogger(__name__, rank_zero_only=True)


class AuxiliaryTensorWriter(BaseBufferedWriter):
    """Merge one named auxiliary tensor payload and optionally publish it to W&B."""

    def __init__(
        self,
        output_dir: str,
        payload_name: str,
        output_filename: str,
        flush_frequency: int = 1000,
        publish_wandb: bool = False,
        artifact_name: str | None = None,
        artifact_type: str | None = None,
        role: str | None = None,
        aliases: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
    ):
        super().__init__(flush_frequency=flush_frequency)
        self.output_dir = output_dir
        self.payload_name = payload_name
        self.output_filename = output_filename
        self.publish_wandb = publish_wandb
        self.artifact_name = artifact_name
        self.artifact_type = artifact_type
        self.role = role
        self.aliases = aliases or ["latest"]
        self.metadata = metadata or {}
        os.makedirs(self.output_dir, exist_ok=True)
        if publish_wandb and not all((artifact_name, artifact_type, role)):
            raise ValueError("W&B auxiliary publication requires artifact_name, artifact_type, and role.")

    @property
    def merged_output_path(self) -> str:
        return os.path.join(self.output_dir, self.output_filename)

    @retry()
    def _flush_buffer(self):
        timestamp = datetime.datetime.now(datetime.UTC).strftime("%Y%m%dT%H%M%S%f")[:-3]
        file_path = os.path.join(
            self.output_dir,
            f"auxiliary_{self.payload_name}_{self.global_rank}_{timestamp}.pkl",
        )
        with open(file_path, "wb") as file:
            pickle.dump(self.buffer, file)

    def on_predict_end(self, trainer: Trainer, pl_module: LightningModule):
        assert trainer.global_rank is not None, "Global rank was not provided."
        super().on_predict_end(trainer, pl_module)
        distributed_barrier()
        if self.global_rank != 0:
            return

        output_path, bundle_metadata = self._merge_files()
        if self.publish_wandb:
            self._publish_file(trainer, output_path, bundle_metadata)

    def _merge_files(self) -> tuple[str, dict[str, Any]]:
        sync_file(self.output_dir)
        prefix = f"auxiliary_{self.payload_name}_"
        shard_names = sorted(
            name for name in os.listdir(self.output_dir) if name.startswith(prefix) and name.endswith(".pkl")
        )
        outputs: list[ModelOutput] = []
        for name in shard_names:
            path = os.path.join(self.output_dir, name)
            with open(path, "rb") as file:
                outputs.extend(pickle.load(file))
            os.remove(path)
        if not outputs:
            raise ValueError(f"No auxiliary payloads named {self.payload_name!r} were produced.")

        keys_parts = []
        labels_parts = []
        trace_parts: dict[str, list[torch.Tensor]] = {}
        schema_version = None
        source_metadata = None
        for output in outputs:
            if self.payload_name not in output.auxiliary:
                raise ValueError(f"ModelOutput is missing auxiliary payload {self.payload_name!r}.")
            payload = output.auxiliary[self.payload_name]
            candidate = {
                "schema_version": payload.get("schema_version"),
                "keys": torch.as_tensor(output.keys),
                "labels": payload.get("labels"),
                "trace": payload.get("trace"),
                "metadata": payload.get("metadata"),
            }
            validate_prefix_trace_bundle(candidate)
            if schema_version is None:
                schema_version = candidate["schema_version"]
                source_metadata = dict(candidate["metadata"])
                trace_parts = {name: [] for name in candidate["trace"]}
            elif candidate["schema_version"] != schema_version or dict(candidate["metadata"]) != source_metadata:
                raise ValueError("Auxiliary tensor shards have inconsistent schema version or metadata.")
            keys_parts.append(candidate["keys"])
            labels_parts.append(candidate["labels"])
            for name, value in candidate["trace"].items():
                trace_parts[name].append(value)

        bundle = {
            "schema_version": schema_version,
            "keys": torch.cat(keys_parts, dim=0).cpu(),
            "labels": torch.cat(labels_parts, dim=0).cpu(),
            "trace": {name: torch.cat(parts, dim=0).cpu() for name, parts in trace_parts.items()},
            "metadata": source_metadata,
        }
        validate_prefix_trace_bundle(bundle)
        sort_index = bundle["keys"].argsort()
        bundle["keys"] = bundle["keys"][sort_index]
        bundle["labels"] = bundle["labels"][sort_index]
        bundle["trace"] = {name: value[sort_index] for name, value in bundle["trace"].items()}
        torch.save(bundle, self.merged_output_path)
        logger.info(f"Saved {len(bundle['keys'])} keyed auxiliary rows to {self.merged_output_path}.")
        publication_metadata = {**(source_metadata or {}), "schema_version": schema_version}
        return self.merged_output_path, publication_metadata

    def _publish_file(self, trainer: Trainer, file_path: str, bundle_metadata: dict[str, Any]):
        path = Path(file_path)
        if not path.is_file():
            raise FileNotFoundError(f"Artifact file does not exist: {file_path}.")
        import wandb

        run = require_wandb_logger_run(trainer, purpose="W&B auxiliary artifact publishing")
        metadata = {
            **bundle_metadata,
            **self.metadata,
            "schema_version": bundle_metadata["schema_version"],
            "role": self.role,
            "bundle_file": path.name,
            "local_output_path": str(path),
        }
        artifact = wandb.Artifact(name=self.artifact_name, type=self.artifact_type, metadata=metadata)
        artifact.add_file(str(path), name=path.name)
        run.log_artifact(artifact, aliases=self.aliases)
