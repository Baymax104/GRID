"""Atomic local serialization and optional W&B publication for structured analysis output."""

from __future__ import annotations

import csv
import json
import os
import tempfile
from pathlib import Path
from typing import Any

from lightning import LightningModule, Trainer
from lightning.pytorch.callbacks import Callback

from src.common.writers.structured_analysis import StructuredAnalysisOutput
from src.utils.wandb import require_wandb_logger_run


class StructuredAnalysisWriter(Callback):
    """Serialize a domain-neutral named-document/table payload after a test batch."""

    def __init__(
        self,
        output_dir: str,
        output_key: str = "structured_analysis",
        publish_wandb: bool = False,
        artifact_name: str = "structured-analysis",
        artifact_type: str = "analysis",
        aliases: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
    ):
        super().__init__()
        self.output_dir = Path(output_dir)
        self.output_key = output_key
        self.publish_wandb = publish_wandb
        self.artifact_name = artifact_name
        self.artifact_type = artifact_type
        self.aliases = aliases or ["latest"]
        self.metadata = metadata or {}

    def on_test_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: dict[str, Any] | None,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        if trainer.global_rank != 0:
            return
        if outputs is None or self.output_key not in outputs:
            raise KeyError(f"Test output does not contain structured analysis key '{self.output_key}'.")
        payload = outputs[self.output_key]
        if not isinstance(payload, StructuredAnalysisOutput):
            raise TypeError(
                f"Expected StructuredAnalysisOutput at '{self.output_key}', got {type(payload).__name__}."
            )
        completed_dir = self._write_atomically(payload)
        if self.publish_wandb:
            self._publish(trainer, completed_dir, payload.metadata)

    def _write_atomically(self, payload: StructuredAnalysisOutput) -> Path:
        if self.output_dir.exists():
            raise FileExistsError(f"Structured analysis output already exists: {self.output_dir}.")
        self.output_dir.parent.mkdir(parents=True, exist_ok=True)
        staging = Path(tempfile.mkdtemp(prefix=f".{self.output_dir.name}-", dir=self.output_dir.parent))
        try:
            for name, document in sorted(payload.documents.items()):
                path = self._safe_path(staging, name)
                path.parent.mkdir(parents=True, exist_ok=True)
                with path.open("w", encoding="utf-8", newline="") as stream:
                    json.dump(document, stream, ensure_ascii=False, indent=2, sort_keys=True)
                    stream.write("\n")
            for name, rows in sorted(payload.tables.items()):
                path = self._safe_path(staging, name)
                path.parent.mkdir(parents=True, exist_ok=True)
                fieldnames = sorted({field for row in rows for field in row})
                with path.open("w", encoding="utf-8", newline="") as stream:
                    writer = csv.DictWriter(stream, fieldnames=fieldnames, extrasaction="raise")
                    if fieldnames:
                        writer.writeheader()
                        writer.writerows(rows)
            effective_metadata = {**self.metadata, **payload.metadata}
            with (staging / "manifest.json").open("w", encoding="utf-8", newline="") as stream:
                json.dump(
                    {
                        "complete": True,
                        "documents": sorted(payload.documents),
                        "tables": sorted(payload.tables),
                        "metadata": effective_metadata,
                    },
                    stream,
                    ensure_ascii=False,
                    indent=2,
                    sort_keys=True,
                )
                stream.write("\n")
            os.replace(staging, self.output_dir)
        except Exception:
            self._remove_empty_or_partial_staging(staging)
            raise
        return self.output_dir

    @staticmethod
    def _safe_path(root: Path, name: str) -> Path:
        path = (root / name).resolve()
        if path.parent != root.resolve() or Path(name).name != name:
            raise ValueError(f"Structured output name must be a plain filename: {name!r}.")
        return path

    @staticmethod
    def _remove_empty_or_partial_staging(staging: Path) -> None:
        if not staging.exists():
            return
        for path in sorted(staging.rglob("*"), reverse=True):
            if path.is_file():
                path.unlink()
            elif path.is_dir():
                path.rmdir()
        staging.rmdir()

    def _publish(self, trainer: Trainer, output_dir: Path, payload_metadata: dict[str, Any]) -> None:
        import wandb

        run = require_wandb_logger_run(trainer, purpose="structured analysis artifact publishing")
        artifact = wandb.Artifact(
            name=self.artifact_name,
            type=self.artifact_type,
            metadata={**self.metadata, **payload_metadata},
        )
        artifact.add_dir(str(output_dir))
        run.log_artifact(artifact, aliases=self.aliases)
