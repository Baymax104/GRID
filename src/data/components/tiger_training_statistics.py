"""TIGER 子序列展开的期望监督次数；独立于 baseline 数据流。"""

import hashlib
import math
from collections.abc import Iterable
from pathlib import Path

import torch

from src.data.components.artifacts import load_model_output
from src.data.components.readers import TFRecordReader


def expected_target_counts(keys: torch.Tensor, sequences: Iterable, max_num_sequences: int = 32) -> torch.Tensor:
    keys = torch.as_tensor(keys)
    if keys.ndim != 1 or not keys.numel() or keys.is_floating_point() or (keys < 0).any():
        raise ValueError("Item keys must be a nonempty non-negative integer vector.")
    if keys.unique().numel() != keys.numel():
        raise ValueError("Duplicate item keys.")
    if isinstance(max_num_sequences, bool) or not isinstance(max_num_sequences, int) or max_num_sequences < 1:
        raise ValueError("max_num_sequences must be a positive integer.")
    lookup = {int(key): index for index, key in enumerate(keys.tolist())}
    counts = torch.zeros(keys.numel(), dtype=torch.float64)
    for sequence in sequences:
        values = torch.as_tensor(sequence)
        if values.ndim != 1 or values.is_floating_point():
            raise ValueError("Training sequence must be a one-dimensional integer array.")
        indices = []
        for value in values.tolist():
            if value not in lookup:
                raise ValueError(f"Training item absent from semantic ID catalog: {value}.")
            indices.append(lookup[value])
        total = len(indices) * (len(indices) - 1) // 2
        if not total:
            continue
        probability = -math.expm1(max_num_sequences * math.log1p(-1 / total)) if total > max_num_sequences else 1.0
        for position, index in enumerate(indices):
            counts[index] += position * probability
    if counts.sum() <= 0:
        raise ValueError("No supervised training targets were found.")
    return counts


def load_training_statistics(
    semantic_id_path: str,
    training_data_dir: str,
    num_hierarchies: int,
    max_num_sequences: int = 32,
    source_split: str = "training",
    wandb_entity: str | None = None,
    wandb_project: str | None = None,
) -> dict:
    directory = Path(training_data_dir)
    if source_split != "training" or directory.name != "training":
        raise ValueError("Training statistics require the training split/directory.")
    files = sorted(directory.rglob("*.tfrecord.gz"))
    if not files:
        raise FileNotFoundError(f"No training TFRecords in {directory}.")
    bundle = load_model_output(
        semantic_id_path, field_name="semantic_id_path", wandb_entity=wandb_entity, wandb_project=wandb_project
    )
    sids = bundle.predictions
    if sids.ndim != 2 or not 0 < num_hierarchies <= sids.size(1):
        raise ValueError("Semantic ID hierarchy shape mismatch.")
    sids = sids[:, :num_hierarchies].long().cpu()
    if sids.size(0) != bundle.keys.numel():
        raise ValueError("Semantic ID keys/rows mismatch.")
    rows = TFRecordReader(list_of_file_paths=[str(p) for p in files], shuffle_rows=False).iterrows()
    counts = expected_target_counts(bundle.keys.cpu(), (row["sequence_data"] for row in rows), max_num_sequences)
    manifest = []
    for path in files:
        with path.open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
        manifest.append({"path": path.relative_to(directory).as_posix(), "sha256": digest})
    sid_digest = hashlib.sha256(bundle.keys.cpu().numpy().tobytes() + sids.numpy().tobytes()).hexdigest()
    return {
        "semantic_ids": sids,
        "expected_counts": counts,
        "metadata": {
            "schema_version": 1,
            "estimator": "expected_unique_causal_subsequences",
            "source_split": source_split,
            "training_data_dir": str(directory),
            "semantic_id_reference": semantic_id_path,
            "semantic_id_sha256": sid_digest,
            "max_num_sequences": max_num_sequences,
            "files": manifest,
            "expected_targets": float(counts.sum()),
            "positive_target_items": int((counts > 0).sum()),
            "limitation": "single file pass expectation; not realized finite-step/DDP/drop_last exposure",
        },
    }
