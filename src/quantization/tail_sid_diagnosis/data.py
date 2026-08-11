from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from src.data.components.readers import TFRecordReader
from src.data.utils import gather_predictions_by_keys, load_model_output


@dataclass(frozen=True)
class SIDViews:
    item_ids: torch.Tensor
    raw_sid: torch.Tensor
    model_sid: torch.Tensor
    dedup_digit: torch.Tensor


def load_sid_views(semantic_id_path: str, raw_num_hierarchies: int) -> SIDViews:
    bundle = load_model_output(semantic_id_path)
    predictions = bundle.predictions.long()
    if predictions.ndim != 2:
        raise ValueError(f"Semantic ID predictions must be 2-D, got shape {tuple(predictions.shape)}.")
    if raw_num_hierarchies <= 0:
        raise ValueError("raw_num_hierarchies must be positive.")
    if predictions.size(1) < raw_num_hierarchies:
        raise ValueError(
            "Semantic ID prediction width must be >= raw_num_hierarchies: "
            f"width={predictions.size(1)}, raw_num_hierarchies={raw_num_hierarchies}."
        )

    dedup_digit = torch.zeros(predictions.size(0), dtype=torch.long)
    if predictions.size(1) > raw_num_hierarchies:
        dedup_digit = predictions[:, -1].long()

    return SIDViews(
        item_ids=bundle.keys.long(),
        raw_sid=predictions[:, :raw_num_hierarchies].long(),
        model_sid=predictions,
        dedup_digit=dedup_digit,
    )


def load_embeddings_for_items(embedding_path: str | None, item_ids: torch.Tensor) -> torch.Tensor | None:
    if embedding_path is None:
        return None
    bundle = load_model_output(embedding_path)
    embeddings = gather_predictions_by_keys(bundle, item_ids)
    if embeddings.ndim != 2:
        raise ValueError(f"Item embeddings must be 2-D, got shape {tuple(embeddings.shape)}.")
    return embeddings.float()


def iter_training_rows(data_dir: str):
    training_dir = Path(data_dir) / "training"
    files = sorted(str(path) for path in training_dir.rglob("*.tfrecord.gz"))
    if not files:
        raise FileNotFoundError(f"No training TFRecord files found under {training_dir}.")
    yield from TFRecordReader(list_of_file_paths=files, shuffle_rows=False).iterrows()


def _sequence_values(row: dict[str, Any]) -> list[int]:
    if "sequence_data" not in row:
        raise KeyError("Training row does not contain 'sequence_data'.")
    sequence = row["sequence_data"]
    if isinstance(sequence, torch.Tensor):
        return [int(value) for value in sequence.reshape(-1).tolist()]
    if hasattr(sequence, "reshape") and hasattr(sequence, "tolist"):
        return [int(value) for value in sequence.reshape(-1).tolist()]
    return [int(value) for value in sequence]


def compute_train_frequencies(rows, item_ids: torch.Tensor) -> dict[int, int]:
    known_items = {int(item_id) for item_id in item_ids.tolist()}
    frequencies = {item_id: 0 for item_id in known_items}
    row_count = 0
    for row in rows:
        row_count += 1
        for item_id in _sequence_values(row):
            if item_id in frequencies:
                frequencies[item_id] += 1
    if row_count == 0:
        raise ValueError("No training sequence records were read.")
    return frequencies
