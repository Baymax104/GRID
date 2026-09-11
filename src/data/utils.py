"""Utilities for data processing."""

import heapq
import random
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch

from src.data.components.artifacts import load_model_output, load_semantic_id_tensor
from src.data.components.data_models import ModelOutput
from src.data.components.readers import TFRecordReader
from src.utils.file import get_file_size

__all__ = [
    "gather_predictions_by_keys",
    "load_model_output",
    "load_semantic_id_tensor",
    "load_training_item_frequency_tensor",
]


def _sequence_item_ids(row: dict[str, Any]) -> list[int]:
    if "sequence_data" not in row:
        raise KeyError("Training row does not contain 'sequence_data'.")
    sequence = row["sequence_data"]
    if isinstance(sequence, torch.Tensor):
        values = sequence.reshape(-1).tolist()
    elif hasattr(sequence, "reshape") and hasattr(sequence, "tolist"):
        values = sequence.reshape(-1).tolist()
    else:
        values = list(sequence)
    item_ids = [int(value) for value in values]
    if any(item_id < 0 for item_id in item_ids):
        raise ValueError("Training sequence contains a negative item id.")
    return item_ids


def load_training_item_frequency_tensor(
    semantic_id_path: str,
    training_data_dir: str,
    *,
    source_split: str = "training",
    wandb_entity: str | None = None,
    wandb_project: str | None = None,
    wandb_cache_dir: str | None = None,
) -> torch.Tensor:
    """Return training item counts aligned to a keyed semantic-ID bundle."""
    if source_split != "training":
        raise ValueError(
            "Prefix allocation frequencies must use source_split='training'; "
            f"got {source_split!r}."
        )

    bundle = load_model_output(
        semantic_id_path,
        field_name="semantic_id_path",
        wandb_entity=wandb_entity,
        wandb_project=wandb_project,
        wandb_cache_dir=wandb_cache_dir,
    )
    keys = bundle.keys.long().reshape(-1)
    if keys.numel() == 0:
        raise ValueError("Semantic ID bundle must contain at least one item key.")
    if keys.unique().numel() != keys.numel():
        raise ValueError("Duplicate keys detected in semantic ID bundle.")
    if torch.any(keys < 0):
        raise ValueError("Semantic ID bundle contains a negative item key.")

    training_dir = Path(training_data_dir)
    files = sorted(str(path) for path in training_dir.rglob("*.tfrecord.gz"))
    if not files:
        raise FileNotFoundError(f"No training TFRecord files found under {training_dir}.")

    key_to_index = {int(item_id): index for index, item_id in enumerate(keys.tolist())}
    frequencies = torch.zeros(keys.numel(), dtype=torch.long)
    row_count = 0
    for row in TFRecordReader(list_of_file_paths=files, shuffle_rows=False).iterrows():
        row_count += 1
        for item_id in _sequence_item_ids(row):
            index = key_to_index.get(item_id)
            if index is not None:
                frequencies[index] += 1
    if row_count == 0:
        raise ValueError("No training sequence records were read.")
    return frequencies


def assign_files_to_workers(
    list_of_files: list[str],
    total_workers: int,
    assign_by_size: bool,
    shuffle_files: bool,
) -> tuple[dict[int, list[str]], bool]:
    """Assign each file path in `list_of_files` across workers.

    - If `total_workers == 0`, then the function returns a single-key dict
      mapping 0 to `list_of_files` as well as a boolean indicating that the
      files are shared among "workers". This is for debugging.
    - Otherwise, if the list of files is shorter than `total_workers`, all files
      are assigned to each worker, and the returned boolean indicates that the
      files are shared among workers.
    - Otherwise, each file gets a single worker, which may be assigned according
      to file size, depending on the value of `assign_by_size`:
        - If `assign_by_size`, files are sorted by size, then assigned in a
          way that encourages even cumulative files size across workers.
        - If not `assign_by_size`, files are assigned randomly to workers.
      In this case, the return boolean is False, indicating that the files are
      not shared among workers.

    :param list_of_files: List of file paths to be assigned.
    :param total_workers: The number of workers among which to assign files.
    :param assign_by_size: Whether to assign files to balance size (if True),
        or to assign randomly.
    :param shuffle_files: Whether to shuffle file ordering before assigning files.

    :return: A dictionary mapping worker indices to file paths and a boolean
        indicating whether files have been assigned to all workers (i.e. each
        file is shared among all workers).
    NOTE: The second returned parameter is currently ignored by the datamodule
    layer but it will be used after an upcoming PR.
    """
    if total_workers == 0:
        return {0: list_of_files}, True

    # If more workers than files, then each worker gets all files, but reads
    # only a fraction of the rows
    if len(list_of_files) < total_workers:
        return {worker: list_of_files.copy() for worker in range(total_workers)}, True

    if not assign_by_size:
        # files are assigned randomly to workers
        list_of_files = list_of_files.copy()
        if shuffle_files:
            random.shuffle(list_of_files)
        worker_to_files = {worker_id: list_of_files[worker_id::total_workers] for worker_id in range(total_workers)}
        return worker_to_files, False

    # Otherwise, assign files to workers balancing by file size
    list_of_files_and_sizes = [(file, get_file_size(file)) for file in list_of_files]
    list_of_files_and_sizes.sort(key=lambda x: x[1], reverse=True)

    worker_to_files = {i: [] for i in range(total_workers)}
    worker_loads = [(0, worker_id) for worker_id in range(total_workers)]

    for file, file_size in list_of_files_and_sizes:
        # assign file to the worker with smallest storage usage
        worker_load, min_worker_load_index = heapq.heappop(worker_loads)
        worker_to_files[min_worker_load_index].append(file)
        # update worker's total storage usage
        heapq.heappush(worker_loads, (worker_load + file_size, min_worker_load_index))

    return worker_to_files, False


def combine_list_of_tensor_dicts(list_of_dicts: list[dict[str, torch.Tensor]]) -> dict[str, list[torch.Tensor]]:
    batch = defaultdict(list)
    for sequence in list_of_dicts:
        for field_name, field_sequence in sequence.items():
            batch[field_name].append(field_sequence)
    return batch


def gather_predictions_by_keys(
    bundle: ModelOutput,
    keys: torch.Tensor,
) -> torch.Tensor:
    """Gather prediction rows from a model output by business keys via binary search."""

    keys = torch.as_tensor(keys, dtype=torch.long)
    original_shape = keys.shape
    flat_keys = keys.reshape(-1)

    indices = torch.searchsorted(bundle.keys, flat_keys)
    # Clamp to valid range to avoid index-out-of-bounds on non-existent keys
    indices = indices.clamp(max=bundle.keys.numel() - 1)
    found = bundle.keys[indices]
    missing = flat_keys[found != flat_keys]
    if missing.numel() > 0:
        preview = missing[:5].tolist()
        raise KeyError(f"Missing keys in model output: {preview} (total missing={missing.numel()}).")

    gathered = bundle.predictions[indices]
    prediction_shape = bundle.predictions.shape[1:]
    return gathered.reshape(*original_shape, *prediction_shape)
