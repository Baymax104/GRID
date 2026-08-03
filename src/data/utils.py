"""Utilities for data processing."""

import heapq
import random
from collections import defaultdict

import torch

from src.utils.file_utils import get_file_size


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
