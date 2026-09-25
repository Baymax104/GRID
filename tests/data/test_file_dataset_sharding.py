import random
from collections import Counter
from types import SimpleNamespace

import pytest

from src.data.datasets import FileDataset


@pytest.mark.parametrize("file_count,workers", [(88, 32), (140, 32), (88, 8), (3, 8), (0, 4), (17, 1)])
@pytest.mark.parametrize("rank,cycle", [(0, 0), (1, 0), (0, 3), (1, 3)])
@pytest.mark.parametrize("shuffle", [False, True])
def test_worker_partitions_cover_rank_files_exactly_once(monkeypatch, file_count, workers, rank, cycle, shuffle):
    files = [f"rank-{rank}/part-{i}" for i in range(file_count)]
    dataset = FileDataset(files, global_rank=rank)
    assigned = []
    state = random.getstate()
    for worker in range(workers):
        monkeypatch.setattr(
            "src.data.datasets.get_worker_info",
            lambda worker=worker: SimpleNamespace(id=worker, num_workers=workers),
        )
        partition = dataset.get_list_of_worker_files(shuffle=shuffle, seed=cycle)
        assert partition == dataset.get_list_of_worker_files(shuffle=shuffle, seed=cycle)
        if not shuffle:
            assert partition == files[worker::workers]
        assigned.extend(partition)
    assert Counter(assigned) == Counter(files)
    assert dataset.list_of_file_paths == files
    assert random.getstate() == state


def test_single_process_without_worker_preserves_coverage_and_changes_cycle_order(monkeypatch):
    monkeypatch.setattr("src.data.datasets.get_worker_info", lambda: None)
    files = [f"part-{i}" for i in range(40)]
    dataset = FileDataset(files, global_rank=0)
    first = dataset.get_list_of_worker_files(shuffle=True, seed=0)
    second = dataset.get_list_of_worker_files(shuffle=True, seed=1)
    assert Counter(first) == Counter(second) == Counter(files)
    assert first != second
    assert dataset.get_list_of_worker_files(shuffle=False) == files
