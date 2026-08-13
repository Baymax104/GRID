import random
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any, TypeAlias

import torch
from torch.utils.data import Dataset, IterableDataset, get_worker_info

from src.common.configs.data import DatasetConfig
from src.data.components.data_models import DiagnosisBatch, SIDViews
from src.data.components.readers import TFRecordReader
from src.data.utils import gather_predictions_by_keys, load_model_output
from src.utils.pylogger import RankedLogger

logger = RankedLogger(__name__, rank_zero_only=True)

Row: TypeAlias = dict[str, Any]
PreprocessingResult: TypeAlias = Row | Iterable[Row] | None


def _iter_preprocessing_result(result: PreprocessingResult) -> Iterator[Row]:
    """Normalize a preprocessing result into a row iterator."""
    if result is None:
        return
    if isinstance(result, dict):
        yield result
        return
    yield from result


class FileDataset:
    def __init__(
        self,
        list_of_file_paths: list[str],
        global_rank: int,
    ):
        self.global_rank = global_rank
        self.list_of_file_paths = list_of_file_paths
        self.cycle_count = 0

    def _get_worker_context(self) -> tuple[int, int, int]:
        # set global dataloader worker id in process level
        # eg. world_size = 2, num_workers = 2
        # Process 0 (global_rank = 0)
        # - Thread 0 (worker_id = 0, global_id = 0 * 2 + 0 = 0)
        # - Thread 1 (worker_id = 1, global_id = 0 * 2 + 1 = 1)
        # Process 1 (global_rank = 1)
        # - Thread 0 (worker_id = 0, global_id = 1 * 2 + 0 = 2)
        # - Thread 1 (worker_id = 1, global_id = 1 * 2 + 1 = 3)
        worker_info = get_worker_info()
        worker_id = worker_info.id if worker_info is not None else 0
        num_workers = worker_info.num_workers if worker_info is not None else 1
        global_dataloader_worker_id = self.global_rank * num_workers + worker_id
        return worker_id, num_workers, global_dataloader_worker_id

    def get_list_of_worker_files(self, shuffle: bool = False, seed: int = 0):
        worker_id, num_workers, global_dataloader_worker_id = self._get_worker_context()
        files = self.list_of_file_paths.copy()
        if shuffle:
            seed += global_dataloader_worker_id
            random.Random(seed).shuffle(files)
        worker_files = files[worker_id::num_workers]
        return worker_files


class SequenceDataset(FileDataset, IterableDataset):
    """
    An unbounded dataset is a dataset that we don't know the size of beforehand.
    For training, we will iterate over the dataset infinitely.
    For evaluation, we will iterate over the dataset once.
    """

    def __init__(
        self,
        dataset_config: DatasetConfig,
        data_folder: str,
        list_of_file_paths: list[str],
        global_rank: int,
        is_for_training: bool = True,
    ):
        super().__init__(list_of_file_paths=list_of_file_paths, global_rank=global_rank)
        self.data_folder = data_folder
        self.data_reader_factory = dataset_config.data_reader
        self.preprocessing_functions = dataset_config.preprocessing_functions
        self.shuffle_files = dataset_config.shuffle_files
        self.is_for_training = is_for_training

    def _load_data(self):
        current_worker_files = self.get_list_of_worker_files(shuffle=self.shuffle_files, seed=self.cycle_count)
        data_reader = self.data_reader_factory(list_of_file_paths=current_worker_files)
        return data_reader.iterrows()

    def _apply_preprocessing_functions(self, row: Row, start_index: int = 0) -> Iterator[Row]:
        """Apply preprocessing functions as a streaming flat-map pipeline."""
        if start_index >= len(self.preprocessing_functions):
            yield row
            return

        preprocessing_function = self.preprocessing_functions[start_index]
        for next_row in _iter_preprocessing_result(preprocessing_function(row)):
            yield from self._apply_preprocessing_functions(next_row, start_index + 1)

    def __iter__(self):
        dataset_iterable = self._load_data()
        # If the dataset is for training, we want to keep iterating over the dataset infinitely.
        # On a streaming dataset, we will always be on Epoch 0.
        finished_iteration = False
        while not finished_iteration:
            for row in dataset_iterable:
                yield from self._apply_preprocessing_functions(row)
            # if the dataset is not for training, we stop the loop. Otherwise, we continue.
            finished_iteration = not self.is_for_training
            if not finished_iteration:
                self.cycle_count += 1
                dataset_iterable = self._load_data()
        return None


class DiagnosisDataset(Dataset):
    """Artifact-backed dataset that yields one full diagnosis batch."""

    def __init__(
        self,
        dataset_config: Any,
        data_folder: str,
        semantic_id_path: str,
        raw_num_hierarchies: int,
        embedding_path: str | None = None,
    ):
        self.dataset_config = dataset_config
        self.data_folder = data_folder
        self.semantic_id_path = semantic_id_path
        self.raw_num_hierarchies = raw_num_hierarchies
        self.embedding_path = embedding_path
        self._batch: DiagnosisBatch | None = None

    def __len__(self) -> int:
        return 1

    def __getitem__(self, index: int) -> DiagnosisBatch:
        if index != 0:
            raise IndexError(index)
        if self._batch is None:
            self._batch = self._build_batch()
        return self._batch

    def _build_batch(self) -> DiagnosisBatch:
        sid_views = self._load_sid_views()
        batch = DiagnosisBatch(
            sid_views=sid_views,
            frequencies=self._compute_train_frequencies(sid_views.item_ids),
            groups_by_item={},
            embeddings=self._load_embeddings_for_items(sid_views.item_ids),
        )
        for preprocessing_function in getattr(self.dataset_config, "preprocessing_functions", []):
            batch = preprocessing_function(batch)
        return batch

    def _load_sid_views(self) -> SIDViews:
        bundle = load_model_output(self.semantic_id_path)
        predictions = bundle.predictions.long()
        if predictions.ndim != 2:
            raise ValueError(f"Semantic ID predictions must be 2-D, got shape {tuple(predictions.shape)}.")
        if self.raw_num_hierarchies <= 0:
            raise ValueError("raw_num_hierarchies must be positive.")
        if predictions.size(1) < self.raw_num_hierarchies:
            raise ValueError(
                "Semantic ID prediction width must be >= raw_num_hierarchies: "
                f"width={predictions.size(1)}, raw_num_hierarchies={self.raw_num_hierarchies}."
            )

        dedup_digit = torch.zeros(predictions.size(0), dtype=torch.long)
        if predictions.size(1) > self.raw_num_hierarchies:
            dedup_digit = predictions[:, -1].long()

        return SIDViews(
            item_ids=bundle.keys.long(),
            raw_sid=predictions[:, :self.raw_num_hierarchies].long(),
            model_sid=predictions,
            dedup_digit=dedup_digit,
        )

    def _load_embeddings_for_items(self, item_ids: torch.Tensor) -> torch.Tensor | None:
        if self.embedding_path is None:
            return None
        bundle = load_model_output(self.embedding_path)
        embeddings = gather_predictions_by_keys(bundle, item_ids)
        if embeddings.ndim != 2:
            raise ValueError(f"Item embeddings must be 2-D, got shape {tuple(embeddings.shape)}.")
        return embeddings.float()

    def _iter_training_rows(self):
        training_dir = Path(self.data_folder) / "training"
        files = sorted(str(path) for path in training_dir.rglob("*.tfrecord.gz"))
        if not files:
            raise FileNotFoundError(f"No training TFRecord files found under {training_dir}.")
        yield from TFRecordReader(list_of_file_paths=files, shuffle_rows=False).iterrows()

    def _sequence_values(self, row: dict[str, Any]) -> list[int]:
        if "sequence_data" not in row:
            raise KeyError("Training row does not contain 'sequence_data'.")
        sequence = row["sequence_data"]
        if isinstance(sequence, torch.Tensor):
            return [int(value) for value in sequence.reshape(-1).tolist()]
        if hasattr(sequence, "reshape") and hasattr(sequence, "tolist"):
            return [int(value) for value in sequence.reshape(-1).tolist()]
        return [int(value) for value in sequence]

    def _compute_train_frequencies(self, item_ids: torch.Tensor) -> dict[int, int]:
        known_items = {int(item_id) for item_id in item_ids.tolist()}
        frequencies = {item_id: 0 for item_id in known_items}
        row_count = 0
        for row in self._iter_training_rows():
            row_count += 1
            for item_id in self._sequence_values(row):
                if item_id in frequencies:
                    frequencies[item_id] += 1
        if row_count == 0:
            raise ValueError("No training sequence records were read.")
        return frequencies
