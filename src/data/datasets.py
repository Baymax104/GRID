import random

from omegaconf import DictConfig
from torch.utils.data import IterableDataset, get_worker_info

from src.utils.pylogger import RankedLogger

logger = RankedLogger(__name__, rank_zero_only=True)


class BaseDataset:
    def __init__(
        self,
        list_of_file_paths: list[str],
        global_rank: int,
    ):
        self.global_rank = global_rank
        self.list_of_file_paths = list_of_file_paths

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

    def get_list_of_worker_files(self, shuffle: bool = False):
        worker_id, num_workers, global_dataloader_worker_id = self._get_worker_context()
        worker_files = self.list_of_file_paths[worker_id::num_workers]
        if shuffle:
            random.seed(global_dataloader_worker_id)
            random.shuffle(worker_files)
        return worker_files


class SequenceDataset(BaseDataset, IterableDataset):
    """
    An unbounded dataset is a dataset that we don't know the size of beforehand.
    For training, we will iterate over the dataset infinitely.
    For evaluation, we will iterate over the dataset once.
    """

    def __init__(
        self,
        dataset_config: DictConfig,
        data_folder: str,
        list_of_file_paths: list[str],
        global_rank: int,
        is_for_training: bool = True,
    ):
        super().__init__(list_of_file_paths=list_of_file_paths, global_rank=global_rank)
        self.data_folder = data_folder
        self.data_reader_factory = dataset_config.data_reader
        self.preprocessing_functions = getattr(dataset_config, "preprocessing_functions", [])
        self.shuffle_files = getattr(dataset_config, "shuffle_files", False)
        self.is_for_training = is_for_training

    def _load_data(self):
        current_worker_files = self.get_list_of_worker_files(shuffle=self.shuffle_files)
        data_reader = self.data_reader_factory(list_of_file_paths=current_worker_files)
        return data_reader.iterrows()

    def __iter__(self):
        dataset_iterable = self._load_data()
        # If the dataset is for training, we want to keep iterating over the dataset infinitely.
        # On a streaming dataset, we will always be on Epoch 0.
        finished_iteration = False
        while not finished_iteration:
            for row_or_batch in dataset_iterable:
                # call preprocessing functions
                for preprocessing_function in self.preprocessing_functions:
                    row_or_batch = preprocessing_function(row_or_batch)
                    if row_or_batch is None:
                        break
                if row_or_batch:
                    yield row_or_batch
            # if the dataset is not for training, we stop the loop. Otherwise, we continue.
            finished_iteration = not self.is_for_training
            if not finished_iteration:
                dataset_iterable = self._load_data()
        return None
