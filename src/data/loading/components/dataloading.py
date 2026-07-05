from typing import List

from torch.utils.data import IterableDataset, get_worker_info

from src.data.loading.components.interfaces import BaseDatasetConfig
from src.utils.pylogger import RankedLogger


command_line_logger = RankedLogger(__name__, rank_zero_only=True)


class BaseDataset:
    def __init__(
        self,
        dataset_config: BaseDatasetConfig,
        data_folder: str,
        should_shuffle_rows: bool = False,
        is_for_training: bool = True,
    ):
        """
        Base class for all datasets. This class is used to set up the dataset and provide the list of files to be used.
        Args:
            dataset_config (BaseDatasetConfig): Configuration for the dataset.
            data_folder (str): Path to the folder where the data is stored.
            should_shuffle_rows (bool): Whether to shuffle the rows of the dataset.
            is_for_training (bool): Whether the dataset is for training or not.
        """
        self.global_worker_id = None
        self.total_workers = None
        self.global_dataloader_worker_id = None
        self.dataset_config = dataset_config
        self.should_shuffle_rows = should_shuffle_rows
        self.data_folder = data_folder
        self.list_of_file_paths = []
        self.is_for_training = is_for_training

    def set_list_of_files(self, list_of_files: List[str]):
        self.list_of_file_paths = list_of_files

    def set_distributed_params(self, total_workers: int, global_worker_id: int):
        self.total_workers = total_workers  # world_size, worker number in process level
        self.global_worker_id = global_worker_id  # global_rank, worker id in process level

    def get_worker_id_and_num_workers(self):
        worker_info = get_worker_info()

        if worker_info is None:
            # Single-worker setup (no multiprocessing)
            worker_id = 0
            num_workers = 1
        else:
            # Multi-worker setup
            worker_id = worker_info.id
            num_workers = worker_info.num_workers

        # set global dataloader worker id in process level
        # eg. world_size = 2, num_workers = 2
        # Process 0 (global_rank = 0)
        # - Thread 0 (worker_id = 0, global_id = 0 * 2 + 0 = 0)
        # - Thread 1 (worker_id = 1, global_id = 0 * 2 + 1 = 1)
        # Process 1 (global_rank = 1)
        # - Thread 0 (worker_id = 0, global_id = 1 * 2 + 0 = 2)
        # - Thread 1 (worker_id = 1, global_id = 1 * 2 + 1 = 3)
        self.global_dataloader_worker_id = self.global_worker_id * num_workers + worker_id

        return worker_id, num_workers

    def get_list_of_worker_files(self):
        # Get information about worker and then separate only files that belong to this worker
        worker_id, num_workers = self.get_worker_id_and_num_workers()
        worker_files = self.list_of_file_paths[worker_id::num_workers]
        command_line_logger.debug(f"GPU Worker: {self.global_worker_id}/{self.total_workers} CPU Worker {worker_id} has {len(worker_files)} files")
        return worker_files

    def setup(self):
        pass


class UnboundedSequenceIterable(BaseDataset, IterableDataset):
    """
    An unbounded dataset is a dataset that we don't know the size of beforehand.
    For training, we will iterate over the dataset infinitely.
    For evaluation, we will iterate over the dataset once.
    """

    def __init__(
        self,
        dataset_config: BaseDatasetConfig,
        data_folder: str,
        should_shuffle_rows: bool = False,
        is_for_training: bool = True,
    ):
        super().__init__(
            dataset_config=dataset_config,
            data_folder=data_folder,
            should_shuffle_rows=should_shuffle_rows,
            is_for_training=is_for_training,
        )
        self.data_iterator = dataset_config.data_iterator
        self.dataset_to_iterate = None

    def setup(self):
        """
        Set up the dataset iterator, it is called lazily when this dataset is iterated.

        - get data files for current worker thread
        - read data files and set real dataset iterator
        """
        # set files that are processed on current worker thread
        current_worker_files = self.get_list_of_worker_files()
        self.data_iterator.update_list_of_file_paths(current_worker_files)

        # here we use global_dataloader_worker_id as the seed for shuffling
        # this keeps file and row shuffling stable across worker processes.
        if self.should_shuffle_rows:
            self.data_iterator = self.data_iterator.shuffle(seed=self.global_dataloader_worker_id)

        self.data_iterator.should_shuffle_rows = self.should_shuffle_rows

        # get real data and dataset real iterator using the only supported row-based mode.
        self.dataset_to_iterate = self.data_iterator.iterrows()

        command_line_logger.debug(
            f"GLOBAL ID {self.global_dataloader_worker_id} GPU Worker: {self.global_worker_id}/{self.total_workers} with {len(self.data_iterator.list_of_file_paths)} files\
                First five files are: {self.data_iterator.list_of_file_paths[:5]}"
        )

    def __iter__(self):
        """
        For IterableDataset, iteration is activated by this function.

        Returns:
            iterator from data_iterator
        """
        if self.dataset_to_iterate is None:
            # If it has not been set up, it means it is a forkserver worker. We need to set it up.
            self.setup()
        # If the dataset is for training, we want to keep iterating over the dataset infinitely.
        # On a streaming dataset, we will always be on Epoch 0.
        finished_iteration = False
        while not finished_iteration:
            for row_or_batch in self.dataset_to_iterate:
                for preprocessing_function in self.dataset_config.preprocessing_functions:
                    row_or_batch = preprocessing_function(row_or_batch, dataset_config=self.dataset_config)
                    if row_or_batch is None:
                        break
                if row_or_batch:
                    yield row_or_batch
            # if the dataset is not for training, we stop the loop. Otherwise, we continue.
            finished_iteration = not self.is_for_training
            if not finished_iteration:
                self.setup()
        # We reset the dataset to iterate to None, so that it is set up again in the next iteration.
        # This is required for validation when persistent_workers = True.
        self.dataset_to_iterate = None
        return None
