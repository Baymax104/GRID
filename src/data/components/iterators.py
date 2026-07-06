import random
from abc import ABC, abstractmethod
from urllib.parse import unquote, urlparse

from pyarrow import parquet as pq
from tfrecord.reader import example_loader
from tfrecord.iterator_utils import shuffle_iterator

from src.utils.decorators import retry
from src.utils.file_utils import open_pyarrow_file


class BaseIterator(ABC):
    """the abstract class for raw data iterator (e.g., parquet, avro, etc.)

    Parameters
    ----------
    list_of_file_paths : list[str]
        the list of file paths to read from
    """

    def __init__(self, **kwargs):
        self.list_of_file_paths = []
        self.should_shuffle_rows = None

    def update_list_of_file_paths(self, list_of_file_paths: list[str]):
        self.list_of_file_paths = list_of_file_paths

    @abstractmethod
    def get_file_suffix(self) -> str:
        raise NotImplementedError("Must be implemented in child classes")

    @abstractmethod
    def iterrows(self):
        raise NotImplementedError("Must be implemented in child classes")

    @abstractmethod
    def shuffle(self, seed=42):
        raise NotImplementedError("Must be implemented in child classes")

    @retry()
    def _get_next_example(self, dataset_iterator):
        try:
            return next(dataset_iterator)
        except StopIteration:
            return None


class ParquetDataIterator(BaseIterator):
    """Data iterator class for parquet files

    Parameters
    ----------
    list_of_file_paths : list[str]
        the list of file paths to read from
    """

    def __init__(self, buffer_size=1000, features_to_consider=None, **kwargs):
        super().__init__(**kwargs)
        self.buffer_size = buffer_size
        self.features_to_consider = features_to_consider or []

    def iterrows(self):
        assert self.list_of_file_paths is not None, "list_of_file_paths is not set"

        for file_path in self.list_of_file_paths:
            with open_pyarrow_file(file_path) as f:
                parquet_file = pq.ParquetFile(f)

                for batch in parquet_file.iter_batches(columns=self.features_to_consider, batch_size=self.buffer_size):
                    for row in batch.to_pylist():
                        yield row

    def shuffle(self, seed=42) -> BaseIterator:
        random.seed(seed)
        random.shuffle(self.list_of_file_paths)
        return self

    def get_file_suffix(self) -> str:
        return "parquet"


class TFRecordIterator(BaseIterator):
    """
    Data iterator class for tfrecord files
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @staticmethod
    def _normalize_example_loader_path(file_path: str) -> str:
        """Convert local file URIs to plain filesystem paths for tfrecord loader."""
        parsed = urlparse(file_path)
        if parsed.scheme != "file":
            return file_path

        normalized_path = unquote(parsed.path)

        if parsed.netloc and parsed.netloc not in {"", "localhost"}:
            normalized_path = f"//{parsed.netloc}{normalized_path}"

        # Windows file URI: /C:/path -> C:/path
        if len(normalized_path) >= 3 and normalized_path[0] == "/" and normalized_path[2] == ":":
            normalized_path = normalized_path[1:]

        return normalized_path

    def iterrows(self):
        """
        Load real data and returns an iterator of dataset.
        The dataset in this iterator is obtained based on current worker thread.

        Returns:
            iterator of dataset, considered as list[sample] (aka. list[row])
            every element in this iterator is a sample.
        """
        assert self.list_of_file_paths is not None, "list_of_file_paths is not set"
        iterators = [
            example_loader(self._normalize_example_loader_path(file_path), None, compression_type="gzip")
            for file_path in self.list_of_file_paths
        ]

        def merged_examples():
            for iterator in iterators:
                yield from iterator

        dataset_iterator = merged_examples()
        if self.should_shuffle_rows:
            dataset_iterator = shuffle_iterator(iter(dataset_iterator), queue_size=128)

        dataset_iterator = iter(dataset_iterator)
        curr_example = self._get_next_example(dataset_iterator)
        while curr_example is not None:
            yield curr_example
            curr_example = self._get_next_example(dataset_iterator)

    def shuffle(self, seed=42) -> BaseIterator:
        # TODO(lneves): Unify the shuffle method for all iterators
        # Currently this one shuffles only files, parquet shuffles rows.
        random.seed(seed)
        random.shuffle(self.list_of_file_paths)
        return self

    def get_file_suffix(self) -> str:
        return "tfrecord.gz"
