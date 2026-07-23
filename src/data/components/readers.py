from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Generator
from urllib.parse import unquote, urlparse

from tfrecord.iterator_utils import shuffle_iterator
from tfrecord.reader import example_loader

from src.utils.decorators import retry


class BaseDataReader(ABC):
    """The abstract class for raw data readers (e.g., parquet, tfrecord, etc.)."""

    def __init__(
        self,
        list_of_file_paths: list[str],
        shuffle_rows: bool = False,
    ):
        self.list_of_file_paths = list_of_file_paths
        self.shuffle_rows = shuffle_rows

    @classmethod
    @abstractmethod
    def get_file_suffix(cls) -> str:
        raise NotImplementedError("Must be implemented in child classes")

    @abstractmethod
    def iterrows(self) -> Generator:
        raise NotImplementedError("Must be implemented in child classes")

    @retry()
    def _get_next_example(self, dataset_iterator):
        try:
            return next(dataset_iterator)
        except StopIteration:
            return None


class TFRecordReader(BaseDataReader):
    """Data reader class for tfrecord files."""

    def __init__(
        self,
        list_of_file_paths: list[str],
        shuffle_rows: bool = False,
    ):
        super().__init__(list_of_file_paths, shuffle_rows)

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
        if self.shuffle_rows:
            dataset_iterator = shuffle_iterator(iter(dataset_iterator), queue_size=1024)

        dataset_iterator = iter(dataset_iterator)
        curr_example = self._get_next_example(dataset_iterator)
        while curr_example is not None:
            yield curr_example
            curr_example = self._get_next_example(dataset_iterator)

    @classmethod
    def get_file_suffix(cls) -> str:
        return "tfrecord.gz"
