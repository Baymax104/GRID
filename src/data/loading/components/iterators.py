import random
from abc import ABC, abstractmethod
from typing import Callable

import tensorflow as tf
from pyarrow import parquet as pq

from src.utils.decorators import retry
from src.utils.file_utils import open_pyarrow_file


tf.config.set_visible_devices([], "GPU")  # Disable all for tensorflow


# if GPU version of TF installed,
# it will automatically occupy the full GPU memory


class RawDataIterator(ABC):
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


class ParquetDataIterator(RawDataIterator):
    """Data iterator class for parquet files

    Parameters
    ----------
    list_of_file_paths : list[str]
        the list of file paths to read from
    """

    def __init__(self, buffer_size=1000, features_to_consider=None, **kwargs):
        super().__init__(**kwargs)
        self.buffer_size = buffer_size
        self.features_to_consider = features_to_consider

    def iterrows(self):
        assert self.list_of_file_paths is not None, "list_of_file_paths is not set"

        for file_path in self.list_of_file_paths:
            with open_pyarrow_file(file_path) as f:
                parquet_file = pq.ParquetFile(f)

                for batch in parquet_file.iter_batches(
                        columns=self.features_to_consider if self.features_to_consider else None,
                        batch_size=self.buffer_size,
                ):
                    for row in batch.to_pylist():
                        yield row

    def shuffle(self, seed=42) -> RawDataIterator:
        random.seed(seed)
        random.shuffle(self.list_of_file_paths)
        return self

    def get_file_suffix(self) -> str:
        return "parquet"


class TFRecordIterator(RawDataIterator):
    """
    Data iterator class for tfrecord files

    Args:
        use_ragged_tensor: bool, Whether to use ragged tensors.
    """

    def __init__(
        self,
        use_ragged_tensor: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.feature_description = None
        self.use_ragged_tensor = use_ragged_tensor

    def initialize_feature_description(self, raw_dataset: tf.data.TFRecordDataset):
        """
        If the feature description is not set, infer the feature description from the first record in the dataset.
        """
        if self.feature_description is None:
            sample_record: tf.Tensor = next(iter(raw_dataset))
            self.feature_description = self.infer_feature_type(self.parse_tfrecord(sample_record).features.feature)

    def iterrows(self):
        """
        Load real data and returns an iterator of dataset.
        The dataset in this iterator is obtained based on current worker thread.

        Returns:
            iterator of dataset, considered as list[sample] (aka. list[row])
            every element in this iterator is a sample.
        """
        assert self.list_of_file_paths is not None, "list_of_file_paths is not set"
        raw_dataset = tf.data.TFRecordDataset(
            self.list_of_file_paths,
            compression_type="GZIP",
            buffer_size=256 * 1024,
        )
        if self.should_shuffle_rows:
            # the buffer here is the number of records to shuffle
            # the larger the buffer, the more memory it will use
            # too large might cause OOM
            raw_dataset = raw_dataset.shuffle(buffer_size=128)

        self.initialize_feature_description(raw_dataset)
        # We create an iterator and manually iterate to allow for retrying the
        # "next" operation in case of a failure.
        dataset_iterator = iter(raw_dataset)
        curr_example = self._get_next_example(dataset_iterator)
        while curr_example:
            example = tf.io.parse_single_example(curr_example, self.feature_description)
            yield example
            curr_example = self._get_next_example(dataset_iterator)

    # dynamic inferring the feature description of tfrecord files
    def infer_feature_type(self, example_proto: tf.Tensor) -> dict:
        feature_description = {}
        # feature has variable length, RaggedFeature in TF 2.x is recommended
        # VarLenFeature is used for compatibility
        tf_feature_type = tf.io.RaggedFeature if self.use_ragged_tensor else tf.io.VarLenFeature
        for key, value in example_proto.items():
            if isinstance(value, tf.train.Feature):
                if value.HasField("bytes_list"):
                    feature_description[key] = tf_feature_type(tf.string)
                elif value.HasField("float_list"):
                    feature_description[key] = tf_feature_type(tf.float32)
                elif value.HasField("int64_list"):
                    feature_description[key] = tf_feature_type(tf.int64)
                else:
                    raise ValueError("Unknown feature type")
        return feature_description

    # parsing the tfrecord files from bytes
    def parse_tfrecord(self, record: tf.Tensor) -> tf.Tensor:
        example = tf.train.Example()
        example.ParseFromString(record.numpy())
        return example

    def shuffle(self, seed=42) -> RawDataIterator:
        # TODO(lneves): Unify the shuffle method for all iterators
        # Currently this one shuffles only files, parquet shuffles rows.
        random.seed(seed)
        random.shuffle(self.list_of_file_paths)
        return self

    def get_file_suffix(self) -> str:
        return "tfrecord.gz"
