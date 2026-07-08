from dataclasses import dataclass, field
from typing import Callable

import torch
import transformers
from torch.utils.data import IterableDataset

from src.data.components.readers import BaseDataReader


@dataclass
class SequenceDatasetConfig:
    """The generic dataset configuration class for datasets of sequence data.

    Parameters:
    ----------
    user_id_field: str
        The user id field name.
    data_reader: Callable[..., BaseDataReader]
        The raw data reader.
    preprocessing_functions: list[callable]
        The list of preprocessing functions. Should be in the order they must be applied.
    shuffle_files: bool
        Whether to shuffle the order of files assigned to the current worker.
    num_placeholder_tokens_map: dict | None
        The number of placeholder tokens map.
    keep_user_id: bool
        Whether to keep the user id feature in the batches.
    field_type_map: dict | None
        The field type map.
    min_sequence_length: int
        The minimum sequence length. Only works if iterating per row.
    feature_map: dict | None
        maps the feature names to the desired feature names.
    features_to_consider: list[str]
        List of features to consider. If not specified, consider all features.
    file_format: str
        The file format of the dataset files. If not specified, the data reader's `get_file_suffix`
        method will be used to determine the file format.
        For example, if the data reader reads tfrecord files at the first level of the data_folder,
        this can be set to "tfrecord.gz". If we want to retrieve all tfrecord files in subdirectories as well,
        we can set it to "*/*tfrecord.gz".
    """

    user_id_field: str
    data_reader: Callable[..., BaseDataReader]
    preprocessing_functions: list[callable] = field(default_factory=list)  # type: ignore
    shuffle_files: bool = False
    keep_user_id: bool = False
    num_placeholder_tokens_map: dict | None = field(default_factory=dict)
    field_type_map: dict | None = field(default_factory=dict)
    min_sequence_length: int = 10
    feature_map: dict | None = None
    features_to_consider: list[str] = field(default_factory=list)
    file_format: str = None


@dataclass
class SequenceDataloaderConfig:
    """The generic dataloader configuration class for datasets of sequence data.

    Each instance of this class is run on one device.

    Parameters:
    ----------
    dataset_class: IterableDataset
        The dataset class.
    data_folder: str
        Path to the folder containingthe dataset files.
    dataset_config: SequenceDatasetConfig
        The dataset configuration.
    labels: dict[str, callable]
        A dictionary mapping from feature names to
    batch_size_per_device: list[callable]
        The batch size per dataloader, also per device (GPU).
    num_workers: int
        The number of workers per dataloader, also per device (GPU).
    assign_files_by_size: dict | None
        Whether to assign files to workers by file size to balance computation
        across workers.
    oov_token: int | None
        The token used to represent OOV items.
    masking_token: int
        The token used to represent masked items.
    collate_fn: callable
        Collate function used to construct batches.
    sequence_length: int = 200
        The length of sequences the dataloader should return. If raw sequences
        are shorter, the dataloader will pad them to reach sequence_length.
    padding_token: int = 0
        The token used for padding sequences.
    drop_last: bool = True
        Whether to drop the last batch if it is smaller than
        batch_size_per_device.
    pin_memory: bool = True
        Whether to allocate memory on CPU to ensure data is always available for
        fast transfer to GPU.
    should_shuffle_rows: bool = False
        Whether to shuffle rows between epochs.
    persistent_workers: bool = False
        Whether to maintain worker processes across epochs.
    """

    dataset_class: IterableDataset
    data_folder: str
    dataset_config: SequenceDatasetConfig
    batch_size_per_device: int
    num_workers: int
    assign_files_by_size: bool
    masking_token: int
    collate_fn: callable  # type: ignore
    labels: dict[str, callable] = field(default_factory=dict)  # type: ignore
    oov_token: int | None = -1
    sequence_length: int = 200
    padding_token: int = 0
    drop_last: bool = True
    pin_memory: bool = True
    should_shuffle_rows: bool = False
    persistent_workers: bool = False
    timeout: int = 0


@dataclass
class SemanticIDDatasetConfig(SequenceDatasetConfig):
    """The dataset configuration class used to store the dataset configuration for pipelines
    that use semantic ids.

    Note that this class inherits from SequenceDatasetConfig, thus inherits all of
    its parameters.

    Parameters:
    -----------
    semantic_id_map: dict[str, torch.Tensor] | None
        The semantic id map from field name to a 2-D tensor.
    keep_user_id: bool
        Whether to keep the user id in the dataset. If set to True, the user id
        will be included in the dataset and can be used for inference or evaluation.
    """

    semantic_id_map: dict[str, torch.Tensor] | None = None
    keep_user_id: bool = False


@dataclass
class TokenizerConfig:
    """The configuration class used to store the tokenizer configuration.

    Parameters:
    ----------
    tokenizer: transformers.PreTrainedTokenizer
        The tokenizer.
    max_length: int
        The maximum length of the tokenized sequences.
    padding: str
        The padding strategy.
    truncation: bool
        Whether to truncate the sequences.
    special_tokens: dict[str, str] | None
        The special tokens.
    add_special_tokens: bool
        Whether to add special tokens.
    postprocess_eos_token: bool | None
        Whether to postprocess the eos token.
    """

    tokenizer: transformers.PreTrainedTokenizer
    max_length: int
    padding: str
    truncation: bool
    special_tokens: dict[str, str] | None = field(default_factory=dict)
    add_special_tokens: bool = True
    postprocess_eos_token: bool | None = False


@dataclass
class ItemDatasetConfig:
    """The configuration class used to store the item dataset configuration.

    Parameters
    ----------
    item_id_field: str
        The item id field.
    data_reader: Callable[..., BaseDataReader]
        The data reader factory.
    preprocessing_functions: list[callable]
        The preprocessing functions to be applied to each row.
    shuffle_files: bool
        Whether to shuffle the order of files assigned to the current worker.
    keep_item_id: bool
        Whether to keep the item id in the data.
    """

    item_id_field: str
    data_reader: Callable[..., BaseDataReader]
    preprocessing_functions: list[callable] = field(default_factory=list)
    keep_item_id: bool = True
    shuffle_files: bool = False


@dataclass
class ItemDataloaderConfig:
    """The dataloader configuration class for item-level pipelines."""

    dataset_class: IterableDataset
    data_folder: str
    dataset_config: ItemDatasetConfig
    batch_size_per_device: int
    num_workers: int
    assign_files_by_size: bool

    collate_fn: callable
    feature_to_input_name: dict[str, str] = field(default_factory=dict)
    drop_last: bool = True
    pin_memory: bool = True
    persistent_workers: bool = False
    timeout: int = 0
    limit_files: int | None = None
