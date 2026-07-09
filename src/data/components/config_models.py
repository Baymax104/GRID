from collections.abc import Callable
from dataclasses import dataclass, field

from torch.utils.data import IterableDataset

from src.data.components.readers import BaseDataReader


@dataclass
class SemanticIDDatasetConfig:
    """The dataset configuration class for sequence pipelines that use semantic ids.

    Parameters
    ----------
    data_reader: Callable[..., BaseDataReader]
        The data reader factory.
    preprocessing_functions: list[callable]
        The preprocessing functions to be applied to each row.
    shuffle_files: bool
        Whether to shuffle the order of files assigned to the current worker.
    """

    data_reader: Callable[..., BaseDataReader]
    preprocessing_functions: list[callable] = field(default_factory=list)  # type: ignore
    shuffle_files: bool = False


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
    dataset_config: SemanticIDDatasetConfig
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
    persistent_workers: bool = False
        Whether to maintain worker processes across epochs.
    """

    dataset_class: IterableDataset
    data_folder: str
    dataset_config: SemanticIDDatasetConfig
    batch_size_per_device: int
    num_workers: int
    assign_files_by_size: bool
    masking_token: int
    collate_fn: callable  # type: ignore
    labels: dict[str, callable] = field(default_factory=dict)  # type: ignore
    sequence_length: int = 200
    padding_token: int = 0
    drop_last: bool = True
    pin_memory: bool = True
    persistent_workers: bool = False
    timeout: int = 0


@dataclass
class ItemDatasetConfig:
    """The configuration class used to store the item dataset configuration.

    Parameters
    ----------
    data_reader: Callable[..., BaseDataReader]
        The data reader factory.
    preprocessing_functions: list[callable]
        The preprocessing functions to be applied to each row.
    shuffle_files: bool
        Whether to shuffle the order of files assigned to the current worker.
    """

    data_reader: Callable[..., BaseDataReader]
    preprocessing_functions: list[callable] = field(default_factory=list)
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
