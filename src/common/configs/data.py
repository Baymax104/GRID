from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from torch.utils.data import IterableDataset


@dataclass
class DatasetConfig:
    """The generic dataset configuration class."""

    data_reader: Callable[..., Any]
    preprocessing_functions: list[callable] = field(default_factory=list)  # type: ignore
    shuffle_files: bool = False


@dataclass
class SequenceDataloaderConfig:
    """The generic dataloader configuration class for sequence data."""

    dataset_class: IterableDataset
    data_folder: str
    dataset_config: DatasetConfig
    batch_size_per_device: int
    num_workers: int
    assign_files_by_size: bool
    collate_fn: callable  # type: ignore
    drop_last: bool = True
    pin_memory: bool = True
    persistent_workers: bool = False
    timeout: int = 0


@dataclass
class ItemDataloaderConfig:
    """The dataloader configuration class for item-level pipelines."""

    dataset_class: IterableDataset
    data_folder: str
    dataset_config: DatasetConfig
    batch_size_per_device: int
    num_workers: int
    assign_files_by_size: bool
    collate_fn: callable  # type: ignore
    feature_to_input_name: dict[str, str] = field(default_factory=dict)
    drop_last: bool = True
    pin_memory: bool = True
    persistent_workers: bool = False
    timeout: int = 0
    limit_files: int | None = None
