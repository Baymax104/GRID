"""Split-safe data module for labeled TIGER prefix tracing."""

import hydra
from omegaconf import DictConfig

from src.data.datamodule.file import FileDataModule


class TigerTraceDataModule(FileDataModule):
    """Reject an implicit or unsupported trace split before dataset setup."""

    def __init__(self, data_split: str, predict_dataloader_config: DictConfig):
        if data_split not in {"evaluation", "testing"}:
            raise ValueError(
                "TIGER prefix trace data_split must be explicitly set to "
                f"'evaluation' or 'testing', got {data_split!r}."
            )
        self.data_split = data_split
        predict_dataloader_config = hydra.utils.instantiate(predict_dataloader_config)
        super().__init__(predict_dataloader_config=predict_dataloader_config)
