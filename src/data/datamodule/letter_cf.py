"""独立LETTER CF目录导出；训练装配复用公共FileDataModule。"""

import torch
from torch.utils.data import DataLoader, TensorDataset

from src.data.datamodule import FileDataModule


class LetterCFDataModule(FileDataModule):
    def __init__(self, catalog, **kwargs):
        super().__init__(**kwargs)
        self.catalog = catalog

    def predict_dataloader(self):
        if self.trainer.world_size != 1:
            raise ValueError("LETTER CF export requires one process.")
        return DataLoader(TensorDataset(torch.arange(len(self.catalog.keys))), batch_size=1024)
