"""LightningDataModule for item-level pipelines."""

from typing import Optional

from omegaconf import DictConfig

from src.data.loading.datamodules.base import BaseDataModule


class ItemDataModule(BaseDataModule):
    """DataModule for item-level features without sequence-only collate binding."""

    def __init__(
        self,
        train_dataloader_config: Optional[DictConfig] = None,
        val_dataloader_config: Optional[DictConfig] = None,
        test_dataloader_config: Optional[DictConfig] = None,
        predict_dataloader_config: Optional[DictConfig] = None,
    ):
        super().__init__(
            train_dataloader_config=train_dataloader_config,
            val_dataloader_config=val_dataloader_config,
            test_dataloader_config=test_dataloader_config,
            predict_dataloader_config=predict_dataloader_config,
        )

    def _build_collate_fn(self, curr_config: DictConfig):
        return curr_config.collate_fn
