"""LightningDataModule for item-level pipelines."""


from omegaconf import DictConfig

from src.data.datamodules.base import BaseDataModule


class ItemDataModule(BaseDataModule):
    """DataModule for item-level features without sequence-only collate binding."""

    def __init__(
        self,
        train_dataloader_config: DictConfig | None = None,
        val_dataloader_config: DictConfig | None = None,
        test_dataloader_config: DictConfig | None = None,
        predict_dataloader_config: DictConfig | None = None,
    ):
        super().__init__(
            train_dataloader_config=train_dataloader_config,
            val_dataloader_config=val_dataloader_config,
            test_dataloader_config=test_dataloader_config,
            predict_dataloader_config=predict_dataloader_config,
        )

    def _build_collate_fn(self, curr_config: DictConfig):
        return curr_config.collate_fn
