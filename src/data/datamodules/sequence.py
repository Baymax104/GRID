"""LightningDataModule for sequential data pipelines."""

from functools import partial

from omegaconf import DictConfig

from src.data.datamodules.base import BaseDataModule


class SequenceDataModule(BaseDataModule):
    """DataModule for sequence tasks with sequence-aware collate binding."""

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
        return partial(
            curr_config.collate_fn,
            labels=curr_config.labels,
            sequence_length=curr_config.sequence_length,
            masking_token=curr_config.masking_token,
            padding_token=curr_config.padding_token,
            oov_token=getattr(curr_config, "oov_token", None),
        )
