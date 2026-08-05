from typing import Any

from lightning import LightningModule, Trainer
from lightning.pytorch.callbacks import Callback

from src.common.metrics.engine import MetricEngine
from src.common.metrics.extractors import MappingMetricExtractor


class MetricCallback(Callback):
    """Lightning callback that owns metric update, logging, and reset hooks."""

    def __init__(
        self,
        engine: MetricEngine,
        extractor: MappingMetricExtractor | None = None,
        train_log_kwargs: dict[str, Any] | None = None,
        validation_log_kwargs: dict[str, Any] | None = None,
        test_log_kwargs: dict[str, Any] | None = None,
    ):
        super().__init__()
        self.engine = engine
        self.extractor = extractor or MappingMetricExtractor()
        self.train_log_kwargs = train_log_kwargs or {
            "on_step": True,
            "on_epoch": False,
            "prog_bar": False,
            "logger": True,
            "sync_dist": True,
        }
        self.validation_log_kwargs = validation_log_kwargs or {
            "on_step": False,
            "on_epoch": True,
            "prog_bar": False,
            "logger": True,
            "sync_dist": True,
        }
        self.test_log_kwargs = test_log_kwargs or {
            "on_step": False,
            "on_epoch": True,
            "prog_bar": False,
            "logger": True,
            "sync_dist": True,
        }

    def setup(self, trainer: Trainer, pl_module: LightningModule, stage: str) -> None:
        self.engine.to(pl_module.device)

    def on_train_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        self.engine.reset("train")

    def on_validation_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        self.engine.reset("val")

    def on_test_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        self.engine.reset("test")

    def on_train_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: Any,
        batch: Any,
        batch_idx: int,
    ) -> None:
        self._update("train", outputs, batch, pl_module)
        self.engine.log(pl_module, "train", only_updated=True, **self.train_log_kwargs)

    def on_validation_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: Any,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        self._update("val", outputs, batch, pl_module)

    def on_test_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: Any,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        self._update("test", outputs, batch, pl_module)

    def on_validation_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        self.engine.log(pl_module, "val", **self.validation_log_kwargs)
        self.engine.reset("val")

    def on_test_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        self.engine.log(pl_module, "test", **self.test_log_kwargs)
        self.engine.reset("test")

    def _update(self, stage: str, outputs: Any, batch: Any, pl_module: LightningModule) -> None:
        if not self.engine.has_stage(stage):
            return
        payload = self.extractor.extract(stage=stage, outputs=outputs, batch=batch, pl_module=pl_module)
        self.engine.update(stage, payload)
