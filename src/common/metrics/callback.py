from typing import Any

from lightning import LightningModule, Trainer
from lightning.pytorch.callbacks import Callback
from torch import Tensor

from src.common.metrics.engine import MetricEngine
from src.common.metrics.extractors import MappingMetricExtractor
from src.utils.pylogger import RankedLogger

logger = RankedLogger(__name__, rank_zero_only=True)

HISTORY_LOGGING_MODE = "history"
SUMMARY_LOGGING_MODE = "summary"
SUPPORTED_LOGGING_MODES = {HISTORY_LOGGING_MODE, SUMMARY_LOGGING_MODE}


class MetricCallback(Callback):
    """Lightning callback that owns metric update, logging, and reset hooks."""

    def __init__(
        self,
        engine: MetricEngine,
        extractor: MappingMetricExtractor | None = None,
        train_log_kwargs: dict[str, Any] | None = None,
        validation_log_kwargs: dict[str, Any] | None = None,
        test_log_kwargs: dict[str, Any] | None = None,
        logging_modes: dict[str, str] | None = None,
    ):
        super().__init__()
        self.engine = engine
        self.extractor = extractor or MappingMetricExtractor()
        self.logging_modes = _validate_logging_modes(logging_modes or {})
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
        self._log_stage(trainer, pl_module, "train", only_updated=True, log_kwargs=self.train_log_kwargs)
        self.engine.reset("train")

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
        self._log_stage(trainer, pl_module, "val", log_kwargs=self.validation_log_kwargs)
        self.engine.reset("val")

    def on_test_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        self._log_stage(trainer, pl_module, "test", log_kwargs=self.test_log_kwargs)
        self.engine.reset("test")

    def _update(self, stage: str, outputs: Any, batch: Any, pl_module: LightningModule) -> None:
        if not self.engine.has_stage(stage):
            return
        payload = self.extractor.extract(stage=stage, outputs=outputs, batch=batch, pl_module=pl_module)
        self.engine.update(stage, payload)

    def _log_stage(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        stage: str,
        only_updated: bool = False,
        log_kwargs: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        mode = self.logging_modes.get(stage, HISTORY_LOGGING_MODE)
        if mode == HISTORY_LOGGING_MODE:
            return self.engine.log(pl_module, stage, only_updated=only_updated, **(log_kwargs or {}))

        metrics = self.engine.compute_prefixed(stage, only_updated=only_updated)
        if metrics:
            _write_summary_metrics(trainer, metrics)
        return metrics


def _validate_logging_modes(logging_modes: dict[str, str]) -> dict[str, str]:
    invalid_modes = {stage: mode for stage, mode in logging_modes.items() if mode not in SUPPORTED_LOGGING_MODES}
    if invalid_modes:
        formatted = ", ".join(f"{stage}={mode!r}" for stage, mode in invalid_modes.items())
        supported = ", ".join(sorted(SUPPORTED_LOGGING_MODES))
        raise ValueError(f"Unsupported metric logging mode(s): {formatted}. Supported modes: {supported}.")
    return dict(logging_modes)


def _write_summary_metrics(trainer: Trainer, metrics: dict[str, Any]) -> None:
    summary_metrics = _to_summary_scalars(metrics)
    if not summary_metrics:
        return

    wrote_summary = False
    for metric_logger in getattr(trainer, "loggers", []):
        experiment = getattr(metric_logger, "experiment", None)
        summary = getattr(experiment, "summary", None)
        if summary is None:
            logger.warning(
                f"Metric summary logging skipped for unsupported logger {type(metric_logger).__name__}: "
                "missing experiment.summary."
            )
            continue

        for name, value in summary_metrics.items():
            summary[name] = value
        wrote_summary = True

    if not wrote_summary:
        logger.warning("Metric summary logging requested, but no configured logger supports run summaries.")


def _to_summary_scalars(metrics: dict[str, Any]) -> dict[str, int | float | bool]:
    summary_metrics: dict[str, int | float | bool] = {}
    for name, value in metrics.items():
        scalar = _to_scalar(value)
        if scalar is None:
            logger.warning(f"Skipping non-scalar metric for summary logging: {name}.")
            continue
        summary_metrics[name] = scalar
    return summary_metrics


def _to_scalar(value: Any) -> int | float | bool | None:
    if isinstance(value, Tensor):
        detached = value.detach()
        if detached.numel() != 1:
            return None
        return detached.item()
    if isinstance(value, bool | int | float):
        return value
    return None
