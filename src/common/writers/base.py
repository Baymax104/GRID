"""Base callbacks for prediction writers."""

from typing import Any

from lightning import LightningModule, Trainer
from lightning.pytorch.callbacks import Callback

from src.data.components.data_models import ModelOutput
from src.utils.pylogger import RankedLogger

logger = RankedLogger(__name__, rank_zero_only=True)


class BaseBufferedWriter(Callback):
    def __init__(
        self,
        flush_frequency: int = 5000,
    ):
        """
        Args:
            flush_frequency: Number of samples to accumulate before flushing.
        """
        super().__init__()
        self.flush_frequency = flush_frequency
        self.buffer: list[ModelOutput] = []
        self.global_rank = None

    def setup(self, trainer: Trainer, pl_module: LightningModule, stage: str):
        self.global_rank = trainer.global_rank if trainer.global_rank else 0
        logger.info(f"Rank {self.global_rank} initialized for inference.")

    def flush_buffer(self):
        """Flush the buffer and then clear it."""
        if self.buffer:
            self._flush_buffer()
            self.buffer.clear()
        else:
            logger.info("Buffer is empty, nothing to flush.")

    def _flush_buffer(self):
        """Override this method to implement the logic for flushing the buffer."""
        raise NotImplementedError("You need to implement the `_flush_buffer` method in your subclass.")

    def _buffer_sample_count(self) -> int:
        """Return the total number of samples currently in the buffer."""
        return sum(len(m.keys) for m in self.buffer)

    def handle_batch(self, model_output: ModelOutput):
        """
        Handles a batch of predictions by appending it to the buffer and
        flushing the buffer if the sample count exceeds flush_frequency.
        """
        if model_output is None:
            logger.warning(
                f"Rank {self.global_rank} received an empty model output. Skipping this batch. This is expected if the batch is a dummy batch."
            )
            return
        self.buffer.append(model_output)
        if self._buffer_sample_count() >= self.flush_frequency:
            self.flush_buffer()

    def on_predict_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: ModelOutput,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ):
        """Called at the end of each prediction batch."""
        self.handle_batch(outputs)

    def on_predict_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
    ):
        """Called at the end of the prediction process."""
        self.flush_buffer()
        logger.info(f"Rank {self.global_rank} finished writing predictions.")
