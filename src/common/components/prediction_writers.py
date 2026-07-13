import datetime
import os
import pickle
from typing import Any, Literal

import torch
from lightning import LightningModule, Trainer
from lightning.pytorch.callbacks import BasePredictionWriter

from src.common.components.model_output import ModelOutput
from src.utils.decorators import retry
from src.utils.file_utils import sync_file
from src.utils.pylogger import RankedLogger

logger = RankedLogger(__name__, rank_zero_only=True)


class BaseBufferedWriter(BasePredictionWriter):
    def __init__(
        self,
        flush_frequency: int = 5000,
        write_interval: Literal["batch", "epoch", "batch_and_epoch"] = "batch",
    ):
        """
        Args:
            flush_frequency: Number of samples to accumulate before flushing.
            write_interval: "batch" or "epoch".
        """
        super().__init__(write_interval)
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

    def write_on_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        prediction: ModelOutput,
        batch_indices: list[int],
        batch: Any,
        batch_idx: int,
        dataloader_idx: int,
    ):
        """Called at the end of each prediction batch."""
        self.handle_batch(prediction)

    def write_on_epoch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        predictions: list[ModelOutput],
        batch_indices: list[list[int]],
    ):
        """Called at the end of a prediction epoch."""
        for batch_pred in predictions:
            self.handle_batch(batch_pred)
        self.flush_buffer()

    def on_predict_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
    ):
        """Called at the end of the prediction process."""
        self.flush_buffer()
        logger.info(f"Rank {self.global_rank} finished writing predictions.")


class LocalPickleWriter(BaseBufferedWriter):
    """
    Callback to write predictions to local pickle files during inference.
    """

    def __init__(
        self,
        output_dir: str,
        flush_frequency: int = 1000,
        write_interval: Literal["batch", "epoch", "batch_and_epoch"] = "batch",
        post_processing_functions: list[callable] | None = None,
    ):
        """
        Args:
            output_dir: Directory to save the pickle files.
            flush_frequency: Number of samples to accumulate before writing to a pickle file.
            write_interval: "batch" or "epoch".
            post_processing_functions: list of ordered post-processing functions to apply to the files.
        """
        super().__init__(write_interval=write_interval, flush_frequency=flush_frequency)
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        self.post_processing_functions = post_processing_functions if post_processing_functions else []

    def _create_file_path(self) -> str:
        """Create a file path for the pickle file."""
        return (
            f"predictions_{self.global_rank}_{datetime.datetime.now(datetime.UTC).strftime('%Y%m%dT%H%M%S%f')[:-3]}.pkl"
        )

    def _local_file_path(self, file_path: str | None = None) -> str:
        """Create a local file path for the pickle file."""
        return f"{self.output_dir}/{file_path if file_path else self._create_file_path()}"

    def _distributed_barrier(self):
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.barrier()
        else:
            logger.info("Distributed not available, skipping distributed barrier.")

    @retry()
    def _flush_buffer(self):
        """Flush the buffer to a local temporary pickle file."""
        file_path = self._create_file_path()
        with open(self._local_file_path(file_path=file_path), "wb") as f:
            pickle.dump(self.buffer, f)

        logger.info(
            f"Global Rank: {self.global_rank} wrote {self._buffer_sample_count()} samples to {self._local_file_path(file_path=file_path)}."
        )

    @retry()
    def on_predict_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
    ):
        assert trainer.global_rank is not None, "Global rank was not provided."

        super().on_predict_end(trainer, pl_module)

        self._distributed_barrier()
        if self.global_rank != 0:
            logger.info(f"Rank {self.global_rank} exits on predict end.")
            return

        logger.info("Merging pickle files on main process.")
        self._merge_files()

        # conducting post-processing functions on the main process
        for process_func in self.post_processing_functions:
            all_files = [f for f in os.listdir(self.output_dir)]
            for file in all_files:
                file_path = os.path.join(self.output_dir, file)
                process_func(file_path)

    def _merge_files(self):
        """Merge all pickle files into a single model output bundle."""
        sync_file(self.output_dir)
        all_files = [f for f in os.listdir(self.output_dir) if f.endswith(".pkl")]
        all_outputs: list[ModelOutput] = []
        for file in all_files:
            with open(os.path.join(self.output_dir, file), "rb") as f:
                all_outputs.extend(pickle.load(f))
            os.remove(os.path.join(self.output_dir, file))

        keys = torch.tensor([int(k) for output in all_outputs for k in output.keys], dtype=torch.long)
        predictions = torch.cat([torch.as_tensor(output.predictions) for output in all_outputs], dim=0)
        cpu_bundle = {"keys": keys.cpu(), "predictions": predictions.cpu()}
        torch.save(cpu_bundle, os.path.join(self.output_dir, "merged_predictions_tensor.pt"))
        logger.info(
            "Merged %s keyed rows into merged_predictions_tensor.pt as model output bundle.",
            len(cpu_bundle["keys"]),
        )
