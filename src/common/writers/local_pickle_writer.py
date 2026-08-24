"""Local prediction bundle writer."""

import datetime
import os
import pickle

import torch
from lightning import LightningModule, Trainer

from src.common.writers.base import BaseBufferedWriter
from src.data.components.data_models import ModelOutput
from src.utils.decorators import retry
from src.utils.distributed import distributed_barrier
from src.utils.file import sync_file
from src.utils.pylogger import RankedLogger

logger = RankedLogger(__name__, rank_zero_only=True)


class LocalPickleWriter(BaseBufferedWriter):
    """
    Callback to write predictions to local pickle files during inference.
    """

    def __init__(
        self,
        output_dir: str,
        flush_frequency: int = 1000,
        post_processing_functions: list[callable] | None = None,
    ):
        """
        Args:
            output_dir: Directory to save the pickle files.
            flush_frequency: Number of samples to accumulate before writing to a pickle file.
            post_processing_functions: list of ordered post-processing functions to apply to the files.
        """
        super().__init__(flush_frequency=flush_frequency)
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        self.post_processing_functions = post_processing_functions if post_processing_functions else []

    def _local_file_path(self, file_path: str) -> str:
        """Create a local file path for the pickle file."""
        return f"{self.output_dir}/{file_path}"

    @retry()
    def _flush_buffer(self):
        """Flush the buffer to a local temporary pickle file."""
        file_path = f"predictions_{self.global_rank}_{datetime.datetime.now(datetime.UTC).strftime('%Y%m%dT%H%M%S%f')[:-3]}.pkl"
        with open(self._local_file_path(file_path=file_path), "wb") as f:
            pickle.dump(self.buffer, f)

        logger.info(
            f"Global Rank: {self.global_rank} wrote {self._buffer_sample_count()} "
            f"samples to {self._local_file_path(file_path=file_path)}."
        )

    @retry()
    def on_predict_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
    ):
        assert trainer.global_rank is not None, "Global rank was not provided."

        super().on_predict_end(trainer, pl_module)

        distributed_barrier()
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
        output_path = os.path.join(self.output_dir, "merged_predictions_tensor.pt")
        torch.save(cpu_bundle, output_path)
        logger.info(f"Merged {len(cpu_bundle['keys'])} keyed rows into model output bundle.")
        logger.info(f"Model output bundle saved to {output_path}.")
