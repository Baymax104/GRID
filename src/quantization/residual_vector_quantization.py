import logging
from collections.abc import Callable
from typing import Any

import torch
from lightning import LightningModule
from lightning.pytorch.trainer.states import TrainerFn
from lightning.pytorch.utilities import rank_zero_only
from torch import nn
from torch.distributions import Categorical
from torchmetrics import MeanMetric

from src.common.components.clustering_initializers import ClusteringInitializer
from src.common.components.distance_functions import DistanceFunction
from src.common.components.loss_functions import WeightedSquaredError
from src.common.components.model_output import OneKeyPerPredictionOutput
from src.common.components.quantization_strategies import QuantizationStrategy
from src.data.components.data_models import ItemBatch


class ResidualVectorQuantization(LightningModule):
    """Residual Vector Quantization model with layer-wise training.

    Each layer performs vector quantization with Straight-Through Estimator
    (STE) on the residuals from the previous layer. Layers are trained one at
    a time, with each layer receiving an equal share of the total training
    steps. Centroid initialization is done via K-Means++ on a buffered subset
    of data, then refined through gradient-based training.
    """

    def __init__(
        self,
        n_layers: int,
        n_clusters: int,
        n_features: int,
        distance_function: DistanceFunction,
        initializer: ClusteringInitializer,
        quantization_strategy: QuantizationStrategy,
        loss_function: nn.Module | None = None,
        init_buffer_size: int = 1000,
        normalize_residuals: bool = True,
        training_loop_function: Callable | None = None,
        quantization_loss_weight: float = 1.0,
        optimizer: Callable[..., torch.optim.Optimizer] | None = None,
        scheduler: Callable[..., torch.optim.lr_scheduler.LRScheduler] | None = None,
        track_residuals: bool = False,
    ):
        super().__init__()

        self.n_layers = n_layers
        self.n_clusters = n_clusters
        self.n_features = n_features
        self.distance_function = distance_function
        self.initializer = initializer
        self.quantization_strategy = quantization_strategy
        self.init_buffer_size = init_buffer_size
        self.normalize_residuals = normalize_residuals
        self.training_loop_function = training_loop_function
        self.quantization_loss_weight = quantization_loss_weight
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.track_residuals = track_residuals

        if loss_function is None:
            loss_function = WeightedSquaredError()
        self.loss_function = loss_function
        self.init_loss_function = WeightedSquaredError()

        # Layer-wise training schedule state
        self.current_layer = 0
        self.steps_per_layer = 0
        self.layer_step_budgets: list[int] = []
        self.layer_step_boundaries: list[int] = []
        self.layer_training_schedule: list[int] = []
        self.current_layer_schedule_index = 0

        # Per-layer parameters and state
        self.centroids_list = nn.ParameterList([
            nn.Parameter(torch.zeros(n_clusters, n_features), requires_grad=True)
            for _ in range(n_layers)
        ])
        self.init_buffers: list[torch.Tensor] = [torch.tensor([]) for _ in range(n_layers)]
        self.is_initialized_list: list[bool] = [False for _ in range(n_layers)]
        self.is_initial_step_list: list[bool] = [False for _ in range(n_layers)]
        self.init_centroids_list: list[torch.Tensor | None] = [None for _ in range(n_layers)]

        if self.training_loop_function is not None:
            logging.info(f"Device {self.device}: Using custom training loop function")
            self.automatic_optimization = False

        # Metrics
        self.train_loss = MeanMetric()
        self.train_quantization_loss = MeanMetric()
        self.train_first_residuals_norm_ratio = MeanMetric()
        self.train_last_residuals_norm_ratio = MeanMetric()
        self.first_centroids_norm = MeanMetric()
        self.last_centroids_norm = MeanMetric()
        self.train_frac_unique_ids = MeanMetric()
        self.train_mse = MeanMetric()
        for layer_idx in range(self.n_layers):
            setattr(self, f"train_layer_coverages_{layer_idx}", MeanMetric())
            setattr(self, f"train_layer_id_entropy_{layer_idx}", MeanMetric())

        self.val_loss = MeanMetric()
        self.val_first_residuals_norm_ratio = MeanMetric()
        self.val_last_residuals_norm_ratio = MeanMetric()
        self.val_mse = MeanMetric()
        self.val_frac_unique_ids = MeanMetric()

        self.test_loss = MeanMetric()
        self.test_first_residuals_norm_ratio = MeanMetric()
        self.test_last_residuals_norm_ratio = MeanMetric()
        self.test_mse = MeanMetric()
        self.test_frac_unique_ids = MeanMetric()

    # ------------------------------------------------------------------ #
    # Per-layer VQ logic (inlined from BaseClusteringModule + VectorQuantization)
    # ------------------------------------------------------------------ #

    def _buffer_points(self, layer_idx: int, batch: torch.Tensor) -> None:
        batch = batch.detach()
        buf = self.init_buffers[layer_idx]
        n_to_add = min(self.init_buffer_size - buf.shape[0], batch.shape[0])
        self.init_buffers[layer_idx] = torch.cat([buf, batch[:n_to_add]], dim=0)

    @rank_zero_only
    def _compute_initial_centroids(self, layer_idx: int, buffer: torch.Tensor) -> None:
        if buffer.shape[0] < self.n_clusters:
            raise ValueError(
                f"Buffer size {buffer.shape[0]} is less than the number of clusters {self.n_clusters}."
            )
        self.init_centroids_list[layer_idx] = self.initializer(buffer)

    def _initialization_step(
        self, layer_idx: int, batch: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        self._buffer_points(layer_idx, batch)

        if self.init_buffers[layer_idx].shape[0] < self.init_buffer_size:
            centroid_zero_embeddings = torch.zeros_like(
                self.centroids_list[layer_idx].data, dtype=batch.dtype, device=self.device
            )
            loss = self.init_loss_function(self.centroids_list[layer_idx], centroid_zero_embeddings)
            batch_zero_embeddings = torch.zeros_like(batch, dtype=batch.dtype, device=self.device)
            batch_zero_assignments = torch.zeros(batch.shape[0], dtype=torch.long, device=self.device)
            return batch_zero_assignments, batch_zero_embeddings, loss

        self.is_initial_step_list[layer_idx] = True
        self.init_centroids_list[layer_idx] = torch.zeros_like(
            self.centroids_list[layer_idx].data, dtype=batch.dtype, device=self.device
        )
        self._compute_initial_centroids(layer_idx=layer_idx, buffer=self.init_buffers[layer_idx])  # noqa
        self.init_buffers[layer_idx] = torch.tensor([], device=self.device)

        init_centroids = self.init_centroids_list[layer_idx]
        loss = self.init_loss_function(self.centroids_list[layer_idx], init_centroids)
        distances = self.distance_function.compute(batch, init_centroids)
        assignments = torch.argmin(distances, dim=1).to(self.device)
        return assignments, init_centroids[assignments], loss

    def _vq_forward(
        self, layer_idx: int, batch: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        """VQ forward: use quantization_strategy to quantize."""
        codebook = self.centroids_list[layer_idx]
        ids, embeddings, reconstruction_loss_embeddings = self.quantization_strategy.quantize(
            codebook=codebook,
            batch=batch,
        )
        return ids, embeddings, reconstruction_loss_embeddings

    def _layer_model_step(
        self, layer_idx: int, batch: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        """Per-layer model step: init check → forward → loss."""
        if batch.device != self.device:
            batch = batch.to(self.device)

        if self.is_initial_step_list[layer_idx]:
            self.is_initial_step_list[layer_idx] = False
            self.is_initialized_list[layer_idx] = True

        if not self.is_initialized_list[layer_idx]:
            return self._initialization_step(layer_idx, batch)

        assignments, embeddings, reconstruction_loss_embeddings = self._vq_forward(layer_idx, batch)
        loss = self.loss_function(batch, embeddings)
        return (
            assignments,
            reconstruction_loss_embeddings if reconstruction_loss_embeddings is not None else embeddings,
            loss,
        )

    def _predict_layer(
        self, layer_idx: int, batch: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch = batch.to(self.device)
        with torch.no_grad():
            centroids = self.centroids_list[layer_idx].data
            distances = self.distance_function.compute(batch, centroids)
            assignments = torch.argmin(distances, dim=1)
            return assignments, centroids[assignments]

    # ------------------------------------------------------------------ #
    # Model-level forward / model_step
    # ------------------------------------------------------------------ #

    def forward(self, embeddings: torch.Tensor):
        cluster_ids = []
        current_residuals = embeddings
        all_residuals = []
        quantized_embeddings = torch.zeros_like(embeddings)
        quantization_loss = torch.tensor(0.0).to(self.device)

        for idx in range(self.n_layers):
            if self.normalize_residuals:
                current_residuals = nn.functional.normalize(current_residuals, dim=-1)

            train_layer = False
            if self.trainer.state.fn == TrainerFn.FITTING:
                train_layer = idx == self.current_layer

            if train_layer:
                layer_ids, layer_embeddings, layer_loss = self._layer_model_step(idx, current_residuals)
                quantization_loss += layer_loss
            else:
                layer_ids, layer_embeddings = self._predict_layer(idx, current_residuals)

            cluster_ids.append(layer_ids)
            quantized_embeddings = quantized_embeddings + layer_embeddings
            current_residuals = current_residuals - layer_embeddings
            if self.track_residuals:
                all_residuals.append(current_residuals)

        cluster_ids = torch.stack(cluster_ids, dim=-1)
        all_residuals = torch.stack(all_residuals, dim=-1) if self.track_residuals else None
        return cluster_ids, all_residuals, quantized_embeddings, quantization_loss

    def model_step(self, model_input: ItemBatch):
        input_embeddings = model_input.features["input_embedding"].to(self.device)
        cluster_ids, all_residuals, quantized_embeddings, quantization_loss = self.forward(input_embeddings)
        return cluster_ids, all_residuals, quantization_loss

    # ------------------------------------------------------------------ #
    # Training
    # ------------------------------------------------------------------ #

    def training_step(self, model_input: ItemBatch) -> torch.Tensor:
        cluster_ids, all_residuals, quantization_loss = self.model_step(model_input)

        loss = self.quantization_loss_weight * quantization_loss
        self.train_loss(loss)
        self.train_quantization_loss(quantization_loss)
        train_dict_to_log = {
            "train/quantization_loss": self.train_quantization_loss,
        }

        with torch.no_grad():
            if self.global_step % self.trainer.log_every_n_steps == 0:
                (
                    train_first_residuals_norm_ratio,
                    train_last_residuals_norm_ratio,
                    first_centroids_norm,
                    last_centroids_norm,
                    train_frac_unique_ids,
                    train_mse,
                    train_layer_coverages,
                    train_layer_id_entropies,
                ) = self._compute_output_stats(
                    cluster_ids=cluster_ids,
                    all_residuals=all_residuals,
                    input_embeddings=model_input.features["input_embedding"],
                )
                self.train_first_residuals_norm_ratio(train_first_residuals_norm_ratio)
                self.train_last_residuals_norm_ratio(train_last_residuals_norm_ratio)
                self.first_centroids_norm(first_centroids_norm)
                self.last_centroids_norm(last_centroids_norm)
                self.train_frac_unique_ids(train_frac_unique_ids)
                self.train_mse(train_mse)
                for layer_idx in range(self.n_layers):
                    getattr(self, f"train_layer_coverages_{layer_idx}")(train_layer_coverages[layer_idx])
                    getattr(self, f"train_layer_id_entropy_{layer_idx}")(train_layer_id_entropies[layer_idx])

                train_dict_to_log.update(
                    {
                        "train/last_residuals_norm_ratio": self.train_last_residuals_norm_ratio,
                        "train/first_residuals_norm_ratio": self.train_first_residuals_norm_ratio,
                        "train/first_centroids_norm": self.first_centroids_norm,
                        "train/last_centroids_norm": self.last_centroids_norm,
                        "train/frac_unique_ids": self.train_frac_unique_ids,
                        "train/mse": self.train_mse,
                    }
                )
                train_dict_to_log.update(
                    {
                        f"train/layer_{layer_idx}/frac_layer_coverages": getattr(self, f"train_layer_coverages_{layer_idx}")
                        for layer_idx in range(self.n_layers)
                    }
                )
                train_dict_to_log.update(
                    {
                        f"train/layer_{layer_idx}/id_entropy": getattr(self, f"train_layer_id_entropy_{layer_idx}")
                        for layer_idx in range(self.n_layers)
                    }
                )

        train_dict_to_log["train/loss"] = self.train_loss

        self.log_dict(
            train_dict_to_log,
            on_step=True,
            on_epoch=False,
            prog_bar=False,
            logger=True,
            sync_dist=True,
        )

        if self.training_loop_function is not None:
            is_initialized = self.is_initialized_list[self.current_layer]
            self.training_loop_function(
                self,
                loss=loss,
                world_size=self.trainer.world_size,
                is_initialized=is_initialized,
            )

        if (
            self.current_layer_schedule_index < len(self.layer_training_schedule) - 1
            and (
                self.current_layer < 0
                or self.is_initialized_list[self.current_layer]
            )
            and self.global_step + 1 >= self.layer_step_boundaries[self.current_layer_schedule_index]
        ):
            logging.info(
                f"Device {self.device}: Finished training {self._format_layer_name(self.current_layer)} at global_step={self.global_step + 1}.",
            )
            self.current_layer_schedule_index += 1
            self.current_layer = self.layer_training_schedule[self.current_layer_schedule_index]

        return loss

    def _format_layer_name(self, layer_index: int) -> str:
        if layer_index < 0:
            return "reconstruction stage"
        return f"layer {layer_index + 1}/{self.n_layers}"

    def on_train_start(self):
        if hasattr(self, "train_loss"):
            self.train_loss.reset()

        self.current_layer = 0
        self.current_layer_schedule_index = 0
        self.layer_step_budgets = []
        self.layer_step_boundaries = []
        self.layer_training_schedule = []

        for idx in range(self.n_layers):
            self.init_buffers[idx] = torch.tensor([], device=self.device)
            self.centroids_list[idx] = self.centroids_list[idx].to(self.device)

        total_steps = self.trainer.max_steps
        self.layer_training_schedule = list(range(self.n_layers))

        eff_n_layers = len(self.layer_training_schedule)
        base_steps_per_layer = total_steps // eff_n_layers
        remainder = total_steps % eff_n_layers
        self.layer_step_budgets = [
            base_steps_per_layer + (1 if layer_idx < remainder else 0) for layer_idx in range(eff_n_layers)
        ]
        cumulative_steps = 0
        for budget in self.layer_step_budgets:
            cumulative_steps += budget
            self.layer_step_boundaries.append(cumulative_steps)

        self.current_layer = self.layer_training_schedule[0]
        self.steps_per_layer = base_steps_per_layer
        schedule_as_text = ", ".join(
            f"{self._format_layer_name(layer)}={budget}"
            for layer, budget in zip(self.layer_training_schedule, self.layer_step_budgets, strict=False)
        )
        logging.info(
            f"Device {self.device}: Training layers one-at-a-time with step budget [{schedule_as_text}] (total={total_steps})."
            " Ensure that early stopping callbacks are disabled."
        )

        self.train_first_residuals_norm_ratio.reset()
        self.train_last_residuals_norm_ratio.reset()
        self.train_frac_unique_ids.reset()
        self.first_centroids_norm.reset()
        self.last_centroids_norm.reset()
        self.train_mse.reset()
        for layer_idx in range(self.n_layers):
            getattr(self, f"train_layer_coverages_{layer_idx}").reset()
            getattr(self, f"train_layer_id_entropy_{layer_idx}").reset()

    # ------------------------------------------------------------------ #
    # Stats / Eval / Predict
    # ------------------------------------------------------------------ #

    def _compute_output_stats(
        self,
        cluster_ids: torch.Tensor,
        all_residuals: torch.Tensor,
        input_embeddings: torch.Tensor,
    ) -> tuple:
        input_embedding_norm = torch.linalg.matrix_norm(input_embeddings)
        first_residuals_norm_ratio = torch.linalg.matrix_norm(all_residuals[:, :, 0]) / input_embedding_norm
        last_residuals_norm = torch.linalg.matrix_norm(all_residuals[:, :, -1])
        last_residuals_norm_ratio = last_residuals_norm / input_embedding_norm
        mse = last_residuals_norm**2 / all_residuals[:, :, -1].numel()

        first_centroids_norm = torch.linalg.matrix_norm(self.centroids_list[0])
        last_centroids_norm = torch.linalg.matrix_norm(self.centroids_list[-1])

        frac_unique_ids = torch.unique(cluster_ids, dim=0).shape[0] / cluster_ids.shape[0]

        layer_coverages = []
        layer_id_entropies = []
        for layer_idx in range(self.n_layers):
            _, cluster_counts = torch.unique(cluster_ids[:, layer_idx], return_counts=True)
            cluster_counts = (cluster_counts / cluster_ids.shape[0]).to(self.device)
            entropy = Categorical(probs=cluster_counts).entropy().to(self.device)
            layer_coverages.append(cluster_counts.shape[0] / self.n_clusters)
            layer_id_entropies.append(entropy)

        return (
            first_residuals_norm_ratio,
            last_residuals_norm_ratio,
            first_centroids_norm,
            last_centroids_norm,
            frac_unique_ids,
            mse,
            layer_coverages,
            layer_id_entropies,
        )

    def eval_step(
        self,
        batch: ItemBatch,
        loss_to_aggregate: MeanMetric,
        first_residuals_norm_ratio_metric: MeanMetric,
        last_residuals_norm_ratio_metric: MeanMetric,
        frac_unique_ids_metric: MeanMetric,
        mse_metric: MeanMetric,
    ):
        cluster_ids, all_residuals, loss = self.model_step(batch)
        loss_to_aggregate(loss)

        (
            first_residuals_norm_ratio,
            last_residuals_norm_ratio,
            _,
            _,
            frac_unique_ids,
            mse,
            _,
            _,
        ) = self._compute_output_stats(
            cluster_ids=cluster_ids,
            all_residuals=all_residuals,
            input_embeddings=batch.features["input_embedding"],
        )
        last_residuals_norm_ratio_metric(last_residuals_norm_ratio)
        first_residuals_norm_ratio_metric(first_residuals_norm_ratio)
        frac_unique_ids_metric(frac_unique_ids)
        mse_metric(mse)

    def validation_step(self, batch: ItemBatch, batch_idx: int):
        self.eval_step(
            batch,
            self.val_loss,
            self.val_first_residuals_norm_ratio,
            self.val_last_residuals_norm_ratio,
            self.val_frac_unique_ids,
            self.val_mse,
        )
        self.log_dict(
            {
                "val/first_residuals_norm_ratio": self.val_first_residuals_norm_ratio,
                "val/last_residuals_norm_ratio": self.val_last_residuals_norm_ratio,
                "val/frac_unique_ids": self.val_frac_unique_ids,
                "val/mse": self.val_mse,
                "val/loss": self.val_loss,
            },
            on_step=False,
            on_epoch=True,
            prog_bar=False,
            logger=True,
            sync_dist=True,
        )

    def on_validation_start(self):
        self.val_loss.reset()
        self.val_first_residuals_norm_ratio.reset()
        self.val_last_residuals_norm_ratio.reset()
        self.val_frac_unique_ids.reset()
        self.val_mse.reset()

    def test_step(self, batch: ItemBatch, batch_idx: int) -> None:
        self.eval_step(
            batch,
            self.test_loss,
            self.test_first_residuals_norm_ratio,
            self.test_last_residuals_norm_ratio,
            self.test_frac_unique_ids,
            self.test_mse,
        )
        self.log_dict(
            {
                "test/first_residuals_norm_ratio": self.test_first_residuals_norm_ratio,
                "test/last_residuals_norm_ratio": self.test_last_residuals_norm_ratio,
                "test/frac_unique_ids": self.test_frac_unique_ids,
                "test/mse": self.test_mse,
                "test/loss": self.test_loss,
            },
            on_step=False,
            on_epoch=True,
            prog_bar=False,
            logger=True,
            sync_dist=True,
        )

    def on_test_start(self):
        self.test_loss.reset()
        self.test_first_residuals_norm_ratio.reset()
        self.test_last_residuals_norm_ratio.reset()
        self.test_frac_unique_ids.reset()
        self.test_mse.reset()

    def predict_step(self, batch: ItemBatch) -> OneKeyPerPredictionOutput:
        cluster_ids, _, _ = self.model_step(batch)
        assert batch.item_ids is not None, "Item ids not provided."
        item_ids = [item_id.item() if isinstance(item_id, torch.Tensor) else item_id for item_id in batch.item_ids]
        return OneKeyPerPredictionOutput(
            keys=item_ids,
            predictions=cluster_ids,
            key_name="item_id",
            prediction_name="cluster_ids",
        )

    # ------------------------------------------------------------------ #
    # Optimizer / Checkpoint
    # ------------------------------------------------------------------ #

    def configure_optimizers(self) -> dict[str, Any]:
        assert self.optimizer is not None, "Optimizer not initialized."
        model = self.trainer.model
        assert model is not None, "Trainer not initialized."
        optimizer = self.optimizer(params=model.parameters())
        if self.scheduler is not None:
            scheduler = self.scheduler(optimizer=optimizer)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "interval": "step",
                    "frequency": 1,
                },
            }
        return {"optimizer": optimizer}

    def on_load_checkpoint(self, checkpoint):
        self.current_layer = checkpoint["current_layer"]
        for idx in range(self.n_layers):
            self.is_initialized_list[idx] = checkpoint["layers_initialized"][idx]
        return super().on_load_checkpoint(checkpoint)

    def on_save_checkpoint(self, checkpoint):
        checkpoint["current_layer"] = self.current_layer
        checkpoint["layers_initialized"] = list(self.is_initialized_list)
        return super().on_save_checkpoint(checkpoint)
