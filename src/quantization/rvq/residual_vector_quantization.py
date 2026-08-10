from collections.abc import Callable
from typing import Any

import torch
from lightning import LightningModule
from lightning.pytorch.trainer.states import TrainerFn
from torch import nn
from torch.distributions import Categorical

from src.common.components.loss_functions import WeightedSquaredError
from src.common.configs.model import TrainingModelConfig
from src.data.components.data_models import ItemBatch, ModelOutput
from src.utils.pylogger import RankedLogger

logger = RankedLogger(__name__, rank_zero_only=True)


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
        sub_layer: Callable[..., nn.Module],
        training_model_config: TrainingModelConfig | None = None,
        normalize_residuals: bool = True,
        quantization_loss_weight: float = 1.0,
    ):
        super().__init__()

        if training_model_config is None:
            training_model_config = TrainingModelConfig()

        self.n_layers = n_layers
        self.n_clusters = n_clusters
        self.n_features = n_features
        self.normalize_residuals = normalize_residuals
        self.quantization_loss_weight = quantization_loss_weight
        self.optimizer = training_model_config.optimizer
        self.scheduler = training_model_config.scheduler

        loss_function = training_model_config.loss_function
        if loss_function is None:
            loss_function = WeightedSquaredError()
        self.loss_function = loss_function

        # Layer-wise training schedule state
        self.current_layer = 0
        self.layer_step_boundaries: list[int] = []

        # Per-layer parameters and state
        self.layers = nn.ModuleList([sub_layer() for _ in range(n_layers)])

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

            layer = self.layers[idx]
            train_layer = False
            if self.trainer.state.fn == TrainerFn.FITTING:
                train_layer = idx == self.current_layer

            if train_layer:
                layer_ids, layer_embeddings, quantization_loss_embeddings = layer(current_residuals)
                if quantization_loss_embeddings is None:
                    layer_loss = layer.centroids.sum() * 0.0
                else:
                    layer_loss = self.loss_function(current_residuals, quantization_loss_embeddings)
                quantization_loss += layer_loss
            else:
                layer_ids, layer_embeddings = layer.predict(current_residuals)

            cluster_ids.append(layer_ids)
            quantized_embeddings = quantized_embeddings + layer_embeddings
            current_residuals = current_residuals - layer_embeddings
            all_residuals.append(current_residuals)

        cluster_ids = torch.stack(cluster_ids, dim=-1)
        all_residuals = torch.stack(all_residuals, dim=-1)
        return cluster_ids, all_residuals, quantized_embeddings, quantization_loss

    # ------------------------------------------------------------------ #
    # Training
    # ------------------------------------------------------------------ #

    def training_step(self, model_input: ItemBatch) -> dict[str, Any]:
        input_embeddings = model_input.features["input_embedding"].to(self.device)
        cluster_ids, all_residuals, _, quantization_loss = self.forward(input_embeddings)

        loss = self.quantization_loss_weight * quantization_loss

        with torch.no_grad():
            output_stats = self._compute_output_stats(
                cluster_ids=cluster_ids,
                all_residuals=all_residuals,
                input_embeddings=model_input.features["input_embedding"],
            )

        metric_payload = {
            "loss": loss,
            "quantization_loss": quantization_loss,
            **output_stats,
        }

        if (
            self.current_layer < self.n_layers - 1
            and self.layers[self.current_layer].is_initialized
            and self.global_step + 1 >= self.layer_step_boundaries[self.current_layer]
        ):
            logger.info(
                f"Device {self.device}: Finished training {self._format_layer_name(self.current_layer)} at global_step={self.global_step + 1}.",
            )
            self.current_layer += 1

        return metric_payload

    def _format_layer_name(self, layer_index: int) -> str:
        if layer_index < 0:
            return "reconstruction stage"
        return f"layer {layer_index + 1}/{self.n_layers}"

    def on_train_start(self):
        for layer in self.layers:
            layer.reset_training_state(self.device)

        total_steps = self.trainer.max_steps
        layer_step_budgets = [total_steps // self.n_layers for _ in range(self.n_layers)]
        layer_step_budgets[-1] += total_steps % self.n_layers
        cumulative_steps = 0
        for budget in layer_step_budgets:
            cumulative_steps += budget
            self.layer_step_boundaries.append(cumulative_steps)

    # ------------------------------------------------------------------ #
    # Stats / Eval / Predict
    # ------------------------------------------------------------------ #

    def _compute_output_stats(
        self,
        cluster_ids: torch.Tensor,
        all_residuals: torch.Tensor,
        input_embeddings: torch.Tensor,
    ) -> dict[str, Any]:
        input_embedding_norm = torch.linalg.matrix_norm(input_embeddings)
        first_residuals_norm_ratio = torch.linalg.matrix_norm(all_residuals[:, :, 0]) / input_embedding_norm
        last_residuals_norm = torch.linalg.matrix_norm(all_residuals[:, :, -1])
        last_residuals_norm_ratio = last_residuals_norm / input_embedding_norm
        mse = last_residuals_norm**2 / all_residuals[:, :, -1].numel()

        first_centroids_norm = torch.linalg.matrix_norm(self.layers[0].centroids)
        last_centroids_norm = torch.linalg.matrix_norm(self.layers[-1].centroids)

        frac_unique_ids = torch.unique(cluster_ids, dim=0).shape[0] / cluster_ids.shape[0]

        layer_coverages = []
        layer_id_entropies = []
        for layer_idx in range(self.n_layers):
            _, cluster_counts = torch.unique(cluster_ids[:, layer_idx], return_counts=True)
            cluster_counts = (cluster_counts / cluster_ids.shape[0]).to(self.device)
            entropy = Categorical(probs=cluster_counts).entropy().to(self.device)
            layer_coverages.append(cluster_counts.shape[0] / self.n_clusters)
            layer_id_entropies.append(entropy)

        return {
            "first_residuals_norm_ratio": first_residuals_norm_ratio,
            "last_residuals_norm_ratio": last_residuals_norm_ratio,
            "first_centroids_norm": first_centroids_norm,
            "last_centroids_norm": last_centroids_norm,
            "frac_unique_ids": frac_unique_ids,
            "mse": mse,
            "layer_coverages": layer_coverages,
            "layer_id_entropies": layer_id_entropies,
        }

    def eval_step(
        self,
        batch: ItemBatch,
    ) -> dict[str, torch.Tensor]:
        input_embeddings = batch.features["input_embedding"].to(self.device)
        cluster_ids, all_residuals, _, loss = self.forward(input_embeddings)

        output_stats = self._compute_output_stats(
            cluster_ids=cluster_ids,
            all_residuals=all_residuals,
            input_embeddings=batch.features["input_embedding"],
        )
        return {
            "loss": loss,
            "first_residuals_norm_ratio": output_stats["first_residuals_norm_ratio"],
            "last_residuals_norm_ratio": output_stats["last_residuals_norm_ratio"],
            "frac_unique_ids": output_stats["frac_unique_ids"],
            "mse": output_stats["mse"],
        }

    def validation_step(self, batch: ItemBatch, batch_idx: int):
        return self.eval_step(batch)

    def test_step(self, batch: ItemBatch, batch_idx: int):
        return self.eval_step(batch)

    def predict_step(self, batch: ItemBatch) -> ModelOutput:
        input_embeddings = batch.features["input_embedding"].to(self.device)
        cluster_ids, _, _, _ = self.forward(input_embeddings)
        assert batch.item_ids is not None, "Item ids not provided."
        item_ids = [item_id.item() if isinstance(item_id, torch.Tensor) else item_id for item_id in batch.item_ids]
        return ModelOutput(keys=item_ids, predictions=cluster_ids)

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
            self.layers[idx].is_initialized = checkpoint["layers_initialized"][idx]
        return super().on_load_checkpoint(checkpoint)

    def on_save_checkpoint(self, checkpoint):
        checkpoint["current_layer"] = self.current_layer
        checkpoint["layers_initialized"] = [layer.is_initialized for layer in self.layers]
        return super().on_save_checkpoint(checkpoint)
