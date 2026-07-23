from collections.abc import Callable
from typing import Any

import torch
import torch.nn.functional as F
from lightning import LightningModule
from lightning.pytorch.trainer.states import TrainerFn
from torch import Tensor, nn
from torch.distributions import Categorical
from torchmetrics import MeanMetric

from src.common.components.model_output import ModelOutput
from src.data.components.data_models import ItemBatch
from src.utils.pylogger import RankedLogger

logger = RankedLogger(__name__, rank_zero_only=True)


class ResidualKMeans(LightningModule):
    """Residual K-Means quantization model with layer-wise training.

    Each layer performs mini-batch K-Means on the residuals from the previous
    layer. Layers are trained one at a time, with each layer receiving an equal
    share of the total training steps. Centroid initialization is done via
    K-Means++ on a buffered subset of data, then refined through gradient-based
    training with WeightedSquaredError loss.
    """

    def __init__(
        self,
        n_layers: int,
        n_clusters: int,
        n_features: int,
        sub_layer: Callable[..., nn.Module],
        loss_function: nn.Module,
        quantization_loss_weight: float = 1.0,
        optimizer: Callable[..., torch.optim.Optimizer] | None = None,
        scheduler: Callable[..., torch.optim.lr_scheduler.LRScheduler] | None = None,
    ):
        super().__init__()

        self.n_layers = n_layers
        self.n_clusters = n_clusters
        self.n_features = n_features
        self.quantization_loss_weight = quantization_loss_weight
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.loss_function = loss_function

        # Layer-wise training schedule state
        self.current_layer = 0
        self.layer_step_boundaries: list[int] = []

        self.layers = nn.ModuleList([sub_layer() for _ in range(n_layers)])
        self.cluster_counts_list: list[torch.Tensor] = [torch.zeros(n_clusters) for _ in range(n_layers)]

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

    def forward(self, embeddings: torch.Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """Layer-wise residual quantization forward pass.

        Args:
            embeddings: (batch_size, n_features)

        Returns:
            cluster_ids: (batch_size, n_layers)
            all_residuals: (batch_size, n_features, n_layers)
            quantized_embeddings: (batch_size, n_features)
            quantization_loss: scalar
        """
        cluster_ids: list[torch.Tensor] = []
        current_residuals = embeddings
        all_residuals: list[torch.Tensor] = []
        quantized_embeddings = torch.zeros_like(embeddings)
        quantization_loss = torch.tensor(0.0).to(self.device)

        for idx in range(self.n_layers):
            current_residuals = F.normalize(current_residuals, dim=-1)

            layer = self.layers[idx]

            if self.trainer.state.fn == TrainerFn.FITTING and idx == self.current_layer:
                layer_ids, layer_embeddings, batch_cluster_counts, batch_cluster_sums = layer(current_residuals)
                if batch_cluster_counts is not None and batch_cluster_sums is not None:
                    batch_cluster_counts = batch_cluster_counts.to(self.device)
                    batch_cluster_sums = batch_cluster_sums.to(self.device)
                    self.cluster_counts_list[idx] += batch_cluster_counts
                    mask = batch_cluster_counts != 0
                    mask_target = batch_cluster_sums[mask] / batch_cluster_counts[mask].unsqueeze(1)
                    centroid_weights = batch_cluster_counts[mask] / self.cluster_counts_list[idx][mask]
                    quantization_loss += self.loss_function(layer.centroids[mask], mask_target, centroid_weights)
                else:
                    quantization_loss += layer.centroids.sum() * 0.0
            else:
                layer_ids, layer_embeddings = layer.predict(current_residuals)

            cluster_ids.append(layer_ids)
            quantized_embeddings = quantized_embeddings + layer_embeddings
            current_residuals = current_residuals - layer_embeddings
            all_residuals.append(current_residuals)

        cluster_ids_tensor = torch.stack(cluster_ids, dim=-1)
        all_residuals_tensor = torch.stack(all_residuals, dim=-1)
        return cluster_ids_tensor, all_residuals_tensor, quantized_embeddings, quantization_loss

    # ------------------------------------------------------------------ #
    # Training
    # ------------------------------------------------------------------ #

    def training_step(self, model_input: ItemBatch) -> torch.Tensor:
        input_embeddings = model_input.features["input_embedding"].to(self.device)
        cluster_ids, all_residuals, _, quantization_loss = self.forward(input_embeddings)

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
                        f"train/layer_{layer_idx}/frac_layer_coverages": getattr(
                            self, f"train_layer_coverages_{layer_idx}"
                        )
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

        if (
            self.current_layer < self.n_layers - 1
            and self.layers[self.current_layer].is_initialized
            and self.global_step > self.layer_step_boundaries[self.current_layer]
        ):
            logger.info(
                f"Device {self.device}: Finished training {self._format_layer_name(self.current_layer)} at global_step={self.global_step + 1}.",
            )
            self.current_layer += 1

        return loss

    def _format_layer_name(self, layer_index: int) -> str:
        if layer_index < 0:
            return "reconstruction stage"
        return f"layer {layer_index + 1}/{self.n_layers}"

    def on_train_start(self):
        if hasattr(self, "train_loss"):
            self.train_loss.reset()

        for layer in self.layers:
            layer.reset_training_state(self.device)
        self.cluster_counts_list = [torch.zeros(self.n_clusters, device=self.device) for _ in range(self.n_layers)]

        total_steps = self.trainer.max_steps
        layer_step_budgets = [total_steps // self.n_layers for _ in range(self.n_layers)]
        layer_step_budgets[-1] += total_steps % self.n_layers
        cumulative_steps = 0
        for budget in layer_step_budgets:
            cumulative_steps += budget
            self.layer_step_boundaries.append(cumulative_steps)


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
        mse = last_residuals_norm ** 2 / all_residuals[:, :, -1].numel()

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
        input_embeddings = batch.features["input_embedding"].to(self.device)
        cluster_ids, all_residuals, _, loss = self.forward(input_embeddings)
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

    def test_step(self, batch: ItemBatch, batch_idx: int):
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

    def predict_step(self, batch: ItemBatch) -> ModelOutput:
        assert batch.item_ids is not None, "Item ids not provided."
        input_embeddings = batch.features["input_embedding"].to(self.device)
        cluster_ids, _, _, _ = self.forward(input_embeddings)
        return ModelOutput(keys=batch.item_ids, predictions=cluster_ids)

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
