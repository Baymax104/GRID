from collections.abc import Callable
from typing import Any

import torch
from lightning import LightningModule
from lightning.pytorch.trainer.states import TrainerFn
from torch import nn
from torch.distributions import Categorical

from src.common.configs.model import TrainingModelConfig
from src.common.loss.weighted_squared_error import WeightedSquaredError
from src.data.components.data_models import ItemBatch, ModelOutput
from src.utils.distributed import broadcast_from_rank_zero, get_distributed_rank
from src.utils.pylogger import RankedLogger

logger = RankedLogger(__name__, rank_zero_only=True)


def _compute_squared_euclidean_distance(x: torch.Tensor, y: torch.Tensor, batch_size: int | None = 256) -> torch.Tensor:
    """Compute squared Euclidean distances between rows of x and rows of y."""
    assert x.dim() == 2, f"Data must be 2D, got {x.dim()} dimensions"
    assert y.dim() == 2, f"Data must be 2D, got {y.dim()} dimensions"
    assert x.size(1) == y.size(1), "Data must have the same number of columns"

    n1 = x.shape[0]
    if batch_size is None or batch_size >= n1:
        x_expanded = x.unsqueeze(1)
        y_expanded = y.unsqueeze(0)
        return torch.sum((x_expanded - y_expanded).pow(2), dim=2)

    all_sq_distances = []
    num_batches = (n1 + batch_size - 1) // batch_size
    for i in range(num_batches):
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, n1)
        x_batch = x[start_idx:end_idx]
        x_batch_expanded = x_batch.unsqueeze(1)
        y_expanded = y.unsqueeze(0)
        sq_diffs_batch = (x_batch_expanded - y_expanded).pow(2)
        all_sq_distances.append(torch.sum(sq_diffs_batch, dim=2))
    return torch.cat(all_sq_distances, dim=0)


def _kmeans_plus_plus_init(
    buffer: torch.Tensor,
    n_clusters: int,
    distance_fn: Callable = _compute_squared_euclidean_distance,
) -> torch.Tensor:
    """Initialize centroids using the k-means++ algorithm."""
    n_samples = buffer.shape[0]
    n_features = buffer.shape[1]
    centroids = torch.zeros((n_clusters, n_features), dtype=buffer.dtype, device=buffer.device)

    first_centroid_idx = torch.randint(0, n_samples, (1,), device=buffer.device)
    centroids[0] = buffer[first_centroid_idx]

    for i in range(1, n_clusters):
        min_distances = torch.min(distance_fn(buffer, centroids[:i]), dim=1)[0]
        if min_distances.sum() == 0:
            centroids[i:] = buffer[torch.randint(0, n_samples, (n_clusters - i,), device=buffer.device)]
            break
        next_centroid_idx = torch.multinomial(min_distances, num_samples=1)
        centroids[i] = buffer[next_centroid_idx]

    return centroids


def _ste_quantize(
    codebook: torch.Tensor,
    batch: torch.Tensor,
    distance_fn: Callable = _compute_squared_euclidean_distance,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Quantize using the Straight-Through Estimator (STE)."""
    dists = distance_fn(batch, codebook)
    ids = torch.argmin(dists, dim=-1)
    embeddings = codebook[ids]
    reconstruction_loss_embeddings = batch + (embeddings - batch).detach()
    return ids, embeddings, reconstruction_loss_embeddings


class ResidualQuantizationVAE(LightningModule):
    """Residual Quantization VAE model with progressive joint training.

    Combines vector quantization (VQ-STE) with an encoder-decoder architecture
    for reconstruction. All layers are trained simultaneously with progressive
    initialization: layer 0 always trains, subsequent layers unlock once the
    previous layer is initialized. Centroid initialization uses K-Means
    convergence (K-Means++ followed by iterative refinement) for stable codebook
    starting points.
    """

    def __init__(
        self,
        n_layers: int,
        n_clusters: int,
        n_features: int,
        training_model_config: TrainingModelConfig | None = None,
        init_buffer_size: int = 1000,
        normalize_residuals: bool = False,
        quantization_loss_weight: float = 1.0,
        reconstruction_loss_weight: float = 0.0,
        normalization_layer: nn.Module | None = None,
        encoder: nn.Module | None = None,
        decoder: nn.Module | None = None,
        kmeans_max_iter: int = 1000,
        kmeans_atol: float = 1e-8,
    ):
        super().__init__()

        if training_model_config is None:
            training_model_config = TrainingModelConfig()

        self.n_layers = n_layers
        self.n_clusters = n_clusters
        self.n_features = n_features
        self.init_buffer_size = init_buffer_size
        self.normalize_residuals = normalize_residuals
        self.quantization_loss_weight = quantization_loss_weight
        self.reconstruction_loss_function = training_model_config.reconstruction_loss_function
        self.reconstruction_loss_weight = reconstruction_loss_weight
        self.kmeans_max_iter = kmeans_max_iter
        self.kmeans_atol = kmeans_atol
        self.optimizer = training_model_config.optimizer
        self.scheduler = training_model_config.scheduler

        loss_function = training_model_config.loss_function
        if loss_function is None:
            loss_function = WeightedSquaredError()
        self.loss_function = loss_function

        self.normalization_layer = normalization_layer if normalization_layer is not None else nn.Identity()
        self.encoder = encoder if encoder is not None else nn.Identity()
        self.decoder = decoder if decoder is not None else nn.Identity()

        # Per-layer parameters and state
        self.centroids_list = nn.ParameterList(
            [nn.Parameter(torch.zeros(n_clusters, n_features), requires_grad=True) for _ in range(n_layers)]
        )
        self.init_buffers: list[torch.Tensor] = [torch.tensor([]) for _ in range(n_layers)]
        self.is_initialized_list: list[bool] = [False for _ in range(n_layers)]

    # ------------------------------------------------------------------ #
    # Per-layer VQ logic + K-Means convergence initialization
    # ------------------------------------------------------------------ #

    def _buffer_points(self, layer_idx: int, batch: torch.Tensor):
        batch = batch.detach()
        buf = self.init_buffers[layer_idx]
        n_to_add = min(self.init_buffer_size - buf.shape[0], batch.shape[0])
        self.init_buffers[layer_idx] = torch.cat([buf, batch[:n_to_add]], dim=0)

    def _compute_initial_centroids_for_current_rank(self, layer_idx: int, buffer: torch.Tensor) -> torch.Tensor:
        """Initialize centroids via K-Means convergence.

        Runs K-Means++ initialization followed by iterative mini-batch K-Means
        refinement until convergence or max_iter. This provides stable codebook
        starting points for the VQ-STE gradient training.
        """
        if buffer.shape[0] < self.n_clusters:
            raise ValueError(f"Buffer size {buffer.shape[0]} is less than the number of clusters {self.n_clusters}.")
        if get_distributed_rank() != 0:
            return torch.zeros_like(self.centroids_list[layer_idx].data)

        # Step 1: K-Means++ initialization
        centroids = _kmeans_plus_plus_init(buffer, self.n_clusters)

        # Step 2: Iterative K-Means refinement (mini-batch K-Means with manual update)
        cluster_counts = torch.zeros(self.n_clusters, device=buffer.device)
        prev_centroids = centroids.clone()

        for step in range(self.kmeans_max_iter):
            distances = _compute_squared_euclidean_distance(buffer, centroids)
            assignments = torch.argmin(distances, dim=1)
            assignments_one_hot = nn.functional.one_hot(assignments, self.n_clusters).float().detach()
            batch_cluster_counts = torch.sum(assignments_one_hot, dim=0)
            cluster_counts += batch_cluster_counts
            batch_cluster_sums = torch.mm(assignments_one_hot.t(), buffer)

            mask = batch_cluster_counts != 0
            mask_target = batch_cluster_sums[mask] / batch_cluster_counts[mask].unsqueeze(1)
            centroid_weights = batch_cluster_counts[mask] / cluster_counts[mask]
            centroids[mask] = centroids[mask] - (centroids[mask] - mask_target) * centroid_weights.unsqueeze(1)

            if step > 0 and torch.allclose(prev_centroids, centroids, atol=self.kmeans_atol):
                logger.info(f"K-Means convergence for layer {layer_idx} after {step} iterations")
                break
            prev_centroids = centroids.clone()

        return centroids.detach()

    def _initialization_step(
        self, layer_idx: int, batch: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        self._buffer_points(layer_idx, batch)

        if self.init_buffers[layer_idx].shape[0] < self.init_buffer_size:
            loss = self.centroids_list[layer_idx].sum() * 0.0
            batch_zero_embeddings = torch.zeros_like(batch, dtype=batch.dtype, device=self.device)
            batch_zero_assignments = torch.zeros(batch.shape[0], dtype=torch.long, device=self.device)
            return batch_zero_assignments, batch_zero_embeddings, loss

        initial_centroids = self._compute_initial_centroids_for_current_rank(
            layer_idx=layer_idx, buffer=self.init_buffers[layer_idx]
        )
        initial_centroids = broadcast_from_rank_zero(initial_centroids)
        with torch.no_grad():
            self.centroids_list[layer_idx].copy_(initial_centroids)
        self.is_initialized_list[layer_idx] = True
        self.init_buffers[layer_idx] = torch.tensor([], device=self.device)

        loss = self.centroids_list[layer_idx].sum() * 0.0
        distances = _compute_squared_euclidean_distance(batch, self.centroids_list[layer_idx].data)
        assignments = torch.argmin(distances, dim=1).to(self.device)
        return assignments, self.centroids_list[layer_idx][assignments], loss

    def _vq_forward(
        self, layer_idx: int, batch: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        codebook = self.centroids_list[layer_idx]
        ids, embeddings, reconstruction_loss_embeddings = _ste_quantize(
            codebook=codebook,
            batch=batch,
        )
        return ids, embeddings, reconstruction_loss_embeddings

    def _layer_model_step(
        self, layer_idx: int, batch: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        if batch.device != self.device:
            batch = batch.to(self.device)

        if not self.is_initialized_list[layer_idx]:
            return self._initialization_step(layer_idx, batch)

        assignments, embeddings, reconstruction_loss_embeddings = self._vq_forward(layer_idx, batch)
        loss = self.loss_function(batch, embeddings)
        return (
            assignments,
            reconstruction_loss_embeddings if reconstruction_loss_embeddings is not None else embeddings,
            loss,
        )

    def _predict_layer(self, layer_idx: int, batch: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        batch = batch.to(self.device)
        with torch.no_grad():
            centroids = self.centroids_list[layer_idx].data
            distances = _compute_squared_euclidean_distance(batch, centroids)
            assignments = torch.argmin(distances, dim=1)
            return assignments, centroids[assignments]

    # ------------------------------------------------------------------ #
    # Model-level forward / model_step
    # ------------------------------------------------------------------ #

    def forward(self, embeddings: torch.Tensor):
        """Progressive residual quantization forward pass.

        Layers are trained simultaneously with progressive unlocking:
        - Layer 0 always trains
        - Layer N trains once layer N-1 is initialized
        - Already-initialized layers are frozen until all layers are initialized
        """
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
                if self.is_initialized_list[idx] and not self.is_initialized_list[-1]:
                    # Already initialized but not all layers ready → freeze
                    train_layer = False
                elif idx == 0:
                    train_layer = True
                elif self.is_initialized_list[idx - 1]:
                    train_layer = True

            if train_layer:
                layer_ids, layer_embeddings, layer_loss = self._layer_model_step(idx, current_residuals)
                quantization_loss += layer_loss
            else:
                layer_ids, layer_embeddings = self._predict_layer(idx, current_residuals)

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
        normalized_input_embeddings = self.normalization_layer(input_embeddings)
        encoded_embeddings = self.encoder(normalized_input_embeddings)
        cluster_ids, all_residuals, quantized_embeddings, quantization_loss = self.forward(encoded_embeddings)

        if self.reconstruction_loss_function is not None and self.is_initialized_list[-1]:
            reconstructed_embeddings = self.decoder(quantized_embeddings)
            reconstruction_loss = self.reconstruction_loss_function(
                reconstructed_embeddings, normalized_input_embeddings
            )
        else:
            reconstruction_loss = torch.tensor(0.0).to(self.device)

        loss = self.quantization_loss_weight * quantization_loss + self.reconstruction_loss_weight * reconstruction_loss

        with torch.no_grad():
            output_stats = self._compute_output_stats(
                cluster_ids=cluster_ids,
                all_residuals=all_residuals,
                input_embeddings=model_input.features["input_embedding"],
            )
        metric_payload = {
            "loss": loss,
            "quantization_loss": quantization_loss,
            "reconstruction_loss": reconstruction_loss,
            **output_stats,
        }
        return metric_payload

    def on_train_start(self):
        for idx in range(self.n_layers):
            self.init_buffers[idx] = torch.tensor([], device=self.device)
            self.centroids_list[idx] = self.centroids_list[idx].to(self.device)

        logger.info(f"Device {self.device}: Training all layers simultaneously")

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
        normalized_input_embeddings = self.normalization_layer(input_embeddings)
        encoded_embeddings = self.encoder(normalized_input_embeddings)
        cluster_ids, all_residuals, _, loss = self.forward(encoded_embeddings)

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
        normalized_input_embeddings = self.normalization_layer(input_embeddings)
        encoded_embeddings = self.encoder(normalized_input_embeddings)
        cluster_ids, _, _, _ = self.forward(encoded_embeddings)
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
        for idx in range(self.n_layers):
            self.is_initialized_list[idx] = checkpoint["layers_initialized"][idx]
        return super().on_load_checkpoint(checkpoint)

    def on_save_checkpoint(self, checkpoint):
        checkpoint["layers_initialized"] = list(self.is_initialized_list)
        return super().on_save_checkpoint(checkpoint)
