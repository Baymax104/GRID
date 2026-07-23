from collections.abc import Callable

import torch
import torch.nn.functional as F
from torch import nn

from src.utils.distributed_utils import broadcast_from_rank_zero, get_distributed_rank


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


class KMeansLayer(nn.Module):
    """Single-layer K-Means centroid and assignment computation for RKMeans."""

    def __init__(
        self,
        n_clusters: int,
        n_features: int,
        init_buffer_size: int = 1000,
    ):
        super().__init__()
        self.n_clusters = n_clusters
        self.n_features = n_features
        self.init_buffer_size = init_buffer_size

        self.centroids = nn.Parameter(torch.zeros(n_clusters, n_features), requires_grad=True)
        self.init_buffer = torch.tensor([])
        self.is_initialized = False

    def reset_training_state(self, device: torch.device | str):
        """Reset runtime training state on the target device."""
        self.init_buffer = torch.tensor([], device=device)

    def forward(self, residuals: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
        """Initialize or compute assignment statistics for current residuals."""
        residuals = residuals.to(self.centroids.device)

        if not self.is_initialized:
            return self._initialization_step(residuals)

        centroids = self.centroids.data
        distances = _compute_squared_euclidean_distance(residuals, centroids)
        ids = torch.argmin(distances, dim=1)
        assignments_one_hot = F.one_hot(ids, self.n_clusters).detach()
        batch_cluster_counts = torch.sum(assignments_one_hot, dim=0)
        batch_cluster_sums = torch.mm(assignments_one_hot.float().t(), residuals)
        return ids, self.centroids[ids], batch_cluster_counts, batch_cluster_sums

    def predict(self, residuals: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return nearest centroid ids and embeddings without updating layer state."""
        residuals = residuals.to(self.centroids.device)
        with torch.no_grad():
            centroids = self.centroids.data
            distances = _compute_squared_euclidean_distance(residuals, centroids)
            assignments = torch.argmin(distances, dim=1)
            return assignments, centroids[assignments]

    def _buffer_points(self, batch: torch.Tensor):
        batch = batch.detach()
        n_to_add = min(self.init_buffer_size - self.init_buffer.shape[0], batch.shape[0])
        self.init_buffer = torch.cat([self.init_buffer, batch[:n_to_add]], dim=0)

    def _initialization_step(
        self, batch: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
        self._buffer_points(batch)

        if self.init_buffer.shape[0] < self.init_buffer_size:
            batch_zero_embeddings = torch.zeros_like(batch, dtype=batch.dtype, device=self.centroids.device)
            batch_zero_assignments = torch.zeros(batch.shape[0], dtype=torch.long, device=self.centroids.device)
            return batch_zero_assignments, batch_zero_embeddings, None, None

        if self.init_buffer.shape[0] < self.n_clusters:
            raise ValueError(
                f"Buffer size {self.init_buffer.shape[0]} is less than the number of clusters {self.n_clusters}."
            )
        if get_distributed_rank() != 0:
            initial_centroids = torch.zeros_like(self.centroids.data)
        else:
            initial_centroids = _kmeans_plus_plus_init(self.init_buffer, self.n_clusters)
        initial_centroids = broadcast_from_rank_zero(initial_centroids)
        with torch.no_grad():
            self.centroids.copy_(initial_centroids)
        self.is_initialized = True
        self.init_buffer = torch.tensor([], device=self.centroids.device)

        distances = _compute_squared_euclidean_distance(batch, self.centroids.data)
        assignments = torch.argmin(distances, dim=1).to(self.centroids.device)
        return assignments, self.centroids[assignments], None, None
