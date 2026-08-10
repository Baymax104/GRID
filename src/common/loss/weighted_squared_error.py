import torch


class WeightedSquaredError(torch.nn.Module):
    def __init__(self):
        """Initialize the WeightedSquaredError loss function."""
        super().__init__()

    def forward(self, x: torch.Tensor, y: torch.Tensor, weights: torch.Tensor | None = None) -> torch.Tensor:
        """
        Compute the weighted squared error loss.

        Args:
            x: Predicted values of shape (n_points, n_features)
            y: Target values of shape (n_points, n_features)
            weights: Weights for each point of shape (n_points,)

        Returns:
            A tensor containing the weighted squared error loss of shape (1,)
        """
        error = x - y
        squared_error = torch.sum(error**2, dim=-1)
        if weights is None:
            return torch.sum(squared_error)
        return torch.sum(weights * squared_error)
