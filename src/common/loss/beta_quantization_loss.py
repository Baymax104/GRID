import torch


class BetaQuantizationLoss(torch.nn.Module):
    def __init__(self, beta: float = 0.25, reduction: str = "sum"):
        """Initialize the Beta Quantization Loss.

        Parameters
        ----------
        beta: float
            Weighting factor for the reconstruction loss.
        reduction: str
            Reduction method to apply to the loss. Options are 'none', 'mean', and 'sum'.
        """
        super().__init__()
        self.beta = beta
        self.criterion = torch.nn.MSELoss(reduction=reduction)

    def forward(self, x: torch.Tensor, xq: torch.Tensor) -> torch.Tensor:
        """
        Compute the beta quantization loss.
        Args:
            x: Original tensor of shape (batch_size, n_features)
            xq: Quantized tensor of shape (batch_size, n_features)
        Returns:
            A tensor containing the beta quantization loss of shape (1,)
        """
        x_no_grad = x.detach()
        xq_no_grad = xq.detach()
        loss = self.criterion(x_no_grad, xq) + self.beta * self.criterion(x, xq_no_grad)
        return loss
