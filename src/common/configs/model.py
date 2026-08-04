from collections.abc import Callable
from dataclasses import dataclass

import torch
from torch import nn


@dataclass
class TrainingModelConfig:
    """Container for model training dependencies instantiated from config."""

    loss_function: nn.Module | None = None
    optimizer: Callable[..., torch.optim.Optimizer] | None = None
    scheduler: Callable[..., torch.optim.lr_scheduler.LRScheduler] | None = None
    reconstruction_loss_function: nn.Module | None = None
