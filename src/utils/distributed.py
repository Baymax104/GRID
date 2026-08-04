import torch
import torch.distributed as dist


def is_distributed_initialized() -> bool:
    """Return whether torch distributed is ready for collective calls."""
    return dist.is_available() and dist.is_initialized()


def get_distributed_rank() -> int:
    """Return the current distributed rank, or zero in single-process mode."""
    if not is_distributed_initialized():
        return 0
    return dist.get_rank()


def broadcast_from_rank_zero(tensor: torch.Tensor) -> torch.Tensor:
    """Broadcast a tensor from rank zero when distributed is initialized."""
    if is_distributed_initialized():
        dist.broadcast(tensor, src=0)
    return tensor
