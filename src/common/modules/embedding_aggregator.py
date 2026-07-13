import torch
import torch.nn as nn

from src.utils.model_utils import create_last_k_mask


class EmbeddingAggregator(nn.Module):
    """Embedding aggregator that computes mean aggregation over token embeddings.

    Args:
        last_k: If specified, only the last K embeddings are considered for aggregation.
    """

    def __init__(self, last_k: int | None = None):
        super().__init__()
        self.last_k = last_k

    def forward(
        self,
        embeddings: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        # embeddings: (batch_size, sequence_length, embedding_dim)
        # attention_mask: (batch_size, sequence_length)

        last_item_index = attention_mask.sum(dim=1) - 1

        # row_ids = [0, 1, 2, ..., batch_size - 1] (traceable with Fx)
        dummy_tensor_for_batch_shape = attention_mask[:, 0]
        ones_tensor = torch.ones_like(dummy_tensor_for_batch_shape, dtype=torch.long)
        row_ids = torch.cumsum(ones_tensor, dim=0) - 1

        return _mean_aggregate(embeddings, row_ids, last_item_index, self.last_k)


def _mean_aggregate(
    embeddings: torch.Tensor,
    row_ids: torch.Tensor,
    last_item_index: torch.Tensor,
    last_k: int | None = None,
) -> torch.Tensor:
    """Aggregate embeddings by computing their mean over the last K tokens per row."""
    embeddings = embeddings[row_ids]
    mask = create_last_k_mask(embeddings.size(1), last_item_index, last_k)
    mask = mask.to(dtype=embeddings.dtype, device=embeddings.device)

    masked_embeddings = embeddings * mask.unsqueeze(2)
    sum_embeddings = torch.sum(masked_embeddings, dim=1)
    count = torch.sum(mask, dim=1).clamp(min=1).unsqueeze(1)
    return sum_embeddings / count
