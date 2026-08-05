import torch


def add_hierarchy_offsets(
    semantic_ids: torch.Tensor,
    codebook_size: int,
    num_hierarchies: int,
    attention_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Map per-hierarchy semantic IDs into a shared embedding table index space."""
    if semantic_ids.ndim != 2:
        raise ValueError("semantic_ids must be 2-dimensional.")

    _, num_cols = semantic_ids.shape
    offsets = torch.arange(num_hierarchies, device=semantic_ids.device) * codebook_size
    num_repeats = (num_cols + num_hierarchies - 1) // num_hierarchies
    repeated_offsets = offsets.repeat(num_repeats)[:num_cols]

    shifted_semantic_ids = semantic_ids + repeated_offsets
    if attention_mask is not None:
        shifted_semantic_ids = shifted_semantic_ids * attention_mask
    return shifted_semantic_ids


def insert_separator_token_between_items(
    id_embeddings: torch.Tensor,
    attention_mask: torch.Tensor,
    sep_token: torch.Tensor,
    num_hierarchies: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Append a separator embedding after each item's hierarchy-level SID embeddings."""
    batch_size, seq_len, emb_dim = id_embeddings.size()
    item_count_per_sequence = seq_len // num_hierarchies

    reshaped_id_embeddings = id_embeddings.view(batch_size, item_count_per_sequence, num_hierarchies, -1)
    reshaped_attention_mask = attention_mask.view(batch_size, item_count_per_sequence, num_hierarchies)
    reshaped_sep_token = sep_token.unsqueeze(0).expand(batch_size, item_count_per_sequence, -1).unsqueeze(-2)

    id_embeddings = torch.cat([reshaped_id_embeddings, reshaped_sep_token], dim=-2)
    attention_mask = torch.cat([reshaped_attention_mask, reshaped_attention_mask[:, :, [-1]]], dim=-1)

    id_embeddings = id_embeddings.reshape(batch_size, -1, emb_dim)
    attention_mask = attention_mask.reshape(batch_size, -1)
    return id_embeddings, attention_mask
