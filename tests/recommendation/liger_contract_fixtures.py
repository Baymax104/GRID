from functools import partial

import torch

from src.recommendation.liger.module import Liger


def catalog():
    return dict(
        keys=torch.arange(12, dtype=torch.long) * 10,
        semantic_ids=torch.tensor([[row // 4, row % 4] for row in range(12)]),
        embeddings=torch.arange(60).reshape(12, 5).float() / 60,
        seen_mask=torch.tensor([True] * 10 + [False] * 2),
    )


def factories():
    shared = dict(
        catalog=None,
        num_hierarchies=2,
        codebook_size=4,
        embedding_dim=8,
        num_layers=1,
        num_heads=2,
        d_kv=4,
        d_ff=16,
        dropout=0.2,
        max_history_items=3,
        projection_hidden_sizes=(7, 6),
        projection_dropout=0.2,
        input_dropout=0.5,
        generation_candidates=3,
        top_k=10,
        catalog_chunk_size=2,
        evaluation_mode="dense",
        prediction_mode="dense",
        residual_lr_multiplier=20,
    )
    return (partial(Liger, **shared),)
