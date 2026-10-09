"""商品 ID 排名转换为共享 Recall/NDCG 的输入协议。"""

from collections.abc import Mapping
from typing import Any

import torch


def item_retrieval_inputs(payload: Mapping[str, Any]) -> dict[str, torch.Tensor]:
    predictions = payload["generated_ids"]
    scores = payload["scores"]
    labels = payload["labels"].to(predictions.device)
    if predictions.ndim != 2 or scores.shape != predictions.shape or labels.shape != predictions.shape[:1]:
        raise ValueError("Item metrics require predictions/scores [B,K] and labels [B].")
    indexes = torch.arange(len(labels), device=predictions.device)[:, None].expand_as(predictions)
    rank_scores = torch.arange(predictions.shape[1], 0, -1, device=predictions.device).expand_as(predictions).float()
    return {
        # 输入已按最终顺序排列，用唯一 rank score 防止 metric 内 topk 重排同分。
        "preds": rank_scores.reshape(-1),
        "target": predictions.eq(labels[:, None]).reshape(-1),
        "indexes": indexes.reshape(-1),
    }
