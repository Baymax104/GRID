"""按 item key 对齐固定目录的 SID 与内容，不读取交互数据。"""

import torch

from src.data.components.artifacts import load_model_output


def load_catalog_content(
    semantic_id_path: str,
    embedding_path: str,
    wandb_entity: str | None = None,
    wandb_project: str | None = None,
) -> dict[str, torch.Tensor]:
    options = {"wandb_entity": wandb_entity, "wandb_project": wandb_project}
    sid = load_model_output(semantic_id_path, field_name="semantic_id_path", **options)
    content = load_model_output(embedding_path, field_name="embedding_path", **options)
    keys, order = sid.keys.reshape(-1).long().sort()
    content_keys, content_order = content.keys.reshape(-1).long().sort()
    if keys.numel() == 0 or content_keys.numel() == 0:
        raise ValueError("Catalog bundles must be nonempty.")
    if keys.unique().numel() != keys.numel() or content_keys.unique().numel() != content_keys.numel():
        raise ValueError("Catalog bundles contain duplicate item keys.")
    positions = torch.searchsorted(content_keys, keys).clamp(max=content_keys.numel() - 1)
    if not torch.equal(content_keys[positions], keys):
        raise ValueError("Embedding bundle is missing semantic-ID catalog item keys.")
    return {
        "keys": keys,
        "semantic_ids": sid.predictions[order],
        "embeddings": content.predictions[content_order[positions]],
    }
