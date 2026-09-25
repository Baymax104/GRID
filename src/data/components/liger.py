"""LIGER 目录与仅来自 training split 的商品集合。"""

from pathlib import Path

import torch

from src.data.components.catalog_content import load_catalog_content
from src.data.components.readers import TFRecordReader


def load_liger_catalog(
    semantic_id_path: str,
    embedding_path: str,
    training_data_dir: str,
    wandb_entity: str | None = None,
    wandb_project: str | None = None,
) -> dict[str, torch.Tensor]:
    catalog = load_catalog_content(semantic_id_path, embedding_path, wandb_entity, wandb_project)
    files = sorted(str(p) for p in Path(training_data_dir).rglob("*.tfrecord.gz"))
    if not files:
        raise ValueError("LIGER requires nonempty training_data_dir to determine seen items.")
    seen = set()
    for row in TFRecordReader(files, shuffle_rows=False).iterrows():
        if "sequence_data" not in row:
            raise ValueError("Training record missing sequence_data.")
        seen.update(int(v) for v in row["sequence_data"].reshape(-1))
    if not seen:
        raise ValueError("Training item set is empty.")
    keys = set(catalog["keys"].tolist())
    if not seen <= keys:
        raise ValueError(f"Training items missing from catalog: {sorted(seen - keys)[:10]}")
    catalog["seen_mask"] = torch.tensor([k in seen for k in catalog["keys"].tolist()], dtype=torch.bool)
    return catalog


def generate_liger_next_item(
    row,
    sequence_field_name="sequence_data",
    input_field_name="input_ids",
    target_field_name="target_ids",
    next_k=4,
):
    """LIGER 历史不包含 TIGER 的目标占位 token，标签是最后一个完整商品。"""
    sequence = row[sequence_field_name]
    if next_k < 1 or sequence.ndim != 1 or len(sequence) < 2 * next_k or len(sequence) % next_k:
        raise ValueError("LIGER requires at least two complete items before label extraction.")
    result = dict(row)
    result[input_field_name] = sequence[:-next_k].clone()
    result[target_field_name] = sequence[-next_k:].clone()
    if sequence_field_name not in {input_field_name, target_field_name}:
        result.pop(sequence_field_name)
    return result
