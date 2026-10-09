"""SASRec 商品目录及官方移位监督/负采样的数据适配。"""

import hashlib

import torch

from src.data.components.artifacts import load_model_output


class ItemCatalog:
    """固定原始 key -> 1..N；真实商品0与模型 padding0分离。"""

    def __init__(self, keys: torch.Tensor) -> None:
        if keys.ndim != 1 or keys.dtype not in (torch.int32, torch.int64):
            raise ValueError("Item catalog keys must be a one-dimensional integer tensor.")
        keys = keys.detach().cpu().long().sort().values
        if keys.numel() == 0 or (keys < 0).any() or keys.unique().numel() != keys.numel():
            raise ValueError("Item catalog keys must be nonempty, unique and nonnegative.")
        self.keys = keys
        self.sha256 = hashlib.sha256(b"sasrec-item-catalog-v1\0" + keys.numpy().astype("<i8").tobytes()).hexdigest()

    def __len__(self) -> int:
        return self.keys.numel()

    def encode(self, raw_ids: torch.Tensor) -> torch.Tensor:
        if raw_ids.dtype not in (torch.int32, torch.int64):
            raise ValueError("Raw item IDs must be integers.")
        keys = self.keys.to(raw_ids.device)
        positions = torch.searchsorted(keys, raw_ids.contiguous())
        if (positions >= len(self)).any() or not torch.equal(keys[positions.clamp_max(len(self) - 1)], raw_ids):
            raise ValueError("Sequence contains an item absent from the fixed catalog.")
        return positions + 1

    def decode(self, model_ids: torch.Tensor) -> torch.Tensor:
        if model_ids.dtype not in (torch.int32, torch.int64):
            raise ValueError("Model item IDs must be integers.")
        if (model_ids < 1).any() or (model_ids > len(self)).any():
            raise ValueError("Only nonpadding model item IDs can be decoded.")
        return self.keys.to(model_ids.device)[model_ids.long() - 1]


def load_sasrec_catalog(
    item_catalog_path: str, wandb_entity: str | None = None, wandb_project: str | None = None
) -> ItemCatalog:
    """仅读 bundle keys；W&B URI 使用显式 role，不消费 SID/内容数值。"""
    bundle = load_model_output(
        item_catalog_path,
        field_name="item_catalog_path",
        wandb_entity=wandb_entity,
        wandb_project=wandb_project,
    )
    return ItemCatalog(bundle.keys.reshape(-1))


def sample_sasrec_negatives(num_items: int, excluded: torch.Tensor, count: int) -> torch.Tensor:
    """等价于官方 uniform rejection；密集排除时有界回退，使用 worker torch RNG。"""
    if num_items < 1 or count < 1:
        raise ValueError("Negative sampling requires positive item and sample counts.")
    excluded = excluded.long().unique()
    if (excluded < 1).any() or (excluded > num_items).any():
        raise ValueError("Excluded IDs must be within 1..num_items.")
    if excluded.numel() == num_items:
        raise ValueError("Training sequence leaves no valid negative items.")
    samples = torch.randint(1, num_items + 1, (count,), device=excluded.device)
    for _ in range(16):
        invalid = torch.isin(samples, excluded)
        if not invalid.any():
            return samples
        samples[invalid] = torch.randint(1, num_items + 1, (int(invalid.sum()),), device=excluded.device)
    invalid = torch.isin(samples, excluded)
    candidates = torch.arange(1, num_items + 1, device=excluded.device)
    candidates = candidates[~torch.isin(candidates, excluded)]
    samples[invalid] = candidates[torch.randint(len(candidates), (int(invalid.sum()),), device=excluded.device)]
    return samples


class SASRecPreprocessor:
    """训练保留逐位置标签；评估保持一用户一目标，不改变输入 split。"""

    def __init__(self, catalog: ItemCatalog, max_history_items: int = 50, training: bool = False) -> None:
        if max_history_items < 1:
            raise ValueError("max_history_items must be positive.")
        self.catalog = catalog
        self.max_history_items = max_history_items
        self.training = training

    def _left_pad(self, sequence: torch.Tensor) -> torch.Tensor:
        sequence = sequence[-self.max_history_items :]
        result = sequence.new_zeros(self.max_history_items)
        result[-sequence.numel() :] = sequence
        return result

    def __call__(self, row: dict) -> dict[str, torch.Tensor] | None:
        if "sequence_data" not in row:
            raise ValueError("SASRec record requires sequence_data.")
        sequence = torch.as_tensor(row["sequence_data"])
        if sequence.ndim != 1:
            raise ValueError("SASRec sequence_data must be one-dimensional.")
        if sequence.numel() < 2:
            if self.training:
                return None
            raise ValueError("Evaluation requires a nonempty history and a target item.")
        sequence = self.catalog.encode(sequence)
        result = {"input_ids": self._left_pad(sequence[:-1])}
        if self.training:
            result["target_ids"] = self._left_pad(sequence[1:])
            negatives = sample_sasrec_negatives(
                len(self.catalog), sequence, min(sequence.numel() - 1, self.max_history_items)
            )
            result["negative_ids"] = self._left_pad(negatives)
        else:
            result["target_ids"] = sequence[-1]
            if "user_id" not in row:
                raise ValueError("Evaluation requires user_id for output alignment.")
        if "user_id" in row:
            user_id = torch.as_tensor(row["user_id"])
            if user_id.numel() != 1 or user_id.dtype not in (torch.int32, torch.int64):
                raise ValueError("user_id must contain one integer key.")
            result["user_id"] = user_id.long().reshape(())
        return result
