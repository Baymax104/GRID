"""LETTER CF专用商品目录与training-only负采样。"""

import hashlib

import torch

from src.data.components.artifacts import load_model_output


class LetterCFCatalog:
    def __init__(self, keys):
        keys = torch.as_tensor(keys)
        if keys.ndim != 1 or keys.dtype not in (torch.int32, torch.int64) or len(keys) < 10:
            raise ValueError("CF requires at least ten integer item keys.")
        if (keys < 0).any() or len(keys.unique()) != len(keys):
            raise ValueError("CF keys must be unique and nonnegative.")
        self.keys = keys.sort().values.cpu().long()
        self.sha256 = hashlib.sha256(self.keys.numpy().astype("<i8").tobytes()).hexdigest()

    def model_ids(self, raw):
        raw = torch.as_tensor(raw)
        if raw.ndim != 1 or raw.dtype not in (torch.int32, torch.int64):
            raise ValueError("CF sequence requires one-dimensional integer keys.")
        indices = torch.searchsorted(self.keys, raw.contiguous())
        if (indices >= len(self.keys)).any() or not torch.equal(self.keys[indices.clamp_max(len(self.keys) - 1)], raw):
            raise ValueError("CF sequence contains unknown items.")
        return indices + 1


def load_letter_cf_catalog(embedding_path, wandb_entity=None, wandb_project=None):
    bundle = load_model_output(
        embedding_path, field_name="embedding_path", wandb_entity=wandb_entity, wandb_project=wandb_project
    )
    return LetterCFCatalog(bundle.keys.reshape(-1))


class LetterCFPreprocessor:
    def __init__(self, catalog, max_history_items=50, training=False):
        if max_history_items < 1:
            raise ValueError("CF history must be positive.")
        self.catalog, self.max_history_items, self.training = catalog, max_history_items, training

    def __call__(self, row):
        ids = self.catalog.model_ids(row["sequence_data"])
        if len(ids) < 2:
            if self.training:
                return []
            raise ValueError("CF evaluation requires nonempty history.")
        history = ids[:-1][-self.max_history_items :]
        inputs = torch.zeros(self.max_history_items, dtype=torch.long)
        inputs[-len(history) :] = history
        if not self.training:
            return {"input_ids": inputs, "target": self.catalog.keys[ids[-1] - 1]}
        positives = torch.zeros_like(inputs)
        positives[-len(history) :] = ids[1:][-self.max_history_items :]
        # 排除完整training行，不使用evaluation/testing，也不漏掉被截断商品。
        allowed = torch.ones(len(self.catalog.keys) + 1, dtype=torch.bool)
        allowed[0] = False
        allowed[ids] = False
        candidates = allowed.nonzero().flatten()
        if not len(candidates):
            raise ValueError("CF training row leaves no legal negative item.")
        negatives = torch.zeros_like(inputs)
        negatives[-len(history) :] = candidates[torch.randint(len(candidates), (len(history),))]
        return {"input_ids": inputs, "positive_ids": positives, "negative_ids": negatives}


def letter_cf_collate(rows):
    return {key: torch.stack([row[key] for row in rows]) for key in rows[0]}
