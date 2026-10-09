"""LETTER专用数据适配；只依赖公共bundle读取协议。"""

import hashlib

import torch
from torch.nn.utils.rnn import pad_sequence

from src.data.components.artifacts import load_model_output
from src.data.datasets import SequenceDataset


class LetterSequenceDataset(SequenceDataset):
    """复用公共reader/flat-map，空训练分片显式失败避免无界空循环。"""

    def __iter__(self):
        while True:
            produced = False
            for row in self._load_data():
                for sample in self._apply_preprocessing_functions(row):
                    produced = True
                    yield sample
            if not self.is_for_training:
                return
            if not produced:
                raise ValueError("LETTER training shard has no usable sequence samples.")
            self.cycle_count += 1


class LetterCatalog:
    def __init__(self, keys, semantic_ids, codebook_size=256, base_vocab_size=32100):
        keys, semantic_ids = torch.as_tensor(keys), torch.as_tensor(semantic_ids)
        if keys.ndim != 1 or keys.dtype not in (torch.int32, torch.int64) or not len(keys):
            raise ValueError("LETTER requires nonempty integer item keys.")
        if keys.unique().numel() != len(keys) or (keys < 0).any():
            raise ValueError("LETTER item keys must be unique and nonnegative.")
        if semantic_ids.shape != (len(keys), 4) or semantic_ids.dtype not in (torch.int32, torch.int64):
            raise ValueError("LETTER requires four integer codes per item.")
        if (semantic_ids < 0).any() or (semantic_ids >= codebook_size).any():
            raise ValueError("LETTER code outside codebook.")
        if semantic_ids.unique(dim=0).shape[0] != len(keys):
            raise ValueError("LETTER requires unique full SIDs.")
        order = keys.argsort()
        self.keys, self.semantic_ids = keys[order].cpu().long(), semantic_ids[order].cpu().long()
        self.codebook_size, self.base_vocab_size = codebook_size, base_vocab_size
        names = sorted({f"<{chr(97 + i)}_{int(c)}>" for row in self.semantic_ids.tolist() for i, c in enumerate(row)})
        mapping = {name: base_vocab_size + index for index, name in enumerate(names)}
        self.tokens = torch.tensor(
            [[mapping[f"<{chr(97 + i)}_{int(c)}>"] for i, c in enumerate(row)] for row in self.semantic_ids.tolist()]
        )
        self.sha256 = hashlib.sha256(
            self.keys.numpy().astype("<i8").tobytes() + self.semantic_ids.numpy().astype("<i8").tobytes()
        ).hexdigest()

    def positions(self, raw):
        if raw.dtype not in (torch.int32, torch.int64):
            raise ValueError("Raw item keys must be integers.")
        keys = self.keys.to(raw.device)
        positions = torch.searchsorted(keys, raw.contiguous())
        if (positions >= len(keys)).any() or not torch.equal(keys[positions.clamp_max(len(keys) - 1)], raw):
            raise ValueError("Sequence contains an item absent from LETTER catalog.")
        return positions


def load_letter_catalog(
    semantic_id_path, codebook_size=256, base_vocab_size=32100, wandb_entity=None, wandb_project=None
):
    bundle = load_model_output(
        semantic_id_path, field_name="semantic_id_path", wandb_entity=wandb_entity, wandb_project=wandb_project
    )
    return LetterCatalog(bundle.keys.reshape(-1), bundle.predictions, codebook_size, base_vocab_size)


def load_letter_embeddings(embedding_path, cf_embedding_path, wandb_entity=None, wandb_project=None):
    content = load_model_output(
        embedding_path, field_name="embedding_path", wandb_entity=wandb_entity, wandb_project=wandb_project
    )
    cf = load_model_output(
        cf_embedding_path, field_name="cf_embedding_path", wandb_entity=wandb_entity, wandb_project=wandb_project
    )
    keys, cf_keys = content.keys.reshape(-1), cf.keys.reshape(-1)
    if not len(keys) or any(raw.dtype not in (torch.int32, torch.int64) or (raw < 0).any() for raw in (keys, cf_keys)):
        raise ValueError("LETTER embeddings require nonempty nonnegative integer item keys.")
    if not torch.equal(keys, cf_keys):
        raise ValueError("LETTER content and CF must cover exactly the same keyed catalog.")
    if content.predictions.ndim != 2 or cf.predictions.shape != (len(keys), 32):
        raise ValueError("LETTER requires matrix content and 32-dimensional CF embeddings.")
    if not content.predictions.isfinite().all() or not cf.predictions.isfinite().all():
        raise ValueError("LETTER embeddings must be finite.")
    return {"keys": keys.long(), "content": content.predictions.float(), "cf": cf.predictions.float()}


class LetterPreprocessor:
    def __init__(self, catalog, max_history_items=20, training=False):
        if max_history_items < 1:
            raise ValueError("History length must be positive.")
        self.catalog, self.max_history_items, self.training = catalog, max_history_items, training

    def __call__(self, row):
        raw = torch.as_tensor(row["sequence_data"])
        if raw.ndim != 1:
            raise ValueError("LETTER sequence must be one-dimensional.")
        positions = self.catalog.positions(raw)
        if len(raw) < 2:
            if self.training:
                return []
            raise ValueError("Evaluation requires history and target.")
        if not self.training and "user_id" not in row:
            raise ValueError("Evaluation requires a user key.")
        user = torch.as_tensor(row.get("user_id", -1))
        if user.numel() != 1 or user.dtype not in (torch.int32, torch.int64):
            raise ValueError("User key must contain one integer.")
        result = []
        for end in range(1, len(raw)) if self.training else [len(raw) - 1]:
            history = self.catalog.tokens[positions[max(0, end - self.max_history_items) : end]].flatten()
            result.append(
                {
                    "input_ids": torch.cat((history, torch.tensor([1]))),
                    "labels": torch.cat((self.catalog.tokens[positions[end]], torch.tensor([1]))),
                    "target": raw[end].long(),
                    "user_id": user.long().reshape(()),
                }
            )
        return result


def letter_collate(rows):
    inputs = pad_sequence([row["input_ids"] for row in rows], batch_first=True, padding_value=0)
    return {
        "input_ids": inputs,
        "attention_mask": inputs.ne(0).long(),
        "labels": torch.stack([row["labels"] for row in rows]),
        "target": torch.stack([row["target"] for row in rows]),
        "user_id": torch.stack([row["user_id"] for row in rows]),
    }
