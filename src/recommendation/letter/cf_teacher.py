"""LETTER专用独立SASRec teacher；依据kang205/SASRec官方公式。"""

import math

import torch
from lightning import LightningModule
from torch import nn

from src.common.configs.model import TrainingModelConfig
from src.data.components.data_models import ModelOutput


class LetterCFBlock(nn.Module):
    def __init__(self, hidden_size=32, dropout=0.5):
        super().__init__()
        self.attention_norm = nn.LayerNorm(hidden_size, eps=1e-8)
        self.ffn_norm = nn.LayerNorm(hidden_size, eps=1e-8)
        self.query = nn.Linear(hidden_size, hidden_size)
        self.key = nn.Linear(hidden_size, hidden_size)
        self.value = nn.Linear(hidden_size, hidden_size)
        self.attention_dropout = nn.Dropout(dropout)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size),
            nn.Dropout(dropout),
        )

    def forward(self, x, valid):
        queries = self.attention_norm(x)
        logits = self.query(queries) @ self.key(x).transpose(-1, -2) / math.sqrt(x.shape[-1])
        causal = torch.ones(x.shape[1], x.shape[1], device=x.device, dtype=torch.bool).tril()
        # 官方有限mask避免全padding query产生NaN；masked query输出最终清零。
        logits = logits.masked_fill(~(causal & valid[:, None, :]), -(2**32) + 1)
        weights = logits.softmax(-1) * queries.abs().sum(-1).ne(0).unsqueeze(-1)
        x = self.attention_dropout(weights) @ self.value(x) + queries
        normalized = self.ffn_norm(x)
        return (normalized + self.ffn(normalized)) * valid.unsqueeze(-1)


class LetterCFTeacher(LightningModule):
    def __init__(self, catalog, training_model_config=None, max_history_items=50, num_blocks=2, dropout=0.5):
        super().__init__()
        if max_history_items < 1 or num_blocks < 1 or not 0 <= dropout < 1:
            raise ValueError("Invalid LETTER CF architecture.")
        self.training_model_config = training_model_config or TrainingModelConfig()
        self.identity = {
            "catalog_sha256": catalog.sha256,
            "hidden_size": 32,
            "max_history_items": max_history_items,
            "num_blocks": num_blocks,
            "dropout": dropout,
            "protocol": "letter-cf-sasrec32-grid-v1",
        }
        self.register_buffer("item_keys", catalog.keys, persistent=False)
        self.items = nn.Embedding(len(catalog.keys) + 1, 32, padding_idx=0)
        self.positions = nn.Embedding(max_history_items, 32)
        self.dropout = nn.Dropout(dropout)
        self.blocks = nn.ModuleList([LetterCFBlock(32, dropout) for _ in range(num_blocks)])
        self.final_norm = nn.LayerNorm(32, eps=1e-8)
        # TF默认embedding/linear为Glorot；不借用其他模型的初始化实现。
        for module in self.modules():
            if isinstance(module, (nn.Embedding, nn.Linear)):
                nn.init.xavier_uniform_(module.weight)
                if isinstance(module, nn.Linear):
                    nn.init.zeros_(module.bias)
        with torch.no_grad():
            self.items.weight[0].zero_()

    def encode(self, input_ids):
        valid = input_ids.ne(0)
        positions = torch.arange(input_ids.shape[1], device=input_ids.device)
        x = self.dropout(self.items(input_ids) * math.sqrt(32) + self.positions(positions)) * valid.unsqueeze(-1)
        for block in self.blocks:
            x = block(x, valid)
        return self.final_norm(x)

    def training_step(self, batch, batch_idx):
        x = self.encode(batch["input_ids"])
        valid = batch["positive_ids"].ne(0)
        if not valid.any():
            raise ValueError("CF training batch has no positive supervision.")
        pos = (x * self.items(batch["positive_ids"])).sum(-1)[valid]
        neg = (x * self.items(batch["negative_ids"])).sum(-1)[valid]
        loss = (-torch.log(pos.sigmoid() + 1e-24) - torch.log(1 - neg.sigmoid() + 1e-24)).mean()
        return {"loss": loss}

    def validation_step(self, batch, batch_idx):
        # 稳定排序使同分按共同目录raw key升序；目录规模小于两万个。
        scores = self.encode(batch["input_ids"])[:, -1] @ self.items.weight[1:].T
        indices = scores.argsort(dim=-1, descending=True, stable=True)[:, :10]
        return {"generated_ids": self.item_keys[indices], "labels": batch["target"], "user_count": len(indices)}

    def predict_step(self, batch, batch_idx=0):
        positions = batch[0]
        return ModelOutput(keys=self.item_keys[positions].cpu(), predictions=self.items(positions + 1).detach().cpu())

    def configure_optimizers(self):
        if self.training_model_config.optimizer is None:
            raise ValueError("CF optimizer must be configured.")
        return self.training_model_config.optimizer(params=self.parameters())

    def on_save_checkpoint(self, checkpoint):
        checkpoint["letter_cf_identity"] = self.identity

    def on_load_checkpoint(self, checkpoint):
        if checkpoint.get("letter_cf_identity") != self.identity:
            raise ValueError("LETTER CF checkpoint catalog or protocol mismatch.")
