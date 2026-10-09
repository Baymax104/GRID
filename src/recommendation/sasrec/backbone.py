"""作者 SASRec 的 PyTorch 转写：保留 Q-only LN、residual 和逐位置 BCE。"""

import math

import torch
from torch import nn
from torch.nn import functional as F

OFFICIAL_COMMIT = "e3738967fddab206d6eeb4fda433e7a7034dd8b1"
OFFICIAL_SOURCE = f"https://github.com/kang205/SASRec/tree/{OFFICIAL_COMMIT}"


class SASRecBlock(nn.Module):
    """对应 modules.py 的 multihead_attention 与 feedforward；无 output projection。"""

    def __init__(self, hidden_size: int, num_heads: int, dropout: float) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.attention_norm = nn.LayerNorm(hidden_size, eps=1e-8)
        self.query = nn.Linear(hidden_size, hidden_size)
        self.key = nn.Linear(hidden_size, hidden_size)
        self.value = nn.Linear(hidden_size, hidden_size)
        self.attention_dropout = nn.Dropout(dropout)
        self.feedforward_norm = nn.LayerNorm(hidden_size, eps=1e-8)
        # Linear 等价于作者 kernel_size=1 的 Conv1D，保持 hidden -> hidden。
        self.feedforward_in = nn.Linear(hidden_size, hidden_size)
        self.feedforward_out = nn.Linear(hidden_size, hidden_size)
        self.feedforward_dropout_in = nn.Dropout(dropout)
        self.feedforward_dropout_out = nn.Dropout(dropout)

    def forward(self, inputs: torch.Tensor, item_mask: torch.Tensor) -> torch.Tensor:
        batch_size, length, hidden_size = inputs.shape
        head_size = hidden_size // self.num_heads
        queries = self.attention_norm(inputs)

        def split_heads(values: torch.Tensor) -> torch.Tensor:
            return values.reshape(batch_size, length, self.num_heads, head_size).transpose(1, 2)

        query = split_heads(self.query(queries))
        key = split_heads(self.key(inputs))
        value = split_heads(self.value(inputs))
        logits = query @ key.transpose(-1, -2) / math.sqrt(head_size)
        # 作者按投影前的输入向量检测 padding；保留这个公式。
        key_mask = inputs.abs().sum(-1).ne(0)
        causal_mask = torch.ones(length, length, device=inputs.device, dtype=torch.bool).tril()
        allowed = key_mask[:, None, None, :] & causal_mask[None, None]
        weights = logits.masked_fill(~allowed, float(-(2**32) + 1)).softmax(-1)
        query_mask = queries.abs().sum(-1).ne(0)
        weights = weights * query_mask[:, None, :, None]
        outputs = self.attention_dropout(weights) @ value
        outputs = outputs.transpose(1, 2).reshape(batch_size, length, hidden_size)
        outputs = outputs + queries
        ff_inputs = self.feedforward_norm(outputs)
        outputs = self.feedforward_dropout_in(F.relu(self.feedforward_in(ff_inputs)))
        outputs = self.feedforward_dropout_out(self.feedforward_out(outputs)) + ff_inputs
        return outputs * item_mask.unsqueeze(-1)


class SASRecBackbone(nn.Module):
    """官方算法骨干；商品 ID 为 1..N，左 padding 为 0，位置为固定槽位。"""

    def __init__(
        self,
        num_items: int,
        max_history_items: int = 50,
        hidden_size: int = 50,
        num_blocks: int = 2,
        num_heads: int = 1,
        dropout: float = 0.5,
        l2_emb: float = 0.0,
    ) -> None:
        super().__init__()
        if min(num_items, max_history_items, hidden_size, num_blocks, num_heads) < 1:
            raise ValueError("SASRec dimensions must be positive.")
        if hidden_size % num_heads:
            raise ValueError("hidden_size must be divisible by num_heads.")
        if not 0 <= dropout < 1 or l2_emb < 0:
            raise ValueError("Invalid SASRec dropout or l2_emb.")
        self.num_items = num_items
        self.max_history_items = max_history_items
        self.hidden_size = hidden_size
        self.l2_emb = l2_emb
        self.item_embedding = nn.Embedding(num_items + 1, hidden_size, padding_idx=0)
        self.position_embedding = nn.Embedding(max_history_items, hidden_size)
        self.embedding_dropout = nn.Dropout(dropout)
        self.blocks = nn.ModuleList(SASRecBlock(hidden_size, num_heads, dropout) for _ in range(num_blocks))
        self.final_norm = nn.LayerNorm(hidden_size, eps=1e-8)
        self.apply(self._initialize)

    @staticmethod
    def _initialize(module: nn.Module) -> None:
        if isinstance(module, (nn.Embedding, nn.Linear)):
            nn.init.xavier_uniform_(module.weight)
            if isinstance(module, nn.Linear):
                nn.init.zeros_(module.bias)

    @property
    def item_table(self) -> torch.Tensor:
        # 官方 raw table 的第零行仍参与 L2，lookup table 的第零行恒零。
        weight = self.item_embedding.weight
        return torch.cat((weight.new_zeros(1, self.hidden_size), weight[1:]), dim=0)

    def _validate_ids(self, ids: torch.Tensor, name: str) -> None:
        if ids.dtype not in (torch.int32, torch.int64):
            raise ValueError(f"{name} must contain integer item IDs.")
        if ids.numel() == 0 or (ids < 0).any() or (ids > self.num_items).any():
            raise ValueError(f"{name} contains empty or out-of-range item IDs.")

    def encode(self, input_ids: torch.Tensor) -> torch.Tensor:
        if input_ids.ndim != 2 or input_ids.shape[1] != self.max_history_items:
            raise ValueError("input_ids must have shape [B, max_history_items].")
        self._validate_ids(input_ids, "input_ids")
        mask = input_ids.ne(0)
        if not mask.any(-1).all():
            raise ValueError("SASRec history must be nonempty.")
        if (mask[:, :-1] & ~mask[:, 1:]).any():
            raise ValueError("SASRec requires contiguous left padding.")
        positions = torch.arange(self.max_history_items, device=input_ids.device)
        values = F.embedding(input_ids.long(), self.item_table) * math.sqrt(self.hidden_size)
        values = self.embedding_dropout(values + self.position_embedding(positions)) * mask.unsqueeze(-1)
        for block in self.blocks:
            values = block(values, mask)
        return self.final_norm(values)

    def forward(
        self, input_ids: torch.Tensor, positive_ids: torch.Tensor, negative_ids: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if positive_ids.shape != input_ids.shape or negative_ids.shape != input_ids.shape:
            raise ValueError("Positive and negative IDs must match input shape.")
        self._validate_ids(positive_ids, "positive_ids")
        self._validate_ids(negative_ids, "negative_ids")
        active = positive_ids.ne(0)
        if not active.any() or (active & (input_ids.eq(0) | negative_ids.eq(0))).any():
            raise ValueError("Valid positive labels require nonpadding input and negative IDs.")
        features = self.encode(input_ids)
        table = self.item_table
        positive_logits = (features * F.embedding(positive_ids.long(), table)).sum(-1)
        negative_logits = (features * F.embedding(negative_ids.long(), table)).sum(-1)
        return positive_logits, negative_logits

    def objective(
        self, positive_logits: torch.Tensor, negative_logits: torch.Tensor, positive_ids: torch.Tensor
    ) -> torch.Tensor:
        if positive_logits.shape != positive_ids.shape or negative_logits.shape != positive_ids.shape:
            raise ValueError("Logits and positive labels must have matching shapes.")
        self._validate_ids(positive_ids, "positive_ids")
        active = positive_ids.ne(0)
        if not active.any():
            raise ValueError("SASRec objective requires at least one positive label.")
        positive = positive_logits[active]
        negative = negative_logits[active]
        loss = (-torch.log(positive.sigmoid() + 1e-24) - torch.log(1 - negative.sigmoid() + 1e-24)).mean()
        regularization = (
            0.5
            * self.l2_emb
            * (self.item_embedding.weight.square().sum() + self.position_embedding.weight.square().sum())
        )
        return loss + regularization

    def score_items(self, query: torch.Tensor, item_ids: torch.Tensor) -> torch.Tensor:
        """用共享 embedding 给一组候选打分；全目录分块由上层负责。"""
        self._validate_ids(item_ids, "item_ids")
        if item_ids.ndim != 1 or query.ndim != 2 or query.shape[-1] != self.hidden_size:
            raise ValueError("Expected query [B,D] and item_ids [C].")
        # 分块评分只查当前块，不反复构造完整 N 商品的有效 table。
        embeddings = self.item_embedding(item_ids.long()) * item_ids.ne(0).unsqueeze(-1)
        # 固定沿 hidden 维的归约顺序，避免不同 GEMM 块形状将数学同分变成微小差异。
        return (query[:, None, :] * embeddings[None, :, :]).sum(-1)
