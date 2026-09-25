"""LIGER 的 GRID 适配：共享内容投影、联合目标、生成后 dense 重排。

算法参考 facebookresearch/liger b6ccc37（CC-BY-NC 4.0），实现与适配说明见
../../../../research/docs/grid-experiments/2026-09-23-liger-baseline.md。
"""

import copy
import hashlib
import math
from typing import Any

import torch
from lightning import LightningModule
from torch import nn
from torch.nn import functional as F
from transformers import T5Config, T5ForConditionalGeneration

from src.common.configs.model import TrainingModelConfig
from src.data.components.data_models import ModelOutput, TigerModelInput
from src.recommendation.liger.candidate_guidance import ProbabilityMixtureProcessor


class LigerT5(T5ForConditionalGeneration):
    """保留官方 wrapper 的初始化，避免默认 T5 按层缩放改变基线。"""

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_factor)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()


class ContentProjection(nn.Module):
    """官方残差 MLP 的等价线性实现，保留 batch=1 维度。"""

    def __init__(self, input_dim, hidden_sizes, output_dim, dropout, eps):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        widths = [input_dim, *hidden_sizes]
        self.blocks = nn.ModuleList(
            [
                nn.Sequential(nn.Linear(a, b), nn.LayerNorm(b, eps=eps), nn.ReLU(), nn.Dropout(dropout))
                for a, b in zip(widths[:-1], widths[1:], strict=True)
            ]
        )
        # 官方 Conv1d(1,b,kernel_size=a) 在单向量上等价于无 bias Linear(a,b)。
        self.residuals = nn.ModuleList(
            [nn.Linear(a, b, bias=False) for a, b in zip(widths[:-1], widths[1:], strict=True)]
        )
        self.output = nn.Linear(widths[-1], output_dim)

    def forward(self, values):
        values = self.dropout(values)
        for block, residual in zip(self.blocks, self.residuals, strict=True):
            values = block(values) + residual(values)
        return self.output(values)


class Liger(LightningModule):
    def __init__(
        self,
        catalog: dict[str, torch.Tensor],
        num_hierarchies: int = 4,
        codebook_size: int = 256,
        embedding_dim: int = 128,
        num_layers: int = 6,
        num_heads: int = 6,
        d_kv: int = 64,
        d_ff: int = 1024,
        dropout: float = 0.2,
        max_history_items: int = 20,
        projection_hidden_sizes: tuple[int, ...] = (768, 512, 256),
        projection_dropout: float = 0.2,
        input_dropout: float = 0.5,
        temperature: float = 0.07,
        sid_loss_weight: float = 1.0,
        content_loss_weight: float = 1.0,
        generation_candidates: int = 20,
        top_k: int = 10,
        evaluation_mode: str = "dense",
        prediction_mode: str = "hybrid",
        catalog_chunk_size: int = 4096,
        candidate_trace: bool = False,
        candidate_strategy: str = "original",
        content_mixture_alpha: float = 0.0,
        training_model_config: TrainingModelConfig | None = None,
    ):
        super().__init__()
        if candidate_strategy not in {"original", "probability_mixture"}:
            raise ValueError("Unknown candidate strategy.")
        self.candidate_strategy = candidate_strategy
        if not math.isfinite(content_mixture_alpha) or not 0 <= content_mixture_alpha <= 1:
            raise ValueError("Invalid content mixture alpha.")
        if candidate_strategy != "probability_mixture" and content_mixture_alpha != 0:
            raise ValueError("Nonzero alpha requires probability_mixture strategy.")
        self.content_mixture_alpha = content_mixture_alpha
        self.dynamic_gate = None
        if min(num_hierarchies, codebook_size, max_history_items, generation_candidates, top_k, catalog_chunk_size) < 1:
            raise ValueError("LIGER dimensions and candidate counts must be positive.")
        if temperature <= 0 or sid_loss_weight < 0 or content_loss_weight < 0:
            raise ValueError("Invalid temperature or loss weights.")
        if sid_loss_weight + content_loss_weight <= 0:
            raise ValueError("At least one loss weight must be positive.")
        if evaluation_mode not in {"dense", "generative", "hybrid"} or prediction_mode not in {
            "dense", "generative", "hybrid"
        }:
            raise ValueError("Unknown LIGER retrieval mode.")
        keys = catalog["keys"].detach().cpu().long().reshape(-1)
        sids = catalog["semantic_ids"].detach().cpu()
        bank = catalog["embeddings"].detach().cpu().float()
        seen = catalog["seen_mask"].detach().cpu()
        if sids.ndim != 2 or sids.shape != (len(keys), num_hierarchies):
            raise ValueError("Catalog requires full semantic IDs including disambiguation digits.")
        if not torch.equal(sids, sids.long()) or ((sids < 0) | (sids >= codebook_size)).any():
            raise ValueError("Catalog contains invalid SID tokens.")
        if not len(keys) or keys.unique().numel() != len(keys) or sids.unique(dim=0).shape[0] != len(keys):
            raise ValueError("Catalog keys and complete SIDs must be nonempty and unique.")
        if bank.ndim != 2 or bank.shape[0] != len(keys) or not torch.isfinite(bank).all():
            raise ValueError("Invalid catalog content embeddings.")
        if seen.shape != keys.shape or seen.dtype != torch.bool or not seen.any():
            raise ValueError("Catalog seen_mask must identify training items.")
        self.num_hierarchies, self.codebook_size = num_hierarchies, codebook_size
        self.candidate_trace = candidate_trace
        self.max_history_items = max_history_items
        self.temperature = temperature
        self.sid_loss_weight, self.content_loss_weight = sid_loss_weight, content_loss_weight
        self.generation_candidates, self.top_k = generation_candidates, top_k
        self.evaluation_mode, self.prediction_mode = evaluation_mode, prediction_mode
        self.catalog_chunk_size = catalog_chunk_size
        self.training_config = training_model_config or TrainingModelConfig()
        self.register_buffer("item_keys", keys)
        self.register_buffer("semantic_ids", sids.long())
        self.register_buffer("content_bank", bank)
        self.register_buffer("seen_mask", seen)
        if codebook_size**num_hierarchies > torch.iinfo(torch.int64).max:
            raise ValueError("SID integer encoding exceeds int64.")
        self.register_buffer("sid_powers", codebook_size ** torch.arange(num_hierarchies - 1, -1, -1))
        codes = (sids.long() * self.sid_powers).sum(-1)
        sorted_codes, order = codes.sort()
        self.register_buffer("sorted_codes", sorted_codes)
        self.register_buffer("sorted_rows", order)
        self.register_buffer("offsets", torch.arange(num_hierarchies) * codebook_size + 1)
        self.eos_id = num_hierarchies * codebook_size + 1
        self.transformer = LigerT5(
            T5Config(
                vocab_size=self.eos_id + 1,
                d_model=embedding_dim,
                num_layers=num_layers,
                num_decoder_layers=num_layers,
                num_heads=num_heads,
                d_kv=d_kv,
                d_ff=d_ff,
                dropout_rate=dropout,
                pad_token_id=0,
                decoder_start_token_id=0,
                eos_token_id=self.eos_id,
                feed_forward_proj="relu",
                layer_norm_epsilon=1e-8,
                initializer_factor=0.02,
            )
        )
        self.content_projection = ContentProjection(
            bank.shape[1],
            list(projection_hidden_sizes),
            embedding_dim,
            projection_dropout,
            1e-8,
        )
        self.item_position = nn.Embedding(max_history_items, embedding_dim)
        self.semantic_position = nn.Embedding(num_hierarchies + 1, embedding_dim)
        self.input_norm = nn.LayerNorm(embedding_dim, eps=1e-8)
        self.input_dropout = nn.Dropout(input_dropout)
        digest = hashlib.sha256()
        for value in (keys, sids.long(), bank, seen):
            digest.update(str(tuple(value.shape)).encode())
            digest.update(value.contiguous().numpy().tobytes())
        self.catalog_sha256 = digest.hexdigest()

    def lookup_rows(self, sids, *, strict=True):
        if sids.shape[-1] != self.num_hierarchies:
            raise ValueError("Incomplete SID.")
        valid = ((sids >= 0) & (sids < self.codebook_size)).all(-1)
        codes = (sids.long() * self.sid_powers).sum(-1)
        positions = torch.searchsorted(self.sorted_codes, codes.reshape(-1)).reshape(codes.shape)
        positions = positions.clamp(max=len(self.item_keys) - 1)
        valid = valid & (self.sorted_codes[positions] == codes)
        if strict and not valid.all():
            raise ValueError("Input or target SID is absent from catalog.")
        return torch.where(valid, self.sorted_rows[positions], -1)

    def encode(self, input_ids, attention_mask):
        if input_ids.ndim != 2 or attention_mask.shape != input_ids.shape:
            raise ValueError("Invalid history batch shape.")
        batch, length = input_ids.shape
        h = self.num_hierarchies
        if length % h or length // h > self.max_history_items:
            raise ValueError("History length must contain complete items within max_history_items.")
        mask = attention_mask.bool()
        grouped_mask = mask.reshape(batch, -1, h)
        if (grouped_mask.any(-1) != grouped_mask.all(-1)).any() or not mask.any(-1).all():
            raise ValueError("History requires nonempty complete unmasked items.")
        if ((~mask[:, :-1]) & mask[:, 1:]).any():
            raise ValueError("LIGER expects right-padded histories.")
        grouped = input_ids.reshape(batch, -1, h)
        valid_items = grouped_mask.all(-1)
        rows = self.lookup_rows(grouped[valid_items])
        contents = self.content_bank.new_zeros(batch, length // h, self.content_bank.shape[-1])
        contents[valid_items] = self.content_bank[rows]
        # 与官方一致：先复制各 SID token 的内容，再执行含 dropout 的投影。
        repeated = contents.repeat_interleave(h, dim=1)
        projected = self.content_projection(repeated)
        item_pos = torch.arange(length, device=input_ids.device) // h
        sem_pos = torch.arange(length, device=input_ids.device) % h
        token_ids = (grouped.long() + self.offsets).reshape(batch, length).masked_fill(~mask, 0)
        embeddings = self.transformer.shared(token_ids) + projected
        embeddings = embeddings + self.item_position(item_pos)[None] + self.semantic_position(sem_pos)[None]
        embeddings = self.input_dropout(self.input_norm(embeddings))
        encoded = self.transformer.encoder(inputs_embeds=embeddings, attention_mask=mask, return_dict=True)
        last = mask.long().sum(-1) - 1
        query = encoded.last_hidden_state[torch.arange(batch, device=input_ids.device), last]
        return encoded, mask, query

    def dense_logits(self, query):
        projected = torch.cat(
            [self.content_projection(part) for part in self.content_bank.split(self.catalog_chunk_size)]
        )
        return F.normalize(query, dim=-1) @ F.normalize(projected, dim=-1).T / self.temperature

    def losses(self, model_input, target_ids, *, training=False):
        rows = self.lookup_rows(target_ids)
        if training and not self.seen_mask[rows].all():
            raise ValueError("Training target is marked as cold-start.")
        encoded, mask, query = self.encode(model_input.input_ids, model_input.attention_mask)
        # 官方 labels 仅含 SID，不附加 EOS；所有位置使用完整词表 CE。
        labels = target_ids.long() + self.offsets
        outputs = self.transformer(encoder_outputs=encoded, attention_mask=mask, labels=labels, use_cache=False)
        logits = self.dense_logits(query)
        if training:
            logits = logits.masked_fill(~self.seen_mask[None], -100.0)
        dense_loss = F.cross_entropy(logits, rows)
        return {
            "loss": self.sid_loss_weight * outputs.loss + self.content_loss_weight * dense_loss,
            "sid_loss": outputs.loss,
            "content_loss": dense_loss,
        }

    def training_step(self, batch, batch_idx):
        model_input, label = batch
        if label is None:
            raise ValueError("LIGER training requires labels.")
        return self.losses(model_input, label.target_ids, training=True)

    def candidate_processor(self, content_logits):
        if self.candidate_strategy == "probability_mixture":
            return ProbabilityMixtureProcessor(
                self.semantic_ids, content_logits, self.codebook_size, self.content_mixture_alpha
            )
        raise ValueError("Original candidates do not use a logits processor.")

    @torch.no_grad()
    def _generate_candidate_rows(self, encoded, mask, *, processor=None, generation_candidates=None):
        generation_candidates = self.generation_candidates if generation_candidates is None else generation_candidates
        extra = {}
        if processor is not None:
            extra = dict(logits_processor=[processor], renormalize_logits=False)
        generated = self.transformer.generate(
            # HF beam search会把ModelOutput中的batch维原地扩展；每次调用使用独立容器，
            # 使同一encoder结果可以安全服务多个固定候选搜索。
            encoder_outputs=copy.copy(encoded),
            attention_mask=mask,
            num_beams=generation_candidates,
            num_return_sequences=generation_candidates,
            max_new_tokens=self.num_hierarchies,
            use_cache=True,
            **extra,
        )[:, 1:]
        # 过早 EOS / PAD 与无效前缀均映射为无效行，不回退到商品0。
        if generated.shape[1] < self.num_hierarchies:
            generated = F.pad(generated, (0, self.num_hierarchies - generated.shape[1]), value=0)
        local = generated[:, : self.num_hierarchies] - self.offsets
        rows = self.lookup_rows(local, strict=False)
        return rows.reshape(mask.shape[0], generation_candidates)

    @torch.no_grad()
    def generate_candidates(self, encoded, mask, content_logits=None):
        processor = None
        if self.candidate_strategy != "original":
            if content_logits is None:
                raise ValueError("Constrained candidates require shared content logits.")
            processor = self.candidate_processor(content_logits)
        rows = self._generate_candidate_rows(encoded, mask, processor=processor)
        return rows

    @torch.no_grad()
    def retrieve(self, model_input, mode=None, target_ids=None):
        mode = mode or self.prediction_mode
        if mode not in {"dense", "generative", "hybrid"}:
            raise ValueError("Unknown retrieval mode.")
        if target_ids is not None and mode != "hybrid":
            raise ValueError("Candidate trace requires hybrid mode.")
        encoded, mask, query = self.encode(model_input.input_ids, model_input.attention_mask)
        batch = mask.shape[0]
        result = self.semantic_ids.new_full((batch, self.top_k, self.num_hierarchies), -1)
        scores = query.new_full((batch, self.top_k), float("-inf"))
        logits = self.dense_logits(query) if mode != "generative" or self.candidate_strategy != "original" else None
        generated = None
        if mode != "dense":
            if self.candidate_strategy == "original":
                generated = self.generate_candidates(encoded, mask)
            else:
                generated = self.generate_candidates(encoded, mask, logits)
        cold = (~self.seen_mask).nonzero().flatten()
        trace_rows = []
        targets = self.lookup_rows(target_ids) if target_ids is not None else None
        for b in range(batch):
            if mode == "dense":
                rows = torch.argsort(logits[b], descending=True, stable=True)[: self.top_k]
            else:
                candidates = generated[b]
                candidates = candidates[candidates >= 0]
                if mode == "hybrid":
                    candidates = torch.cat([candidates, cold]).unique(sorted=True)
                    order = torch.argsort(logits[b, candidates], descending=True, stable=True)
                    rows = candidates[order[: self.top_k]]
                    if targets is not None:
                        target = targets[b]
                        dense_order = torch.argsort(logits[b], descending=True, stable=True)
                        hybrid_order = candidates[order]
                        found = (hybrid_order == target).nonzero().flatten()
                        trace_rows.append(
                            {
                                "generated_rows": generated[b],
                                "generated_unique_count": generated[b][generated[b] >= 0].unique().numel(),
                                "invalid_generated_count": (generated[b] < 0).sum(),
                                "candidate_count": len(candidates),
                                "target_row": target,
                                "target_cold": ~self.seen_mask[target],
                                "target_generated": (generated[b] == target).any(),
                                "target_covered": found.numel() > 0,
                                "dense_rank": (dense_order == target).nonzero().flatten()[0] + 1,
                                "hybrid_rank": found[0] + 1 if found.numel() else 0,
                                "dense_topk_rows": F.pad(
                                    dense_order[: self.top_k], (0, max(0, self.top_k - len(dense_order))), value=-1
                                ),
                            }
                        )
                else:
                    # 保留 beam 顺序，而不是 unique() 的商品索引顺序。
                    ordered = list(dict.fromkeys(candidates.tolist()))[: self.top_k]
                    rows = torch.tensor(ordered, device=query.device, dtype=torch.long)
            result[b, : len(rows)] = self.semantic_ids[rows]
            if logits is not None:
                scores[b, : len(rows)] = logits[b, rows]
            else:
                scores[b, : len(rows)] = 0
        if targets is not None:
            trace = {
                name: torch.stack([torch.as_tensor(row[name], device=query.device) for row in trace_rows])
                for name in trace_rows[0]
            }
            trace["hybrid_topk_sids"] = result
            return result, scores, trace
        return result, scores

    def eval_step(self, batch):
        model_input, label = batch
        if label is None:
            raise ValueError("LIGER evaluation requires labels.")
        losses = self.losses(model_input, label.target_ids)
        sids, scores = self.retrieve(model_input, self.evaluation_mode)
        return {**losses, "generated_ids": sids, "labels": label.target_ids, "marginal_probs": scores}

    def validation_step(self, batch, batch_idx):
        return self.eval_step(batch)

    def test_step(self, batch, batch_idx):
        return self.eval_step(batch)

    def predict_step(self, batch, batch_idx=0):
        model_input = batch if isinstance(batch, TigerModelInput) else batch[0]
        if model_input.output_keys is None:
            raise ValueError("Prediction output_keys are required.")
        auxiliary = {}
        if self.candidate_trace:
            if isinstance(batch, TigerModelInput) or batch[1] is None:
                raise ValueError("Candidate trace requires labels.")
            labels = batch[1].target_ids
            retrieved = self.retrieve(model_input, target_ids=labels)
            sids, _, trace = retrieved
            auxiliary["liger_candidates"] = dict(
                schema_version="liger_candidates_v1",
                labels=labels,
                trace=trace,
                metadata=dict(
                    catalog_sha256=self.catalog_sha256,
                    top_k=self.top_k,
                    generation_candidates=self.generation_candidates,
                    catalog_size=len(self.semantic_ids),
                    cold_count=int((~self.seen_mask).sum()),
                    candidate_strategy=self.candidate_strategy,
                    content_mixture_alpha=self.content_mixture_alpha,
                    candidate_normalization=(
                        "legal_conditional"
                        if self.candidate_strategy == "probability_mixture"
                        else "full_vocabulary"
                    ),
                ),
            )
        else:
            sids, _ = self.retrieve(model_input)
        return ModelOutput(keys=model_input.output_keys, predictions=sids, auxiliary=auxiliary)

    def configure_optimizers(self) -> dict[str, Any]:
        if self.training_config.optimizer is None:
            raise ValueError("Optimizer is required for training.")
        optimizer = self.training_config.optimizer(params=self.parameters())
        result = {"optimizer": optimizer}
        if self.training_config.scheduler is not None:
            result["lr_scheduler"] = {
                "scheduler": self.training_config.scheduler(optimizer=optimizer),
                "interval": "step",
            }
        return result

    def on_save_checkpoint(self, checkpoint):
        checkpoint["liger_catalog_sha256"] = self.catalog_sha256

    def on_load_checkpoint(self, checkpoint):
        if checkpoint.get("liger_catalog_sha256") != self.catalog_sha256:
            raise ValueError("LIGER checkpoint catalog/content/training-item identity mismatch.")

    def load_state_dict(self, state_dict, strict=True, assign=False):
        for name in ("item_keys", "semantic_ids", "content_bank", "seen_mask"):
            if name in state_dict and not torch.equal(state_dict[name].cpu(), getattr(self, name).cpu()):
                raise ValueError(f"LIGER checkpoint input mismatch: {name}")
        return super().load_state_dict(state_dict, strict=strict, assign=assign)
