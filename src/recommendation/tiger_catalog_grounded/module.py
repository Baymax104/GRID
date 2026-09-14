"""独立 CGBS 模型：固定目录证据参与训练和逐层生成。"""

import math
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

from src.recommendation.tiger.tiger import Tiger

from .catalog import PrefixCatalog

ARMS = ("original", "mask_ce", "token_content_init", "single_prototype", "full", "no_aux", "shuffled", "hybrid")
CONTENT_ARMS = {"single_prototype", "full", "no_aux", "shuffled"}


class TigerCatalogGrounded(Tiger):
    def __init__(
        self,
        catalog: dict[str, torch.Tensor],
        arm: str = "full",
        projection_dim: int = 128,
        prototypes: int = 4,
        cluster_iterations: int = 5,
        temperature: float = 0.1,
        auxiliary_weight: float = 0.1,
        alpha_initial: float = 0.1,
        alpha_maximum: float = 0.5,
        bank_seed: int = 42,
        hybrid_weight: float = 0.5,
        **kwargs: Any,
    ):
        if arm not in ARMS:
            raise ValueError(f"Unknown catalog-grounded arm {arm!r}; expected {ARMS}.")
        if not 0 < alpha_initial < alpha_maximum < 1 or temperature <= 0 or auxiliary_weight < 0:
            raise ValueError("Invalid temperature, auxiliary weight or mixture bounds.")
        if not 0 < hybrid_weight < 1:
            raise ValueError("hybrid_weight must be strictly between zero and one.")
        super().__init__(**kwargs)
        if not self.should_check_prefix or self.decoder.prefix_allocation.enabled:
            raise ValueError("CGBS requires legal-prefix checking and disabled quota allocation.")
        if arm == "hybrid" and self.trace_prefix_survival:
            raise ValueError("Hybrid dense union has no compatible beam-only trace; set trace_prefix_survival=false.")
        if not torch.equal(self.semantic_ids.cpu(), catalog["semantic_ids"].long().cpu()):
            raise ValueError("Model semantic IDs do not match the keyed content catalog.")
        self.arm = arm
        self.temperature = temperature
        self.alpha_maximum = alpha_maximum
        self.hybrid_weight = hybrid_weight
        self.auxiliary_weight = auxiliary_weight if arm in (CONTENT_ARMS - {"no_aux"}) | {"hybrid"} else 0.0
        self.catalog = PrefixCatalog(
            catalog,
            self.num_embeddings_per_hierarchy,
            self.num_hierarchies,
            projection_dim,
            1 if arm == "single_prototype" else prototypes,
            cluster_iterations,
            bank_seed if arm == "shuffled" else None,
        )
        if not 1 <= self.top_k_for_generation <= len(self.catalog.keys):
            raise ValueError("Beam width must be between one and the number of catalog items.")
        if arm == "original" and self.top_k_for_generation > len(self.catalog.codes_0):
            raise ValueError("Original TIGER beam requires at least beam_width distinct first-level prefixes.")
        dimension = self.catalog.features.shape[1]
        if arm in CONTENT_ARMS or arm == "hybrid":
            # 额外模块使用局部随机状态，保持各 arm 的 backbone/dropout 随机流一致。
            with torch.random.fork_rng(devices=[]):
                torch.random.default_generator.manual_seed(bank_seed)
                self.content_query = nn.Sequential(
                    nn.Linear(self.embedding_dim, self.embedding_dim),
                    nn.GELU(),
                    nn.Linear(self.embedding_dim, dimension),
                )
        if arm in CONTENT_ARMS:
            initial = math.log(alpha_initial / (alpha_maximum - alpha_initial))
            self.mixture_logits = nn.Parameter(torch.full((self.num_hierarchies,), initial))
        if arm == "token_content_init":
            self._initialize_tokens()
        self.catalog_contract = {
            "version": 1,
            "arm": arm,
            "fingerprint": self.catalog.fingerprint,
            "temperature": temperature,
            "auxiliary_weight": self.auxiliary_weight,
            "alpha_maximum": alpha_maximum,
            "alpha_initial": alpha_initial,
            "bank_seed": bank_seed,
            "hybrid_weight": hybrid_weight,
            "loss_reduction": "hierarchy_mean",
        }
        self.prefix_trace_metadata = {**self.prefix_trace_metadata, "catalog_grounded": self.catalog_contract}

    @torch.no_grad()
    def _initialize_tokens(self):
        features = self.catalog.features
        if features.shape[1] > self.embedding_dim:
            raise ValueError("token_content_init requires projection_dim <= embedding_dim.")
        features = F.pad(features, (0, self.embedding_dim - features.shape[1]))
        for level in range(self.num_hierarchies - 1):
            table = self.sid_embedding_table.weight[
                level * self.num_embeddings_per_hierarchy : (level + 1) * self.num_embeddings_per_hierarchy
            ]
            values, indices = [], []
            for token in self.catalog.sids[:, level].unique().tolist():
                values.append(features[self.catalog.sids[:, level] == token].mean(0))
                indices.append(token)
            means = torch.stack(values)
            means = means * table.std(unbiased=False) / means.std(unbiased=False).clamp_min(1e-8)
            table[indices] = means

    def on_save_checkpoint(self, checkpoint: dict[str, Any]) -> None:
        checkpoint["catalog_grounded"] = dict(self.catalog_contract)

    def on_load_checkpoint(self, checkpoint: dict[str, Any]) -> None:
        if checkpoint.get("catalog_grounded") != self.catalog_contract:
            raise ValueError(
                "CGBS checkpoint catalog/arm/scoring contract mismatch; use the original training configuration."
            )

    def _query(self, encoded: torch.Tensor, mask: torch.Tensor) -> torch.Tensor | None:
        if not hasattr(self, "content_query"):
            return None
        mask = mask.to(encoded.dtype).unsqueeze(-1)
        pooled = (encoded * mask).sum(1) / mask.sum(1).clamp_min(1)
        return F.normalize(self.content_query(pooled).float(), dim=-1)

    def conditional_log_probs(self, logits, prefixes, query=None):
        _, legal = self.catalog.lookup_children(prefixes)
        if not bool(legal.any(-1).all()):
            raise ValueError("No catalog continuation exists for a supplied prefix.")
        token_log = F.log_softmax(logits.float().masked_fill(~legal, -torch.inf), dim=-1)
        if self.arm not in CONTENT_ARMS:
            return token_log
        mass = self.catalog.log_mass(query, prefixes, self.temperature)
        content_log = F.log_softmax(mass, dim=-1)
        alpha = self.alpha_maximum * self.mixture_logits[prefixes.shape[1]].sigmoid()
        # 在非法位置使用有限中间值，避免 logaddexp(-inf, -inf) 的 NaN 梯度。
        token_safe = token_log.masked_fill(~legal, 0)
        content_safe = content_log.masked_fill(~legal, 0)
        mixed = torch.logaddexp(torch.log1p(-alpha) + token_safe, alpha.log() + content_safe)
        return mixed.masked_fill(~legal, -torch.inf)

    def _teacher(self, encoded, mask, targets, query):
        self.catalog.item_indices(targets)
        raw = self.decoder(future_ids=targets, encoder_output=encoded, encoder_attention_mask=mask)
        if self.arm == "original":
            return raw
        width = self.num_embeddings_per_hierarchy
        result = raw.new_full(raw.shape, -torch.inf)
        for level in range(self.num_hierarchies):
            result[:, level, level * width : (level + 1) * width] = self.conditional_log_probs(
                raw[:, level, level * width : (level + 1) * width], targets[:, :level], query
            )
        return result

    def forward(self, attention_mask_encoder, input_ids, future_ids):
        if self.arm == "original":
            return super().forward(attention_mask_encoder, input_ids, future_ids)
        encoded, mask = self.encoder(input_ids=input_ids, attention_mask=attention_mask_encoder)
        return self._teacher(encoded, mask, future_ids, self._query(encoded, mask))

    def training_step(self, batch, batch_idx):
        if self.arm == "original":
            output = super().training_step(batch, batch_idx)
            return {
                **output,
                "generation_loss": output["loss"],
                "content_loss": output["loss"].new_zeros(()),
                "mixture_alpha": output["loss"].new_zeros(()),
            }
        inputs, labels = batch
        if labels is None:
            raise ValueError("Catalog-grounded training requires target labels.")
        encoded, mask = self.encoder(input_ids=inputs.input_ids, attention_mask=inputs.attention_mask)
        query = self._query(encoded, mask)
        logits = self._teacher(encoded, mask, labels.target_ids, query)
        generation_loss = self._compute_loss(logits, labels.target_ids)
        auxiliary_loss = generation_loss.new_zeros(())
        if self.auxiliary_weight:
            auxiliary_loss = F.cross_entropy(
                query @ self.catalog.features.T / self.temperature, self.catalog.item_indices(labels.target_ids)
            )
        return {
            "loss": generation_loss + self.auxiliary_weight * auxiliary_loss,
            "generation_loss": generation_loss,
            "content_loss": auxiliary_loss,
            "mixture_alpha": self.alpha_maximum * self.mixture_logits.sigmoid().mean()
            if hasattr(self, "mixture_logits")
            else generation_loss.new_zeros(()),
        }

    def generate(self, attention_mask, input_ids, target_ids=None, trace_enabled=False):
        if self.arm == "original":
            return super().generate(attention_mask, input_ids, target_ids, trace_enabled)
        if trace_enabled and (target_ids is None or self.arm == "hybrid"):
            raise ValueError("Beam tracing requires targets and a non-hybrid arm.")
        encoded, mask = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        query = self._query(encoded, mask)
        generated, log_scores, trace = self._beam(encoded, mask, query, target_ids, trace_enabled)
        if self.arm == "hybrid":
            generated, log_scores = self._hybrid(encoded, mask, query, generated)
        result = (generated, log_scores.exp())
        return (*result, trace) if trace_enabled else result

    def _beam(self, encoded, mask, query, targets, trace_enabled):
        batch = encoded.shape[0]
        width = self.num_embeddings_per_hierarchy
        prefixes = torch.empty((batch, 1, 0), dtype=torch.long, device=encoded.device)
        scores = encoded.new_zeros((batch, 1), dtype=torch.float32)
        steps = []
        if targets is not None:
            targets = targets.to(encoded.device).long()
            if targets.shape != (batch, self.num_hierarchies):
                raise ValueError("Trace targets have incorrect shape.")
            self.catalog.item_indices(targets)
        for level in range(self.num_hierarchies):
            previous, previous_scores = prefixes, scores
            beams = prefixes.shape[1]
            flat = prefixes.reshape(batch * beams, level)
            offsets = torch.arange(level, device=encoded.device) * width
            tokens = self.sid_embedding_table(flat + offsets)
            bos = self.decoder.bos_token.unsqueeze(0).expand(batch * beams, 1, -1)
            hidden = self.decoder.decoder(
                inputs_embeds=torch.cat((bos, tokens), 1),
                encoder_hidden_states=encoded.repeat_interleave(beams, 0),
                encoder_attention_mask=mask.repeat_interleave(beams, 0),
                use_cache=False,
            ).last_hidden_state[:, -1]
            logits = self.decoder.lm_head(hidden)[:, level * width : (level + 1) * width]
            local = self.conditional_log_probs(
                logits, flat, None if query is None else query.repeat_interleave(beams, 0)
            )
            local = local.reshape(batch, beams, width)
            candidates = (scores[:, :, None] + local).flatten(1)
            count = min(self.top_k_for_generation, candidates.shape[1])
            scores, positions = candidates.topk(count, dim=1)
            parents = positions // width
            prefixes = torch.cat(
                (previous.gather(1, parents[:, :, None].expand(-1, -1, level)), (positions % width)[:, :, None]), dim=-1
            )
            # 窄树早期不足 B 条路径时保留 inactive 槽；其合法占位前缀不参与候选竞争。
            inactive = ~scores.isfinite()
            prefixes = torch.where(inactive[:, :, None], prefixes[:, :1], prefixes)
            if trace_enabled:
                steps.append(self._observe(previous, previous_scores, local, prefixes, scores, targets, level))
        if not scores.isfinite().all():
            raise RuntimeError("Beam could not produce the requested number of unique catalog items.")
        self.catalog.item_indices(prefixes)
        trace = {}
        if trace_enabled:
            trace = {key: torch.stack([step[key] for step in steps], 1) for key in steps[0]}
            trace["first_failure_depth"] = self.decoder._first_failure_depth(trace["target_prefix_survived"])
        return prefixes, scores, trace

    def _observe(self, previous, previous_scores, local, prefixes, scores, targets, level):
        batch = targets.shape[0]
        rows = torch.arange(batch, device=targets.device)
        matches = (previous == targets[:, None, :level]).all(-1) & previous_scores.isfinite()
        parent_exists, parent = matches.any(-1), matches.long().argmax(-1)
        path_score = (previous_scores[rows, parent] + local[rows, parent, targets[:, level]]).exp()
        path_score = path_score.masked_fill(~parent_exists, torch.nan)
        selected = (prefixes == targets[:, None, : level + 1]).all(-1) & scores.isfinite()
        survived = selected.any(-1)
        cutoff = scores[:, -1].exp()
        return {
            "target_prefix_survived": survived,
            "target_parent_beam_rank": torch.where(parent_exists, parent + 1, -1),
            "target_beam_rank": torch.where(survived, selected.long().argmax(-1) + 1, -1),
            "target_path_score": path_score,
            "beam_cutoff_score": cutoff,
            "cutoff_margin": path_score - cutoff,
            "legal_candidate_count": self.catalog.lookup_children(targets[:, :level])[1].sum(-1),
        }

    def _hybrid(self, encoded, mask, query, generated):
        dense_log = F.log_softmax(query @ self.catalog.features.T / self.temperature, dim=-1)
        dense = dense_log.topk(self.top_k_for_generation, dim=-1).indices
        generated_indices = self.catalog.item_indices(generated)
        union = torch.cat((generated_indices, dense), dim=1)
        batch, count = union.shape
        target_sids = self.catalog.sids[union]
        logits = self._teacher(
            encoded.repeat_interleave(count, 0),
            mask.repeat_interleave(count, 0),
            target_sids.flatten(0, 1),
            query.repeat_interleave(count, 0),
        )
        offsets = torch.arange(self.num_hierarchies, device=encoded.device) * self.num_embeddings_per_hierarchy
        token_log = logits.gather(-1, (target_sids.flatten(0, 1) + offsets).unsqueeze(-1)).squeeze(-1).sum(-1)
        mixture = torch.logaddexp(
            token_log.reshape(batch, count) + math.log1p(-self.hybrid_weight),
            dense_log.gather(1, union) + math.log(self.hybrid_weight),
        )
        duplicates = union[:, :, None] == union[:, None, :]
        duplicates = (duplicates & torch.ones(count, count, dtype=torch.bool, device=union.device).tril(-1)).any(-1)
        mixture = mixture.masked_fill(duplicates, -torch.inf)
        scores, chosen = mixture.topk(self.top_k_for_generation, dim=-1)
        return self.catalog.sids[union.gather(1, chosen)], scores
