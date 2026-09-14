"""固定 SID 下的多深度 item 解析及匹配对照。"""

import hashlib
import time

import torch
from torch import nn
from torch.nn import functional as F

from src.data.components.data_models import ModelOutput
from src.recommendation.tiger.tiger import Tiger

from .catalog import ResolutionCatalog

ARMS = ("mask_ce", "token_content_init", "dense", "hybrid", "cobra", "earliest", "depth2", "depth_gate", "mir")
RESOLUTION_ARMS = {"earliest", "depth2", "depth_gate", "mir"}


class TigerItemResolution(Tiger):
    def __init__(
        self,
        catalog,
        arm="mir",
        projection_dim=128,
        max_bucket=128,
        temperature=0.1,
        route_weight=0.1,
        resolve_weight=0.1,
        warmup_steps=2000,
        max_states=64,
        max_item_scores=4096,
        expansion_batch_size=8,
        hybrid_weight=0.5,
        trace_resolution=False,
        inference_policy="standard",
        calibration=None,
        checkpoint_reference=None,
        data_split=None,
        initialization_seed=42,
        **kwargs,
    ):
        super().__init__(**kwargs)
        if arm not in ARMS or not self.should_check_prefix or self.trace_prefix_survival:
            raise ValueError("MIR requires a known arm, legal prefixes, and disabled legacy prefix trace.")
        if temperature <= 0 or min(route_weight, resolve_weight, warmup_steps) < 0:
            raise ValueError("Invalid loss temperature, weights or warmup.")
        if min(max_states, max_item_scores, expansion_batch_size) < 1 or not 0 < hybrid_weight < 1:
            raise ValueError("Invalid inference budgets or hybrid weight.")
        if inference_policy not in ("standard", "calibrate", "wide") or (
            inference_policy != "standard" and arm != "hybrid"
        ):
            raise ValueError("Only hybrid supports calibrate/wide inference policies.")
        self.catalog = ResolutionCatalog(
            catalog, self.num_embeddings_per_hierarchy, self.num_hierarchies, projection_dim, max_bucket
        )
        if not torch.equal(self.semantic_ids.cpu(), self.catalog.sids.cpu()):
            raise ValueError("Model SID order does not match keyed catalog.")
        if not 1 <= self.top_k_for_generation <= len(self.catalog.keys):
            raise ValueError("Invalid top K.")
        self.arm, self.max_bucket, self.temperature = arm, max_bucket, temperature
        self.route_weight, self.resolve_weight, self.warmup_steps = route_weight, resolve_weight, warmup_steps
        self.max_states, self.max_item_scores = max_states, max_item_scores
        self.expansion_batch_size, self.hybrid_weight = expansion_batch_size, hybrid_weight
        self.trace_resolution, self.inference_policy = trace_resolution, inference_policy
        self.checkpoint_reference, self.data_split = checkpoint_reference, data_split
        self.calibration = calibration
        self.loaded_checkpoint_fingerprint = None
        self.semantic_depth = self.num_hierarchies - 1
        dim = self.catalog.features.shape[1]
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(initialization_seed)
            if arm not in ("mask_ce", "token_content_init"):
                self.item_projection = nn.Linear(dim, dim, bias=False)
                nn.init.eye_(self.item_projection.weight)
                self.depth_embedding = nn.Embedding(self.num_hierarchies + 1, self.embedding_dim)
                self.query_head = nn.Sequential(
                    nn.Linear(self.embedding_dim * 2, self.embedding_dim), nn.GELU(), nn.Linear(self.embedding_dim, dim)
                )
            if arm == "mir":
                self.gate_head = nn.Sequential(
                    nn.Linear(self.embedding_dim * 2 + 1, self.embedding_dim),
                    nn.GELU(),
                    nn.Linear(self.embedding_dim, 1),
                )
            if arm == "depth_gate":
                self.gate_logits = nn.Parameter(torch.zeros(self.num_hierarchies))
        if arm != "mask_ce":
            self._initialize_tokens()
        if arm == "dense":
            self.decoder.lm_head.requires_grad_(False)
        self.contract = dict(
            version=1,
            arm=arm,
            catalog=self.catalog.fingerprint,
            max_bucket=max_bucket,
            temperature=temperature,
            route_weight=route_weight,
            resolve_weight=resolve_weight,
            warmup_steps=warmup_steps,
            hybrid_weight=hybrid_weight,
            initialization_seed=initialization_seed,
            projection_dim=projection_dim,
            baseline_adaptation="tiger_fixed_content_v1" if arm in ("cobra", "hybrid") else None,
        )

    @torch.no_grad()
    def _initialize_tokens(self):
        features = self.catalog.features
        if features.shape[1] > self.embedding_dim:
            raise ValueError("projection_dim must not exceed embedding_dim for token initialization.")
        features = F.pad(features, (0, self.embedding_dim - features.shape[1]))
        for level in range(self.semantic_depth):
            table = self.sid_embedding_table.weight[
                level * self.num_embeddings_per_hierarchy : (level + 1) * self.num_embeddings_per_hierarchy
            ]
            tokens = self.catalog.sids[:, level].unique()
            means = torch.stack([features[self.catalog.sids[:, level] == token].mean(0) for token in tokens])
            table[tokens] = means * table.std(unbiased=False) / means.std(unbiased=False).clamp_min(1e-8)

    def on_save_checkpoint(self, checkpoint):
        checkpoint["item_resolution"] = dict(self.contract)

    def on_predict_start(self):
        if self.trainer.world_size != 1:
            raise ValueError("Item resolution inference/calibration requires a single GPU process.")

    def on_load_checkpoint(self, checkpoint):
        if checkpoint.get("item_resolution") != self.contract:
            raise ValueError("Item resolution checkpoint/catalog/arm contract mismatch.")
        digest = hashlib.sha256()
        for name, value in sorted(checkpoint["state_dict"].items()):
            digest.update(name.encode())
            digest.update(value.detach().cpu().contiguous().numpy().tobytes())
        self.loaded_checkpoint_fingerprint = digest.hexdigest()
        if self.inference_policy == "wide":
            if (
                self.calibration is None
                or self.calibration["checkpoint_fingerprint"] != self.loaded_checkpoint_fingerprint
            ):
                raise ValueError("WIDE calibration checkpoint fingerprint mismatch.")
            if self.calibration["catalog_fingerprint"] != self.catalog.fingerprint:
                raise ValueError("WIDE calibration catalog fingerprint mismatch.")

    def states(self, encoded, mask, prefixes):
        """状态位置 d 只读取 BOS 与 prefix[:d]，不接受未生成的 label token。"""
        level = prefixes.shape[1]
        offsets = torch.arange(level, device=prefixes.device) * self.num_embeddings_per_hierarchy
        tokens = self.sid_embedding_table(prefixes + offsets)
        bos = self.decoder.bos_token[None].expand(len(prefixes), 1, -1)
        return self.decoder.decoder(
            inputs_embeds=torch.cat((bos, tokens), 1),
            encoder_hidden_states=encoded,
            encoder_attention_mask=mask,
            use_cache=False,
        ).last_hidden_state

    def route_log(self, hidden, prefixes):
        level = prefixes.shape[1]
        width = self.num_embeddings_per_hierarchy
        logits = self.decoder.lm_head(hidden)[:, level * width : (level + 1) * width].float()
        return F.log_softmax(logits.masked_fill(~self.catalog.legal(prefixes), -torch.inf), -1)

    def query(self, hidden, depth):
        depths = torch.full((len(hidden),), depth, device=hidden.device, dtype=torch.long)
        features = torch.cat((hidden, self.depth_embedding(depths)), -1)
        return F.normalize(self.query_head(features).float(), dim=-1)

    def item_vectors(self):
        return F.normalize(self.item_projection(self.catalog.features).float(), dim=-1)

    def resolve_log(self, hidden, nodes, depth, vectors):
        indices, valid = self.catalog.members(nodes)
        query = self.query(hidden, depth)
        logits = torch.einsum("bd,bmd->bm", query, vectors[indices]) / self.temperature
        return F.log_softmax(logits.masked_fill(~valid, -torch.inf), -1), indices, valid

    def gates(self, hidden, nodes, depth):
        count = self.catalog.counts[nodes]
        if depth == 0:
            return hidden.new_zeros(len(hidden), dtype=torch.float32)
        if depth == self.semantic_depth:
            return hidden.new_ones(len(hidden), dtype=torch.float32)
        if self.arm == "earliest":
            result = hidden.new_ones(len(hidden), dtype=torch.float32)
        elif self.arm == "depth2":
            result = hidden.new_full((len(hidden),), float(depth >= min(2, self.semantic_depth)), dtype=torch.float32)
        elif self.arm == "depth_gate":
            result = self.gate_logits[depth].sigmoid().expand(len(hidden))
        else:
            depths = torch.full_like(nodes, depth)
            features = torch.cat((hidden, self.depth_embedding(depths), count.float().log()[:, None]), -1)
            result = self.gate_head(features).squeeze(-1).float().sigmoid()
        return torch.where(count == 1, 1.0, torch.where(count <= self.max_bucket, result, 0.0))

    def objective(self, encoded, mask, targets, *, warmup=False):
        target_items = self.catalog.item_indices(targets)
        states = self.states(encoded, mask, targets[:, : self.semantic_depth])
        routes = torch.stack(
            [
                self.route_log(states[:, depth], targets[:, :depth]).gather(1, targets[:, depth : depth + 1]).squeeze(1)
                for depth in range(self.num_hierarchies)
            ],
            1,
        )
        route_loss = -routes.mean()
        if self.arm in ("mask_ce", "token_content_init"):
            return dict(
                loss=route_loss,
                target_log_probability=routes.sum(1),
                item_nll=-routes.sum(1).mean(),
                route_loss=route_loss,
                resolve_loss=route_loss.new_zeros(()),
                gate_mean=route_loss.new_zeros(()),
            )
        vectors = self.item_vectors()
        if self.arm in ("dense", "hybrid", "cobra"):
            depth = self.semantic_depth if self.arm == "cobra" else 0
            dense_log = F.log_softmax(self.query(states[:, depth], depth) @ vectors.T / self.temperature, -1)
            dense_loss = F.nll_loss(dense_log, target_items)
            if self.arm == "dense":
                loss = dense_loss
            elif self.arm == "cobra":
                loss = -routes[:, : self.semantic_depth].mean() + dense_loss
            else:
                loss = route_loss + self.resolve_weight * dense_loss
            return dict(
                loss=loss,
                target_log_probability=dense_log.gather(1, target_items[:, None]).squeeze(1),
                item_nll=dense_loss,
                route_loss=route_loss,
                resolve_loss=dense_loss,
                gate_mean=loss.new_zeros(()),
            )
        reach = routes[:, 0]
        terms, resolve_losses, available, gate_values = [], [], [], []
        for depth in range(1, self.semantic_depth + 1):
            nodes = self.catalog.nodes(targets[:, :depth])
            feasible = self.catalog.counts[nodes] <= self.max_bucket
            # 不对超容量桶建立分母；不使用 all-catalog 隐性训练。
            target_log = reach.new_zeros(len(targets))
            if feasible.any():
                lp, indices, valid = self.resolve_log(states[feasible, depth], nodes[feasible], depth, vectors)
                matching = indices == target_items[feasible, None]
                positions = (matching & valid).long().argmax(-1)
                target_log[feasible] = lp.gather(1, positions[:, None]).squeeze(1)
            gates = self.gates(states[:, depth], nodes, depth)
            log_stop = torch.where(gates > 0, gates, torch.ones_like(gates)).log().masked_fill(gates == 0, -torch.inf)
            terms.append(reach + log_stop + target_log)
            resolve_losses.append(-target_log)
            available.append(feasible)
            gate_values.append(gates)
            if depth < self.semantic_depth:
                log_continue = (
                    torch.where(gates < 1, 1 - gates, torch.ones_like(gates)).log().masked_fill(gates == 1, -torch.inf)
                )
                reach = reach + log_continue + routes[:, depth]
        log_terms = torch.stack(terms, 1)
        nll = -log_terms.logsumexp(1).mean()
        weights = torch.stack(available, 1)
        resolver_loss = (torch.stack(resolve_losses, 1) * weights).sum() / weights.sum().clamp_min(1)
        # 主路由只到 raw SID；去重位不参与 MIR 目标。
        route_loss = -routes[:, : self.semantic_depth].mean()
        loss = (
            route_loss + resolver_loss
            if warmup
            else nll + self.route_weight * route_loss + self.resolve_weight * resolver_loss
        )
        return dict(
            loss=loss,
            target_log_probability=log_terms.logsumexp(1),
            target_depth_responsibility=log_terms.softmax(1),
            item_nll=nll,
            route_loss=route_loss,
            resolve_loss=resolver_loss,
            gate_mean=torch.stack(gate_values, 1).mean(),
        )

    def training_step(self, batch, batch_idx):
        inputs, labels = batch
        if labels is None:
            raise ValueError("Training requires labels.")
        encoded, mask = self.encoder(input_ids=inputs.input_ids, attention_mask=inputs.attention_mask)
        output = self.objective(encoded, mask, labels.target_ids.long(), warmup=self.global_step < self.warmup_steps)
        return {key: value if key == "loss" else value.detach() for key, value in output.items()}

    def forward(self, attention_mask_encoder, input_ids, future_ids):
        """返回目标 item 的训练分布分数；MIR 不暴露伪造的逐 token logits。"""
        encoded, mask = self.encoder(input_ids=input_ids, attention_mask=attention_mask_encoder)
        return self.objective(encoded, mask, future_ids.long())["target_log_probability"]

    def eval_step(self, batch):
        inputs, labels = batch
        encoded, mask = self.encoder(input_ids=inputs.input_ids, attention_mask=inputs.attention_mask)
        output = self.objective(encoded, mask, labels.target_ids.long())
        sids, scores, _ = self.generate_encoded(encoded, mask, collect_trace=False)
        return {**output, "generated_ids": sids, "marginal_probs": scores, "labels": labels.target_ids}

    def generate(self, attention_mask, input_ids, target_ids=None, trace_enabled=False):
        if trace_enabled:
            raise ValueError("Use item_resolution_trace, not legacy prefix tracing.")
        encoded, mask = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        sids, scores, _ = self.generate_encoded(encoded, mask, collect_trace=False)
        return sids, scores

    def generate_encoded(self, encoded, mask, *, collect_trace=True):
        from .search import generate

        return generate(self, encoded, mask, collect_trace=collect_trace)

    def predict_step(self, batch):
        inputs, labels = batch if isinstance(batch, tuple) else (batch, None)
        if inputs.output_keys is None:
            raise ValueError("Prediction requires stable user keys.")
        if self.trace_resolution and inputs.input_ids.is_cuda:
            torch.cuda.synchronize(inputs.input_ids.device)
        started = time.perf_counter()
        encoded, mask = self.encoder(input_ids=inputs.input_ids, attention_mask=inputs.attention_mask)
        if self.inference_policy == "calibrate":
            from .calibration import calibration_output

            return calibration_output(self, encoded, mask, inputs, labels)
        sids, scores, trace = self.generate_encoded(encoded, mask, collect_trace=self.trace_resolution)
        if self.trace_resolution and encoded.is_cuda:
            torch.cuda.synchronize(encoded.device)
        auxiliary = {}
        if self.trace_resolution:
            trace["model_batch_seconds"] = torch.full(
                (len(encoded),), time.perf_counter() - started, device=encoded.device
            )
            if labels is None:
                raise ValueError("Resolution trace requires labels for keyed downstream analysis.")
            if self.arm in RESOLUTION_ARMS:
                evidence = self.objective(encoded, mask, labels.target_ids.long())
                trace["target_item_probability"] = evidence["target_log_probability"].exp()
                trace["target_depth_responsibility"] = evidence["target_depth_responsibility"]
            auxiliary["item_resolution_trace"] = dict(
                schema_version="item_resolution_v1",
                labels=labels.target_ids.cpu(),
                trace={
                    key: value.cpu()
                    for key, value in {
                        **trace,
                        "topk_scores": scores,
                        "topk_item_keys": self.catalog.keys[self.catalog.item_indices(sids)],
                    }.items()
                },
                metadata=dict(
                    contract=self.contract,
                    data_split=self.data_split,
                    checkpoint_reference=self.checkpoint_reference,
                    checkpoint_fingerprint=self.loaded_checkpoint_fingerprint,
                    inference_policy=self.inference_policy,
                    max_states=self.max_states,
                    max_item_scores=self.max_item_scores,
                    event_capacity=trace["expanded_node"].shape[1],
                    beam_width=self.top_k_for_generation,
                    num_hierarchies=self.num_hierarchies,
                    trace_mode="item_resolution",
                ),
            )
        return ModelOutput(keys=inputs.output_keys.cpu(), predictions=sids.cpu(), auxiliary=auxiliary)
