"""冻结 checkpoint 的全目录精确评分与预算搜索取证。"""

import time

import torch

from src.data.components.data_models import ModelOutput
from src.data.components.item_resolution_audit import input_fingerprint

from .module import TigerItemResolution


@torch.no_grad()
def exact_catalog_distribution(model, encoded, mask, chunk_size):
    """遍历全部目录 SID，不读取用户真值；分块只影响计算内存。"""
    if len(encoded) != 1 or chunk_size < 1:
        raise ValueError("Exact audit requires one user and positive chunk size.")
    probabilities, responsibilities = [], []
    for targets in model.catalog.sids.split(chunk_size):
        output = model.objective(encoded.expand(len(targets), -1, -1), mask.expand(len(targets), -1), targets)
        probabilities.append(output["target_log_probability"])
        responsibilities.append(output["target_depth_responsibility"])
    log_probability = torch.cat(probabilities)
    responsibility = torch.cat(responsibilities)
    if not log_probability.isfinite().all() or not torch.allclose(
        log_probability.exp().sum(), log_probability.new_tensor(1.0), atol=1e-4, rtol=1e-4
    ):
        raise ValueError("Exact item marginal is nonfinite or not normalized.")
    return log_probability, responsibility


def _sync(tensor):
    if tensor.is_cuda:
        torch.cuda.synchronize(tensor.device)


class TigerItemResolutionAudit(TigerItemResolution):
    def __init__(
        self,
        audit_chunk_size=128,
        audit_state_budgets=(64, 128, 256),
        audit_users=128,
        audit_sampling_seed=20260915,
        **kwargs,
    ):
        super().__init__(**kwargs)
        budgets = tuple(int(q) for q in audit_state_budgets)
        if self.arm not in ("mir", "depth2") or self.inference_policy != "standard" or self.data_split != "evaluation":
            raise ValueError("Score/search audit supports MIR/depth2 on evaluation only.")
        if not budgets or budgets != tuple(sorted(set(budgets))) or min(budgets) < 1 or max(budgets) > 1024:
            raise ValueError("Audit Q budgets must be unique, increasing, and between 1 and 1024.")
        if not 1 <= audit_chunk_size <= 1024 or not 1 <= audit_users <= 512 or audit_sampling_seed < 0:
            raise ValueError("Invalid bounded audit configuration.")
        self.audit_chunk_size, self.audit_state_budgets = audit_chunk_size, budgets
        self.audit_users, self.audit_sampling_seed = audit_users, audit_sampling_seed

    def training_step(self, *args, **kwargs):
        raise RuntimeError("Audit checkpoints must remain frozen; training is disabled.")

    def on_predict_start(self):
        super().on_predict_start()
        if self.loaded_checkpoint_fingerprint is None:
            raise ValueError("Audit requires a loaded checkpoint.")

    @torch.no_grad()
    def predict_step(self, batch):
        inputs, labels = batch
        if len(inputs.input_ids) != 1 or inputs.output_keys is None or labels is None:
            raise ValueError("Audit requires batch1 with keyed inputs and labels.")
        if not self.loaded_checkpoint_fingerprint:
            raise ValueError("Audit requires a loaded checkpoint.")
        _sync(inputs.input_ids)
        started = time.perf_counter()
        encoded, mask = self.encoder(input_ids=inputs.input_ids, attention_mask=inputs.attention_mask)
        _sync(encoded)
        encoder_seconds = time.perf_counter() - started
        started = time.perf_counter()
        logp, responsibility = exact_catalog_distribution(self, encoded, mask, self.audit_chunk_size)
        _sync(encoded)
        exact_seconds = time.perf_counter() - started
        # 同分时按已排序 item key 升序，不利用真值打破平局。
        order = logp.argsort(descending=True, stable=True)
        k = self.top_k_for_generation
        exact_top = order[:k]
        target = self.catalog.item_indices(labels.target_ids.long()).reshape(-1)[0]
        exact_rank = (order == target).nonzero()[0, 0] + 1
        output_fields = {
            name: []
            for name in [
                "search_target_rank",
                "search_ndcg",
                "search_seconds",
                "search_remaining_mass",
                "search_topk_certified",
                "search_states",
                "search_item_scores",
                "search_topk_keys",
                "search_topk_scores",
                "search_topk_probability_capture",
                "exact_topk_overlap",
            ]
        }
        original_budget, first_sids = self.max_states, None
        try:
            for q in self.audit_state_budgets:
                self.max_states = q
                sids, scores, trace = self.generate_encoded(encoded, mask, collect_trace=True)
                if first_sids is None:
                    first_sids = sids
                items = self.catalog.item_indices(sids)[0]
                hits = (items == target).nonzero().flatten()
                rank = int(hits[0]) + 1 if len(hits) else -1
                values = dict(
                    search_target_rank=torch.tensor(rank),
                    search_ndcg=torch.tensor(1.0 / torch.log2(torch.tensor(rank + 1.0)).item() if rank > 0 else 0.0),
                    search_seconds=trace["decoder_batch_seconds"][0],
                    search_remaining_mass=trace["remaining_mass"][0],
                    search_topk_certified=trace["topk_certified"][0],
                    search_states=trace["states_evaluated"][0],
                    search_item_scores=trace["items_scored"][0],
                    search_topk_keys=self.catalog.keys[items],
                    search_topk_scores=scores[0],
                    search_topk_probability_capture=scores[0] / logp[items].exp(),
                    exact_topk_overlap=torch.isin(items, exact_top).float().mean(),
                )
                for name, value in values.items():
                    output_fields[name].append(value.detach().cpu())
        finally:
            self.max_states = original_budget
        evidence = {name: torch.stack(values)[None] for name, values in output_fields.items()}
        evidence.update(
            input_sha256=input_fingerprint(inputs),
            exact_log_probability=logp.cpu()[None],
            exact_target_rank=exact_rank.cpu().reshape(1),
            exact_target_log_probability=logp[target].cpu().reshape(1),
            exact_topk_keys=self.catalog.keys[exact_top].cpu()[None],
            target_depth_responsibility=responsibility[target].cpu()[None],
            global_depth_mass=(logp.exp()[:, None] * responsibility).sum(0).cpu()[None],
            exact_seconds=torch.tensor([exact_seconds]),
            encoder_seconds=torch.tensor([encoder_seconds]),
        )
        metadata = dict(
            data_split=self.data_split,
            checkpoint_reference=self.checkpoint_reference,
            checkpoint_fingerprint=self.loaded_checkpoint_fingerprint,
            catalog_fingerprint=self.catalog.fingerprint,
            contract=self.contract,
            item_keys=self.catalog.keys.cpu().tolist(),
            semantic_depth=self.semantic_depth,
            state_budgets=list(self.audit_state_budgets),
            item_score_budget=self.max_item_scores,
            top_k=k,
            requested_users=self.audit_users,
            sampling_seed=self.audit_sampling_seed,
            sampling="smallest_sha256_seed_user_key_v1",
            exact_chunk_size=self.audit_chunk_size,
        )
        return ModelOutput(
            keys=inputs.output_keys.cpu(),
            predictions=first_sids.cpu(),
            auxiliary={
                "item_resolution_audit": dict(
                    schema_version="item_resolution_score_search_v1",
                    labels=labels.target_ids.cpu(),
                    trace=evidence,
                    metadata=metadata,
                )
            },
        )
