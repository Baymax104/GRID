"""全目录联合 CE 与合法前缀 mixture 监督。"""

import torch
from torch.nn import functional as F

from src.recommendation.copmrec.protocol import ScratchTrainingProtocol
from src.recommendation.liger.candidate_guidance import mix_log_probs


class FullCatalogObjective(ScratchTrainingProtocol):
    copmrec_version = "v2"

    @property
    def dense_ce_support(self):
        return dict(
            protocol="copmrec-all-catalog-dense-ce-v1",
            training_support="all_catalog",
            cold_items_in_denominator=True,
            training_targets_must_be_seen=True,
            validation_support="all_catalog",
        )

    @property
    def unified_scratch_contract(self):
        return {**super().unified_scratch_contract, "dense_ce_support": self.dense_ce_support}

    def on_load_checkpoint(self, checkpoint):
        contract = checkpoint.get("copmrec_unified_scratch")
        support = contract.get("dense_ce_support") if isinstance(contract, dict) else None
        expected = self.dense_ce_support
        # dict 相等会接受 True == 1；新的支持集契约必须同时匹配类型和值。
        if (
            not isinstance(support, dict)
            or set(support) != set(expected)
            or any(type(support[key]) is not type(value) or support[key] != value for key, value in expected.items())
        ):
            raise ValueError("Unified full-catalog CE checkpoint support contract mismatch.")
        super().on_load_checkpoint(checkpoint)

    def _joint_losses(self, target_ids, encoded, mask, query, rows, *, training=False, content_logits=None):
        # 保留父 loss 的 decoder→目录投影顺序，避免改变共享 dropout 的随机数使用。
        tokens = target_ids.long() + self.offsets
        outputs = self.transformer(encoder_outputs=encoded, attention_mask=mask, labels=tokens, use_cache=False)
        logits = self.dense_logits(query) if content_logits is None else content_logits
        dense_loss = F.cross_entropy(logits, rows)
        processor = self.training_candidate_processor(logits)
        prefix = tokens.new_full((len(tokens), 1), self.transformer.config.decoder_start_token_id)
        target_log_probs = []
        for depth in range(self.num_hierarchies):
            _, gen, content, _ = processor.distributions(prefix, outputs.logits[:, depth].float())
            target = target_ids[:, depth, None].long()
            target_log_probs.append(
                mix_log_probs(gen.gather(1, target), content.gather(1, target), self.dynamic_gate.bias)
            )
            prefix = torch.cat((prefix, tokens[:, depth, None]), dim=1)
        mixture_loss = -torch.stack(target_log_probs).mean()
        return dict(
            loss=outputs.loss + dense_loss + mixture_loss,
            sid_loss=outputs.loss,
            content_loss=dense_loss,
            mixture_loss=mixture_loss,
            mixture_alpha=self.dynamic_gate.bias.sigmoid().detach(),
        )
