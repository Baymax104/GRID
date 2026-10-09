"""从零联合优化 SID、内容和合法前缀混合概率。"""

import torch
from torch import nn
from torch.nn import functional as F

from src.recommendation.liger.candidate_guidance import ProbabilityMixtureProcessor, mix_log_probs
from src.recommendation.liger.module import Liger


class JointConstantGate(nn.Module):
    def __init__(self):
        super().__init__()
        # 不消耗随机数，保持基座初始化及后续 dropout 随机序列。
        self.bias = nn.Parameter(torch.zeros(()))

    def forward(self, features):
        return self.bias.expand(*features.shape[:-1], 1)


class JointMixtureCoPMRec(Liger):
    copmrec_version = "v2"

    def __init__(
        self,
        *,
        content_only=False,
        mechanism_control="learned_mass",
        inference_mixture_alpha=None,
        fixed_mixture_alpha=None,
        **kwargs,
    ):
        # 旧参数只作明确拒绝，正式运行面不再实现扫描或其他混合支路。
        if (
            content_only
            or mechanism_control != "learned_mass"
            or inference_mixture_alpha is not None
            or fixed_mixture_alpha is not None
        ):
            raise ValueError("CoPMRec v2 only supports learned mass alpha; iteration controls were retired.")
        if kwargs.get("candidate_strategy", "probability_mixture") != "probability_mixture":
            raise ValueError("Joint mixture requires probability_mixture candidates.")
        kwargs["candidate_strategy"] = "probability_mixture"
        kwargs.setdefault("content_mixture_alpha", 0.5)
        if kwargs["content_mixture_alpha"] != 0.5:
            raise ValueError("Joint alpha is learned from its checkpoint.")
        super().__init__(**kwargs)
        if self.sid_loss_weight != 1 or self.content_loss_weight != 1:
            raise ValueError("Joint protocol requires unit SID and content loss weights.")
        self.dynamic_gate = JointConstantGate()

    def candidate_processor(self, content_logits):
        return ProbabilityMixtureProcessor(
            self.semantic_ids, content_logits, self.codebook_size, 0.0, gate=self.dynamic_gate
        )

    def losses(self, model_input, target_ids, *, training=False):
        rows = self.lookup_rows(target_ids)
        if training and not self.seen_mask[rows].all():
            raise ValueError("Training target is marked as cold-start.")
        encoded, mask, query = self.encode(model_input.input_ids, model_input.attention_mask)
        return self._joint_losses(target_ids, encoded, mask, query, rows, training=training)

    def training_candidate_processor(self, content_logits):
        """训练混合项使用逐层合法后代 mass。"""
        return ProbabilityMixtureProcessor(self.semantic_ids, content_logits, self.codebook_size, 0.5)

    def _joint_losses(self, target_ids, encoded, mask, query, rows, *, training=False, content_logits=None):
        """同一表示和目录投影服务联合目标。"""
        tokens = target_ids.long() + self.offsets
        outputs = self.transformer(encoder_outputs=encoded, attention_mask=mask, labels=tokens, use_cache=False)
        # 同一次内容投影的 logits 同时服务内容 CE 与混合项。
        logits = self.dense_logits(query) if content_logits is None else content_logits
        dense_loss = F.cross_entropy(logits.masked_fill(~self.seen_mask[None], -100) if training else logits, rows)
        processor = self.training_candidate_processor(logits)
        prefix = tokens.new_full((len(tokens), 1), self.transformer.config.decoder_start_token_id)
        target_log_probs = []
        for depth in range(self.num_hierarchies):
            _, gen, content, _ = processor.distributions(prefix, outputs.logits[:, depth].float())
            target = target_ids[:, depth, None].long()
            # 只取合法目标后再混合，避免无效分支 -inf/-inf 的反向 NaN。
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

    def predict_step(self, batch, batch_idx=0):
        output = super().predict_step(batch, batch_idx)
        if self.candidate_trace:
            metadata = output.auxiliary["liger_candidates"]["metadata"]
            metadata.update(
                joint_mixture_protocol="liger-joint-mixture-v1",
                copmrec_version=self.copmrec_version,
                mechanism_control="learned_mass",
                prefix_aggregation="mass",
                content_mixture_alpha=self.dynamic_gate.bias.sigmoid().item(),
                mixture_alpha_source="checkpoint",
                fixed_mixture_alpha=None,
            )
            if self.path_trace:
                output.auxiliary["liger_paths"]["metadata"].update(metadata)
        return output

    def on_save_checkpoint(self, checkpoint):
        super().on_save_checkpoint(checkpoint)
        checkpoint["liger_joint_mixture"] = "liger-joint-mixture-v1"
        checkpoint["copmrec_version"] = self.copmrec_version
        checkpoint["copmrec_alpha_policy"] = "learned"
        checkpoint["copmrec_fixed_mixture_alpha"] = None

    def on_load_checkpoint(self, checkpoint):
        super().on_load_checkpoint(checkpoint)
        if checkpoint.get("liger_joint_mixture") != "liger-joint-mixture-v1":
            raise ValueError("Joint mixture requires a matching joint checkpoint.")
        if checkpoint.get("copmrec_version") != self.copmrec_version:
            raise ValueError("CoPMRec checkpoint version mismatch.")
        if (
            checkpoint.get("copmrec_alpha_policy") != "learned"
            or checkpoint.get("copmrec_fixed_mixture_alpha") is not None
        ):
            raise ValueError("CoPMRec checkpoint alpha policy mismatch.")
