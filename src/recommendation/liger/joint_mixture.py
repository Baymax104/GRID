"""从零联合优化 SID、内容和合法前缀混合概率。"""

import math

import torch
from torch import nn
from torch.nn import functional as F

from src.recommendation.liger.candidate_guidance import ProbabilityMixtureProcessor, mix_log_probs
from src.recommendation.liger.module import Liger


class JointConstantGate(nn.Module):
    def __init__(self, content_only=False):
        super().__init__()
        # 不消耗随机数，保持基座初始化及后续 dropout 随机序列。
        self.bias = nn.Parameter(torch.zeros(()))
        self.content_only = content_only

    def forward(self, features):
        if self.content_only:
            return features.new_full((*features.shape[:-1], 1), torch.inf)
        return self.bias.expand(*features.shape[:-1], 1)


class JointMixtureLiger(Liger):
    def __init__(
        self, *, content_only=False, mechanism_control="learned_mass", inference_mixture_alpha=None, **kwargs
    ):
        if mechanism_control not in {
            "learned_mass",
            "legal_generation",
            "max_mixture",
        }:
            raise ValueError("Unknown mechanism control.")
        if content_only and mechanism_control != "learned_mass":
            raise ValueError("Mechanism control cannot be combined with content_only.")
        if inference_mixture_alpha is not None:
            if not math.isfinite(inference_mixture_alpha) or not 0 <= inference_mixture_alpha <= 1:
                raise ValueError("Inference mixture alpha must be finite and in [0, 1].")
            if content_only or mechanism_control != "learned_mass":
                raise ValueError("Inference mixture alpha cannot be combined with another inference ablation.")
        self.mechanism_control = mechanism_control
        self.inference_mixture_alpha = inference_mixture_alpha
        if kwargs.get("candidate_strategy", "probability_mixture") != "probability_mixture":
            raise ValueError("Joint mixture requires probability_mixture candidates.")
        kwargs["candidate_strategy"] = "probability_mixture"
        kwargs.setdefault("content_mixture_alpha", 0.5)
        if kwargs["content_mixture_alpha"] != 0.5:
            raise ValueError("Joint alpha comes from its checkpoint; use content_only for the alpha1 ablation.")
        super().__init__(**kwargs)
        if self.sid_loss_weight != 1 or self.content_loss_weight != 1:
            raise ValueError("Joint protocol requires unit SID and content loss weights.")
        self.dynamic_gate = JointConstantGate(content_only)

    def candidate_processor(self, content_logits):
        if self.mechanism_control == "learned_mass":
            if self.inference_mixture_alpha is not None:
                return ProbabilityMixtureProcessor(
                    self.semantic_ids,
                    content_logits,
                    self.codebook_size,
                    self.inference_mixture_alpha,
                )
            return ProbabilityMixtureProcessor(
                self.semantic_ids,
                content_logits,
                self.codebook_size,
                0.0,
                gate=self.dynamic_gate,
            )
        aggregation = "max" if self.mechanism_control == "max_mixture" else "mass"
        return ProbabilityMixtureProcessor(
            self.semantic_ids, content_logits, self.codebook_size, 0.0,
            gate=self.dynamic_gate if self.mechanism_control == "max_mixture" else None,
            aggregation=aggregation,
        )

    def losses(self, model_input, target_ids, *, training=False):
        if training and (
            self.dynamic_gate.content_only
            or self.mechanism_control != "learned_mass"
            or self.inference_mixture_alpha is not None
        ):
            raise ValueError("Inference ablations are inference-only.")
        rows = self.lookup_rows(target_ids)
        if training and not self.seen_mask[rows].all():
            raise ValueError("Training target is marked as cold-start.")
        encoded, mask, query = self.encode(model_input.input_ids, model_input.attention_mask)
        tokens = target_ids.long() + self.offsets
        outputs = self.transformer(encoder_outputs=encoded, attention_mask=mask, labels=tokens, use_cache=False)
        # 同一次内容投影的 logits 同时服务内容 CE 与混合项。
        logits = self.dense_logits(query)
        dense_loss = F.cross_entropy(logits.masked_fill(~self.seen_mask[None], -100) if training else logits, rows)
        processor = ProbabilityMixtureProcessor(self.semantic_ids, logits, self.codebook_size, 0.5)
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
            prefix_aggregation = {
                "max_mixture": "max",
            }.get(self.mechanism_control, "mass")
            metadata["joint_mixture_protocol"] = "liger-joint-mixture-v1"
            metadata["mechanism_control"] = self.mechanism_control
            metadata["prefix_aggregation"] = prefix_aggregation
            metadata["content_mixture_alpha"] = (
                0.0 if self.mechanism_control == "legal_generation" else
                1.0 if self.dynamic_gate.content_only else
                self.inference_mixture_alpha if self.inference_mixture_alpha is not None else
                self.dynamic_gate.bias.sigmoid().item()
            )
            metadata["mixture_alpha_source"] = (
                "fixed_inference" if self.inference_mixture_alpha is not None else
                "content_only" if self.dynamic_gate.content_only else
                "legal_generation" if self.mechanism_control == "legal_generation" else
                "checkpoint"
            )
        return output

    def on_save_checkpoint(self, checkpoint):
        super().on_save_checkpoint(checkpoint)
        checkpoint["liger_joint_mixture"] = "liger-joint-mixture-v1"

    def on_load_checkpoint(self, checkpoint):
        super().on_load_checkpoint(checkpoint)
        if checkpoint.get("liger_joint_mixture") != "liger-joint-mixture-v1":
            raise ValueError("Joint mixture requires a matching joint checkpoint.")
