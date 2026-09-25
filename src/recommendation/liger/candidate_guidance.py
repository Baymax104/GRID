"""在合法 SID 树上用最大内容势差引导有限 beam 搜索。"""

import math

import torch
from transformers import LogitsProcessor


def mix_log_probs(generation: torch.Tensor, content: torch.Tensor, logit: torch.Tensor) -> torch.Tensor:
    """用单个可学习 logit 稳定混合两路对数概率。"""
    log_alpha = -torch.nn.functional.softplus(-logit)
    log_one_minus_alpha = -torch.nn.functional.softplus(logit)
    return torch.logaddexp(generation + log_one_minus_alpha, content + log_alpha)


class ContentPrefixProcessor(LogitsProcessor):
    def __init__(self, semantic_ids, content_logits, codebook_size, weight):
        if not math.isfinite(weight) or weight < 0:
            raise ValueError("Guidance weight must be finite and nonnegative.")
        if semantic_ids.ndim != 2 or content_logits.ndim != 2 or content_logits.shape[1] != len(semantic_ids):
            raise ValueError("Content scores and SID catalog must align.")
        if not torch.isfinite(content_logits).all():
            raise ValueError("Guidance requires finite content scores.")
        self.k = codebook_size
        self.h = semantic_ids.shape[1]
        self.batch = content_logits.shape[0]
        self.weight = weight
        self.codes, self.potentials = [], []
        codes = torch.zeros(len(semantic_ids), device=semantic_ids.device, dtype=torch.long)
        # 每个深度只存一个 batch × 前缀数表，不展开 beam × 商品数。
        for depth in range(self.h):
            codes = codes * self.k + semantic_ids[:, depth]
            unique, inverse = torch.unique(codes, sorted=True, return_inverse=True)
            values = content_logits.new_full((self.batch, len(unique)), -torch.inf)
            values.scatter_reduce_(1, inverse[None].expand(self.batch, -1), content_logits, reduce="amax")
            self.codes.append(unique)
            self.potentials.append(values)
        self.root = content_logits.max(-1).values

    def _lookup(self, depth, codes, users):
        catalog_codes = self.codes[depth]
        indices = torch.searchsorted(catalog_codes, codes).clamp_max(len(catalog_codes) - 1)
        valid = catalog_codes[indices] == codes
        return self.potentials[depth][users, indices], valid

    def _continuations(self, input_ids, scores):
        depth = input_ids.shape[1] - 1  # 排除 decoder BOS。
        if not 0 <= depth < self.h or len(scores) % self.batch:
            raise ValueError("Unexpected beam layout or SID depth.")
        users = torch.arange(self.batch, device=scores.device).repeat_interleave(len(scores) // self.batch)
        prefix = torch.zeros(len(scores), device=scores.device, dtype=torch.long)
        valid_parent = torch.ones(len(scores), device=scores.device, dtype=torch.bool)
        for j in range(depth):
            token = input_ids[:, j + 1] - (1 + j * self.k)
            valid_parent &= (token >= 0) & (token < self.k)
            prefix = prefix * self.k + token
        if depth:
            parent, exists = self._lookup(depth - 1, prefix, users)
            valid_parent &= exists
        else:
            parent = self.root[users]
        children = prefix[:, None] * self.k + torch.arange(self.k, device=scores.device)[None]
        potential, valid = self._lookup(depth, children, users[:, None])
        valid &= valid_parent[:, None]
        return depth, parent, potential, valid

    def __call__(self, input_ids, scores):
        depth, parent, potential, valid = self._continuations(input_ids, scores)
        start = 1 + depth * self.k
        local = scores[:, start : start + self.k]
        # 不重新归一化：累计势差恰为 weight*(c(item)-V(root))。
        guided = local + self.weight * (potential - parent[:, None])
        output = torch.full_like(scores, -torch.inf)
        output[:, start : start + self.k] = guided.masked_fill(~valid, -torch.inf)
        return output


class ProbabilityMixtureProcessor(ContentPrefixProcessor):
    """合法分支生成概率与精确内容质量的条件概率混合。"""

    def __init__(self, semantic_ids, content_logits, codebook_size, alpha, gate=None, aggregation="mass"):
        if aggregation not in {"mass", "max"}:
            raise ValueError("Unknown prefix aggregation.")
        if not math.isfinite(alpha) or not 0 <= alpha <= 1:
            raise ValueError("Mixture alpha must be finite and in [0, 1].")
        super().__init__(semantic_ids, content_logits, codebook_size, weight=0)
        self.alpha = alpha
        self.gate = gate
        self.aggregation = aggregation
        if aggregation == "max":
            return
        codes = torch.zeros(len(semantic_ids), device=semantic_ids.device, dtype=torch.long)
        for depth in range(self.h):
            codes = codes * self.k + semantic_ids[:, depth]
            inverse = torch.searchsorted(self.codes[depth], codes)[None].expand(self.batch, -1)
            maxima = self.potentials[depth]
            sums = torch.zeros_like(maxima).scatter_add_(1, inverse, (content_logits - maxima.gather(1, inverse)).exp())
            if aggregation == "mass":
                self.potentials[depth] = maxima + sums.log()
        if aggregation == "mass":
            self.root = content_logits.logsumexp(-1)

    def distributions(self, input_ids, scores):
        depth, _, mass, valid = self._continuations(input_ids, scores)
        start = 1 + depth * self.k
        active = valid.any(-1, keepdim=True)

        def conditional(values):
            # 无效beam先用有限占位值归一化，再恢复全负无穷，避免NaN。
            masked = values.masked_fill(~valid, -torch.inf)
            return torch.where(active, masked, 0).log_softmax(-1).masked_fill(~valid, -torch.inf)

        gen = conditional(scores[:, start : start + self.k])
        content = conditional(mass)
        return depth, gen, content, valid

    def __call__(self, input_ids, scores):
        depth, gen, content, valid = self.distributions(input_ids, scores)
        start = 1 + depth * self.k
        if self.gate is not None:
            mixed = mix_log_probs(gen, content, self.gate.bias)
        elif self.alpha == 0:
            mixed = gen
        elif self.alpha == 1:
            mixed = content
        else:
            mixed = torch.logaddexp(gen + math.log1p(-self.alpha), content + math.log(self.alpha))
        output = torch.full_like(scores, -torch.inf)
        output[:, start : start + self.k] = mixed
        return output
