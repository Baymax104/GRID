"""作者 LETTER-TIGER 的标准 T5、温度训练与目录约束生成。"""

import math

import torch
import torch.nn.functional as F
from torch import nn
from transformers import LogitsProcessor, LogitsProcessorList, T5Config, T5ForConditionalGeneration


class LetterPrefixLogitsProcessor(LogitsProcessor):
    """批量查询目录 trie，避免 HF 回调逐 beam 读取 CUDA 前缀。"""

    def __init__(self, prefixes: dict[tuple[int, ...], list[int]]):
        self.prefixes = prefixes

    def __call__(self, input_ids: torch.Tensor, scores: torch.Tensor):
        row_indices, token_indices = [], []
        # 每步仅一次设备到主机传输；索引也合并后一次性写入 mask。
        for row_index, prefix in enumerate(input_ids.tolist()):
            allowed = self.prefixes.get(tuple(prefix))
            if allowed is None:
                if prefix[-1] in (0, 1):
                    allowed = [0]
                else:
                    raise ValueError("Generation reached an unknown LETTER prefix.")
            if not allowed:
                raise ValueError("LETTER prefix has an empty allowed token set.")
            row_indices.extend([row_index] * len(allowed))
            token_indices.extend(allowed)
        rows = torch.tensor(row_indices, dtype=torch.long, device=scores.device)
        tokens = torch.tensor(token_indices, dtype=torch.long, device=scores.device)
        mask = torch.full_like(scores, -math.inf)
        mask[rows, tokens] = 0
        # 保留 HF 的加性约束和完整词表概率，不重新归一化合法词。
        return scores + mask


class LetterBackbone(nn.Module):
    def __init__(
        self,
        item_keys: torch.Tensor,
        semantic_ids: torch.Tensor,
        codebook_size: int = 256,
        base_vocab_size: int = 32100,
        d_model: int = 128,
        d_ff: int = 1024,
        d_kv: int = 64,
        num_heads: int = 6,
        num_layers: int = 4,
        dropout: float = 0.1,
        temperature: float = 1.0,
        generation_candidates: int = 20,
        top_k: int = 10,
    ):
        super().__init__()
        if not math.isfinite(temperature) or temperature <= 0:
            raise ValueError("LETTER temperature must be finite and positive.")
        if item_keys.ndim != 1 or item_keys.dtype not in (torch.int32, torch.int64):
            raise ValueError("LETTER item keys must be one-dimensional integers.")
        if semantic_ids.shape != (len(item_keys), 4) or semantic_ids.dtype not in (torch.int32, torch.int64):
            raise ValueError("LETTER semantic IDs must be four learned integer codes.")
        if item_keys.unique().numel() != len(item_keys) or (item_keys < 0).any():
            raise ValueError("LETTER item keys must be unique and nonnegative.")
        if (semantic_ids < 0).any() or (semantic_ids >= codebook_size).any():
            raise ValueError("LETTER semantic IDs are outside codebooks.")
        if torch.unique(semantic_ids, dim=0).shape[0] != len(item_keys):
            raise ValueError("LETTER requires unique full semantic IDs.")
        if (
            not 1 <= top_k <= generation_candidates <= len(item_keys)
            or generation_candidates < 2
            or base_vocab_size < 2
        ):
            raise ValueError("LETTER top_k/beam must fit the full catalog.")
        self.temperature, self.top_k = temperature, top_k
        self.generation_candidates, self.codebook_size = generation_candidates, codebook_size
        order = item_keys.argsort()
        self.register_buffer("item_keys", item_keys[order].long())
        self.register_buffer("semantic_ids", semantic_ids[order].long())
        # 作者 tokenizer.add_tokens(sorted(observed_codes))，不是数值顺序offset。
        names = sorted(
            {f"<{chr(97 + level)}_{int(code)}>" for row in semantic_ids.tolist() for level, code in enumerate(row)}
        )
        mapping = {name: base_vocab_size + index for index, name in enumerate(names)}
        tokens = torch.tensor(
            [
                [mapping[f"<{chr(97 + level)}_{int(code)}>"] for level, code in enumerate(row)]
                for row in self.semantic_ids.tolist()
            ],
            dtype=torch.long,
        )
        self.register_buffer("catalog_tokens", tokens)
        self.t5 = T5ForConditionalGeneration(
            T5Config(
                vocab_size=base_vocab_size + len(names),
                d_model=d_model,
                d_ff=d_ff,
                d_kv=d_kv,
                num_heads=num_heads,
                num_layers=num_layers,
                num_decoder_layers=num_layers,
                dropout_rate=dropout,
                pad_token_id=0,
                eos_token_id=1,
                decoder_start_token_id=0,
                tie_word_embeddings=True,
            )
        )
        self.prefixes: dict[tuple[int, ...], list[int]] = {}
        self.token_to_item = {}
        for index, row in enumerate(tokens.tolist()):
            sequence = [0, *row, 1]
            for end in range(1, len(sequence)):
                self.prefixes.setdefault(tuple(sequence[:end]), []).append(sequence[end])
            self.token_to_item[tuple(row)] = index
        self.prefixes = {prefix: sorted(set(values)) for prefix, values in self.prefixes.items()}

    def token_rows(self, raw_keys: torch.Tensor):
        keys = self.item_keys.to(raw_keys.device)
        positions = torch.searchsorted(keys, raw_keys.contiguous())
        if (positions >= len(keys)).any() or not torch.equal(keys[positions.clamp_max(len(keys) - 1)], raw_keys):
            raise ValueError("Item absent from LETTER catalog.")
        return self.catalog_tokens.to(raw_keys.device)[positions]

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, labels: torch.Tensor):
        outputs = self.t5(input_ids=input_ids, attention_mask=attention_mask, labels=labels, use_cache=False)
        loss = F.cross_entropy(
            outputs.logits.reshape(-1, outputs.logits.shape[-1]) / self.temperature,
            labels.reshape(-1),
            ignore_index=-100,
        )
        return loss, outputs.logits

    def allowed_tokens(self, batch_id: int, prefix: torch.Tensor):
        values = self.prefixes.get(tuple(prefix.tolist()))
        if values is None:
            # 完成的EOS路径只允许padding；HF不会对未完成非法路径继续生成。
            if int(prefix[-1]) in (0, 1):
                return [0]
            raise ValueError("Generation reached an unknown LETTER prefix.")
        return values

    @torch.no_grad()
    def generate(self, input_ids: torch.Tensor, attention_mask: torch.Tensor):
        output = self.t5.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=5,
            num_beams=self.generation_candidates,
            num_return_sequences=self.generation_candidates,
            logits_processor=LogitsProcessorList([LetterPrefixLogitsProcessor(self.prefixes)]),
            return_dict_in_generate=True,
            output_scores=True,
            early_stopping=True,
            length_penalty=1.0,
            renormalize_logits=False,
            do_sample=False,
        )
        sequences = output.sequences.reshape(len(input_ids), self.generation_candidates, -1)
        if sequences.shape[-1] != 6 or not (sequences[:, :, -1] == 1).all():
            raise ValueError("LETTER generation did not produce four codes followed by EOS.")
        indices = []
        for rows in sequences[:, :, 1:5].tolist():
            ids = [self.token_to_item[tuple(row)] for row in rows]
            if len(set(ids)) != len(ids):
                raise ValueError("LETTER generation returned duplicate items.")
            indices.append(ids)
        indices = torch.tensor(indices, device=self.item_keys.device)
        scores = output.sequences_scores.reshape(len(input_ids), self.generation_candidates)
        order = scores.argsort(dim=-1, descending=True, stable=True)[:, : self.top_k]
        indices = indices.gather(1, order)
        return self.item_keys[indices], scores.gather(1, order)
