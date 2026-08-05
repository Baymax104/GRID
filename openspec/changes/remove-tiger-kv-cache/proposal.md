## Why

TIGER generation 当前包含 KV cache 分支，但 cache 生命周期不完整：首步 decoder 返回的 cache 会被后续 beam-search 分支覆盖，实际收益有限且增加了 generation 逻辑复杂度。移除该分支可以让 TIGER generation 更直接，并为后续替换为原生 PyTorch Transformer decoder 降低耦合。

## What Changes

- 移除 TIGER generation/decoder forward 中的 `use_cache`、`past_key_values`、`DynamicCache`、`EncoderDecoderCache` 相关逻辑。
- 移除 cache 有效性判断和 beam search 中的 cache reorder。
- generation 每个 hierarchy step 都使用当前完整 generated prefix 重新计算 decoder 输出。
- 保持 beam search 输出语义、prefix validity filtering、train/eval/predict 输入输出形状不变。

## Capabilities

### New Capabilities

- None.

### Modified Capabilities

- `self-contained-tiger-generation-model`: TIGER generation SHALL NOT rely on decoder KV cache state.

## Impact

- Affected code: `src/recommendation/tiger/tiger.py`, `src/recommendation/tiger/decoder.py`.
- No checkpoint compatibility guarantee is required for this in-flight TIGER refactor.
- Inference may recompute decoder prefixes each hierarchy step, but `num_hierarchies` is small and behavior becomes easier to validate.
