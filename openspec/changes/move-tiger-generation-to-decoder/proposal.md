## Why

TIGER generation 目前由 `Tiger` 同时协调 encoder、decoder 输入组装、beam search、prefix validation 和 Lightning 训练流程，导致 `Tiger` 类过深且 decoder 行为边界不清晰。将 decoder forward 与 generation 细节收敛到 `TigerDecoder` 可以让 `Tiger` 保持 orchestration 角色，同时让 decoder 自己承载 teacher-forcing 与 autoregressive 生成语义。

## What Changes

- 将 decoder 输入组装逻辑从 `Tiger.decoder_forward_pass()` 移入 `TigerDecoder.forward()`。
- 将自回归 generation 循环和 per-step beam search 从 `Tiger.generate()` / `_beam_search_one_step()` 移入 `TigerDecoder.generate()`。
- 将 decoder-side prefix validation 逻辑迁入 `TigerDecoder`，由 decoder 持有模型侧 `semantic_ids` 与 `should_check_prefix`。
- 保持共享 SID embedding table 由 `Tiger` 创建，并通过构造参数传给 `TigerEncoder` 和 `TigerDecoder`。
- 保持 `Tiger` 对 Lightning train/eval/predict lifecycle、encoder 调用和 evaluator 调用的协调职责。

## Capabilities

### New Capabilities

- None.

### Modified Capabilities

- `self-contained-tiger-generation-model`: TIGER generation SHALL coordinate through `Tiger`, while decoder-specific forward and autoregressive generation behavior SHALL be owned by `TigerDecoder`.

## Impact

- Affected code: `src/recommendation/tiger/tiger.py`, `src/recommendation/tiger/decoder.py`.
- No external Hydra `_target_` change is required.
- No checkpoint compatibility guarantee is required for this in-flight TIGER refactor.
