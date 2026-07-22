## Context

TIGER train/eval preprocessing 已经负责 SID causal duplicate row expansion 与 next-k label generation，并在 collate 前产出 `input_ids` / `target_ids`。旧的 `collate_fn_train` 和 `collate_fn_inference_for_sequence` 仍会承担 batch-level sequence normalization 或专用推理拼装职责。统一为 `collate_fn_sequence` 后，collate 只消费 preprocessing 产物。

`SequenceDataset` 的 preprocessing contract 是 row-level streaming flat-map：每个 preprocessing function 接收单条 row，可返回 `None`、单条 row 或 row iterable。因此不能直接把 batch-level `normalize_sequence_batch(sequences=...)` 作为 preprocessing step；需要一个 row-level helper。

## Goals / Non-Goals

**Goals:**

- 新增名为 `normalize_sequence` 的 row-level preprocessing function。
- 在 `generate_next_k_labels` 之后规范化 `input_ids`，并生成与规范化后 `input_ids` 对齐的 `attention_mask`。
- 让 `collate_fn_sequence` 只 stack input、`attention_mask`、可选 `target_ids` 和可选 output keys，并构造 `TigerModelInput` / `TigerLabelData`。
- 将 TIGER train/eval/inference 的 `sequence_length` 从 collate 配置迁移到 preprocessing 配置。

**Non-Goals:**

- 不改变推理 output key 语义；`user_id` 仍只进入 `TigerModelInput.output_keys`。
- 不改变 SID causal duplicate sampling 语义。
- 不改变 `generate_next_k_labels` 的 target 选择和 masking 语义。
- 不改变 `TigerModelInput` / `TigerLabelData` dataclass 字段名。

## Decisions

### Decision 1: 新增 row-level `normalize_sequence`，不直接复用 batch-level callable 作为配置目标

`SequenceDataset` preprocessing 每次处理一条 row，而 `normalize_sequence_batch` 接收 list of tensors 并返回 batch tensor。新增 `normalize_sequence(row, input_field_name, attention_mask_field_name, sequence_length, padding_token)` 可以符合现有 preprocessing contract，并让配置直接表达模型输入准备步骤。

备选方案是让 dataset 支持 batch-level preprocessing，但这会破坏当前 streaming row expansion 模型，并扩大变更范围。

### Decision 2: `attention_mask` 在 `normalize_sequence` 内生成

`attention_mask` 的正确长度和 padding 位置依赖规范化后的 `input_ids`。如果在 `generate_next_k_labels` 内生成，后续 pad/trim 可能导致 mask 与最终输入不一致。因此 `normalize_sequence` 同时负责输出 fixed-length `input_ids` 和同长度 `attention_mask`。

### Decision 3: preprocessing 顺序固定为 expansion → label generation → normalization

SID causal duplicate expansion 必须先于 label generation，使每个上采样子序列独立生成自己的 next-k label。label generation 必须先于 normalization，使 `target_ids` 来自真实或上采样后的 semantic-ID 序列末尾，而不是来自已 pad/trim 的固定长度输入。

### Decision 4: `collate_fn_sequence` 读取预处理字段并 stack

变更后 `collate_fn_sequence` 仅接收 `list[dict[str, torch.Tensor]]` rows，对 row 字段做存在性检查，然后直接：

- `torch.stack(batch[input_field_name], dim=0)`
- `torch.stack(batch[attention_mask_field_name], dim=0)`
- 可选 `torch.stack(batch[target_field_name], dim=0)`
- 可选 `torch.stack(batch[output_key_field_name], dim=0)`

`padding_token` 和 `sequence_length` 不再属于 `collate_fn_sequence` 的必要参数；`sequence_length` 属于 `normalize_sequence`，`padding_token` 仅由 `normalize_sequence` 用于 padding 与 mask 生成。

## Risks / Trade-offs

- **Risk:** 旧配置若直接调用 `collate_fn_sequence` 且未预先生成 `attention_mask` 会失败。→ **Mitigation:** 更新 `configs/data/tiger_train.yaml` / `configs/data/tiger_inference.yaml` 并在错误信息中明确要求缺失字段。
- **Risk:** row-level normalization 与 `normalize_sequence_batch` 的 trim 语义不一致。→ **Mitigation:** 复用或抽取同一单序列 pad/trim 逻辑，并增加 pad/trim smoke。
- **Risk:** `sequence_length` 迁移后 specs 中仍残留 collate-owned 参数描述。→ **Mitigation:** 同步更新 TIGER data/collate/batch specs，并运行 OpenSpec validation。
