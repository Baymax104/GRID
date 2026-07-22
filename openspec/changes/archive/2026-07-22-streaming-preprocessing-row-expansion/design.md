## Context

`SequenceDataset.__iter__()` 当前按一条 row 顺序调用 `dataset_config.preprocessing_functions`，每个 preprocessing function 只能返回单条 row 或 `None`。这使得 TIGER 的 SID causal duplicate augmentation 和 next-item label generation 留在 `collate_fn_train` 中，因为 augmentation 会把一条 sequence 扩展成多条训练样本，而现有 dataset preprocessing 无法表达 row expansion。

直接把 pipeline 改成 list pipeline 会让每个 preprocessing step 遍历扩展后的整个列表，并可能在长序列或大数据集上物化大量中间 rows。本设计采用 streaming flat-map：每个 preprocessing step 可以返回一个 generator，dataset 逐条继续执行后续 preprocessing 并按需 yield。

## Goals / Non-Goals

**Goals:**
- 让 `SequenceDataset` 支持 preprocessing 返回 `None | row | Iterable[row]`。
- 使用 generator / flat-map 语义避免物化完整扩展列表。
- 将 TIGER SID causal duplicate expansion 移到 preprocessing helper。
- 将 TIGER next-item label generation 移到 preprocessing helper，产出 `input_ids` 和 `target_ids`。
- 让 `collate_fn_train` 只负责拼装：normalize / stack / attention mask / `TigerModelInput` / `TigerLabelData`。

**Non-Goals:**
- 不改变 TFRecord reader、worker file assignment 或 dataloader retry 机制。
- 不改变 semantic ID lookup 语义。
- 不改变 TIGER 模型训练目标、target shape 或 decoder loss 语义。
- 不要求 item-level embedding / quantization pipeline 使用 row expansion。

## Decisions

### D1: 使用 streaming flat-map，而不是 list pipeline

`SequenceDataset` 增加内部 helper，将 preprocessing function 的返回值规范化为 iterable rows：

```python
PreprocessingResult = dict[str, Any] | Iterable[dict[str, Any]] | None
```

dataset 对每个原始 row 从第一个 preprocessing step 开始递归/迭代地继续处理：

```text
row -> step_i -> zero/one/many rows -> step_i+1 -> ... -> yield
```

这样扩展出的 rows 不会作为完整 list 反复传给后续每个 step。

### D2: Row expansion helper 返回 generator

新增 `expand_sid_causal_duplicate_sequences(...)`，输入单条 row，按 `sequence_field_name` 和 `sid_hierarchy` 枚举至少两个 item 的连续 subsequences，并逐条 `yield` 新 row。

为了避免长序列先枚举全部候选再采样，helper SHALL 先根据 item 数计算候选总数，必要时采样候选 index，然后边枚举边产出被选中的 subsequence。

### D3: Label generation helper 产出显式字段

新增 row-level label generation preprocessing helper，将 flattened semantic ID sequence 转为：

- `input_ids`: masked encoder input sequence
- `target_ids`: shape `(num_hierarchies,)` 的目标 semantic ID

该 helper 替代当前 collate 内部 `label_generate_functions` 调用路径。它可以复用 `next_k_token_masking` 的核心语义，但以单条 row 为输入输出。

### D4: `collate_fn_train` 只做拼装

`collate_fn_train` 改为接收已预处理字段名：

- `input_field_name: str = "input_ids"`
- `target_field_name: str = "target_ids"`
- `sequence_length`
- `padding_token`

它只负责：

1. 合并 rows。
2. normalize `input_ids`。
3. stack `target_ids`。
4. 计算 `attention_mask`。
5. 返回 `(TigerModelInput, TigerLabelData)`。

`masking_token`、`label_generate_functions`、SID augmentation 参数从 collate 配置移除。

### D5: 训练与评估 preprocessing 分离

`tiger_train` 使用独立 preprocessing chains：

- train chain：semantic ID lookup -> SID causal duplicate expansion -> label generation
- eval/test chain：semantic ID lookup -> label generation，不做 duplicate expansion

这样 eval/test 保持 deterministic，训练扩展逻辑也在配置中显式可见。

## Risks / Trade-offs

- **Risk: row-level `max_num_sequences` 与旧 batch-level `max_batch_size` 语义不同** → 在 spec 和配置中明确新语义是每条原始 row 的最大扩展样本数；训练 batch size 仍由 dataloader 控制。
- **Risk: preprocessing 返回 iterable 后，dict 也属于 Iterable，可能被误判为多 row** → helper 必须先判断 `dict`，再判断 generic iterable。
- **Risk: generator 只能消费一次** → dataset pipeline 只逐条向后流式传递，不缓存 generator。
- **Risk: label generation 前移后 collate 不再知道 masking_token** → masking_token 必须在 preprocessing config 中声明。
- **Risk: active changes 中仍有旧 collate contract 文案** → 同步更新 living specs 和相关 active change artifacts，保证 OpenSpec validation 通过。

## Migration Plan

1. 扩展 `SequenceDataset` preprocessing execution 为 streaming flat-map。
2. 新增 row-level SID expansion 和 label generation preprocessing helpers。
3. 简化 `collate_fn_train` 接口与实现。
4. 迁移 `configs/data/tiger_train.yaml` 的 train/eval preprocessing 和 collate blocks。
5. 更新 specs 和 active change artifacts。
6. 运行 compile、ruff、Hydra smoke、dataset expansion smoke、collate smoke、OpenSpec validation。
