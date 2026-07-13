## Context

当前推理结果保存流程存在 tensor→dict→tensor 往返转换。`prediction_step` 产出 `(keys_tensor, predictions_tensor)`，通过 `ModelOutput.list_of_row_format` 按 `key_name`/`prediction_name` 展开为 `list[dict]`，writer 再通过 `merge_list_of_keyed_tensors_to_single_tensor` 从 dict 中按同样的名字提取回 tensor。`key_name`/`prediction_name` 同时存在于 YAML 配置、writer、module、ModelOutput 四处，映射职责分散。

关键现状：
- `merged_predictions.pkl`（行格式中间产物）无代码消费者
- `merge_list_of_keyed_tensors_to_single_tensor` 唯一消费者是 writer 的 `_merge_files`
- `list_of_row_format` 唯一消费者是 writer 的 `handle_batch`
- tiger 用 `prediction_value_name`（构造函数参数）而非 writer 注入的 `prediction_name`，注入静默失效
- 最终产物 `merged_predictions_tensor.pt` 格式 `{"keys": tensor, "predictions": tensor}` 不变，下游协议不变

## Goals / Non-Goals

**Goals:**
- 消除 tensor→dict→tensor 往返转换
- 将映射职责收敛到 `prediction_step`（直接构造 `keys` + `predictions`）
- `ModelOutput` 作为结果写入的字段规范层（直接持有 `keys` + `predictions`）
- 去除 writer 中的 `prediction_key_name`/`prediction_name` 配置和 `setup()` 注入逻辑
- 保持最终产物格式和下游协议不变

**Non-Goals:**
- 不改变 `merged_predictions_tensor.pt` 的文件格式（`{"keys": tensor, "predictions": tensor}`）
- 不改变下游 `load_keyed_prediction_bundle`/`lookup_values_in_keyed_prediction_bundle` 协议
- 不改变 `post_processing_functions` 机制（`deduplicate_rows_in_tensor` 作用于 `.pt` 文件）
- 不重构 writer 的分片/分布式 barrier 逻辑

## Decisions

### 决策 1：ModelOutput 直接持有 keys + predictions

**选择**：`ModelOutput.__init__(self, keys, predictions)`，去除 `key_name`/`prediction_name`/`list_of_row_format`/`_convert_to_list`。

**理由**：`ModelOutput` 的职责是定义写入数据的字段规范。当前 `keys`/`predictions` 是所有 inference pipeline 的唯一公共结构，不需要 `key_name`/`prediction_name` 的间接命名——字段名（item_id/embedding 等）是语义概念，在 `prediction_step` 的变量命名中已体现。

**替代方案**：保持 `ModelOutput` 作为命名字段容器（`**fields`），允许每个 pipeline 定义不同字段。否决：当前所有 pipeline 都只产出 keys + predictions 两字段，通用容器引入不必要的复杂性，且 writer 仍需知道哪个字段是 key。

### 决策 2：writer 直接缓存 ModelOutput 对象，保持行数语义的 flush_frequency

**选择**：`rows_buffer: list[dict]` → `buffer: list[ModelOutput]`。`handle_batch` 用 `len(model_output.keys)` 代替 `len(rows)` 判断 flush 时机，保持 `flush_frequency` 的行数语义不变。

**理由**：`flush_frequency=100000`（rkmeans/tiger）如果改为 batch 数语义会导致所有数据缓存在内存中直到结束才 flush。保持行数语义确保内存使用行为不变。

### 决策 3：_flush_buffer 直接 pickle list[ModelOutput]

**选择**：分片文件内容从 `list[dict]` 变为 `list[ModelOutput]`。`ModelOutput` 只持有 `keys`（list 或 tensor）和 `predictions`（tensor），均可 pickle。

**理由**：消除行格式中间层。分片文件是内部实现细节，无外部消费者。

### 决策 4：_merge_files 直接 torch.cat，不再产出 merged_predictions.pkl

**选择**：合并时遍历所有分片，收集所有 `ModelOutput`，对 `keys` 和 `predictions` 分别 `torch.cat`。`keys` 统一转为 `torch.long` tensor（与当前 `merge_list_of_keyed_tensors_to_single_tensor` 的 `torch.tensor([int(row[index_key]) ...], dtype=torch.long)` 行为一致）。不产出 `merged_predictions.pkl`。

**理由**：`merged_predictions.pkl` 无代码消费者。`merged_predictions_tensor.pt` 是唯一有意义的最终产物。

### 决策 5：删除 merge_list_of_keyed_tensors_to_single_tensor

**选择**：从 `tensor_utils.py` 删除该函数。

**理由**：唯一消费者是 writer 的 `_merge_files`，重构后不再需要。

### 决策 6：去除 TransformerBaseModule 的 prediction_key_name/prediction_name 属性

**选择**：删除 `_prediction_key_name`/`_prediction_name` 属性及 setter。

**理由**：这些属性仅在 `predict_step` 中用于构造 `ModelOutput`，而 `TransformerBaseModule.predict_step` 已在前序变更中删除。各子类的 `predict_step` 重构后不再引用这些属性。`BaseBufferedWriter.setup()` 的注入逻辑同步删除。

### 决策 7：去除 SemanticIDEncoderDecoder 的 prediction_key_name/prediction_value_name 构造参数

**选择**：从构造函数删除这两个参数及赋值。

**理由**：`predict_step` 重构后直接 `ModelOutput(keys=ids, predictions=generated_sids)`，不再需要字段名。YAML 配置中也未显式设置这两个参数（使用默认值），删除不影响配置。

## Risks / Trade-offs

- **[不产出 merged_predictions.pkl]** → 无代码消费者，AGENTS.md 文档需更新。风险低。
- **[分片文件格式变化]** → 分片 `.pkl` 文件是内部实现细节，无外部消费者。风险低。
- **[失去多字段扩展性]** → 当前行格式理论上支持每行携带额外字段，但没有任何 pipeline 实际使用。如果未来需要，可重新引入。风险低。
- **[keys 类型混合]** → `predict_step` 中 keys 可能是 `list[int]` 或 `tensor`。writer merge 时统一 `torch.tensor(keys_list, dtype=torch.long)`，与当前行为一致。风险低。
- **[BREAKING 配置变更]** → 3 个 callback YAML 去除字段，旧配置文件不兼容。但这是内部项目，无外部消费者。风险低。
