## ADDED Requirements

### Requirement: TIGER batch containers SHALL expose explicit model semantics
TIGER sequence 数据链路 SHALL 使用专用 runtime containers 表达模型输入、训练标签和 generated labels 输出，不再复用通用 sequential masked-token dataclass 名称或字段。

#### Scenario: TIGER model input has explicit fields
- **WHEN** 维护者检查 TIGER sequence collate 返回的模型输入对象
- **THEN** 该对象 MUST 暴露 `input_ids` 字段保存 encoder 输入 token tensor
- **AND** 该对象 MUST 暴露 `attention_mask` 字段保存 encoder attention mask tensor
- **AND** 该对象 MUST 暴露 `output_keys` 字段保存推理输出归属 key，训练时可为 `None`
- **AND** 该对象 MUST NOT 暴露 `transformed_sequences` 或 `user_id_list` 字段

#### Scenario: TIGER label data has target semantic IDs
- **WHEN** TIGER training collate 返回 label data
- **THEN** label data MUST 暴露 `target_ids` 字段
- **AND** `target_ids` MUST 具有 `(batch_size, num_hierarchies)` 形状
- **AND** label data MUST NOT 暴露 `labels`、`label_location` 或 `attention_mask` 字段

#### Scenario: GeneratedLabels has only input and target tensors
- **WHEN** TIGER label callable 生成 `GeneratedLabels`
- **THEN** `GeneratedLabels` MUST 暴露 `input_ids` 字段保存 masked input sequence
- **AND** `GeneratedLabels` MUST 暴露 `target_ids` 字段保存目标 semantic IDs
- **AND** `GeneratedLabels` MUST NOT 暴露 `sequence`、`labels`、`label_location` 或 `attention_mask` 字段

### Requirement: TIGER collate SHALL construct TIGER-specific batch objects
TIGER train and inference collate functions SHALL construct TIGER-specific model input and label data objects directly from the configured sequence field and output key field.

#### Scenario: training collate returns model input and label data
- **WHEN** `collate_fn_train` receives rows containing the configured sequence field and label callable
- **THEN** it MUST return `(TigerModelInput, TigerLabelData)`
- **AND** `TigerModelInput.input_ids` MUST come from the label output masked input IDs
- **AND** `TigerModelInput.attention_mask` MUST be computed from `TigerModelInput.input_ids != padding_token`
- **AND** `TigerLabelData.target_ids` MUST come from the label output target IDs

#### Scenario: inference collate preserves output keys outside model input
- **WHEN** `collate_fn_inference_for_sequence` receives a field matching `id_field_name`
- **THEN** it MUST store that field in `TigerModelInput.output_keys`
- **AND** it MUST NOT store that field in `TigerModelInput.input_ids`
- **AND** attention masks MUST be computed from the non-id sequence input field
