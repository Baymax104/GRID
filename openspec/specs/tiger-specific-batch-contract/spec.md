# TIGER Specific Batch Contract

## Purpose

定义 TIGER sequence 数据链路的专用 batch、label 和 generated label fields，使模型输入、训练标签与推理输出 key 的语义直接对应生成式推荐任务。

## Requirements

### Requirement: TIGER batch containers SHALL expose explicit model semantics
TIGER sequence 数据链路 SHALL 使用专用 runtime containers 表达模型输入、训练标签和 generated label fields，不再复用通用 sequential masked-token dataclass 名称或字段。

#### Scenario: TIGER model input has explicit fields
- **WHEN** 维护者检查 TIGER sequence collate 返回的模型输入对象
- **THEN** 该对象 MUST 暴露 `input_ids` 字段保存 encoder 输入 token tensor
- **AND** 该对象 MUST 暴露 `attention_mask` 字段保存 encoder attention mask tensor
- **AND** 该对象 MUST 暴露 `output_keys` 字段保存推理输出归属 key，训练时可为 `None`
- **AND** 该对象 MUST NOT 暴露旧的 dict-based sequence container 或 user-id-specific key 字段

#### Scenario: TIGER label data has target semantic IDs
- **WHEN** TIGER training collate 返回 label data
- **THEN** label data MUST 暴露 `target_ids` 字段
- **AND** `target_ids` MUST 具有 `(batch_size, num_hierarchies)` 形状
- **AND** label data MUST NOT 暴露旧的 dict labels、label-position metadata 或 label-only mask 字段

#### Scenario: Generated label fields have only input and target tensors
- **WHEN** TIGER label preprocessing generates fields for a row
- **THEN** it MUST produce `input_ids` storing the masked input sequence
- **AND** it MUST produce `target_ids` storing target semantic IDs
- **AND** it MUST NOT produce旧的 generic sequence、flatten labels、label-position metadata 或 label-only mask 字段

### Requirement: TIGER collate SHALL construct TIGER-specific batch objects
TIGER train and inference SHALL use `collate_fn_sequence` to construct TIGER-specific model input and optional label data objects directly from configured input, attention mask, target, and output key fields on row dictionaries.

#### Scenario: training collate returns model input and label data
- **WHEN** `collate_fn_sequence` receives `list[dict[str, torch.Tensor]]` rows containing the configured input, attention mask, and target fields
- **THEN** it MUST return `(TigerModelInput, TigerLabelData)`
- **AND** `TigerModelInput.input_ids` MUST come from the preprocessed input field
- **AND** `TigerModelInput.attention_mask` MUST come from the preprocessed attention mask field
- **AND** `TigerLabelData.target_ids` MUST come from the preprocessed target field

#### Scenario: training collate does not generate labels
- **WHEN** `collate_fn_sequence` assembles a training batch
- **THEN** it MUST NOT call label generator functions
- **AND** it MUST NOT apply SID causal duplicate sampling

#### Scenario: inference collate preserves output keys outside model input
- **WHEN** `collate_fn_sequence` receives a field matching `output_key_field_name`
- **THEN** it MUST store that field in `TigerModelInput.output_keys`
- **AND** it MUST NOT store that field in `TigerModelInput.input_ids`
- **AND** attention masks MUST come from the configured preprocessed attention mask field
