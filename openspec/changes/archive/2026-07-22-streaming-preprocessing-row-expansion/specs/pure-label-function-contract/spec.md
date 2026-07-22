## MODIFIED Requirements

### Requirement: Label generators SHALL be pure callable functions
数据管线中的 TIGER label generator SHALL 以纯函数 callable 暴露。label callable MUST run as dataset preprocessing, consume a single semantic-ID-flattened sequence row, and produce explicit `input_ids` and `target_ids` fields. It MUST NOT require callers to use class instances, `.transform_label(...)`, or collate-owned label generator maps.

#### Scenario: preprocessing directly calls label callable
- **WHEN** `SequenceDataset` executes TIGER label preprocessing
- **THEN** it MUST call the configured label generator as a normal callable within `dataset_config.preprocessing_functions`
- **AND** `collate_fn_train` MUST NOT call `label_generate_functions[field_name](...)`

#### Scenario: label_functions 模块不暴露类协议
- **WHEN** 维护者检查 `src/data/components/label_functions.py`
- **THEN** 该模块 MUST NOT 定义 `LabelFunction` 抽象基类
- **AND** 该模块 MUST NOT 定义 `Identity` 或 `NextKTokenMasking` label function 类

#### Scenario: label_functions 模块不暴露未使用 identity label
- **WHEN** 维护者检查 `src/data/components/label_functions.py`
- **THEN** 该模块 MUST NOT 定义或导出未使用的 identity-style label callable

### Requirement: next_k_token_masking SHALL preserve existing masking semantics
`next_k_token_masking` or its row-level replacement SHALL 保持 TIGER 当前行为：每条 sequence 最后 `next_k` 个 token 作为目标 semantic IDs，第一个 label 位置替换为 `masking_token`，其余 label 位置替换为 `padding_token`。

#### Scenario: next_k target IDs are row-level
- **WHEN** label preprocessing receives one semantic-ID-flattened sequence row
- **THEN** produced `target_ids` MUST have shape `(next_k,)`
- **AND** produced `input_ids` MUST be the masked input sequence for that row
- **AND** produced output MUST NOT contain label-position metadata fields

### Requirement: Label config SHALL use direct Hydra partial callables
官方配置中的 label generator SHALL 通过 Hydra `_partial_` 直接声明纯函数 callable in preprocessing configuration，而不是声明类实例、额外 `transform` wrapper，或 collate-local label generator map。

#### Scenario: TIGER label config is preprocessing-owned
- **WHEN** 维护者检查 `configs/data/tiger_train.yaml`
- **THEN** label generation MUST appear inside train/eval dataset preprocessing chains
- **AND** collate blocks MUST NOT contain `label_generate_functions`
- **AND** label entries MUST NOT contain `transform:` wrapper
