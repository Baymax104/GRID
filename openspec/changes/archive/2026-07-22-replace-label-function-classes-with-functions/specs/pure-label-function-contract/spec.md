## ADDED Requirements

### Requirement: Label generators SHALL be pure callable functions
数据管线中的 label generator SHALL 以纯函数 callable 暴露。label callable MUST 接收 `sequence`、`padding_token`、`masking_token` 等运行时参数并返回 `GeneratedLabels`，且 MUST NOT 要求调用方通过类实例的 `.transform_label(...)` 方法间接执行。

#### Scenario: collate 直接调用 label callable
- **WHEN** `collate_fn_train` 为某个字段生成 labels
- **THEN** 它 MUST 直接调用 `label_generate_functions[field_name](sequence=..., padding_token=..., masking_token=...)`
- **AND** 它 MUST NOT 访问 `label_generate_functions[field_name].transform` 或 `.transform_label(...)`

#### Scenario: label_functions 模块不暴露类协议
- **WHEN** 维护者检查 `src/data/components/label_functions.py`
- **THEN** 该模块 MUST NOT 定义 `LabelFunction` 抽象基类
- **AND** 该模块 MUST NOT 定义 `Identity` 或 `NextKTokenMasking` label function 类

### Requirement: next_k_token_masking SHALL preserve existing masking semantics
`next_k_token_masking` SHALL 保持旧 `NextKTokenMasking` 行为：每行最后 `next_k` 个非 padding token 作为 labels，第一个 label 位置替换为 `masking_token`，其余 label 位置替换为 `padding_token`。

#### Scenario: next_k labels and locations are stable
- **WHEN** `next_k_token_masking` 接收 shape 为 `(batch_size, sequence_length)` 的序列 batch
- **THEN** 返回的 `labels` MUST 具有 shape `(batch_size * next_k,)`
- **AND** 返回的 `label_location` MUST 具有 shape `(batch_size * next_k, 2)`
- **AND** 每行的 `next_k` 个 label locations MUST 指向该行第一个被 mask 的位置

### Requirement: Label config SHALL use direct Hydra partial callables
官方配置中的 label generator SHALL 通过 Hydra `_partial_` 直接声明纯函数 callable，而不是声明类实例或额外 `transform` wrapper。

#### Scenario: TIGER label config points to pure function
- **WHEN** 维护者检查 `configs/data/tiger_train.yaml` 的 `label_functions`
- **THEN** 每个 label entry MUST 直接 `_target_` 到 `src.data.components.label_functions.next_k_token_masking`
- **AND** 每个 label entry MUST 设置 `_partial_: true`
- **AND** 每个 label entry MUST NOT 包含 `transform:` wrapper
