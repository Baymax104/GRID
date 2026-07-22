## MODIFIED Requirements

### Requirement: sequence 链路 config 类 SHALL 精简为运行时字段
`DatasetConfig` 和 `SequenceDataloaderConfig` SHALL 只暴露运行时消费的字段，删除旧架构遗留字段和无消费方的兼容字段。TIGER sequence 链路 SHALL 使用统一 `DatasetConfig`，而不是语义 ID 专用 dataset config 类。

#### Scenario: DatasetConfig 不保留 sequence 旧协议字段
- **WHEN** 维护者检查 `DatasetConfig` 定义和 TIGER dataset config blocks
- **THEN** 它 MUST 不再保留 `semantic_id_map`、`keep_user_id`、`user_id_field`、`features_to_consider`、`num_placeholder_tokens_map`、`field_type_map`、`min_sequence_length`、`feature_map`、`file_format`
- **AND** `SemanticIDDatasetConfig` 和 `SequenceDatasetConfig` MUST 被删除

#### Scenario: SequenceDataloaderConfig 不保留旧 shuffle 字段
- **WHEN** 维护者检查 `SequenceDataloaderConfig` 定义
- **THEN** 它 MUST 不再保留 `should_shuffle_rows`，shuffle 语义只通过 `dataset_config.shuffle_files` + reader `shuffle_rows` 表达

#### Scenario: SequenceDataloaderConfig 不保留 collate-only 字段
- **WHEN** 维护者检查 `SequenceDataloaderConfig` 定义
- **THEN** 它 MUST 不再保留 `labels`、`sequence_length`、`masking_token`、`padding_token` 这类仅用于调用 collate function 的字段
