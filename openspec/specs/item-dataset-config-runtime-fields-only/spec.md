# item-dataset-config-runtime-fields-only Specification

## Purpose
TBD - created by archiving change remove-item-id-field-from-item-dataset-config. Update Purpose after archive.
## Requirements
### Requirement: ItemDatasetConfig SHALL only contain runtime-consumed fields
`ItemDatasetConfig` SHALL 只保留 `SequenceDataset` 或 `BaseDataModule` 在运行时实际读取的字段。collate 域消费的配置（如 `item_id_field`）SHALL NOT 放在 dataset config 上，而应由 collate 配置直接声明。

#### Scenario: ItemDatasetConfig 不包含 collate 域字段
- **WHEN** 维护者检查 `ItemDatasetConfig` 的字段集合
- **THEN** 它 MUST NOT 包含 `item_id_field` 或其他仅被 collate 函数消费的字段

#### Scenario: collate 配置直接声明 item_id_field
- **WHEN** 维护者查看 item 链路实验的 collate 配置
- **THEN** `item_id_field` MUST 在 collate 的 `_partial_` 配置中直接声明，而不是通过 Hydra 插值从 dataset_config 引用

#### Scenario: ItemDatasetConfig 字段全部有运行时消费方
- **WHEN** 维护者检查 `ItemDatasetConfig` 的每个字段
- **THEN** 每个字段 MUST 被 `SequenceDataset.__init__` 或 `BaseDataModule` 在运行时读取（如 `data_reader`、`preprocessing_functions`、`shuffle_files`）

