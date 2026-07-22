# Unified Dataset Config Contract

## Purpose

定义统一 dataset runtime config 的类名、字段集合与 Hydra target 约定，避免 item 与 semantic ID 链路继续维护字段完全相同的重复 dataset config 类。

## Requirements

### Requirement: DatasetConfig SHALL be the only dataset runtime config class
数据管线 SHALL 使用统一的 `DatasetConfig` 表达 dataset runtime config。`DatasetConfig` MUST 包含 `data_reader`、`preprocessing_functions`、`shuffle_files` 三个字段，且 MUST NOT 按 item / semantic ID 链路拆分重复 dataset config 类。

#### Scenario: 配置模型只暴露统一 DatasetConfig
- **WHEN** 维护者检查 `src/data/components/config_models.py`
- **THEN** 该模块 MUST 定义 `DatasetConfig`
- **AND** 该模块 MUST NOT 定义 `SemanticIDDatasetConfig` 或 `ItemDatasetConfig`

#### Scenario: DatasetConfig 字段集合保持运行时最小化
- **WHEN** 维护者检查 `DatasetConfig` 字段集合
- **THEN** 字段 MUST 只包含 `data_reader`、`preprocessing_functions`、`shuffle_files`
- **AND** collate-only 或链路派生字段 MUST NOT 出现在 `DatasetConfig` 上

### Requirement: Data YAML SHALL target DatasetConfig for dataset configs
官方 data YAML 中的 dataset config block SHALL 统一通过 Hydra `_target_` 指向 `src.data.components.config_models.DatasetConfig`。

#### Scenario: 官方 experiments 不引用旧 dataset config 类
- **WHEN** 维护者检查 `configs/data/*.yaml` 中的 dataset config `_target_`
- **THEN** `_target_` MUST 使用 `src.data.components.config_models.DatasetConfig`
- **AND** `_target_` MUST NOT 使用 `SemanticIDDatasetConfig` 或 `ItemDatasetConfig`

### Requirement: Dataloader configs SHALL type dataset_config as DatasetConfig
`SequenceDataloaderConfig` 与 `ItemDataloaderConfig` SHALL 将 `dataset_config` 字段类型注解为统一 `DatasetConfig`。

#### Scenario: dataloader config 类型注解统一
- **WHEN** 维护者检查 `SequenceDataloaderConfig` 和 `ItemDataloaderConfig`
- **THEN** 二者的 `dataset_config` 字段 MUST 注解为 `DatasetConfig`
- **AND** 它们 MUST NOT 继续引用 `SemanticIDDatasetConfig` 或 `ItemDatasetConfig`
