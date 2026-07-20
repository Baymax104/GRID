## ADDED Requirements

### Requirement: dataset 配置中的原始数据源组件 SHALL 统一命名为 data_reader
项目中的 dataset 配置与相关调用侧，对外暴露原始文件数据源组件时必须统一使用 `data_reader` 术语，而不是继续使用 `data_iterator`。

#### Scenario: 配置 dataclass 使用 data_reader 字段
- **WHEN** 维护者检查 `SequenceDatasetConfig` 与 `ItemDatasetConfig`
- **THEN** 这些 dataclass MUST 使用 `data_reader` 字段承载原始数据源组件
- **THEN** 这些 dataclass MUST NOT 再暴露 `data_iterator` 作为当前字段名

#### Scenario: Hydra data 配置使用 data_reader key
- **WHEN** 维护者检查 `configs/data/*.yaml` 中的原始数据读取组件定义
- **THEN** 顶层配置 key 与插值引用 MUST 使用 `data_reader`
- **THEN** `_target_` 路径 MUST 指向 reader 语义的模块与类名

#### Scenario: 调用侧使用 reader 语义类型名
- **WHEN** 维护者检查 dataset / datamodule / 文档中对该组件的引用
- **THEN** 这些引用 MUST 使用 reader 语义的模块名、类型名或字段名
