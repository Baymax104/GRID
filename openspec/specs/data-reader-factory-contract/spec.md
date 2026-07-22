# data-reader-factory-contract Specification

## Purpose
TBD - created by archiving. Update Purpose after archive.

## Requirements
### Requirement: dataset config 中的 data_reader SHALL 采用 factory contract
在新 data pipeline 中，`DatasetConfig` 暴露的 `data_reader` SHALL 是可调用的 reader factory（通常由 Hydra `_partial_` 生成），由 dataset 在运行时基于当前文件列表按需实例化。

#### Scenario: DatasetConfig 暴露 reader factory
- **WHEN** 维护者检查 `DatasetConfig`
- **THEN** 其中的 `data_reader` 字段 MUST 表示 reader factory，而不是长期复用的 reader instance

#### Scenario: SequenceDataset 运行时实例化 reader
- **WHEN** `SequenceDataset` 开始为某个 worker 加载数据
- **THEN** 它 MUST 基于当前 worker 的 `list_of_file_paths` 调用 `data_reader` factory 构造 reader
- **THEN** 它 MUST NOT 依赖对长期持有 reader instance 的内部状态 mutation 来切换文件列表

#### Scenario: reader suffix 获取与 factory contract 一致
- **WHEN** datamodule 需要根据 dataset config 判断文件后缀
- **THEN** 相关逻辑 MUST 与 `data_reader` 为 factory 的 contract 保持一致
