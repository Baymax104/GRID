# data-shuffle-contract Specification

## Purpose
TBD - created by archiving. Update Purpose after archive.

## Requirements
### Requirement: 新 data pipeline SHALL 使用 shuffle_files 与 shuffle_rows 表达 shuffle 语义
在新的 data pipeline contract 中，shuffle 语义 SHALL 拆分为文件层与样本层两个维度，不再以 `should_shuffle_rows` 作为目标表达方式。

#### Scenario: 文件层 shuffle 由 dataset config 控制
- **WHEN** 维护者检查 dataset config 中与文件顺序相关的配置
- **THEN** 文件顺序 shuffle MUST 通过 `shuffle_files` 表达

#### Scenario: 样本层 shuffle 由 data_reader 控制
- **WHEN** 维护者检查 reader 内部的样本顺序 shuffle 配置
- **THEN** 该行为 MUST 通过 `data_reader` 上的 `shuffle_rows` 表达

#### Scenario: rkmeans_train 不再以 should_shuffle_rows 作为目标 contract
- **WHEN** 维护者检查 `rkmeans_train` 的新 data 链路
- **THEN** 其稳定后的主链路 MUST NOT 依赖 `should_shuffle_rows` 作为目标 contract
- **THEN** 若仓库中仍存在该字段，也只可视为其他未迁移实验的过渡残留，而不是新模板的一部分
