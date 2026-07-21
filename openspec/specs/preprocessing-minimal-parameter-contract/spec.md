# preprocessing-minimal-parameter-contract Specification

## Purpose
TBD - created by archiving. Update Purpose after archive.

## Requirements
### Requirement: preprocessing 函数 SHALL 仅接收最小必要参数
preprocessing 函数 SHALL 保持纯函数风格：接收输入行以及完成当前变换所需的最小必要参数，而不是接收模块配置对象或宽泛的 metadata 容器。

#### Scenario: field type conversion 使用局部参数
- **WHEN** preprocessing 需要把字段转换为张量
- **THEN** 它 MUST 只接收该步骤所需的局部字段类型映射，而不是整个 dataset config

#### Scenario: embedding lookup 使用局部参数
- **WHEN** preprocessing 需要把 sparse id 映射为 embedding
- **THEN** 它 MUST 只接收所需的 embedding lookup 依赖（如单个 bundle 或最小映射），而不是整个 dataset config

#### Scenario: feature filtering 使用局部参数
- **WHEN** preprocessing 需要筛选或重命名字段
- **THEN** 它 MUST 只接收该步骤所需的 feature 名称、保留策略或重命名映射等局部参数
