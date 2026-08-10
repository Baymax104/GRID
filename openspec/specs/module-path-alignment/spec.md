# module-path-alignment Specification

## Purpose
TBD - created by archiving change align-module-paths-and-rename-common. Update Purpose after archive.
## Requirements
### Requirement: Experiment modules SHALL use top-level `src` paths
实验模块在代码与配置中 SHALL 统一使用 `src.embedding.*`、`src.quantization.*`、`src.recommendation.*` 路径，而 SHALL NOT 继续引用旧的 `src.models.*` 实验路径。

#### Scenario: Experiment config instantiates embedding module
- **WHEN** 实验配置实例化 embedding 模块
- **THEN** `_target_` 必须使用 `src.embedding.*` 路径

#### Scenario: Experiment code imports quantization module
- **WHEN** 代码导入 quantization 或 recommendation 相关模块
- **THEN** 必须使用新的 `src.quantization.*` 或 `src.recommendation.*` 路径

### Requirement: Common modules SHALL live under `src.common`
公有模块 SHALL 统一位于 `src.common` 命名空间下；实验专属模块 SHALL 位于对应实验领域命名空间下。跨实验复用的推理输出写入组件 SHALL 位于 `src.common.inference` 命名空间下；共享 loss 与 scheduler 实现 SHALL 分别位于 `src.common.loss` 与 `src.common.scheduler` 命名空间下。

#### Scenario: Code imports shared loss components
- **WHEN** 代码导入共享 loss 实现
- **THEN** 必须使用 `src.common.loss.*` 路径

#### Scenario: Code imports shared scheduler components
- **WHEN** 代码导入共享 scheduler 实现
- **THEN** 必须使用 `src.common.scheduler.*` 路径

#### Scenario: Code imports experiment-owned modules
- **WHEN** 代码导入 embedding、quantization 或 recommendation 专属模块
- **THEN** 必须使用 `src.embedding.*`、`src.quantization.*` 或 `src.recommendation.*` 路径
- **THEN** 不得通过 `src.common.modules.*` 引用实验专属模块

#### Scenario: Code imports inference output components
- **WHEN** 代码导入 prediction writer 或推理结果后处理函数
- **THEN** 必须使用 `src.common.inference.*` 路径

#### Scenario: Config instantiates shared loss components
- **WHEN** 配置实例化共享 loss 实现
- **THEN** `_target_` 必须使用 `src.common.loss.*` 路径

#### Scenario: Config instantiates shared scheduler components
- **WHEN** 配置实例化共享 scheduler 实现
- **THEN** `_target_` 必须使用 `src.common.scheduler.*` 路径

#### Scenario: Config instantiates experiment-owned modules
- **WHEN** 配置实例化 embedding、quantization 或 recommendation 专属模块
- **THEN** `_target_` 必须使用对应 `src.embedding.*`、`src.quantization.*` 或 `src.recommendation.*` 路径
- **THEN** `_target_` 不得使用 `src.common.modules.*` 路径

#### Scenario: Config instantiates inference output components
- **WHEN** 配置实例化 prediction writer 或推理结果后处理函数
- **THEN** `_target_` 必须使用 `src.common.inference.*` 路径

### Requirement: Migration SHALL preserve default pipeline composition
路径对齐与目录重命名之后，默认训练/推理主链路与实验配置 SHALL 仍能完成装配。

#### Scenario: Default pipeline resolves updated paths
- **WHEN** 默认训练或推理入口加载配置并实例化对象
- **THEN** 不得因为旧路径残留导致导入失败或 Hydra 目标解析失败
