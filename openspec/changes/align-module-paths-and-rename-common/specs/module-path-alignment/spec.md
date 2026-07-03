## ADDED Requirements

### Requirement: Experiment modules SHALL use top-level `src` paths
实验模块在代码与配置中必须统一使用 `src.embedding.*`、`src.quantization.*`、`src.recommendation.*` 路径，而不得继续引用旧的 `src.models.*` 实验路径。

#### Scenario: Experiment config instantiates embedding module
- **WHEN** 实验配置实例化 embedding 模块
- **THEN** `_target_` 必须使用 `src.embedding.*` 路径

#### Scenario: Experiment code imports quantization module
- **WHEN** 代码导入 quantization 或 recommendation 相关模块
- **THEN** 必须使用新的 `src.quantization.*` 或 `src.recommendation.*` 路径

### Requirement: Common modules SHALL live under `src.common`
公有模块必须统一位于 `src.common` 命名空间下。

#### Scenario: Code imports shared components
- **WHEN** 代码导入共享 components 或 modules
- **THEN** 必须使用 `src.common.components.*` 或 `src.common.modules.*` 路径

#### Scenario: Config instantiates shared modules
- **WHEN** 配置实例化共享 components 或 modules
- **THEN** `_target_` 必须使用 `src.common.*` 路径

### Requirement: Migration SHALL preserve default pipeline composition
路径对齐与目录重命名之后，默认训练/推理主链路与实验配置必须仍能完成装配。

#### Scenario: Default pipeline resolves updated paths
- **WHEN** 默认训练或推理入口加载配置并实例化对象
- **THEN** 不得因为旧路径残留导致导入失败或 Hydra 目标解析失败
