## MODIFIED Requirements

### Requirement: Common modules SHALL live under `src.common`
公有模块 SHALL 统一位于 `src.common` 命名空间下；实验专属模块 SHALL 位于对应实验领域命名空间下。具有明确推理输出语义的组件与协议工具 SHALL 位于 `src.inference` 命名空间下。

#### Scenario: Code imports shared components
- **WHEN** 代码导入共享 components
- **THEN** 必须使用 `src.common.components.*` 路径

#### Scenario: Code imports experiment-owned modules
- **WHEN** 代码导入 embedding、quantization 或 recommendation 专属模块
- **THEN** 必须使用 `src.embedding.*`、`src.quantization.*` 或 `src.recommendation.*` 路径
- **THEN** 不得通过 `src.common.modules.*` 引用实验专属模块

#### Scenario: Code imports inference output components
- **WHEN** 代码导入推理输出协议、prediction writer、推理结果后处理或 keyed prediction bundle 工具
- **THEN** 必须使用 `src.inference.*` 路径

#### Scenario: Config instantiates shared components
- **WHEN** 配置实例化共享 components
- **THEN** `_target_` 必须使用 `src.common.components.*` 路径

#### Scenario: Config instantiates experiment-owned modules
- **WHEN** 配置实例化 embedding、quantization 或 recommendation 专属模块
- **THEN** `_target_` 必须使用对应 `src.embedding.*`、`src.quantization.*` 或 `src.recommendation.*` 路径
- **THEN** `_target_` 不得使用 `src.common.modules.*` 路径

#### Scenario: Config instantiates inference output components
- **WHEN** 配置实例化推理输出相关 components
- **THEN** `_target_` 必须使用 `src.inference.*` 路径
