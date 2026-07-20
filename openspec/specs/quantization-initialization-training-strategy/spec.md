# quantization-initialization-training-strategy Specification

## Purpose
TBD - created by archiving change scope-custom-training-loop-to-quantization. Update Purpose after archive.
## Requirements
### Requirement: quantization 初始化特殊训练路径 SHALL 局部封装在 quantization 子域
用于量化初始化阶段的特殊 manual optimization 路径必须局部封装在 quantization 子域内，而不是以项目通用 training hook 的形式暴露。

#### Scenario: 初始化策略位于 quantization 命名空间
- **WHEN** 维护者查找 DDP 初始化阶段的 loss 缩放 / 临时 optimizer 逻辑
- **THEN** 这些实现 MUST 位于 quantization 更贴近的模块路径
- **THEN** 不得继续以 `src.common.components.training_loop_functions` 的“项目级 common hook”形态暴露

#### Scenario: inference 配置不再暴露训练策略
- **WHEN** 维护者检查 quantization inference 配置
- **THEN** inference 配置 MUST NOT 再暴露仅在 training_step 中生效的 `training_loop_function` 配置项

