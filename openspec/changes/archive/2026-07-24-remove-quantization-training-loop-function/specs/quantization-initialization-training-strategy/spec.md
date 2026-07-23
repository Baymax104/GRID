## MODIFIED Requirements

### Requirement: quantization 初始化 SHALL 使用 direct centroid assignment
用于量化初始化阶段的 centroids 初始化 SHALL 局部封装在 quantization 模型内，并通过 rank-zero compute、distributed broadcast、direct parameter assignment 完成，而不是通过特殊 manual optimization training loop 暴露。

#### Scenario: 初始化策略位于 quantization 命名空间
- **WHEN** 维护者查找 DDP 初始化阶段的 centroid 计算、broadcast、assignment 逻辑
- **THEN** 这些实现 MUST 位于 quantization 模型或同模型目录的专用模块中
- **AND** 不得继续以 `src.common.components.training_loop_functions` 或 `src.quantization.training_loop_functions` 的外部 training hook 形态暴露

#### Scenario: inference 配置不再暴露训练策略
- **WHEN** 维护者检查 quantization inference 配置
- **THEN** inference 配置 MUST NOT 再暴露仅在 training_step 中生效的 `training_loop_function` 配置项

#### Scenario: training 配置不再暴露初始化训练 loop
- **WHEN** 维护者检查 quantization training 配置
- **THEN** training 配置 MUST NOT 暴露 `training_loop_function`
- **AND** 初始化所需的 DDP 同步 MUST 由模型初始化逻辑内部处理
