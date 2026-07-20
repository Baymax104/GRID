## ADDED Requirements

### Requirement: layer-wise 训练日志 SHALL 准确表达 step budget
对于按 layer 分配训练 step 的实验，日志必须准确表达每层分配到的 step budget 以及与全局 step 的关系，不得使用会掩盖余数分配的简化表达。

#### Scenario: 启动日志明确 layer budgets
- **WHEN** `ResidualQuantization` 以 layer-wise 模式开始训练
- **THEN** 日志 MUST 明确表达每层的 step budget 或等价的全局 step 边界
- **THEN** 若 `max_steps` 不能被层数整除，日志 MUST 反映余数如何分配

#### Scenario: 层切换日志与全局 step 对齐
- **WHEN** layer-wise 训练切换到下一层
- **THEN** 日志 MUST 指出当前完成的是哪一层
- **THEN** 日志 SHOULD 同时包含对应的全局 step 信息，以便与训练 progress bar 对齐理解

#### Scenario: layer 特有信息不进入通用 progress bar
- **WHEN** 项目为所有训练实验使用统一的通用 progress bar
- **THEN** `layer`、`layer_step` 等仅适用于 layer-wise 训练的特有信息 MUST NOT 成为通用 progress bar 的默认展示项
- **THEN** 这类信息 SHOULD 继续通过实验特定日志表达
