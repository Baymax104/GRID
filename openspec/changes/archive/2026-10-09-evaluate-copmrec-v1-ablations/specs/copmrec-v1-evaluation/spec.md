## ADDED Requirements

### Requirement: 无历史排除的消融推理

系统 SHALL 使用每臂原 validation-selected own-best 进行仅推理的完整目录 dense 无历史排除排名，并保留消融来源契约及稳定 catalog-row 排序。

#### Scenario: 五臂独立恢复

- **WHEN** 提供对应变体的审计 checkpoint URI、SHA 和 seed
- **THEN** 恢复契约通过后冻结参数，仅省略最终历史掩码，输出合法唯一 Top10；不同变体 checkpoint 必须拒绝

### Requirement: v1 M1 证据

系统 SHALL 基于 v1 Full 和五臂 keyed bundle 计算原值、带符号差、预定切片与配对区间，不执行模型 forward。

#### Scenario: 无排除列表包含历史商品

- **WHEN** 明确设置无历史排除的评价规则
- **THEN** 历史重叠允许但输出仍须合法唯一，v0 默认历史排除检查保持

### Requirement: 有界执行与可核验结果

系统 SHALL 固定 Beauty/seed42、0训练/0独立Validation/5单卡Testing，复用既有Full，保留来源归档和独立指标核验后才登记完成。

#### Scenario: 运行完成

- **WHEN** W&B 运行 finished
- **THEN** 仍需核验 checkpoint/input/source、完整用户标签、输出合法性及独立指标，完成与观测方向无关
