## ADDED Requirements

### Requirement: 固定基础得分和合法支持
系统 SHALL 使用同一联合 checkpoint 的内容得分与生成模型，提供 learned_mass、legal_generation、max_mixture 三种推理模式，不改变合法目录与排序。

#### Scenario: 聚合控制
- **WHEN** 选择 max_mixture
- **THEN** 仅将后代 logsumexp 改为 max，保持 learned alpha、合法归一化及算术混合。

#### Scenario: 合法生成控制
- **WHEN** 选择 legal_generation
- **THEN** 使用相同合法子节点上的生成条件概率，忽略内容引导。

### Requirement: 控制隔离与可审计性
系统 SHALL 拒绝使用控制模式训练，拒绝与 content_only 同时使用，记录 trace 中的控制机制与有效 alpha，并保持旧 checkpoint 兼容。

#### Scenario: 错误训练
- **WHEN** 非默认控制用于训练
- **THEN** 明确报错。
