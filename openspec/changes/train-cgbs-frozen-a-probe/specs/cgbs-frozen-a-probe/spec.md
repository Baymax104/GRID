## ADDED Requirements

### Requirement: 显式加载并冻结已训练 A
系统 SHALL 严格加载已训练 A，拒绝不匹配目录、契约或权重，只优化新增内容头和混合系数。

#### Scenario: 有效来源
- **WHEN** 提供匹配 A checkpoint 并执行优化
- **THEN** A 参数逐位不变，关闭内容评分的生成结果与 A 一致，新增参数有有限梯度并更新

#### Scenario: 无效来源
- **WHEN** 来源非 A、目录不同、缺失权重或含非有限值
- **THEN** 在训练前明确报错

### Requirement: 有界训练与来源记录
系统 SHALL 通过统一入口执行最多2k更新并独立记录验证 content loss，checkpoint 保存来源与冻结协议。

#### Scenario: 恢复与推理
- **WHEN** 加载探针 checkpoint
- **THEN** 恢复训练核验来源协议，原 CGBS 推理模型可严格加载相同结构权重

#### Scenario: 启动与预算
- **WHEN** 真实根脚本传递来源 URI、notes、dry-run 与额外 override
- **THEN** Hydra 保留原值，dry-run 仍限一步，超出2k训练被拒绝
