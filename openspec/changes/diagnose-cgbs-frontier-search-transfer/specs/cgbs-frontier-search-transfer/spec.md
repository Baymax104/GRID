## ADDED Requirements
### Requirement: 有界匹配来源
系统 MUST 仅使用evaluation固定用户子集与匹配的A/global/branch checkpoint，禁止训练和来源不一致。
#### Scenario: 主干被改变
- **WHEN** 任一头checkpoint的主干张量或源/缓存契约不匹配
- **THEN** 诊断拒绝执行，不发布方法结论
### Requirement: 评分与搜索有效性
系统 SHALL 同时检查off和零残差的有序输出及分数复现，记录固定frontier和真实搜索的不同证据。
#### Scenario: 零残差排序不一致
- **WHEN** 任一用户零残差输出不能精确复现A
- **THEN** 报告有效性失败，保留原始差异，不放宽容差或给出机制通过结论
### Requirement: 完整产物与人工入口
系统 MUST 经src.main和Trainer.test执行，writer SHALL 检查数量和唯一key，启动脚本 MUST 支持dry-run、notes和额外override。
#### Scenario: 部分样本结束或dry-run
- **WHEN** 样本不完整或启用dry-run
- **THEN** 不发布正式诊断结果
