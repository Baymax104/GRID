## ADDED Requirements
### Requirement: 从头联合高位排序训练
系统 SHALL 从随机推荐模型参数开始，在固定总预算内完成基础联合训练和高位排序联合训练，不加载旧推荐checkpoint或旧候选分数。训练teacher SHALL 在基础阶段结束时由当前模型冻结，并仅在training数据标定cap。
#### Scenario: 阶段切换
- **WHEN** 达到固定基础阶段更新数
- **THEN** 冻结当前双分支参照，完成有限training校准后启用capped NDCG与双保护，学生全部推荐参数保持可训练
#### Scenario: 恢复和评价
- **WHEN** 加载checkpoint或执行validation
- **THEN** 严格核对阶段契约，恢复teacher与cap状态，评价不得初始化或改变校准状态
### Requirement: 训练推理边界与固定用户划分
系统 SHALL 仅training候选可补目标，推理使用真实候选，selection与audit沿用固定用户hash划分。
#### Scenario: 目标不在检索候选
- **WHEN** training目标未召回
- **THEN** 在训练短列表补入目标并记录自然覆盖率，推理不得采用同样补入
### Requirement: 可复现运行入口
系统 SHALL 通过src.main和独立experiment启动，支持dry-run、notes及额外override，保存来源和阶段契约，不改变baseline。
#### Scenario: 手动启动
- **WHEN** 使用新根脚本
- **THEN** 默认从随机初始化训练，完整实验由用户手动启动

### Requirement: 双卡训练一致性
系统 SHALL 支持双卡 DDP，保持有效 batch 256、每更新排序样本16和50k更新预算；校准累计量跨卡汇总，验证和推理用户无重复分片。
#### Scenario: 双卡校准与评价
- **WHEN** 两个进程进入校准或完成验证
- **THEN** 校准和cap状态在各进程一致，验证指标由全局命中、折损收益和真实用户数汇总，不对局部均值简单平均
