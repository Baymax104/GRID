# tail-static-risk-standardization Specification

## Purpose
TBD - created by archiving change analyze-tail-search-ranking-loss. Update Purpose after archive.
## Requirements
### Requirement: Static-risk comparison SHALL use an explicit item-level estimand
系统 SHALL 先聚合同一 target item 的 evaluation 用户，再在静态 raw-damage 共同支持内比较 Head/Tail 的逐层 item-macro survival；MUST 不以用户记录顺序选择单个 Head 对照，也不得宣称 frequency 已被完全匹配。

#### Scenario: Item has multiple users
- **WHEN** 同一 item 有多个不同 survival 的用户
- **THEN** 所有用户 MUST 进入该 item 聚合，打乱输入行 MUST 不改变估计结果

#### Scenario: Frequency defines groups
- **WHEN** Head/Tail 来自训练频次划分
- **THEN** 比较 MUST 标明 frequency 差异仍存在，连续频次曲线只能作描述性证据

### Requirement: Common support and balance SHALL be auditable
系统 SHALL 用唯一目标 item 的 pooled raw damage 构建不依赖 outcome 的固定分箱，默认 5 箱、每组每箱至少 20 items，以归一化 min(nHead,nTail) 作为共同权重。

#### Scenario: Overlap qualifies
- **WHEN** 两组 item 保留率均至少 50% 且绝对 raw-damage SMD 不超过 0.1
- **THEN** 输出 MUST 报告箱边界、支持、权重、保留率、平衡和标准化点估计，并标明这些是预声明质量阈值

#### Scenario: Common support is weak
- **WHEN** 有效箱为空、保留率不足或平衡失败
- **THEN** 结果 MUST 分别标为 insufficient_overlap 或 imbalanced，不得静默外推
- **AND** 重复边界 MUST 合并，零方差 SMD MUST 按设计约定处理

### Requirement: Cluster intervals SHALL recompute the reported estimator
系统 SHALL 按前两层 model SID prefix 聚类重采样 item 聚合量，保留风险箱边界并重算支持、共同权重和同一层级估计量。默认 1000 次、95% percentile CI，随机 seed MUST 可追溯。

#### Scenario: Standardized confidence interval is produced
- **WHEN** 输出标准化 Tail−Head 差距的 CI
- **THEN** 重采样 MUST 计算同一标准化差距，不能附上旧未匹配差距的 CI
- **AND** 原始与标准化估计 MUST 使用不同标识，区间 MUST 标注为 pointwise 探索性区间

#### Scenario: Bootstrap lacks support
- **WHEN** 有效重采样不足总次数的 90% 或条件退出分母为零
- **THEN** 对应 CI/率 MUST 标为 unavailable/null，并报告重复数、cluster 数与原因

### Requirement: Analysis settings SHALL be fixed before pilot reanalysis
系统 SHALL 记录 estimator version、分箱、支持/平衡规则、bootstrap 参数和样本单位；主分析固定 5 箱，敏感性仅用预声明 3/5/10 箱。

#### Scenario: Sensitivity changes the sign
- **WHEN** 不同预声明分箱得到不同方向
- **THEN** 报告 MUST 如实呈现不稳定性，不得改选最显著配置作为主结果

#### Scenario: Reanalysis uses evaluation
- **WHEN** 已查看过的四组 evaluation 用于修正分析
- **THEN** 输出 MUST 保持探索性标记，不能宣称独立确认或测试集验证
