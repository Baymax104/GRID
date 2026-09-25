## ADDED Requirements

### Requirement: 冻结来源和标签隔离
系统 SHALL 只加载已核验的成熟C权重，同query生成item及root评分，且不训练或运行beam。

#### Scenario: 评价目标改变
- **WHEN** 只修改目标
- **THEN** 商品和root评分不变，仅评价统计改变

### Requirement: 复用A明细必须匹配
系统 SHALL 对每用户核验key、历史指纹和完整target SID，并检查A原审计有效。

#### Scenario: 输入或目标不一致
- **WHEN** 当前用户历史或目标不同于A明细
- **THEN** 拒绝生成可解释比较报告

### Requirement: 固定预算和聚合门槛
系统 SHALL 使用512用户和2000次配对bootstrap，在完整性通过后给出固定门槛结论。

#### Scenario: 数据不完整或dry-run
- **WHEN** 用户缺失重复、冻结状态变化或dry-run
- **THEN** 不发布正式证据
