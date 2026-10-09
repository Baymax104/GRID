## ADDED Requirements

### Requirement: Noninterfering path observation
系统 SHALL 默认关闭路径记录，开启后 MUST 不改变 legal/max/mass 的候选或预测，不改变 checkpoint state_dict。

#### Scenario: Trace on and off
- **WHEN** 同模型、输入、随机状态分别开启与关闭路径记录
- **THEN** 三种机制的候选和最终输出一致，旧 checkpoint 严格加载成功

### Requirement: Actual frontier and target observability
系统 SHALL 记录实际逐层 beam 前缀、目标存活及首次丢失深度；父节点不可达时的分支概率 MUST 标为缺失。最终生成命中与最后一层存活一致。

#### Scenario: Target pruned
- **WHEN** 目标前缀在中间层被剪枝
- **THEN** 首次丢失层正确，后续存活为 false，后续分支概率为 NaN，验证器拒绝不一致字段

### Requirement: Keyed local path artifact
系统 SHALL 通过共享 writer 保存独立 keyed liger_paths_v1 bundle 并发布 node1 file reference；命令 MUST 同时开启 candidate 和 path trace。

#### Scenario: Mechanism command composition
- **WHEN** 配置选择路径回调与 path_trace=true
- **THEN** prediction、candidate、path writer 和 lineage callback 同时存在，保存路径产物且不改变 candidate v1
