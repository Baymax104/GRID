# liger-path-trace Specification

## Purpose

规定当前可达的概率混合路径观察能力。2026-09-29 提案中的 legal/max/mass 三机制控制已退出运行面，原提案仅作为历史档案保存。

## Requirements

### Requirement: Noninterfering path observation

系统 SHALL 默认关闭路径记录；在 LIGER 的 hybrid probability_mixture 与正式 CoPMRec v2 learned_mass 路径上启用后，MUST 不改变候选、预测或 checkpoint state_dict。目标标签 SHALL 仅用于观察，不参与候选选择或排序。

#### Scenario: Trace on and off

- **WHEN** 同一受支持模型、输入与随机状态分别开启和关闭路径记录
- **THEN** 候选及最终输出一致，checkpoint 能严格恢复，改变观察标签不改变预测

### Requirement: Actual frontier and target observability

系统 SHALL 记录实际逐层 beam 前缀、目标存活及首次丢失深度；父节点不可达时的分支概率 MUST 标为缺失。最终生成命中 SHALL 与最后一层存活一致。

#### Scenario: Target pruned

- **WHEN** 目标前缀在中间层被剪枝
- **THEN** 首次丢失层正确，后续存活为 false，后续分支概率为 NaN，验证器拒绝不一致字段

### Requirement: Keyed local path artifact

系统 SHALL 通过共享 writer 保存独立 keyed liger_paths_v1 bundle，发布时使用运行机器上的 file reference；路径观察 SHALL 同时启用 candidate trace 和 path trace。

#### Scenario: Path command composition

- **WHEN** 受支持配置选择路径回调并启用 candidate_trace 和 path_trace
- **THEN** prediction、candidate、path writer 和 lineage callback 同时存在，路径产物通过 validator，candidate v1 产物契约保持一致
