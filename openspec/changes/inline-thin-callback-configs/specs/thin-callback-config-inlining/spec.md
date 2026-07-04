## ADDED Requirements

### Requirement: Thin callback wrapper configs SHALL be reducible
对于只承担单层 defaults 转发或单用途命名包装的 callback 配置，系统应允许将其收敛到更直接的配置入口，而不改变默认行为。

#### Scenario: Inference callback wrapper is inlined
- **WHEN** 一个 inference callback 配置文件只承担 defaults 转发作用
- **THEN** 该包装层可以被内联到更直接的 inference 配置入口
- **AND** inference 的默认 callback 行为必须保持不变

### Requirement: Active callback behavior SHALL remain unchanged after inlining
在收敛薄包装 callback 配置时，train / inference 当前启用的 callback 行为必须与变更前一致。

#### Scenario: Default train callback behavior remains stable
- **WHEN** 收敛薄包装 callback 配置
- **THEN** `train.yaml` 仍必须装配当前默认训练 callback 组合

#### Scenario: Default inference progress bar behavior remains stable
- **WHEN** 收敛 inference 侧的薄包装 callback 配置
- **THEN** inference 仍必须保持当前 one-based epoch progress bar 行为
