## Why

A 的历史编码和目标前缀共用输入表。需要验证角色更新是否存在有害干扰，避免把负梯度余弦误判为论文瓶颈。

## What Changes

- 新增冻结 checkpoint 的单卡有界角色审计；训练用户生成方向、互斥评价用户测量排名及CE。
- 比较共享、独立角色、历史单侧、前缀单侧、匹配范数随机干预；不运行优化器或保存新模型。
- 共享 writer 发布可追溯证据，提供统一入口配置与手动脚本。

## Capabilities

### New Capabilities
- `a-role-sharing-audit`: 数值等价、路径梯度分解、独立样本有限扰动及证据导出。

### Modified Capabilities

无。

## Impact

新增独立数据装配、模型审计子类、配置、脚本和测试；复用A的评分及checkpoint协议，不改训练模型行为。完整诊断由用户手动启动。
