## Why

真实 CoPMRec v0 已改善候选准入，但 Beauty/Sports/Toys seed42 仍有97/191/141个自身 dense Top10目标遗漏，主要发生在前两层；Max 对照保留根前缀的表现提供继续验证的正面依据。用户授权将首层 Max、后续 Mass 实现为从 v0 派生的 v3，并交付与 v0 同训练协议的双卡命令。

## What Changes

- 新增固定 `root_max_mass` 聚合：第一层合法子分支用后代内容最大值归一化，后续层用后代 logsumexp 归一化。
- v3 训练混合 NLL 与推理共用该聚合；保留 SID CE、content CE、混合 NLL 三项等权以及可学习全局 alpha。
- 新增独立模型、checkpoint/trace 契约、v3 train/inference 配置与根脚本，直接继承真实 v0 数据、优化器、trainer、dense 验证选点和 content 终排。
- 验证算法、梯度、恢复、脚本、Hydra配置与双进程执行；通过 Mutagen 同步并交付手动训练命令。完整训练不由 agent 启动。

## Capabilities

### New Capabilities

- `copmrec-v3-prefix-aggregation`: 从 v0 派生的逐层聚合与匹配训练/推理、版本及运行入口契约。

### Modified Capabilities

无；v0/v1/v1.1/v2既有默认行为保留。

## Impact

影响共享候选 processor、v0混合损失的一个聚合装配扩展点及新增v3组件/配置/脚本/测试。无需新依赖。更新版本说明、研究状态和当前计划；保留过去负向证据、关闭预算和 testing 开发使用边界。
