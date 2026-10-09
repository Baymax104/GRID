## Why

成熟A上的全局mean128内容头未通过增量门槛。用户已接受三个顺序门槛，授权推进实现，以匹配对照判断候选分支条件化交互是否产生增量；完整实验仍手动启动。

## What Changes

- 固化结构增量、搜索机制、跨seed复现三个门槛及预算，后两步条件执行。
- 实现冻结A训练集竞争缓存、全局/分支条件化残差头和跨父节点frontier目标。
- 新增统一入口配置、脚本、来源契约、分片writer和聚焦测试。
- 初始残差为零，off走原A路径；保留旧CGBS及其失败结论。

## Capabilities

### New Capabilities
- `cgbs-branch-residual-probe`: 三门槛协议与第一门槛缓存、训练、推理契约。

### Modified Capabilities
无。

## Impact

新增recommendation/data/writer组件与Hydra配置、根脚本和测试，不增加依赖，不自动启动GPU任务。研究依据见../../../../research/docs/grid-experiments/2026-09-21-cgbs-structural-redesign-research.md。
