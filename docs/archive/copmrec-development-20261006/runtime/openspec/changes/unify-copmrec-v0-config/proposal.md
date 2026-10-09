# 恢复CoPMRec v0真实配置

## Why

用户明确BMX-116已完成的基础方法就是v0。当前版本化v0入口曾将原全evaluation dense验证/testing hybrid推理改为selection/audit与2500间隔，导致名称和可执行行为不一致。以W&B run7y54j4m6实际完整配置作为基准恢复，不改写历史结果。

## What Changes

- v0训练/推理入口恢复原LIGER joint组件装配和评分/数据/验证配置。
- 显式隔离v1/v1.1既有开发协议，防止v0更改产生隐式行为变更。
- 核对v2对照配置，记录用户确认的评价数据范围；保留固定lambda0.01、beta0.5和融合终排。
- 对比完整resolved配置、验证入口和同步，更新方法版本文档及研究状态。

## Capabilities

### New Capabilities
- `copmrec-v0-config-identity`: 真实v0配置身份及版本隔离。

### Modified Capabilities
无。

## Impact

配置、文档与聚焦测试；不新增依赖、不修改历史run，不启动/停止正式实验。
