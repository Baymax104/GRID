## Why

已完成的性能诊断显示，v1 同批预热前向加反向约346 ms，基础目标约78 ms；4个排序用户的候选分块造成8次串行decoder评分。用户要求将批量评分优化实现为v1.1，并提供独立训练命令。

## What Changes

- 新增v1.1模型，跨用户合并候选对并按显式全局分块评分，保留完整SID末状态和可训练主干。
- 训练与hybrid推理复用批量评分，保留候选规则、beam20、排序抽样、全部loss与selection/audit评价。
- 保留v0/v1，新增v1.1配置、LF脚本和checkpoint执行契约。
- 用关闭dropout的评分/梯度对照、非零dropout的训练检查、DDP回归和零更新生产计时验证。

## Capabilities

### New Capabilities

- `copmrec-batched-relevance`: 跨用户批量候选评分、版本化入口和验证边界。

### Modified Capabilities

无。

## Impact

推荐模型新增子类；配置与根目录脚本新增独立入口；新增内存单元测试和更新版本文档。不引入依赖，不冻结推荐参数，不新增正式实验预算，不自动开始完整训练。
