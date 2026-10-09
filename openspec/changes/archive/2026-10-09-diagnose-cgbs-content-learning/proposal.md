## Why

CGBS content loss下降缓慢，但现有曲线不能区分内容排序、梯度竞争和小样本可优化性。需要有界诊断，避免直接调权或加模块。

## What Changes

- 新增统一入口的冻结checkpoint诊断，导出全目录内容评分和编码器/query两项损失梯度。
- 仅training样本允许临时小批拟合；分别query-only和encoder+query，各100步，恢复原权重，不保存新checkpoint。
- 确定性抽样、共享Artifact writer、CPU回归和手动命令。

## Capabilities

### New Capabilities
- `cgbs-content-learning-diagnosis`: 有界内容分支诊断与不可变checkpoint契约。

### Modified Capabilities

无。

## Impact

新增推荐诊断子类、数据抽样组件、配置、根脚本和测试；不改原CGBS训练，不新增依赖。正式诊断由用户手动启动。
