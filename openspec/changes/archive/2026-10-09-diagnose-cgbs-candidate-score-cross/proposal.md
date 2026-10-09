# 候选池与评分交叉诊断

## Why
已有on/off收益混合候选与排序变化，旧dense后置重排评分不同，不能隔离搜索作用。

## What Changes
- 新增统一predict入口的冻结checkpoint重评分；不训练、不运行beam。
- 读取已有on/off候选与trace，核对来源、用户标签和复现排序。
- 导出四格逐用户结果，通过共享writer发布诊断Artifact。

## Capabilities
### New Capabilities
- `cgbs-candidate-score-cross`: 匹配评分函数的候选池交叉诊断。

## Impact
新增诊断model/config/script/test，保持原训练和推理行为不变。
