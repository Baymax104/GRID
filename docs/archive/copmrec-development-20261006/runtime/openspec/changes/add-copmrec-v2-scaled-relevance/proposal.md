# CoPMRec v2：尺度受约束的decoder相关性

## Why

用户确定v1迭代结束，v2与v1平级并独立基于v0。已有排序等权验证仅支持优化失衡，未证明完整推荐收益；用户采纳d-only head与尺度约束方案，并指定loss lambda固定0.01、分数beta固定0.5。此次实现不将这些值称为校准或最优结果。

## What Changes

- 新增直接继承v0的v2组件、独立训练/推理配置和根入口。
- relevance=0.5*stop_gradient(sqrt(Var_natural(content)+epsilon²))*tanh(head(d_ui))。
- 保留v0三项loss，加入固定0.01的融合候选CE，全部推荐参数共同训练。
- 独立checkpoint/trace契约与尺度日志，继承共享候选/产物协议。
- 保留v1/v1.1运行字节，完成测试、远端同步和从零双卡命令。

## Capabilities

### New Capabilities
- `copmrec-scaled-decoder-relevance`: 从v0派生、尺度受约束的decoder终排相关性。

### Modified Capabilities

无。

## Impact

新增推荐组件与配置/脚本/测试；共享候选trace校验增加v2分支。不修改v1模型、训练权重或运行入口，无新依赖，不启动或停止完整训练。
