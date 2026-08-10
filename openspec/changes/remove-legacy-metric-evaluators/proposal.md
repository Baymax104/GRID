## Why

通用 metric runtime 与 TIGER/quantization 迁移完成后，旧 `Evaluator` / `SIDRetrievalEvaluator` 包装层已不再被配置或模型引用。保留它会造成两个指标框架并存，增加后续维护歧义。

## What Changes

- 删除 `src.common.components.eval_metrics` 中的旧 evaluator 和基础 retrieval metric 实现。
- 将 `NDCG`、`Recall` 移入 TIGER 领域目录，作为 TIGER SID retrieval runtime metric。
- 增加测试确保旧 evaluator 不再作为公开入口存在。

## Capabilities

### New Capabilities
- `legacy-metric-evaluator-removal`: 移除旧 evaluator 包装层，统一指标生命周期到 metric runtime。

### Modified Capabilities

## Impact

- 影响 `src/common/components/eval_metrics.py` 和 `src/recommendation/tiger/metrics.py`。
- 影响 focused metric tests。
