## Why

TIGER 当前仍通过 `Evaluator`、`MeanMetric` 属性和模型内 validation/test hooks 管理指标，这与新的低耦合指标运行时方向冲突。迁移 TIGER 是验证通用指标框架能承接复杂 retrieval 指标的最小模型级变更。

## What Changes

- 将 TIGER 训练指标从模型内部迁移到 `MetricCallback` / `MetricEngine`。
- 新增 TIGER SID retrieval metric group，用于复用现有 NDCG/Recall 计算逻辑。
- 修改 TIGER step 输出，使 callback 能从返回值中读取 loss、generated IDs、marginal probabilities 和 labels。
- 修改 `configs/model/tiger_train.yaml`，用 `model.metrics` 声明训练/验证/测试指标。
- 在 pipeline 初始化中自动根据 `cfg.model.metrics` 追加 metric callback。
- 移除 TIGER 中的指标属性、`Evaluator` 参数、指标 `setattr` 和指标相关 epoch hooks。

## Capabilities

### New Capabilities
- `tiger-runtime-metrics`: TIGER 使用通用指标运行时记录 loss 与 SID retrieval 指标。

### Modified Capabilities

## Impact

- 影响 `src/recommendation/tiger/tiger.py`。
- 影响 `src/common/metrics/`，新增 TIGER retrieval metric group。
- 影响 `src/utils/launcher.py`，自动追加 metric callback。
- 影响 `configs/model/tiger_train.yaml`。
- 新增 focused unit tests，不运行完整 experiment。
