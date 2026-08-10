## Why

当前 `MetricEngine` 同时支持简单 `spec` 映射和 `update_from_payload` 特例；TIGER SID retrieval 又通过 `SIDRetrievalMetricGroup` 把输入适配、指标容器和 `compute()` 汇总绑在一起。这样虽然能运行，但指标定义不再是“一条配置对应一个指标实例”，后续复杂指标会继续产生新的 group 特例。

本变更将 metric 输入统一收敛到配置化 adapter 机制：每个 metric 都是独立实例，所有输入都由同一套 `spec` resolver 产生，复杂领域转换通过纯函数 adapter 完成。

## What Changes

- 为 `MetricEngine` 的 metric definition 增加 `spec.adapter` 支持，adapter 负责把 stage payload 转成 `metric.update(**kwargs)` 参数。
- 将普通 scalar、`args`、`kwargs`、indexed value 和 adapter 都归入同一个 `spec` 解析机制。
- 将 TIGER SID retrieval 转换提取为领域纯函数 adapter，放在 TIGER 相关模块下。
- 将 TIGER retrieval metrics 展平成普通 metric entries，例如 `ndcg@5`、`recall@10`。
- 删除 `SIDRetrievalMetricGroup` 及其导出、配置引用和测试。
- 更新当前 metric/TIGER runtime OpenSpec artifacts，使其描述 adapter 机制而不是 metric group 特例。

## Capabilities

### New Capabilities
- `metric-input-adapters`: 配置驱动的 metric spec 适配机制，支持每个 metric 通过纯函数 adapter 从模型 payload 生成 `update()` 入参。

### Modified Capabilities

## Impact

- 影响代码：`src/common/metrics/engine.py`、`src/common/metrics/groups.py`、`src/common/metrics/__init__.py`、TIGER metric adapter 新模块、`configs/model/tiger_train.yaml`。
- 影响测试：`tests/common/metrics/`、`tests/recommendation/test_tiger_metrics_runtime.py`。
- 影响 OpenSpec artifacts：当前 metric runtime 和 TIGER runtime 相关 active change 文档需要同步改为 adapter 设计。
- 不新增外部依赖，不改变 NDCG/Recall 计算公式，不改变 TIGER step payload 字段。
