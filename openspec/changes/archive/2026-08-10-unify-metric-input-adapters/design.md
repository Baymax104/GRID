## Context

当前 metric runtime 已经把指标生命周期从 `LightningModule` 迁移到 `MetricCallback` / `MetricEngine`，但输入解析机制还不完全统一：

- scalar 指标通过 `spec` 从 payload 取字段。
- 多参数指标通过 `args` / `kwargs` 从 payload 取字段。
- 复杂领域指标可通过 `update_from_payload(payload)` 绕过普通 spec resolver。
- TIGER SID retrieval 使用 `SIDRetrievalMetricGroup` 同时承担 payload 适配、多个 metric 容器和 `compute()` 汇总。

用户要求彻底统一指标机制：每个指标配置块对应一个 metric 实例，复杂输入适配通过配置化纯函数完成，而不是继续引入 metric group 特例。

## Goals / Non-Goals

**Goals:**

- `MetricEngine` 使用单一 spec resolver 更新所有指标。
- 每个 metric definition 对应一个真实 metric 实例和一个输出名。
- 支持 `spec.adapter`，由纯函数将 stage payload 转为 `metric.update(**kwargs)`。
- TIGER SID retrieval 指标展平成普通 `ndcg@K` / `recall@K` metric entries。
- 删除 `SIDRetrievalMetricGroup` 及其配置引用。
- 保留现有 NDCG/Recall 公式、TIGER eval payload 字段和日志名称。

**Non-Goals:**

- 不改变 TIGER generation、loss、beam search 或模型 step 语义。
- 不引入 adapter 结果缓存；当前 TIGER 仅少量 retrieval metrics，重复 adapter 计算可接受。
- 不新增外部依赖。
- 不把 TIGER SID 字段语义放进 `src.common.metrics`。

## Decisions

### adapter 是 spec resolver 的一种形式

metric definition 统一通过 `spec` 描述 `metric.update()` 入参：

```yaml
loss:
  metric:
    _target_: torchmetrics.MeanMetric
  spec:
    key: loss

ndcg@5:
  metric:
    _target_: src.recommendation.tiger.metrics.NDCG
    top_k: 5
  spec:
    adapter:
      _target_: src.recommendation.tiger.metrics.sid_retrieval_inputs
      _partial_: true
```

`MetricEngine.update(stage, payload)` 根据 metric spec 解析输入：

- `adapter`: 调用 `adapter(payload)`，返回值必须是 mapping，并以 `metric.update(**returned_mapping)` 更新。
- `kwargs`: 对每个 kwarg 按 key/index 从 payload 解析，并以 `metric.update(**kwargs)` 更新。
- `args`: 按顺序解析 payload 字段，并以 `metric.update(*args)` 更新。
- `key` 或字符串 shorthand: 解析单个 payload 值，并以 `metric.update(value)` 更新。

理由：所有指标都走同一条 metric spec 路径，`update_from_payload` 不再是运行时特例。

### adapter 是无状态纯函数

adapter contract 固定为：

```python
def adapter(payload: Mapping[str, Any]) -> Mapping[str, Any]:
    ...
```

adapter 不接收 metric 实例，不调用 `compute/reset/log`，不产生副作用，只返回 `update()` kwargs。

理由：这样 adapter 可以独立测试，且不会重新把指标生命周期泄漏到领域模块。

### TIGER SID adapter 留在领域目录

新增 `src/recommendation/tiger/metrics.py`，实现 `sid_retrieval_inputs(payload)`：

- 读取 `marginal_probs`、`generated_ids`、`labels`。
- 通过完整 SID 匹配构造 retrieval `target`。
- 展平 `preds` / `target` 并生成 batch-level `indexes`。
- 返回 `{"preds": preds, "target": target, "indexes": indexes}`。

理由：SID 字段和匹配规则属于 TIGER 输出语义，不属于通用 metric runtime。

### 删除 SIDRetrievalMetricGroup

TIGER config 直接声明：

- `ndcg@5`
- `ndcg@10`
- `recall@5`
- `recall@10`

每个指标都有自己的 `metric` 和同一个 `spec.adapter`。`MetricEngine.compute()` 只处理普通 metric 返回值，不再需要 group 返回 dict 的特殊输出。

理由：配置和运行时都恢复为“一条 metric definition 对应一个 metric 实例”。

## Risks / Trade-offs

- [Risk] 多个 TIGER retrieval metrics 会重复调用同一个 adapter。  
  Mitigation: 当前只有 4 个指标，计算量相对 generation 较小；如后续指标数量显著增加，再增加 stage-batch 级 adapter cache。

- [Risk] adapter 返回非法类型时错误可能较晚暴露。  
  Mitigation: `MetricEngine` 在 update 时检查 adapter 返回 mapping，并用单元测试覆盖错误路径。

- [Risk] 删除 `update_from_payload` 会影响未来尚未发现的自定义 metric。  
  Mitigation: 先用 residual scan 确认当前仅 `SIDRetrievalMetricGroup` 使用该协议，再删除；后续复杂指标统一写 adapter。

## Migration Plan

1. 扩展 `MetricEngine` spec resolver，支持 `spec.adapter` 和 `spec: {key: ...}` 形式，同时保持字符串 shorthand 兼容。
2. 新增 TIGER SID retrieval adapter 纯函数和单元测试。
3. 将 `configs/model/tiger_train.yaml` 的 retrieval metrics 展平成普通 metric entries。
4. 删除 `SIDRetrievalMetricGroup`、导出和旧测试。
5. 更新 active OpenSpec artifacts 中关于 metric group 的描述。
6. 运行 focused metric/TIGER tests、scoped ruff、OpenSpec strict validation。

## Open Questions

无。
