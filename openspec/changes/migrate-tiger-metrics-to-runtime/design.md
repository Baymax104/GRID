## Context

TIGER 的指标逻辑包括：

- train loss 使用 `MeanMetric` 并在 `training_step` 内 `self.log`。
- validation/test loss 由 `MeanMetric` 聚合并在 epoch end log。
- SID retrieval 指标由 `SIDRetrievalEvaluator` 将生成结果转换成 ranking target，然后更新 NDCG/Recall。

新框架要求指标生命周期由 callback 接管，因此 TIGER 需要返回指标 payload，而不是直接 update/log/reset。

## Goals / Non-Goals

**Goals:**

- TIGER 不再持有单个 metric 属性。
- TIGER 不再实现指标专用 validation/test epoch hooks。
- TIGER train/val/test step 返回 metric runtime 可消费的 payload。
- TIGER retrieval 指标名称保持 `ndcg@K`、`recall@K`，日志输出保持 `val/`、`test/` stage 前缀。
- 官方 TIGER train config 使用 `model.metrics` 声明指标。

**Non-Goals:**

- 不改变 TIGER loss 公式、generation 逻辑或 optimizer/scheduler。
- 不迁移 quantization 指标。
- 不删除 `Evaluator` 兼容类；彻底清理留给最终 cleanup 变更。

## Decisions

### 使用 MetricGroup 承接 SID retrieval

新增 `SIDRetrievalMetricGroup`，内部持有按 top-k 展开的 NDCG/Recall 指标，并提供 `update_from_payload(payload)`。

理由：

- SID retrieval 需要从 `marginal_probs`、`generated_ids`、`labels` 构造 ranking target，属于 TIGER 领域转换，不应塞进通用 engine。
- group 对外仍表现为一个 metric module，适配 `MetricEngine`。

### launcher 自动注入 MetricCallback

`initialize_pipeline_modules` 在实例化常规 callbacks 后，如果 `cfg.model.metrics` 存在，则实例化 `MetricCallback(engine=cfg.model.metrics)` 并追加。

理由：

- model config 是指标定义来源。
- experiment 不需要重复声明 callback。
- 迁移后模型无需感知 callback。

### TIGER eval step 返回完整 payload

`validation_step` / `test_step` 调用共享 eval 方法，返回：

- `loss`
- `marginal_probs`
- `generated_ids`
- `labels`

理由：

- loss 和 retrieval 指标都可从同一 payload 更新。
- 保持模型业务计算仍在模型内，指标生命周期在 callback 内。

## Risks / Trade-offs

- [Risk] `training_step` 返回 dict 可能影响 Lightning 对 loss 的自动反向传播。  
  Mitigation: 返回包含 `"loss"` key 的 dict，Lightning 支持从 mapping output 取 loss。

- [Risk] callback 自动注入可能在没有指标配置的模型上误触发。  
  Mitigation: 仅当 `cfg.model.metrics` 存在且非空时追加。

- [Risk] retrieval group 与旧 `SIDRetrievalEvaluator` 同时存在造成重复概念。  
  Mitigation: 本阶段保留兼容，后续全量迁移完成后统一清理。
