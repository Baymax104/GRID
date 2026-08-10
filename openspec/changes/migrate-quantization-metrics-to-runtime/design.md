## Context

三个 quantization 模型的指标逻辑结构基本一致：

- train 阶段记录 `loss`、`quantization_loss`、residual norm ratio、centroid norm、`frac_unique_ids`、`mse`。
- RQVAE 额外记录 `reconstruction_loss`。
- train 阶段按 layer 记录 `frac_layer_coverages` 和 `id_entropy`。
- val/test 阶段记录 `loss`、residual norm ratio、`frac_unique_ids`、`mse`。

新 runtime 已支持 stage 隔离、callback 生命周期和 repeat 展开，因此模型只需要返回 payload。

## Goals / Non-Goals

**Goals:**

- 移除 quantization 模型中的 `MeanMetric` 属性和动态指标 `setattr`。
- 移除 quantization 模型中的指标 `log_dict` 和指标 reset hooks。
- 保持原有指标名称和阶段前缀。
- 保持 per-layer 指标数量由 `num_hierarchies` 配置生成。
- 用 focused tests 覆盖模型不再暴露指标属性、配置声明 repeat 指标。

**Non-Goals:**

- 不改变指标公式。
- 不改变 layer 初始化、训练调度、checkpoint 字段。
- 不运行完整 experiment。

## Decisions

### 复用模型内 `_compute_output_stats`

迁移时保留 `_compute_output_stats`，因为它计算的是模型输出统计事实，而不是指标生命周期。`training_step` / `eval_step` 调用它后把结果放入 payload。

### 用配置 repeat 表达 per-layer 指标

`configs/model/*_train.yaml` 的 `model.metrics.stages.train` 使用 repeat：

- `layer_{layer_idx}/frac_layer_coverages`
- `layer_{layer_idx}/id_entropy`

`spec.index` 使用 `{layer_idx}` 模板。

### 训练 step 返回 dict

训练 step 返回包含 `"loss"` 的 mapping，Lightning 可继续使用该 loss 做反向传播。

## Risks / Trade-offs

- [Risk] 训练指标原先只在 `log_every_n_steps` 时计算 output stats，迁移后每步计算会增加开销。  
  Mitigation: 这是用户确认的行为变更；三个 quantization 模型统一每步返回完整 metric payload，避免 callback 侧出现缺字段或旧值日志。
