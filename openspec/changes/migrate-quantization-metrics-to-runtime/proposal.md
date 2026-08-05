## Why

RKMeans、RVQ、RQVAE 当前在模型内部声明大量 `MeanMetric`，并用动态 `setattr` 创建 per-layer 指标。这是项目中指标逻辑最重的残留点，也是验证 metric runtime 动态 repeat 能力的必要迁移。

## What Changes

- 将三个 quantization 训练模型的指标迁移到 `MetricCallback` / `MetricEngine`。
- 训练、验证、测试 step 返回指标 payload，不再直接 update/log/reset 指标。
- 使用 `model.metrics` 声明 loss、residual norm、centroid norm、unique ids、MSE 等指标。
- 使用 repeat 配置按 `${num_hierarchies}` 生成 per-layer coverage / entropy 指标。
- 保留现有量化 forward、初始化、层训练调度、checkpoint state 行为。

## Capabilities

### New Capabilities
- `quantization-runtime-metrics`: Quantization 模型使用通用指标运行时记录 scalar 和动态 per-layer 指标。

### Modified Capabilities

## Impact

- 影响 `src/quantization/rkmeans/residual_kmeans.py`。
- 影响 `src/quantization/rvq/residual_vector_quantization.py`。
- 影响 `src/quantization/rqvae/residual_quantization_vae.py`。
- 影响 `configs/model/rkmeans_train.yaml`、`configs/model/rvq_train.yaml`、`configs/model/rqvae_train.yaml`。
- 扩展 quantization 单元测试，不运行完整 experiment。
