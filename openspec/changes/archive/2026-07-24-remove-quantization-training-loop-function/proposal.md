## Why

量化模型当前通过 `training_loop_function` 暴露一条特殊 manual optimization 路径，用于在 DDP 初始化阶段把 rank 0 计算出的 centroids 通过 loss 缩放和临时 SGD optimizer 间接写入参数。RKMeans 已经改为 direct centroid assignment + rank-zero broadcast，继续保留该 loop 函数会让训练入口、配置和初始化语义变复杂。

本变更将 RVQ / RQVAE 的初始化也改为 direct assignment，从而删除 quantization 自定义训练 loop，回到 Lightning 默认 automatic optimization。

## What Changes

- **BREAKING**: 从 RKMeans、RVQ、RQVAE model constructor 和训练配置中移除 `training_loop_function` 字段。
- 删除 `src/quantization/training_loop_functions.py` 中的 `scale_loss_by_world_size_for_initialization_training_loop`，以及所有配置引用。
- RKMeans 删除已不再需要的 manual optimization 分支，继续保留当前 direct centroid assignment + rank-zero broadcast 初始化。
- RVQ 初始化从“init loss + custom loop 间接更新 centroids”改为“rank 0 计算 K-Means++ centroids → broadcast → `copy_` 到 centroids”。
- RQVAE 初始化从“K-Means++ + convergence 后通过 init loss 间接更新 codebook”改为“rank 0 完成 K-Means++ + convergence → broadcast → `copy_` 到 centroids”。
- 删除 RVQ / RQVAE 的初始化过渡状态（如 `is_initial_step_list` / `init_centroids_list` / `init_loss_function`）中不再需要的部分。
- 更新测试与 smoke check，确保三种量化模型不再进入 manual optimization。

## Capabilities

### New Capabilities

- `quantization-default-training-loop`: 约束量化模型使用 Lightning 默认 training loop，初始化 centroids 通过 direct assignment 而不是 custom manual optimization 完成。

### Modified Capabilities

- `quantization-initialization-training-strategy`: 初始化策略从“quantization 子域中的特殊 manual optimization”改为“rank-zero direct centroid initialization + broadcast”。
- `config-manual-input-clarity`: 量化训练配置不得暴露不需要用户选择的 `training_loop_function` 实现细节。
- `independent-quantization-models`: 各量化模型继续自包含，并在模型自身初始化逻辑内完成 direct centroid assignment。

## Impact

- Affected code:
  - `src/quantization/training_loop_functions.py`
  - `src/quantization/rkmeans/residual_kmeans.py`
  - `src/quantization/rkmeans/kmeans_layer.py`
  - `src/quantization/rvq/residual_vector_quantization.py`
  - `src/quantization/rqvae/residual_quantization_vae.py`
- Affected config:
  - `configs/model/rkmeans_train.yaml`
  - `configs/model/rvq_train.yaml`
  - `configs/model/rqvae_train.yaml`
- Tests:
  - Extend quantization unit tests to assert constructor/config cleanup and direct initialization behavior.
- Existing external configs or overrides that still pass `model.training_loop_function` will fail and must remove the field.
- Checkpoint centroids layout, prediction output bundle format, and script command shape are not intended to change.
