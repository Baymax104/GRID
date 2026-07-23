## Why

`KMeansLayer` 当前用 `is_initial_step` 与 `is_initialized` 两个布尔状态表达初始化流程，实际是为了让初始化 centroid 通过一次 loss/optimizer step 生效。这让单层初始化状态难以理解，并且让 `ResidualKMeans` 的“当前层是否已初始化”语义依赖一个隐式过渡 step。

本次变更采用更直接的初始化模型：rank 0 使用 `init_buffer` 运行 K-Means++，将结果直接写入 centroid 参数，并通过 distributed broadcast 同步到所有 rank；layer 只保留一个简单的 initialized 状态。

## What Changes

- 删除 RKMeans `KMeansLayer` 的 `is_initial_step` 过渡状态。
- 删除 `init_centroids` 作为跨 step 暂存状态的需求。
- 当 `init_buffer` 满足初始化条件时，直接将 K-Means++ 结果写入 `layer.centroids`。
- 在 distributed 训练时，由 rank 0 初始化 centroid，并 broadcast 到所有 rank，保持多卡初始 centroid 一致。
- `KMeansLayer.train_step()` 在初始化完成后即可返回基于新 centroid 的 ids/embeddings，并不再返回初始化 loss。
- `ResidualKMeans` 继续只依赖 `layer.is_initialized` / `layer.initialized` 判断调度与 checkpoint 状态。
- 保持 Hydra 配置接口、训练命令、推理输出协议不变。

## Capabilities

### New Capabilities

- `rkmeans-direct-centroid-initialization`: 定义 RKMeans KMeans layer 使用直接 centroid 初始化与 distributed broadcast 的单状态初始化行为。

### Modified Capabilities

- `quantization-initialization-training-strategy`: 明确 RKMeans 的初始化同步策略仍局部位于 quantization 子域，但不再依赖初始化 loss/manual optimizer step 过渡。

## Impact

- Affected code:
  - `src/quantization/rkmeans/kmeans_layer.py`
  - `src/quantization/rkmeans/residual_kmeans.py`
  - `tests/quantization/rkmeans/test_kmeans_layer.py`
- Affected specs:
  - 新增 `rkmeans-direct-centroid-initialization`
  - 修改 `quantization-initialization-training-strategy`
- Checkpoint impact:
  - 不改变 centroid 参数路径；runtime 初始化状态 checkpoint 仍只需保存每层 initialized 布尔值。
- No external dependency changes.
