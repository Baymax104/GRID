## Why

`ResidualKMeans` 当前同时维护多层 residual 编排、逐层训练调度、K-Means 初始化状态、centroid 参数和单层聚类更新逻辑，导致每层 K-Means 行为需要通过多个 `*_list[layer_idx]` 状态追踪，难以理解和后续拆分。

本次变更将 RKMeans 的单层 K-Means 聚类参数与行为抽取为独立 `nn.Module`，让 `ResidualKMeans` 专注于 Lightning 生命周期、layer-wise 调度和 residual quantization 编排。

## What Changes

- 新增单层 K-Means layer module，每个 residual hierarchy 对应一个独立 module。
- 将每层 centroid 参数、初始化 buffer、初始化状态、cluster counts 和单层 train/predict 逻辑从 `ResidualKMeans` 中迁移到 layer module。
- `ResidualKMeans` 改用 `nn.ModuleList` 组合 n 个 K-Means layer module。
- 保留现有配置接口、训练命令、推理输出协议和 layer-wise step budget 行为。
- 不兼容旧 checkpoint 结构；新 checkpoint 使用 module 化后的参数路径。
- 不将临时初始化 buffer 作为持久化 checkpoint 状态。

## Capabilities

### New Capabilities

- `rkmeans-kmeans-layer-modules`: 定义 RKMeans 使用独立 KMeans layer modules 封装单层聚类参数、初始化状态和单层 train/predict 行为。

### Modified Capabilities

- `independent-quantization-models`: 放宽 RKMeans “单文件完整追踪”的实现约束，允许模型目录内的专用 layer module 承载 K-Means 聚类逻辑，同时保持模型不依赖共享基类或跨模型抽象。

## Impact

- Affected code:
  - `src/quantization/rkmeans/residual_kmeans.py`
  - 新增 `src/quantization/rkmeans/kmeans_layer.py` 或等价模块
- Affected specs:
  - 新增 `rkmeans-kmeans-layer-modules`
  - 修改 `independent-quantization-models`
- Checkpoint impact:
  - **BREAKING**: 不兼容旧 RKMeans checkpoint 的 `centroids_list.*` 参数路径。
- No external dependency changes.
