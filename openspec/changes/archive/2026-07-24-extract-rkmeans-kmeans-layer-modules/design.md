## Context

`ResidualKMeans` 已迁移到 `src/quantization/rkmeans/residual_kmeans.py`。它当前通过多个 parallel lists 管理每层 K-Means 状态：`centroids_list`、`init_buffers`、`is_initialized_list`、`is_initial_step_list`、`cluster_counts_list` 和 `init_centroids_list`。这些状态与 `_buffer_points()`、`_initialization_step()`、`_kmeans_forward()`、`_layer_model_step()`、`_predict_layer()` 混在 `ResidualKMeans` 内，使维护者需要同时理解 LightningModule 生命周期、layer-wise schedule 和单层 K-Means 聚类细节。

用户明确接受不兼容旧 checkpoint 结构，因此本设计不做 `centroids_list.*` 到新 module key 的 checkpoint remap。

## Goals / Non-Goals

**Goals:**

- 用 `nn.ModuleList` 创建 n 个独立 K-Means layer modules。
- 每个 layer module 拥有自己的 centroid 参数、初始化状态、cluster counts 和单层 train/predict 行为。
- 让 `ResidualKMeans` 保留多层 residual traversal、layer-wise schedule、Lightning hooks、metrics、eval/predict 和 optimizer 配置。
- 保持现有 Hydra 配置接口、训练命令、推理输出协议和 step budget 行为不变。
- 不兼容旧 RKMeans checkpoint 参数路径。

**Non-Goals:**

- 不拆分为多个独立 Trainer run。
- 不引入跨 RKMeans/RVQ/RQVAE 共享的通用 KMeans 基类。
- 不重构 RVQ 或 RQVAE。
- 不持久化初始化 buffer。
- 不实现旧 checkpoint 兼容加载。

## Decisions

### Decision 1: 使用 n 个 `KMeansLayer` module，而不是一个 module 管理 n 层参数

每个 residual hierarchy SHALL 对应一个 `KMeansLayer(nn.Module)`，由 `ResidualKMeans.layers: nn.ModuleList` 持有。

Rationale:

- 当前状态天然是 per-layer ownership，`layer.centroids`、`layer.is_initialized` 比 `centroids_list[layer_idx]`、`is_initialized_list[layer_idx]` 更直接。
- 单层 train/predict 行为可以在一个小模块内完整追踪。
- 后续若继续拆 layer training schedule，`KMeansLayer` 是稳定原子边界。

Alternative considered: `ResidualKMeansClustering(nn.Module)` 管理所有层的参数和状态。该方案 checkpoint key 更集中，但仍保留 `*_list[layer_idx]` 风格，不能充分降低单层行为的认知负担。

### Decision 2: `ResidualKMeans` 负责 residual 编排，`KMeansLayer` 负责单层聚类

`ResidualKMeans.forward()` SHALL 继续循环 residual layers，判断当前 layer 是否训练，并调用：

- `layer.train_step(current_residuals)`：初始化或更新当前层 centroid。
- `layer.predict(current_residuals)`：只做最近 centroid 查询。

`KMeansLayer` 不应了解 `current_layer`、step budget、Lightning trainer state、metrics 或 logging。

### Decision 3: centroid 是 module parameter，运行期状态保持非持久化

`KMeansLayer.centroids` SHALL 是 `nn.Parameter`，随 `state_dict` 保存为 `layers.<idx>.centroids`。

`cluster_counts` SHOULD 使用 `register_buffer(..., persistent=False)` 或等价运行期状态；`init_buffer` 和 `init_centroids` SHALL 保持为训练期临时状态，不作为持久化 checkpoint contract。

### Decision 4: checkpoint 不兼容旧 RKMeans 参数路径

新 checkpoint 使用 `layers.<idx>.centroids`。旧 `centroids_list.<idx>` checkpoint 不保证可加载，也不新增 key remap。

Rationale: 用户已明确不需要兼容旧 checkpoint，避免在一次结构性重构中增加迁移分支。

### Decision 5: DDP 初始化路径保持原语义

K-Means++ 初始化仍由 rank zero 计算目标 centroids，再通过当前 manual optimization / world-size loss scaling 路径同步参数。`training_loop_function` 仍由 `ResidualKMeans.training_step()` 调用，但 `is_initialized` 来源改为当前 layer module 的状态。

## Risks / Trade-offs

- [Checkpoint breaking change] 旧 RKMeans checkpoint 无法直接加载 → 明确记录为 breaking change，不实现兼容迁移。
- [DDP 初始化状态错位] `is_initial_step` / `is_initialized` 迁入 layer module 后可能与 manual optimization 的 `is_initialized` 参数不同步 → 保持当前状态转换顺序，并用单元测试覆盖 buffer 未满、buffer 满、初始化后第一步。
- [非持久化 buffer 设备问题] `init_buffer` / `cluster_counts` 迁入 module 后可能留在 CPU → 提供 `reset_training_state(device)`，由 `ResidualKMeans.on_train_start()` 对所有 layer 调用。
- [Metric 访问路径变化] `_compute_output_stats()` 当前直接读取 `centroids_list[0/-1]` → 改为读取 `layers[0].centroids` / `layers[-1].centroids`，保持指标含义不变。
