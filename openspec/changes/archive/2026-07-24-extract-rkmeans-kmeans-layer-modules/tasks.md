## 1. KMeans Layer Module

- [x] 1.1 新增 `src/quantization/rkmeans/kmeans_layer.py`，定义 `KMeansLayer(nn.Module)` 并迁入 `_compute_squared_euclidean_distance` / `_kmeans_plus_plus_init` 可复用函数或保持同目录可导入。
- [x] 1.2 在 `KMeansLayer` 中实现 `centroids: nn.Parameter`、初始化 buffer、初始化状态、cluster counts、init centroids 等单层状态。
- [x] 1.3 在 `KMeansLayer` 中实现 `reset_training_state(device)`，重置初始化 buffer 与 cluster counts 到目标 device。
- [x] 1.4 在 `KMeansLayer` 中实现 `train_step(residuals)`，覆盖 buffer 未满、buffer 满后 K-Means++ 初始化、初始化完成后的 mini-batch KMeans update。
- [x] 1.5 在 `KMeansLayer` 中实现 `predict(residuals)`，只执行最近 centroid 查询并返回 ids 与 embeddings。

## 2. ResidualKMeans 集成

- [x] 2.1 将 `ResidualKMeans` 的 `centroids_list`、`init_buffers`、`is_initialized_list`、`is_initial_step_list`、`cluster_counts_list`、`init_centroids_list` 替换为 `self.layers = nn.ModuleList([...KMeansLayer...])`。
- [x] 2.2 更新 `ResidualKMeans.forward()`，按 hierarchy 顺序调用当前 layer 的 `train_step()` 或 `predict()`，保持 residual normalization、residual stacking 和 quantization loss 聚合行为不变。
- [x] 2.3 更新 `training_step()` 的 initialization 判断，改为读取当前 `KMeansLayer.is_initialized`。
- [x] 2.4 更新 `on_train_start()`，对每个 layer 调用 `reset_training_state(self.device)`，并保留 layer-wise step budget 初始化与日志行为。
- [x] 2.5 更新 `_compute_output_stats()`，从 `self.layers[0].centroids` / `self.layers[-1].centroids` 读取 centroid norm。
- [x] 2.6 更新 checkpoint save/load hook，保存和加载 `[layer.is_initialized for layer in self.layers]`，不实现旧 `centroids_list.*` key remap。

## 3. 清理与一致性

- [x] 3.1 删除 `ResidualKMeans` 中已迁移的单层 K-Means 私有方法，确保 model 文件只保留 residual 编排、训练调度、metrics 和 Lightning hooks。
- [x] 3.2 确认 `configure_optimizers()` 仍通过 `model.parameters()` 覆盖所有 `KMeansLayer.centroids` 参数。
- [x] 3.3 保持 `configs/model/rkmeans_train.yaml` 与 `configs/model/rkmeans_inference.yaml` 的 public 配置接口不变。

## 4. 验证

- [x] 4.1 添加 CPU 单元测试覆盖 `KMeansLayer` buffer 未满、buffer 满后初始化、初始化后 train update、predict 不更新参数。
- [x] 4.2 添加或更新 CPU 单元测试覆盖 `ResidualKMeans` 创建 N 个 layer modules、只训练 current layer、输出 shape 与 residual stacking 不变。
- [x] 4.3 运行与触达范围一致的 smoke check，例如 `uv run pytest` 或针对新增测试文件的 `uv run pytest <path>`。
- [x] 4.4 运行导入检查，确认 `src.quantization.rkmeans.residual_kmeans.ResidualKMeans` 仍可被 Hydra target 解析。
