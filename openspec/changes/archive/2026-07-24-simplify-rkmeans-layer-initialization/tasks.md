## 1. KMeansLayer 初始化状态简化

- [x] 1.1 删除 `KMeansLayer.is_initial_step` 状态以及所有读写逻辑。
- [x] 1.2 删除 `KMeansLayer.init_centroids` 作为跨 step 暂存状态的字段和 reset 逻辑。
- [x] 1.3 调整 `reset_training_state(device)`，仅重置 `init_buffer`、`cluster_counts` 和必要运行期状态，并保持 `is_initialized` 的 checkpoint 恢复语义不被意外覆盖。
- [x] 1.4 调整 `train_step(residuals)`，使未初始化 layer 只在 buffer 未满时走 buffer 收集分支，buffer 满后直接完成 centroid 初始化并标记 initialized。

## 2. Direct Centroid Initialization 与 Broadcast

- [x] 2.1 新增 quantization-local distributed helper，用于判断 distributed 是否初始化并获取当前 rank。
- [x] 2.2 实现 rank 0 计算 K-Means++ initial centroids、非 rank 0 准备同 shape 接收 tensor、所有 rank broadcast 的初始化路径。
- [x] 2.3 在非 distributed 环境保持单进程直接 K-Means++ 初始化路径。
- [x] 2.4 使用 `torch.no_grad()` 将初始化结果 `copy_` 到 `layer.centroids`，初始化后清空 `init_buffer`。
- [x] 2.5 初始化完成的同一个 train step 返回基于新 centroids 的 ids/embeddings，并返回不会触发初始化 optimizer 过渡的 loss 语义。

## 3. ResidualKMeans 集成与一致性

- [x] 3.1 确认 `ResidualKMeans.forward()` 和 `training_step()` 只依赖 `layer.is_initialized` 判断当前层训练与 schedule advance。
- [x] 3.2 确认 checkpoint save/load 仍保存和恢复 `[layer.is_initialized for layer in self.layers]`，不保存 `init_buffer` 或临时 init centroid。
- [x] 3.3 保持 `configs/model/rkmeans_train.yaml` 与 `configs/model/rkmeans_inference.yaml` public 配置接口不变。
- [x] 3.4 确认 `src/quantization/training_loop_functions.py` 不再需要处理 RKMeans 初始化 loss 过渡状态，但仍兼容 initialized 参数。

## 4. 测试与验证

- [x] 4.1 更新 `KMeansLayer` CPU 单元测试，覆盖 buffer 未满时不初始化、buffer 满时同 step 直接 initialized、`is_initial_step` 不存在、predict 不更新状态。
- [x] 4.2 添加或更新 distributed helper / broadcast 路径的可测试单元，使用 monkeypatch 或最小 mock 验证 rank 0/non-rank0 分支不需要真实多进程训练。
- [x] 4.3 更新 `ResidualKMeans` CPU 单元测试，确认 current layer 初始化后 schedule 判断读取单一 initialized 状态，输出 shape 与 residual stacking 不变。
- [x] 4.4 运行触达范围一致的测试，例如 `uv run pytest tests/quantization/rkmeans/test_kmeans_layer.py`。
- [x] 4.5 运行 Hydra target 导入检查，确认 `src.quantization.rkmeans.residual_kmeans.ResidualKMeans` 仍可解析。
