## Why

`ResidualQuantization`（704 行）通过 `train_layer_wise` 配置 flag 在同一个类中处理三种完全不同的模型（rkmeans / rvq / rqvae），导致 forward / on_train_start / training_step 中充满条件分支，难以理解和维护。量化逻辑（K-Means 更新、VQ-STE）被埋在三层继承抽象（BaseClusteringModule → MiniBatchKMeans / VectorQuantization → ResidualQuantization）之下，增加了不必要的间接性。需要将三个模型的实现拆分为独立自包含的类，内联各自的量化逻辑。

## What Changes

- 新建 `ResidualKMeans`（`src/quantization/residual_kmeans.py`）：K-Means 量化逻辑内联 + layer-wise 逐层训练，替代 rkmeans_train / rkmeans_inference 的模型实现。
- 新建 `ResidualVectorQuantization`（`src/quantization/residual_vector_quantization.py`）：VQ(STE) 量化逻辑内联 + layer-wise 逐层训练，替代 rvq_train 的模型实现。
- 新建 `ResidualQuantizationVAE`（`src/quantization/residual_quantization_vae.py`）：VQ(STE) 量化逻辑内联 + 渐进式联合训练 + encoder/decoder/reconstruction + K-Means 收敛初始化自包含，替代 rqvae_train 的模型实现。
- **BREAKING**：删除 `residual_quantization.py`、`base_clustering_module.py`、`mini_batch_kmeans.py`、`vector_quantization.py`。已有 checkpoint 不兼容，需重新训练。
- 删除 `clustering_initializers.py` 中的 `ClusteringModuleInitializer` 类（K-Means 收敛逻辑内联到 RQVAE）。
- 更新 4 个 model 配置（rkmeans_train / rkmeans_inference / rvq_train / rqvae_train）：`_target_` 指向新类、删除 `train_layer_wise` / `quantization_layer` 等旧参数、清理 stale keys（`verbose` / `quantization_layer_list` / `evaluator`）。
- 更新 `launcher_utils.py`：移除 dry-run 对 `train_layer_wise` 的 override 逻辑。
- 保留 `training_loop_functions.py`（外部工具函数，通过配置注入）及所有可注入组件（DistanceFunction、STEQuantization、WeightedSquaredError 等）。

## Capabilities

### New Capabilities
- `independent-quantization-models`: 三个量化模型（ResidualKMeans / ResidualVectorQuantization / ResidualQuantizationVAE）各自独立自包含，不再通过 config flag 在同一类中切换行为，量化逻辑直接内联在模型实现中。

### Modified Capabilities
<!-- 无。这是模型层重构，不改变 data contract 或 pipeline 行为。 -->

## Impact

- **删除文件**（4 个）：`src/quantization/residual_quantization.py`、`base_clustering_module.py`、`mini_batch_kmeans.py`、`vector_quantization.py`
- **新建文件**（3 个）：`src/quantization/residual_kmeans.py`、`residual_vector_quantization.py`、`residual_quantization_vae.py`
- **修改文件**：
  - `configs/model/rkmeans_train.yaml`、`rkmeans_inference.yaml`、`rvq_train.yaml`、`rqvae_train.yaml`
  - `src/utils/launcher_utils.py`（dry-run override 调整）
  - `src/common/components/clustering_initializers.py`（删除 ClusteringModuleInitializer）
- **Checkpoint 兼容性**：**BREAKING** — 三类新模型的 state_dict 结构与旧 `ResidualQuantization` 不同，已有 checkpoint 无法加载，需重新训练。
- **保留不变**：`training_loop_functions.py`、所有可注入组件（DistanceFunction / STEQuantization / WeightedSquaredError / BetaQuantizationLoss / KMeansPlusPlusInitInitializer / 各 scheduler / optimizer）、data pipeline 配置。
