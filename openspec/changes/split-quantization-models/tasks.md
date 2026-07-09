## 1. 新建 ResidualKMeans（rkmeans 模型）

- [x] 1.1 创建 `src/quantization/residual_kmeans.py`：`ResidualKMeans(LightningModule)`，内联 K-Means 量化逻辑。包含：`centroids: nn.ParameterList`、`init_buffer` list、`is_initialized` / `is_initial_step` flags、`cluster_counts` list
- [x] 1.2 实现 `forward`：layer-wise 逐层训练（`train_layer = idx == current_layer`），每层计算 assignments + batch_cluster_counts + batch_cluster_sums
- [x] 1.3 实现 `model_step`：初始化检查 → 缓冲数据 → KMeansPlusPlus 初始化 → K-Means mini-batch 更新或 WeightedSquaredError loss
- [x] 1.4 实现 `on_train_start`：step budget + layer schedule + boundary 计算
- [x] 1.5 实现 `training_step`：调用 training_loop_function + layer schedule 推进
- [x] 1.6 实现 eval/val/test/predict_step、`_compute_output_stats`、`configure_optimizers`、`on_save_checkpoint`/`on_load_checkpoint`、metrics 初始化
- [x] 1.7 实现 `normalize_residuals` 逻辑（作为构造参数）

## 2. 新建 ResidualVectorQuantization（rvq 模型）

- [x] 2.1 创建 `src/quantization/residual_vector_quantization.py`：`ResidualVectorQuantization(LightningModule)`，内联 VQ-STE 量化逻辑
- [x] 2.2 实现 `forward`：layer-wise 逐层训练，每层用 quantization_strategy.quantize(codebook, batch) → ids + embeddings
- [x] 2.3 实现 `model_step`：初始化检查 → 缓冲数据 → initializer 初始化 → VQ forward → loss_function(batch, embeddings)
- [x] 2.4 复用 ResidualKMeans 的 `on_train_start` / `training_step` / eval / checkpoint / metrics 结构（layer-wise 一致）
- [x] 2.5 实现 `normalize_residuals` 逻辑

## 3. 新建 ResidualQuantizationVAE（rqvae 模型）

- [x] 3.1 创建 `src/quantization/residual_quantization_vae.py`：`ResidualQuantizationVAE(LightningModule)`，内联 VQ-STE + 渐进式联合训练 + encoder/decoder/reconstruction + K-Means 收敛初始化
- [x] 3.2 实现 `forward`：渐进式初始化逻辑——第一层总训练，后续层在前一层初始化后解锁，已初始化层在非全部初始化时冻结
- [x] 3.3 实现 K-Means 收敛初始化自包含：KMeansPlusPlus 初始化 → 迭代 K-Means 更新到收敛（max 1000 次）→ centroids 作为 VQ codebook
- [x] 3.4 实现 `model_step`：含 encoder forward → 量化 → decoder → reconstruction loss
- [x] 3.5 实现 `on_train_start`：仅日志（无 step budget）
- [x] 3.6 实现 `training_step`：`layer_to_check=-1`（检查最后一层），无 schedule 推进
- [x] 3.7 实现 encoder/decoder（MLP）、normalization_layer、reconstruction_loss 的构造与 forward 集成
- [x] 3.8 复用 eval/val/test/predict/checkpoint/metrics 结构

## 4. 更新 model 配置

- [x] 4.1 更新 `configs/model/rkmeans_train.yaml`：`_target_` → ResidualKMeans，删旧参数，内联 K-Means 量化层参数
- [x] 4.2 更新 `configs/model/rkmeans_inference.yaml`：`_target_` → ResidualKMeans，同上清理
- [x] 4.3 更新 `configs/model/rvq_train.yaml`：`_target_` → ResidualVectorQuantization，删旧参数，内联 VQ 量化层参数
- [x] 4.4 更新 `configs/model/rqvae_train.yaml`：`_target_` → ResidualQuantizationVAE，删旧参数 + ClusteringModuleInitializer 引用，内联 VQ + encoder/decoder/reconstruction 参数

## 5. 删除旧文件与类

- [x] 5.1 删除 `src/quantization/residual_quantization.py`
- [x] 5.2 删除 `src/quantization/base_clustering_module.py`
- [x] 5.3 删除 `src/quantization/mini_batch_kmeans.py`
- [x] 5.4 删除 `src/quantization/vector_quantization.py`
- [x] 5.5 删除 `src/common/components/clustering_initializers.py` 中的 `ClusteringModuleInitializer` 类 + 清理 unused import

## 6. 更新 launcher_utils.py

- [x] 6.1 删除 `launcher_utils.py` 中 dry-run 对 `train_layer_wise` 的 override 逻辑

## 7. 验证

- [x] 7.1 grep 确认旧类名零残留（排除 openspec/，仅新类名 + 注释引用）
- [x] 7.2 grep 确认旧配置参数零残留（train_layer_wise / quantization_layer_list / quantization_layer 在 model 配置中为零；verbose/evaluator 仅在 callbacks 和 tiger model 中存在，不涉及本次变更）
- [x] 7.3 import 3 个新模型文件全部通过
- [x] 7.4 Hydra `--cfg job` 验证 4 个实验 config composition 成功
- [x] 7.5 model 实例化验证通过（ResidualKMeans / ResidualVectorQuantization / ResidualQuantizationVAE 均成功实例化，参数正确）
