## 1. 展平 distance_functions + clustering_initializers 到 quantization 模型

- [x] 1.1 在 `residual_kmeans.py` 中新增 `_compute_squared_euclidean_distance(x, y, batch_size=256)` 函数（从 `SquaredEuclideanDistance.compute` 提取）和 `_kmeans_plus_plus_init(buffer, n_clusters, initialize_on_cpu=True, distance_fn=...)` 函数（从 `KMeansPlusPlusInitInitializer.forward` 提取）；`_compute_initial_centroids` 改为调用 `_kmeans_plus_plus_init`；删除 `initializer` 构造参数和 `distance_function` 构造参数
- [x] 1.2 在 `residual_vector_quantization.py` 中新增同上两个函数 + `_ste_quantize(codebook, batch, distance_fn=..., compute_reconstruction_loss_embeddings=False)` 函数（从 `STEQuantization.quantize` 提取）；`_vq_forward` 改为调用 `_ste_quantize`；删除 `initializer`/`distance_function`/`quantization_strategy` 构造参数
- [x] 1.3 在 `residual_quantization_vae.py` 中新增同上三个函数；`_compute_initial_centroids` 改用 `_kmeans_plus_plus_init`（含 K-Means 收敛迭代逻辑）；`_vq_forward` 改为调用 `_ste_quantize`；删除 `initializer`/`distance_function`/`quantization_strategy` 构造参数
- [x] 1.4 删除 `src/common/components/distance_functions.py` 和 `src/common/components/clustering_initializers.py`

## 2. 展平 quantization_strategies 到 VQ 模型

- [x] 2.1 确认 Task 1.2/1.3 中已将 `STEQuantization.quantize` 逻辑内联为 `_ste_quantize` 函数
- [x] 2.2 删除 `src/common/components/quantization_strategies.py`

## 3. 展平 aggregation_strategy 到 EmbeddingAggregator

- [x] 3.1 在 `src/common/modules/embedding_aggregator.py` 中新增 `_mean_aggregate(embeddings, row_ids, last_item_index, last_k=None)` 函数（从 `MeanAggregation.aggregate` 提取）；`forward` 改为调用该函数；`__init__` 接收 `last_k` 参数替代 `aggregation_strategy`
- [x] 3.2 删除 `src/common/components/aggregation_strategy.py`；移除 `embedding_aggregator.py` 中对 `aggregation_strategy` 的 import

## 4. 清理死代码

- [x] 4.1 删除 `src/common/components/optimizer.py`（整个文件死代码）
- [x] 4.2 `src/common/components/loss_functions.py` 删除 `FullBatchCrossEntropyLoss` 类
- [x] 4.3 `src/common/components/eval_metrics.py` 删除 `RetrievalEvaluator` 类

## 5. 更新配置

- [x] 5.1 `configs/model/rkmeans_train.yaml`：删除 `distance_function` 和 `initializer` 的 `_target_` 块及其参数；将 `n_clusters`/`initialize_on_cpu` 等参数扁平化到 model root
- [x] 5.2 `configs/model/rkmeans_inference.yaml`：同上
- [x] 5.3 `configs/model/rvq_train.yaml`：同上 + 删除 `quantization_strategy` 的 `_target_` 块及其参数
- [x] 5.4 `configs/model/rqvae_train.yaml`：同上 + 删除 `quantization_strategy` 的 `_target_` 块及其参数
- [x] 5.5 `configs/model/sem_embeds_inference.yaml`：删除 `aggregation_strategy` 的 `_target_` 块；将 `last_k` 参数移到 `EmbeddingAggregator` 直接参数

## 6. 验证

- [x] 6.1 import 检查：3 个 quantization 模型 + `embedding_aggregator` + `hf_language_model` 均可正常 import
- [x] 6.2 grep 确认零残留：`DistanceFunction`/`ClusteringInitializer`/`QuantizationStrategy`/`AggregationStrategy`/`PassThroughOptimizer`/`FullBatchCrossEntropyLoss`/`RetrievalEvaluator`/`RandomInitializer`/`GumbelSoftmax`/`RotationTrick`/`LastAggregation`/`FirstAggregation` 在 .py/.yaml 中零引用
- [x] 6.3 grep 确认 5 个展平文件已删除（distance_functions.py/clustering_initializers.py/quantization_strategies.py/aggregation_strategy.py/optimizer.py）
- [x] 6.4 Hydra `--cfg job` 验证 5 个实验 config composition 全部通过（rkmeans_train/rkmeans_inference/rvq_train/rqvae_train/sem_embeds_inference）
- [x] 6.5 dry-run smoke check 验证 datamodule + model 实例化通过（至少 rkmeans_train + sem_embeds_inference）
