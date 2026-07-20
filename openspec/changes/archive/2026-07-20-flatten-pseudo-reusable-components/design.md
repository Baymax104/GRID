## Context

`src/common/components/` 当前有 9 个文件，其中 4 个（distance_functions、clustering_initializers、aggregation_strategy、quantization_strategies）采用 ABC + 多备选实现模式，但实际只用 1 个实现。另外 `optimizer.py` 整个文件为死代码，保留模块中也存在多个死类。这与之前 quantization 模块拆分的理念一致：消除伪复用，让每个模型自包含。

当前依赖链：
```
quantization model
  ├── initializer (ClusteringInitializer) → 用 distance_function
  ├── quantization_strategy (QuantizationStrategy) → 用 distance_function  [VQ only]
  └── distance_function (DistanceFunction) → SquaredEuclideanDistance
```

## Goals / Non-Goals

**Goals:**
- 将 4 个伪复用模块展平为消费方文件内的函数
- 删除全部死代码（死文件 + 死类 + 重复类）
- 更新配置，移除展平组件的注入参数

**Non-Goals:**
- 不移动保留模块（model_output、loss_functions、scheduler、eval_metrics）的位置
- 不展平 `common/modules/`（embedding_aggregator 仅内联 aggregation_strategy 逻辑，文件位置不变）
- 不改变模型的核心训练逻辑

## Decisions

### 决策 1：核心逻辑抽取为函数，非完全展开

**选择**：将展平组件的核心逻辑抽取为独立函数放在消费方模型文件中，而非在调用处完全展开。

**理由**：保持函数封装使代码可读性更好——函数名表达意图，调用处简洁。完全展开会让模型方法过长且难以快速理解。

**备选**：完全在调用处展开。放弃——降低可读性。

### 决策 2：distance_function 重复到每个 quantization 模型文件

**选择**：`_compute_squared_euclidean_distance` 函数在 3 个 quantization 模型文件中各放一份。

**理由**：与 quantization 模块拆分的决策一致——模型完全独立，允许重复代码。distance computation 是无状态纯函数，重复无副作用。若提取到 utils 会重新引入跨模块依赖。

**备选**：提取到 `src/utils/`。放弃——会重新引入 common 层依赖，与"消除伪复用"目标矛盾。

### 决策 3：保留 last_k 参数于 EmbeddingAggregator

**选择**：`MeanAggregation` 的 `last_k` 参数移到 `EmbeddingAggregator` 构造函数中，聚合函数接收它作为参数。

**理由**：`last_k` 是 `MeanAggregation` 的唯一配置参数，展平后自然归属到 `EmbeddingAggregator`。sem_embeds_inference 配置中 `aggregation_strategy.last_k` 变为 `EmbeddingAggregator` 的直接参数。

### 决策 4：STE 函数签名接收 distance 函数

**选择**：`_ste_quantize(codebook, batch, distance_fn)` 接收 distance 函数作为参数（默认值为同文件的 `_compute_squared_euclidean_distance`）。

**理由**：STE 逻辑依赖距离计算，通过参数注入而非硬编码，使函数可测试且语义清晰。由于 distance 函数在同一文件内，不引入跨模块依赖。

### 决策 5：KMeansPlusPlus 函数接收 n_clusters 和 distance_fn

**选择**：`_kmeans_plus_plus_init(buffer, n_clusters, initialize_on_cpu=True, distance_fn=_compute_squared_euclidean_distance)` 作为模型文件内的函数。

**理由**：`n_clusters` 和 `initialize_on_cpu` 原为 `KMeansPlusPlusInitInitializer` 构造参数，展平后作为函数参数传递。`distance_fn` 默认指向同文件的距离函数。

## Risks / Trade-offs

- **[代码重复]** 3 个 quantization 模型各有一份 distance + K-Means++ 函数。→ **缓解**：与既定决策一致（独立模型允许重复）；这些函数简短且稳定。
- **[配置不兼容]** 删除注入参数使旧配置失效。→ **缓解**：BREAKING CHANGE，一并更新所有配置。
- **[checkpoint 不兼容]** 模型构造参数变化。→ **缓解**：与 quantization 拆分时一致，接受重新训练。
