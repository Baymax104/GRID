## Why

`src/common/components/` 中有 4 个模块（distance_functions、clustering_initializers、aggregation_strategy、quantization_strategies）采用 ABC 基类 + 多个备选实现的模式，但当前可用链路中每个模块只用到 1 个实现，其余均为死代码。这种伪复用增加了不必要的间接层和配置注入开销，使代码难以直读。同时 `optimizer.py` 整个文件为死代码，保留模块中也存在多个死类。需要将这些伪复用模块展平为消费方文件内的函数，并清理全部死代码。

## What Changes

- **展平 distance_functions**：将 `SquaredEuclideanDistance.compute` 逻辑抽取为函数，放入使用它的 quantization 模型文件中。删除 `distance_functions.py`（含其中重复的 `WeightedSquaredError`/`BetaQuantizationLoss` 副本）。
- **展平 clustering_initializers**：将 `KMeansPlusPlusInitInitializer.forward` 逻辑抽取为函数，放入 3 个 quantization 模型文件中。删除 `clustering_initializers.py`（含死类 `RandomInitializer`）。
- **展平 aggregation_strategy**：将 `MeanAggregation.aggregate` 逻辑抽取为函数，放入 `embedding_aggregator.py`。删除 `aggregation_strategy.py`（含死类 `LastAggregation`、`FirstAggregation`）。
- **展平 quantization_strategies**：将 `STEQuantization.quantize` 逻辑抽取为函数，放入 `residual_vector_quantization.py` 和 `residual_quantization_vae.py`。删除 `quantization_strategies.py`（含死类 `GumbelSoftmaxQuantization`、`RotationTrickQuantization`）。
- **删除 `optimizer.py`**：整个文件为死代码（`PassThroughOptimizer` 零引用）。
- **清理保留模块死类**：`loss_functions.py` 删除 `FullBatchCrossEntropyLoss`；`eval_metrics.py` 删除 `RetrievalEvaluator`。
- **配置更新**：4 个 quantization model configs 删除 `distance_function`/`initializer` 注入参数；rvq_train/rqvae_train 删除 `quantization_strategy` 注入参数；sem_embeds_inference model config 删除 `aggregation_strategy` 注入参数。
- **BREAKING**：checkpoint 不兼容（模型构造参数变化）。
