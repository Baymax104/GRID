## ADDED Requirements

### Requirement: 每个量化模型 SHALL 独立自包含
ResidualKMeans、ResidualVectorQuantization、ResidualQuantizationVAE 各自直接继承 LightningModule，不通过共享基类或 config flag 切换行为。每个模型的完整训练/推理逻辑（forward、training_step、初始化、eval、checkpoint save/load）可在单文件内追踪。

#### Scenario: 模型 _target_ 直接指向独立类
- **WHEN** 维护者查看 rkmeans_train / rvq_train / rqvae_train 的 model 配置
- **THEN** `_target_` MUST 直接指向 `ResidualKMeans` / `ResidualVectorQuantization` / `ResidualQuantizationVAE`，而非统一的 `ResidualQuantization`

#### Scenario: 无 train_layer_wise 配置开关
- **WHEN** 维护者查看量化模型的配置
- **THEN** 配置中 MUST 不存在 `train_layer_wise` 字段，训练策略由模型类本身固定

#### Scenario: 量化逻辑内联在模型中
- **WHEN** 维护者追踪某个量化模型的 centroids 更新逻辑
- **THEN** 该逻辑 MUST 直接在模型类实现中可见，不需要跨文件跳转到 BaseClusteringModule / MiniBatchKMeans / VectorQuantization

### Requirement: 旧抽象层 SHALL 完全删除
`ResidualQuantization`、`BaseClusteringModule`、`MiniBatchKMeans`、`VectorQuantization` 四个类定义 MUST 删除，不再存在于代码库中。

#### Scenario: 旧文件不存在
- **WHEN** 维护者检查 `src/quantization/` 目录
- **THEN** `residual_quantization.py`、`base_clustering_module.py`、`mini_batch_kmeans.py`、`vector_quantization.py` MUST 不存在

#### Scenario: 无残留引用
- **WHEN** 维护者全局搜索旧类名
- **THEN** Python 代码中 MUST 无 `ResidualQuantization`、`BaseClusteringModule`、`MiniBatchKMeans`、`VectorQuantization` 的 import 或引用（配置文件中也不应有）

### Requirement: RQVAE SHALL 自包含 K-Means 收敛初始化
ResidualQuantizationVAE 的 K-Means 收敛初始化逻辑（KMeansPlusPlus 初始化 → 迭代到收敛 → centroids 作为 VQ codebook）MUST 直接在模型类中实现，不依赖外部 `ClusteringModuleInitializer`。

#### Scenario: 无 ClusteringModuleInitializer 依赖
- **WHEN** 维护者查看 rqvae_train 的 model 配置
- **THEN** 初始化器配置 MUST 不引用 `ClusteringModuleInitializer`

#### Scenario: ClusteringModuleInitializer 类已删除
- **WHEN** 维护者检查 `clustering_initializers.py`
- **THEN** `ClusteringModuleInitializer` 类 MUST 不存在
