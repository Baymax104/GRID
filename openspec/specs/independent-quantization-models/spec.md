# independent-quantization-models Specification

## Purpose
TBD - created by archiving change split-quantization-models. Update Purpose after archive.
## Requirements
### Requirement: 每个量化模型 SHALL 独立自包含
ResidualKMeans、ResidualVectorQuantization、ResidualQuantizationVAE SHALL 各自直接继承 LightningModule，不通过共享基类或 config flag 切换行为。ResidualVectorQuantization SHALL keep its complete training/inference logic traceable within `src/quantization/rvq/` and MAY将单层 VQ codebook 参数、初始化状态与 VQ-STE 行为封装到同目录下的专用 layer module 中。ResidualQuantizationVAE 的完整训练/推理逻辑（forward、training_step、初始化、eval、checkpoint save/load）SHALL 可在单文件内追踪。ResidualKMeans SHALL 可在 `src/quantization/rkmeans/` 模型目录内追踪完整行为，并 MAY 将单层 K-Means 聚类参数与行为封装到同目录下的专用 layer module 中。量化模型 SHALL NOT 保留未使用的 `initialize_on_cpu` 初始化配置开关或 CPU 初始化分支。量化模型初始化 SHALL 在模型内部通过 direct centroid assignment 完成，不依赖外部 `training_loop_function` 或特殊 manual optimization hook。

#### Scenario: 模型 _target_ 直接指向按模型分组的独立类
- **WHEN** 维护者查看 rkmeans_train / rvq_train / rqvae_train 的 model 配置
- **THEN** `_target_` MUST 直接指向 `src.quantization.rkmeans.residual_kmeans.ResidualKMeans` / `src.quantization.rvq.residual_vector_quantization.ResidualVectorQuantization` / `src.quantization.rqvae.residual_quantization_vae.ResidualQuantizationVAE`，而非统一的 `ResidualQuantization`

#### Scenario: 无 train_layer_wise 配置开关
- **WHEN** 维护者查看量化模型的配置
- **THEN** 配置中 MUST 不存在 `train_layer_wise` 字段，训练策略由模型类本身固定

#### Scenario: RKMeans 量化逻辑位于模型目录内
- **WHEN** 维护者追踪 ResidualKMeans 的 centroids 更新逻辑
- **THEN** 该逻辑 MUST 位于 `src/quantization/rkmeans/` 模型目录内
- **THEN** 该逻辑 MUST NOT 依赖 BaseClusteringModule / MiniBatchKMeans / VectorQuantization 或跨模型共享抽象

#### Scenario: RVQ 量化逻辑位于模型目录内
- **WHEN** 维护者追踪 ResidualVectorQuantization 的 centroids 更新逻辑
- **THEN** 该逻辑 MUST 位于 `src/quantization/rvq/` 模型目录内
- **THEN** 该逻辑 MAY 通过同目录下的专用 layer module 暴露单层 VQ codebook 行为
- **THEN** 该逻辑 MUST NOT 依赖 BaseClusteringModule / MiniBatchKMeans / VectorQuantization 或跨模型共享抽象

#### Scenario: RQVAE 量化逻辑仍在模型类中可见
- **WHEN** 维护者追踪 ResidualQuantizationVAE 的 centroids 更新逻辑
- **THEN** 该逻辑 MUST 直接在对应模型类实现中可见，不需要跨文件跳转到 BaseClusteringModule / MiniBatchKMeans / VectorQuantization

#### Scenario: 无 initialize_on_cpu 配置开关
- **WHEN** 维护者查看 RKMeans / RVQ / RQVAE 的 model 配置和构造函数
- **THEN** 配置和构造函数中 MUST 不存在 `initialize_on_cpu` 字段
- **AND** K-Means 初始化路径 MUST 不包含基于该字段的 CPU 分支

#### Scenario: 无外部 training_loop_function
- **WHEN** 维护者查看 RKMeans / RVQ / RQVAE 的 model 配置和构造函数
- **THEN** 配置和构造函数中 MUST 不存在 `training_loop_function` 字段
- **AND** 初始化逻辑 MUST 不依赖 `scale_loss_by_world_size_for_initialization_training_loop`

### Requirement: 旧抽象层 SHALL 完全删除
`ResidualQuantization`、`BaseClusteringModule`、`MiniBatchKMeans`、`VectorQuantization` 四个类定义 SHALL 删除，不再存在于代码库中。

#### Scenario: 旧文件不存在
- **WHEN** 维护者检查 `src/quantization/` 目录
- **THEN** `residual_quantization.py`、`base_clustering_module.py`、`mini_batch_kmeans.py`、`vector_quantization.py` MUST 不存在

#### Scenario: 无残留引用
- **WHEN** 维护者全局搜索旧类名
- **THEN** Python 代码中 MUST 无 `ResidualQuantization`、`BaseClusteringModule`、`MiniBatchKMeans`、`VectorQuantization` 的 import 或引用（配置文件中也不应有）

### Requirement: RQVAE SHALL 自包含 K-Means 收敛初始化
ResidualQuantizationVAE 的 K-Means 收敛初始化逻辑（KMeansPlusPlus 初始化 → 迭代到收敛 → centroids 作为 VQ codebook）SHALL 直接在模型类中实现，不依赖外部 `ClusteringModuleInitializer`。

#### Scenario: 无 ClusteringModuleInitializer 依赖
- **WHEN** 维护者查看 rqvae_train 的 model 配置
- **THEN** 初始化器配置 MUST 不引用 `ClusteringModuleInitializer`

#### Scenario: ClusteringModuleInitializer 类已删除
- **WHEN** 维护者检查 `clustering_initializers.py`
- **THEN** `ClusteringModuleInitializer` 类 MUST 不存在

