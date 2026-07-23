## MODIFIED Requirements

### Requirement: 每个量化模型 SHALL 独立自包含
ResidualKMeans、ResidualVectorQuantization、ResidualQuantizationVAE SHALL 各自直接继承 LightningModule，不通过共享基类或 config flag 切换行为。ResidualVectorQuantization 与 ResidualQuantizationVAE 的完整训练/推理逻辑（forward、training_step、初始化、eval、checkpoint save/load）SHALL 可在单文件内追踪。ResidualKMeans SHALL 可在 `src/quantization/rkmeans/` 模型目录内追踪完整行为，并 MAY 将单层 K-Means 聚类参数与行为封装到同目录下的专用 layer module 中。

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

#### Scenario: RVQ 和 RQVAE 量化逻辑仍在模型类中可见
- **WHEN** 维护者追踪 ResidualVectorQuantization 或 ResidualQuantizationVAE 的 centroids 更新逻辑
- **THEN** 该逻辑 MUST 直接在对应模型类实现中可见，不需要跨文件跳转到 BaseClusteringModule / MiniBatchKMeans / VectorQuantization
