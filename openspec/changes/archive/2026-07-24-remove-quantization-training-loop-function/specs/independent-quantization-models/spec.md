## MODIFIED Requirements

### Requirement: 每个量化模型 SHALL 独立自包含
ResidualKMeans、ResidualVectorQuantization、ResidualQuantizationVAE SHALL 各自直接继承 LightningModule，不通过共享基类或 config flag 切换行为。每个模型的完整训练/推理逻辑（forward、training_step、初始化、eval、checkpoint save/load）SHALL 可在对应模型目录或模型文件内追踪。量化模型初始化 SHALL 在模型内部通过 direct centroid assignment 完成，不依赖外部 `training_loop_function` 或特殊 manual optimization hook。

#### Scenario: 模型 _target_ 直接指向按模型分组的独立类
- **WHEN** 维护者查看 rkmeans_train / rvq_train / rqvae_train 的 model 配置
- **THEN** `_target_` MUST 直接指向 `src.quantization.rkmeans.residual_kmeans.ResidualKMeans` / `src.quantization.rvq.residual_vector_quantization.ResidualVectorQuantization` / `src.quantization.rqvae.residual_quantization_vae.ResidualQuantizationVAE`，而非统一的 `ResidualQuantization`

#### Scenario: 无 train_layer_wise 配置开关
- **WHEN** 维护者查看量化模型的配置
- **THEN** 配置中 MUST 不存在 `train_layer_wise` 字段，训练策略由模型类本身固定

#### Scenario: 量化逻辑内联在模型中
- **WHEN** 维护者追踪某个量化模型的 centroids 更新逻辑
- **THEN** 该逻辑 MUST 直接在对应模型目录或模型类实现中可见，不需要跨文件跳转到 BaseClusteringModule / MiniBatchKMeans / VectorQuantization

#### Scenario: 无外部 training_loop_function
- **WHEN** 维护者查看 RKMeans / RVQ / RQVAE 的 model 配置和构造函数
- **THEN** 配置和构造函数中 MUST 不存在 `training_loop_function` 字段
- **AND** 初始化逻辑 MUST 不依赖 `scale_loss_by_world_size_for_initialization_training_loop`
