# self-contained-tiger-generation-model Specification

## Purpose
TBD - created by archiving change consolidate-tiger-generation-model. Update Purpose after archive.
## Requirements
### Requirement: TIGER generation model SHALL be self-contained
TIGER 生成推荐模型 SHALL 由一个自包含的 LightningModule 承载训练、验证、测试、预测和 generation 运行时行为，不得依赖 `TransformerBaseModule` 或 `SemanticIDGenerativeRecommender` 作为运行时父类。

#### Scenario: 模型类直接承载 Lightning contract
- **WHEN** 维护者检查 TIGER Hydra `_target_` 指向的模型类
- **THEN** 该类 MUST 直接或等价地暴露 LightningModule 所需的 `configure_optimizers`、`training_step`、`validation_step`、`test_step` 行为
- **THEN** 该类 MUST NOT 通过继承 `TransformerBaseModule` 或 `SemanticIDGenerativeRecommender` 获得这些运行时行为

#### Scenario: generation 逻辑位于 TIGER 模型主体
- **WHEN** TIGER 执行验证、测试或预测 generation
- **THEN** semantic ID prefix 校验、beam search 和 evaluator 调用 MUST 由 TIGER 模型主体直接协调

### Requirement: TIGER model configuration SHALL expose only active runtime dependencies
TIGER train/inference 模型配置 SHALL 只声明当前生成式推荐路径实际消费的运行时依赖，不得继续暴露旧 embedding retrieval 路径、通用 feature mapping 路径的空配置参数，或自定义 T5 FFN 覆盖开关。官方 TIGER 模型 SHALL 使用 HuggingFace T5 原生 feed-forward block，不得在模型构造阶段用项目自定义模块替换 `T5LayerFF`。

#### Scenario: 配置不包含旧 postprocessor 和 aggregator
- **WHEN** 维护者检查 `configs/model/tiger_train.yaml` 与 `configs/model/tiger_inference.yaml`
- **THEN** 配置 MUST NOT 包含 `postprocessor` 字段
- **THEN** 配置 MUST NOT 包含 `aggregator` 字段

#### Scenario: 配置不包含通用 feature mapping 字段
- **WHEN** 维护者检查 `configs/model/tiger_train.yaml` 与 `configs/model/tiger_inference.yaml`
- **THEN** 配置 MUST NOT 包含旧的通用 feature-to-model-input mapping 字段
- **AND** TIGER model MUST read `input_ids` and `attention_mask` directly from `TigerModelInput`

#### Scenario: inference 构造不依赖训练专属对象
- **WHEN** Hydra instantiate `tiger_inference` 模型配置
- **THEN** 模型构造 MUST NOT 因缺少训练专属 loss 或 evaluator 对象而失败

#### Scenario: 配置不包含自定义 T5 FFN 覆盖
- **WHEN** 维护者检查 `configs/model/tiger_train.yaml` 与 `configs/model/tiger_inference.yaml`
- **THEN** 配置 MUST NOT 包含 `mlp_layers` 字段
- **AND** `Tiger` MUST NOT expose `mlp_layers` as a constructor parameter
- **AND** TIGER model construction MUST NOT replace HuggingFace `T5LayerFF` modules with a project custom FFN module

### Requirement: TIGER model SHALL use semantic ID tensors for prefix validation
TIGER 模型侧 SHALL 接收 semantic ID tensor 作为 prefix 校验数据源，不得将完整 keyed `ModelOutput` bundle 作为 `codebooks` 传入模型。

#### Scenario: 模型侧 semantic IDs shape 明确
- **WHEN** TIGER 模型从 semantic ID artifact 构造 prefix 校验数据
- **THEN** 模型侧输入 tensor MUST 具有 `(num_items, num_hierarchies)` 形状
- **THEN** 每个 semantic ID 值 MUST 表示对应 hierarchy 上的 code id

#### Scenario: 模型构造参数使用 semantic_ids 命名
- **WHEN** 维护者检查 TIGER 模型构造参数和模型配置
- **THEN** 模型侧 semantic ID tensor 参数 MUST 使用 `semantic_ids` 或等价明确名称
- **THEN** 模型配置 MUST NOT 将完整 keyed bundle 命名为 `codebooks`
