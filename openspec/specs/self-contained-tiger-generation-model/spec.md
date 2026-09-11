# self-contained-tiger-generation-model Specification

## Purpose
TBD - created by archiving change consolidate-tiger-generation-model. Update Purpose after archive.
## Requirements
### Requirement: TIGER generation model SHALL be self-contained
TIGER 生成推荐模型 SHALL 由一个自包含的 LightningModule 承载训练、验证、测试、预测和 generation 运行时行为，不得依赖 `TransformerBaseModule` 或 `SemanticIDGenerativeRecommender` 作为运行时父类。TIGER 模型主体 SHALL 协调整体 generation 流程，decoder-specific teacher-forcing、autoregressive generation、beam search step 和 decoder-side prefix validation SHALL 由 `TigerDecoder` 承载。

#### Scenario: 模型类直接承载 Lightning contract
- **WHEN** 维护者检查 TIGER Hydra `_target_` 指向的模型类
- **THEN** 该类 MUST 直接或等价地暴露 LightningModule 所需的 `configure_optimizers`、`training_step`、`validation_step`、`test_step` 行为
- **THEN** 该类 MUST NOT 通过继承 `TransformerBaseModule` 或 `SemanticIDGenerativeRecommender` 获得这些运行时行为

#### Scenario: generation 逻辑由 TIGER 模型主体协调
- **WHEN** TIGER 执行验证、测试或预测 generation
- **THEN** TIGER 模型主体 MUST coordinate encoder execution, decoder generation invocation, and evaluator calls
- **AND** decoder-specific autoregressive generation and beam-search step behavior MUST be owned by `TigerDecoder`

#### Scenario: decoder owns decoder-side generation behavior
- **WHEN** 维护者检查 TIGER decoder implementation
- **THEN** `TigerDecoder` MUST expose a `forward` path for teacher-forcing decoder hidden states
- **AND** `TigerDecoder` MUST expose a `generate` path for autoregressive semantic ID generation
- **AND** `TigerDecoder` MUST own decoder-side prefix validation when `should_check_prefix` is enabled

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

### Requirement: TIGER decoder SHALL use a global SID lm head
`TigerDecoder` SHALL project decoder hidden states with one global SID `lm_head` whose output width is `num_hierarchies * codebook_size`. The global logits SHALL align with the same hierarchy-offset SID vocabulary used by the shared SID embedding table.

#### Scenario: Teacher-forcing forward returns global logits
- **WHEN** TIGER executes teacher-forcing decoder forward for a labeled batch
- **THEN** `TigerDecoder.forward()` MUST return raw logits with shape `(batch_size, sequence_length, num_hierarchies * codebook_size)`
- **AND** it MUST NOT apply softmax to those logits
- **AND** it MUST NOT use one projection module per hierarchy

#### Scenario: Loss uses global targets
- **WHEN** TIGER computes training or evaluation loss
- **THEN** local target SID values MUST be converted to hierarchy-offset global SID target values
- **AND** loss MUST be computed against global logits

#### Scenario: Generation slices active hierarchy logits
- **WHEN** `TigerDecoder.generate()` computes candidate scores for hierarchy `h`
- **THEN** it MUST slice global logits to `[h * codebook_size, (h + 1) * codebook_size)`
- **AND** beam search MUST continue to emit local SID values in `[0, codebook_size)`

#### Scenario: Weight tying is not introduced
- **WHEN** maintainers inspect `TigerDecoder`
- **THEN** `lm_head` MUST NOT be tied to `sid_embedding_table.weight` in this change

### Requirement: TIGER decoder SHALL own logits projection
`TigerDecoder` SHALL own projection from decoder hidden states to semantic-ID logits. Teacher-forcing decoder forward SHALL return raw logits rather than hidden states or softmax probabilities, and TIGER loss computation SHALL consume those logits without reaching into decoder projection internals.

#### Scenario: Teacher-forcing forward returns raw logits
- **WHEN** TIGER executes teacher-forcing decoder forward for a labeled batch
- **THEN** `TigerDecoder.forward()` MUST return raw logits with shape `(batch_size, sequence_length, codebook_size)`
- **AND** it MUST NOT apply softmax to those logits

#### Scenario: Loss consumes logits only
- **WHEN** TIGER computes training or evaluation loss
- **THEN** the loss helper MUST consume logits and target semantic IDs
- **AND** it MUST NOT call `TigerDecoder.decoder_mlp` or otherwise access decoder projection internals

#### Scenario: Generation keeps decoder-side projection
- **WHEN** `TigerDecoder.generate()` computes candidate scores for beam search
- **THEN** candidate logits MUST be produced by decoder-owned projection layers
- **AND** probability conversion MUST remain local to beam-search scoring logic

### Requirement: TIGER SHALL rely on Lightning lifecycle for train and eval mode
TIGER SHALL NOT define custom lifecycle hooks whose only responsibility is manually switching wrapped encoder or decoder modules between train and eval mode. Standard training, validation, test, and prediction loops SHALL rely on Lightning lifecycle behavior for module mode management.

#### Scenario: Manual mode hooks are absent
- **WHEN** maintainers inspect the TIGER LightningModule
- **THEN** it MUST NOT define helper methods that manually set wrapped encoder or decoder train/eval mode for Lightning loops
- **AND** it MUST NOT set custom `is_training` attributes on wrapped encoder or decoder modules

#### Scenario: Metric lifecycle hooks remain
- **WHEN** TIGER starts train, validation epoch, or test epoch lifecycle events
- **THEN** metric reset behavior MUST remain available
- **AND** those hooks MUST NOT perform manual train/eval mode switching

#### Scenario: Step methods remain mode-neutral
- **WHEN** maintainers inspect TIGER `training_step`, validation, test, and prediction step methods
- **THEN** those methods MUST NOT call `train()` or `eval()` directly

### Requirement: TIGER decoder forward SHALL be teacher-forcing only
`TigerDecoder.forward()` SHALL be a teacher-forcing decoder computation path. It MUST require future semantic IDs, MUST construct decoder inputs from BOS plus future semantic ID embeddings, and MUST NOT use `future_ids=None` as a generation bootstrap signal.

#### Scenario: Decoder forward requires future semantic IDs
- **WHEN** maintainers inspect `TigerDecoder.forward()`
- **THEN** `future_ids` MUST be treated as a required tensor input
- **AND** the method MUST NOT contain a BOS-only branch for missing future IDs

#### Scenario: Decoder generation owns BOS bootstrap
- **WHEN** `TigerDecoder.generate()` performs the first autoregressive hierarchy step
- **THEN** it MUST construct the BOS-only decoder input inside `generate()`
- **AND** it MUST NOT call `TigerDecoder.forward()` for that BOS-only generation step

#### Scenario: Decoder generation remains inline
- **WHEN** maintainers inspect `TigerDecoder.generate()`
- **THEN** generation decoder input assembly MUST remain in the method body
- **AND** the change MUST NOT introduce separate small helper methods solely for decoder input assembly or wrapped decoder execution

### Requirement: TIGER Lightning steps SHALL use explicit train, evaluation, and prediction paths
TIGER Lightning step methods SHALL call explicit computation paths for their runtime role instead of relying on a mode-dependent `model_step` dispatcher. `forward()` SHALL remain a pure teacher-forcing computation path and MUST NOT perform loss aggregation, metric updates, logging, or autoregressive generation.

#### Scenario: Training step computes only teacher-forcing loss
- **WHEN** TIGER executes `training_step`
- **THEN** it MUST compute teacher-forcing decoder outputs and loss from `TigerModelInput` plus `TigerLabelData`
- **AND** it MUST NOT call autoregressive generation for that batch

#### Scenario: Evaluation step computes loss and generation explicitly
- **WHEN** TIGER executes validation or test evaluation for a labeled batch
- **THEN** it MUST compute teacher-forcing loss through the explicit loss path
- **AND** it MUST call autoregressive generation exactly once for evaluator metrics

#### Scenario: Prediction step produces generated outputs only
- **WHEN** TIGER executes `predict_step`
- **THEN** it MUST call autoregressive generation for the batch
- **AND** it MUST wrap generated semantic IDs in `ModelOutput`
- **AND** it MUST NOT return placeholder loss values

#### Scenario: Step methods defer mode switching to Lightning lifecycle
- **WHEN** maintainers inspect `training_step`, validation, test, and prediction step methods
- **THEN** those methods MUST NOT call `train()` or `eval()` to switch module mode
- **AND** mode-sensitive behavior MUST remain controlled by Lightning loops and model lifecycle hooks

### Requirement: TIGER generation SHALL NOT depend on decoder KV cache
TIGER generation SHALL compute decoder outputs without maintaining or reordering HuggingFace decoder KV cache state.

#### Scenario: Generation recomputes from current prefix
- **WHEN** TIGER generates semantic IDs for validation, testing, or prediction
- **THEN** decoder invocation MUST receive the current generated semantic ID prefix embeddings or BOS embedding as input
- **AND** decoder invocation MUST NOT receive `past_key_values`
- **AND** decoder invocation MUST NOT request cache output

#### Scenario: Beam search does not mutate cache state
- **WHEN** TIGER beam search selects top-k candidate prefixes
- **THEN** it MUST reorder generated semantic ID prefixes and marginal probabilities only
- **AND** it MUST NOT reorder decoder cache state

### Requirement: TIGER decoder SHALL own optional generation tracing
`TigerDecoder` SHALL own derivation of beam parent, target-prefix survival, rank, score, and cutoff observations from its constrained beam-search state. TIGER model orchestration MAY request those observations but MUST NOT reimplement beam state reconstruction outside the decoder.

#### Scenario: Decoder tracing is requested
- **WHEN** TIGER invokes decoder generation with labeled target IDs and tracing enabled
- **THEN** `TigerDecoder` MUST derive trace observations from the same tensors used for constrained beam selection
- **AND** TIGER MUST associate the returned trace with the batch output keys

#### Scenario: Decoder tracing is not requested
- **WHEN** ordinary validation, testing, or prediction runs without tracing
- **THEN** the decoder MUST preserve its existing generation behavior and output compatibility

### Requirement: Teacher-forcing trace statistics SHALL consume raw decoder logits
TIGER SHALL derive target token diagnostic statistics from the raw global logits returned by the existing teacher-forcing path. The diagnostic path MUST NOT modify loss inputs or introduce an additional trainable projection.

#### Scenario: Teacher-forcing trace is computed
- **WHEN** a labeled trace batch is evaluated
- **THEN** the model MUST use the active hierarchy slice of the existing global logits
- **AND** training/evaluation loss MUST continue to consume the original logits unchanged

### Requirement: TIGER decoder SHALL own optional candidate allocation
`TigerDecoder` SHALL own candidate shortlist construction, prefix-priority lookup, reserved-slot selection, original-score backfill, and final beam ordering for prefix-balanced generation. TIGER model orchestration MAY provide configuration and aligned frequency inputs but MUST NOT reimplement decoder selection.

#### Scenario: Candidate allocation is enabled
- **WHEN** TIGER invokes generation with a valid prefix allocation configuration
- **THEN** `TigerDecoder` MUST apply allocation after legal-prefix filtering and cumulative path scoring
- **AND** TIGER MUST continue to coordinate encoder execution and decoder invocation through its existing generation path

#### Scenario: Candidate allocation is disabled
- **WHEN** ordinary training, validation, testing, or prediction does not enable the probe
- **THEN** decoder-owned beam selection MUST preserve the existing generation contract
