## MODIFIED Requirements

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
