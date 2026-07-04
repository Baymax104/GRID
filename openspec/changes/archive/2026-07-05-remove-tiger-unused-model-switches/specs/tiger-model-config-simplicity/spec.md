## ADDED Requirements

### Requirement: Official TIGER experiments SHALL not expose unused generic model switches
官方 TIGER experiment 不得继续暴露未启用或未形成真实选择语义的通用模型开关。

#### Scenario: Inspect official TIGER experiment config
- **WHEN** 开发者查看 `tiger_train` 或 `tiger_inference` 的模型配置
- **THEN** 配置中不得继续包含 `compile`
- **AND** 不得继续包含 `weight_tying`

### Requirement: Base transformer evaluation path SHALL use encoder input embeddings by default
基类 transformer 模块在评估路径中必须固定使用 encoder input embeddings 作为 retrieval evaluator 的 key embeddings 来源。

#### Scenario: Evaluate a model inheriting the base transformer module
- **WHEN** `TransformerBaseModule` 执行 validation 或 test 阶段的 evaluator 更新
- **THEN** `key_embeddings` 必须来源于 `self.encoder.get_input_embeddings().weight`
- **AND** 不得再依赖 `weight_tying` 配置分支决定来源
