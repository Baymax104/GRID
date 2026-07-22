## MODIFIED Requirements

### Requirement: TIGER model configuration SHALL expose only active runtime dependencies
TIGER train/inference 模型配置 SHALL 只声明当前生成式推荐路径实际消费的运行时依赖，不得继续暴露旧 embedding retrieval 路径或通用 feature mapping 路径的空配置参数。

#### Scenario: 配置不包含旧 postprocessor 和 aggregator
- **WHEN** 维护者检查 `configs/model/tiger_train.yaml` 与 `configs/model/tiger_inference.yaml`
- **THEN** 配置 MUST NOT 包含 `postprocessor` 字段
- **THEN** 配置 MUST NOT 包含 `aggregator` 字段

#### Scenario: 配置不包含 feature_to_model_input_map
- **WHEN** 维护者检查 `configs/model/tiger_train.yaml` 与 `configs/model/tiger_inference.yaml`
- **THEN** 配置 MUST NOT 包含 `feature_to_model_input_map`
- **AND** TIGER model MUST read `input_ids` and `attention_mask` directly from `TigerModelInput`

#### Scenario: inference 构造不依赖训练专属对象
- **WHEN** Hydra instantiate `tiger_inference` 模型配置
- **THEN** 模型构造 MUST NOT 因缺少训练专属 loss 或 evaluator 对象而失败
