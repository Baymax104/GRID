## Why

当前 TIGER 模型实现分散在 `TransformerBaseModule`、`SemanticIDGenerativeRecommender`、`SemanticIDEncoderDecoder` 三层继承中，但实际只有 TIGER 使用这条继承链。通用 transformer 基类中的 `postprocessor`、`aggregator`、embedding retrieval 风格 `eval_step` 等职责已不符合当前 TIGER 生成式推荐路径，导致模型构造参数、配置和 semantic ID artifact 语义不清晰。

## What Changes

- 将 TIGER 生成推荐模型收敛为自包含的 LightningModule：保留现有 Hydra `_target_` 类名入口时，令 `SemanticIDEncoderDecoder` 直接承载训练、验证、测试、推理、优化器配置和 generation 逻辑。
- 移除 TIGER 对 `TransformerBaseModule` 与 `SemanticIDGenerativeRecommender` 的运行时继承依赖，并删除或停用无引用的旧基类文件。
- 从 TIGER train/inference 模型配置中删除旧基类残留参数 `postprocessor: null` 与 `aggregator: null`。
- 明确区分数据侧与模型侧的 semantic ID artifact 语义：数据侧继续使用完整 keyed bundle 执行 `item_id -> semantic_id` lookup；模型侧只接收 semantic ID tensor 用于 prefix 校验，不再把 `ModelOutput` bundle 直接命名为 `codebooks` 传入模型。
- **BREAKING**：不再支持从外部导入或继承 `TransformerBaseModule` / `SemanticIDGenerativeRecommender` 作为 TIGER 运行时扩展点；TIGER 模型构造参数将移除 `postprocessor`、`aggregator`，并将模型侧 `codebooks` 语义重命名为 semantic IDs。

## Capabilities

### New Capabilities
- `self-contained-tiger-generation-model`: TIGER 生成推荐模型的自包含 LightningModule 契约、训练/推理构造参数和 semantic ID artifact 语义。

### Modified Capabilities
- `transformer-base-training-contract`: 通用 transformer 训练基类不再作为当前 recommendation/TIGER 主链的运行时依赖。
- `keyed-prediction-bundle-artifact`: 明确 keyed semantic ID bundle 可以派生出模型侧 semantic ID tensor，而模型构造不得直接接收完整 bundle 作为 codebooks。

## Impact

- Affected code:
  - `src/recommendation/tiger_generation_model.py`
  - `src/recommendation/base_recommender.py`
  - `src/common/modules/transformer_base_module.py`
  - `src/utils/tensor_utils.py`
- Affected configs:
  - `configs/model/tiger_train.yaml`
  - `configs/model/tiger_inference.yaml`
- Verification focus:
  - Hydra compose/instantiate for `tiger_train` and `tiger_inference`
  - minimal train/eval/predict method smoke checks for the self-contained TIGER class
  - semantic ID tensor loader shape and prefix-check compatibility
