## Context

当前 TIGER 生成推荐模型由三层继承组成：

- `TransformerBaseModule`：早期通用 transformer Lightning 训练壳，包含 optimizer、metric reset/log、training/validation/test step、embedding retrieval 风格 eval 逻辑。
- `SemanticIDGenerativeRecommender`：TIGER 专属推荐基类，包含 semantic ID prefix 校验、beam search generation、TIGER eval 逻辑。
- `SemanticIDEncoderDecoder`：当前 Hydra 配置实际实例化的 TIGER encoder-decoder 模型。

代码调查显示这条继承链只有 TIGER 使用；`TransformerBaseModule` 中的 `postprocessor`、`aggregator` 和 embedding retrieval eval 不再服务当前生成式推荐路径。与此同时，TIGER 模型配置将 `semantic_id_path` 通过 `load_model_output` 注入到模型侧 `codebooks`，但 `load_model_output` 返回 keyed `ModelOutput` bundle，而模型实现实际需要 semantic ID tensor 做 prefix 校验。

## Goals / Non-Goals

**Goals:**

- 让 TIGER 模型成为自包含 LightningModule，模型主体清楚表达训练、验证、测试、预测和 generation 行为。
- 移除 TIGER 对旧通用基类与推荐基类的运行时继承依赖。
- 删除 TIGER 配置中无行为意义的 `postprocessor` / `aggregator` 参数。
- 明确 semantic ID artifact 在数据侧和模型侧的不同消费方式。
- 保持 TIGER 主运行入口与现有实验配置路径尽量稳定，优先保留当前 Hydra `_target_` 类入口。

**Non-Goals:**

- 不改变 TIGER 的生成算法、loss、metric 语义或 beam search 结果。
- 不重写 encoder/decoder 子模块内部实现。
- 不改变数据侧 `map_sparse_id_to_semantic_id` 使用完整 keyed bundle 的协议。
- 不为旧基类提供长期兼容 shim；本变更允许删除无引用基类。

## Decisions

### 保留 `SemanticIDEncoderDecoder` 作为 Hydra `_target_`

`SemanticIDEncoderDecoder` SHALL 直接继承 `lightning.LightningModule`，并内联原先分散在两个父类中的 TIGER 运行时逻辑。这样可以减少配置迁移面，同时避免引入新类名导致所有 checkpoint/config 文档一次性重命名。

Alternatives considered:

- 新建 `TIGERGenerationModel` 并迁移 `_target_`：命名更准确，但配置与 checkpoint 迁移面更大。
- 只把 `TransformerBaseModule` 重命名为推荐基类：继承层次仍然存在，无法解决 semantic ID 语义混乱。

### 将有用 Lightning 训练壳逻辑内联到 TIGER 模型

自包含模型 SHALL 显式持有 `optimizer`、`scheduler`、`loss_function`、`evaluator` 等依赖，并提供 `configure_optimizers`、`training_step`、`validation_step`、`test_step`、metric reset/log hooks。`model_step`、`eval_step`、`generate` 等 TIGER 专属逻辑 SHALL 与模型主体放在同一运行时类中。

### 删除 `postprocessor` 与 `aggregator` 构造参数

这两个参数只服务旧 embedding retrieval eval 路径，当前 TIGER 配置中均为 `null` 且没有行为贡献。删除参数可以让模型构造契约与实际行为一致。

### 模型侧使用 semantic ID tensor，而非完整 keyed bundle

数据侧仍 SHALL 使用 `ModelOutput(keys, predictions)` bundle 进行 `item_id -> semantic_id` 查询。模型侧 SHALL 通过独立 loader 从同一 artifact 中提取 `predictions` tensor，并以 `semantic_ids` 命名注入模型。该 tensor 的形状 SHALL 为 `(num_items, num_hierarchies)` 或可按模型所需 hierarchy 截断到该形状。

## Risks / Trade-offs

- **Checkpoint 兼容风险** → 保留当前 `_target_` 类名与核心属性名；必要时在实现中避免无意义重命名 state_dict key。
- **训练/推理配置差异风险** → 让 train-only 依赖保持 optional 或只在对应 step 使用时要求非空，确保 inference instantiate 不被 loss/evaluator 依赖阻断。
- **semantic ID shape 方向风险** → 增加 loader 和 smoke check，显式验证模型侧 `semantic_ids.shape == (num_items, num_hierarchies)`，并覆盖 prefix 校验路径。
- **误删仍有引用的基类风险** → 删除前使用静态引用检查与 Hydra instantiate smoke 验证，确保 `TransformerBaseModule` / `SemanticIDGenerativeRecommender` 没有运行时依赖。

## Migration Plan

1. 在 `SemanticIDEncoderDecoder` 内联 TIGER 所需 Lightning hooks、optimizer 配置、metric reset/log、train/eval/test wrapper 和 generation 逻辑。
2. 将 `SemanticIDGenerativeRecommender` 的 semantic ID、prefix 校验与 beam search 行为迁移到模型主体。
3. 新增模型侧 semantic ID tensor loader，并将 TIGER 模型配置从 `codebooks: load_model_output(...)` 迁移为 `semantic_ids: <tensor loader>`。
4. 删除 TIGER 配置中的 `postprocessor` 与 `aggregator`。
5. 删除或停用无引用的 `TransformerBaseModule`、`SemanticIDGenerativeRecommender`。
6. 运行 Hydra compose/instantiate、最小 train/eval/predict smoke、静态引用检查和 OpenSpec validation。

Rollback strategy: 如果自包含模型迁移失败，可恢复旧三层继承文件与配置参数；因为本变更不改变数据产物格式，semantic ID artifact 本身不需要回滚。

## Open Questions

无。当前改造范围和兼容策略已明确。
