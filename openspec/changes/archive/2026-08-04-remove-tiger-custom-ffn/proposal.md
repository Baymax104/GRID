## Why

TIGER 当前通过 `mlp_layers` 动态替换 HuggingFace T5 的原生 `T5LayerFF`，把标准 T5 FFN 改成项目自定义多层 MLP。该覆盖没有独立规格或测试支撑，会扩大模型结构、耦合 transformers 内部类名，并让官方 TIGER 配置暴露一个不必要的结构变体。

## What Changes

- 删除 TIGER 官方 train/inference 模型配置中的 `mlp_layers` 字段。
- 删除 `SemanticIDEncoderDecoder` 中遍历 `named_modules()` 并替换 `T5LayerFF` 的逻辑。
- 删除 `T5MultiLayerFF` 自定义模块，让 TIGER 使用 HuggingFace T5 原生 FFN。
- **BREAKING**: 使用旧 `mlp_layers=2` 结构训练出的 TIGER checkpoint 不要求兼容；移除后需要使用原生 T5 FFN 结构重新训练或加载匹配结构的 checkpoint。

## Capabilities

### New Capabilities

None.

### Modified Capabilities

- `self-contained-tiger-generation-model`: 收紧 TIGER 模型配置契约，要求官方配置不暴露自定义 T5 FFN 覆盖开关，并使用 T5 原生 FFN。

## Impact

- Affected code: `src/recommendation/tiger_generation_model.py`, `src/recommendation/t5_multi_layer_ff.py`.
- Affected config: `configs/model/tiger_train.yaml`, `configs/model/tiger_inference.yaml`.
- Affected specs: `openspec/specs/self-contained-tiger-generation-model/spec.md`.
- No dependency changes.
