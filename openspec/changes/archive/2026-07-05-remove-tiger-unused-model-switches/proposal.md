## Why

当前官方 TIGER experiment 在模型配置中仍暴露 `weight_tying` 与 `compile` 两个开关，但它们都没有为当前官方链路提供真实的可选价值：`compile` 只是一个未启用的预留参数，当前代码不会执行 `torch.compile(...)`；`weight_tying` 虽然在基类中保留了分支，但官方 TIGER 实现已经通过自己的 embedding table 路径覆盖了主要使用场景，配置层始终固定为 `true`，并未形成真实可切换接口。

现在需要删除这两个无效/无用的模型开关，收紧官方 TIGER experiment 的配置接口，并把评估路径固定为当前默认语义。

## What Changes

- 删除官方 TIGER experiment 中的 `weight_tying` 与 `compile` 配置项。
- 删除 `TransformerBaseModule` 中对应的构造参数与未使用逻辑。
- 将基类评估路径使用的 embedding table 来源固定为当前默认的 encoder input embeddings。
- 同步清理相关注释与说明，避免继续暗示支持未启用的可选开关。

## Capabilities

### Modified Capabilities
- `tiger-model-config-simplicity`: 官方 TIGER 模型配置不再暴露未启用或未被实际消费的通用开关。

## Impact

- 受影响代码：`src/common/modules/transformer_base_module.py`，以及可能提及这两个字段的推荐模型相关代码
- 受影响配置：`configs/experiment/tiger_train.yaml`、`configs/experiment/tiger_inference.yaml`
- **BREAKING**：官方 TIGER experiment 不再接受 `weight_tying` 与 `compile` 作为模型配置字段
