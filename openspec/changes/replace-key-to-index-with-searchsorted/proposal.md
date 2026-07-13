## Why

`load_model_output` 加载时通过 `_build_key_to_index` 构建 Python dict（`key_to_index`）作为运行时索引，用于 `gather_predictions_by_keys` 的 key→行号翻译。这个机制引入了不必要的复杂性：一个单独的辅助函数、一个注入到 bundle dict 中的运行时字段、以及 `gather_predictions_by_keys` 中的懒构建逻辑。既然 `keys` 是 tensor，完全可以用 `torch.searchsorted` 替代 dict 查找，消除这些额外状态和代码。

## What Changes

- `load_model_output` 加载后按 key 值排序 keys + predictions（`argsort` + 重排）
- `gather_predictions_by_keys` 改用 `torch.searchsorted` 二分查找行号，替代 dict lookup
- 删除 `_build_key_to_index` 函数
- 删除 `bundle["key_to_index"]` 运行时字段及懒构建逻辑

## Capabilities

### New Capabilities
（无）

### Modified Capabilities
- `keyed-prediction-bundle-artifact`: `load_model_output` 加载时按 key 排序；`gather_predictions_by_keys` 用 `torch.searchsorted` 替代 dict 查找；去除 `key_to_index` 运行时字段

## Impact

- `src/utils/tensor_utils.py` — 重写 `load_model_output` 和 `gather_predictions_by_keys`，删除 `_build_key_to_index`
- 无外部消费者变更（函数签名不变，行为等价）
