## Why

`LocalPickleWriter` 当前继承 Lightning 的 `BasePredictionWriter`，即使业务逻辑不使用 `batch_indices`，Lightning 也会在 prediction loop 中尝试推断 dataloader batch indices。对于当前基于 `IterableDataset` 和 `DataloaderWithIterationRetry` 的推理数据链路，这会产生无业务价值的 warning。

## What Changes

- 将推理结果 writer 明确收窄为 batch-only callback。
- 移除 writer 的 epoch 写入分支和 `write_interval` 配置语义。
- 保留现有 batch buffering、flush、rank 0 merge、post-processing 行为。
- 保持 `ModelOutput(keys=..., predictions=...)` 和 `merged_predictions_tensor.pt` bundle 输出协议不变。

## Capabilities

### New Capabilities

无。

### Modified Capabilities

- `prediction-output-protocol`: writer 的推理写入协议从 Lightning `BasePredictionWriter` interval 语义收窄为项目自有 batch-only callback 语义。

## Impact

- Affected code: `src/inference/prediction_writers.py`
- Affected configs: inference callback configs that currently set `write_interval`
- Affected tests: prediction writer tests or config-level smoke tests
- Dependencies: no new dependencies
