## Why

当前 `WandbArtifactWriter` 只发布 `source_path` 指向的已有文件，inference 配置因此隐式依赖 `LocalPickleWriter` 先产出 `${paths.output_dir}/pickle/merged_predictions_tensor.pt`。当只启用 W&B writer 时，W&B artifact 链路无法独立产出和上传推理结果。

## What Changes

- **BREAKING**: 移除 `WandbArtifactWriter.source_path` 已有文件发布模式。
- 将 `WandbArtifactWriter` 改为独立 prediction writer：直接消费 `ModelOutput`，自行缓存、flush shard、merge keyed prediction bundle、执行 post-processing，并发布到 W&B Artifact。
- 为 W&B writer 增加独立 `output_dir`、`flush_frequency`、`post_processing_functions` 等写入配置，使其可单独启用。
- 调整 inference callback 配置，使 W&B writer 不再指向 `LocalPickleWriter` 的 `${paths.output_dir}/pickle/merged_predictions_tensor.pt`。
- 保持 writer 与 logger 解耦：W&B writer 自行复用 active W&B run 或创建/结束自身 run，不要求启用 W&B logger。
- 保持 launcher 主链路无 W&B 专用装配逻辑。

## Capabilities

### New Capabilities

### Modified Capabilities
- `prediction-output-protocol`: W&B Artifact writer 也必须作为 batch-only prediction writer 直接消费 `ModelOutput`，并独立合并 keyed prediction bundle。
- `keyed-prediction-bundle-artifact`: W&B Artifact writer 发布的推理产物必须遵守 keyed prediction bundle 协议，且不依赖 local pickle writer 的输出目录。

## Impact

- Affected code:
  - `src/common/writers/wandb_artifact_writer.py`
  - `configs/callbacks/*_inference.yaml`
  - `tests/common/writers/test_artifact_writers.py`
- No new dependency is expected.
- Existing configs using `source_path` for `WandbArtifactWriter` must be migrated to `output_dir`.
