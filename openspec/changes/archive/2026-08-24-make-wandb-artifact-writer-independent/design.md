## Context

> Superseded by `require-wandb-logger-run-for-artifacts`: W&B writer independence still applies to batch writing and local writer coupling, but run lifecycle and `fail_on_error` fallback are removed.

`LocalPickleWriter` 当前承担 prediction writer 职责：在 `on_predict_batch_end` 接收 `ModelOutput`，按样本数 flush 临时 `.pkl` shard，并在 `on_predict_end` 由 rank 0 合并为 `merged_predictions_tensor.pt`。`WandbArtifactWriter` 当前只在 `on_predict_end` 发布 `source_path` 指向的已有文件，因此 inference 配置需要 local writer 先写出 `${paths.output_dir}/pickle/merged_predictions_tensor.pt`。

用户要求 W&B artifact writer 与 local pickle writer 独立工作，并明确不保留 W&B writer 的旧 `source_path` 发布模式。模块边界要求是不在 launcher 主链路加入 W&B 特判，writer 与 logger 保持低耦合。

## Goals / Non-Goals

**Goals:**
- 让 `WandbArtifactWriter` 成为独立 batch-only prediction writer。
- 让 `WandbArtifactWriter` 自己完成缓存、flush、merge、post-processing 和 W&B Artifact 发布。
- 允许只启用 W&B writer，不启用 local writer。
- 允许 W&B writer 与 local writer 同时启用且互不读写对方目录。
- 保持 W&B run 管理位于 writer 内部：复用 active run，或创建并按配置结束自身 run。

**Non-Goals:**
- 不改 launcher 的 callback 装配逻辑。
- 不引入 W&B logger 与 W&B writer 的配套关系。
- 不改变 `WandbCheckpointWriter` 的 checkpoint artifact 行为。
- 不保留 `WandbArtifactWriter.source_path` 发布已有文件模式。

## Decisions

1. `BaseBufferedWriter` 放在中性 `src.common.writers.base` 模块

   复用中性的 batch 缓存协议，使 W&B writer 与 local writer 在 Lightning callback 形态上保持一致。W&B writer 不继承 `LocalPickleWriter`，不从 `local_pickle_writer` 模块导入基础类，也不读取 local writer 的输出目录。

2. W&B writer 使用独立 `output_dir`

   每个 writer 都只操作自己的目录。默认合并文件名仍为 `merged_predictions_tensor.pt`，但 W&B writer 的文件位于自身 `output_dir` 下，例如 `${paths.output_dir}/wandb_artifact`。这避免两个 writer 同时启用时发生 `.pkl` shard 冲突、删除冲突或 post-processing 重复处理同一文件。

3. 移除 `source_path` 参数

   `WandbArtifactWriter` 的职责收敛为“将 inference `ModelOutput` 写成 W&B Artifact”。如果未来需要发布任意已有文件，应使用单独 publisher，而不是复用 prediction writer。

4. W&B writer 内部独立实现 merge

   由于用户明确要求两个 writer 可以独立工作，W&B writer 不通过 helper 耦合 `LocalPickleWriter`。实现可以保留与 local writer 同等的 bundle 协议：临时 `.pkl` shard 保存 `ModelOutput` 列表，rank 0 合并 `keys` 与 `predictions`，保存 `{"keys": ..., "predictions": ...}`。

5. post-processing 属于 writer 自身配置

   W&B writer 支持自己的 `post_processing_functions`。如果 semantic ID artifact 需要去重扩展列，W&B writer 在自身 bundle 上执行同样的 post-processing，不依赖 local writer 对 `${paths.output_dir}/pickle` 的处理结果。

## Risks / Trade-offs

- [Risk] W&B writer 与 local writer 同时启用会各自缓存一份预测输出，增加临时磁盘和内存压力 → 使用 flush_frequency 控制缓冲规模，并让两个 writer 写不同目录。
- [Risk] 两个 writer 的 merge 逻辑存在重复 → 当前优先满足模块独立和低耦合；后续如需收敛，可抽取不带 W&B/Local 语义的更底层 bundle assembler。
- [Risk] 移除 `source_path` 会破坏旧配置 → 本 change 同步迁移 inference callback 配置，并用测试覆盖不再引用 `source_path`。
- [Risk] W&B publish 失败可能导致 inference 产物不可见 → 保留 `fail_on_error` 语义；当为 false 时跳过发布但本地合并 bundle 保留在 W&B writer 的 `output_dir` 中用于排查。

## Migration Plan

1. 修改 `WandbArtifactWriter` 构造参数：删除 `source_path`，新增 `output_dir`、`flush_frequency`、`post_processing_functions`。
2. 在 W&B writer 内实现 shard flush、rank 0 merge、post-processing 和 publish。
3. 将 inference callback 配置中的 `wandb_artifact_writer.source_path` 替换为独立 `output_dir`。
4. 为 semantic ID inference 的 W&B writer 添加与 local writer 对齐的 post-processing 配置。
5. 更新测试，覆盖只启用 W&B writer 和两个 writer 共存。

## Open Questions

无阻塞实现的未澄清问题。
