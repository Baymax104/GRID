## ADDED Requirements

### Requirement: Prediction artifact writing SHALL be separate from local bundle writing
推理 Artifact 发布 SHALL 由与 `LocalPickleWriter` 平级的 W&B writer 承担，二者 SHOULD 位于 `src/common/writers/`，MUST 分别放在独立 writer 文件中，MUST NOT 保留旧 `src/common/inference/` 兼容导入，MUST NOT 改变 `LocalPickleWriter` 的本地 batch 缓存、flush、merge、post-processing 职责。

#### Scenario: Local writer works without W&B writer
- **WHEN** 推理实验使用 `LocalPickleWriter`
- **AND** 未启用 W&B Artifact 发布
- **THEN** writer MUST 继续写入 `${paths.output_dir}/pickle/merged_predictions_tensor.pt`
- **THEN** 推理流程 MUST NOT 要求存在 W&B logger 或 W&B Artifact writer

#### Scenario: WandbArtifactWriter consumes completed local output
- **WHEN** 推理实验启用 W&B Artifact 发布
- **THEN** `WandbArtifactWriter` MUST 在本地 writer 完成 merge 和 post-processing 后读取最终输出文件
- **THEN** `WandbArtifactWriter` MUST NOT 参与 `LocalPickleWriter` 的 batch flush 或 rank-local shard 合并

#### Scenario: W&B logger does not imply W&B writer
- **WHEN** 推理或训练实验配置了 W&B logger
- **AND** 未显式启用 W&B Artifact 发布
- **THEN** 系统 MUST NOT 自动发布推理 output Artifact
- **THEN** 系统 MUST NOT 要求使用 W&B-specific writer

#### Scenario: W&B writer is not paired to LocalPickleWriter by type
- **WHEN** `WandbArtifactWriter` 被配置为发布 `${paths.output_dir}/pickle/merged_predictions_tensor.pt`
- **THEN** 它 MUST 只依赖配置中的文件路径或 artifact source path
- **THEN** 它 MUST NOT 通过 Python 类型检查或实例引用强制要求同一实验存在 `LocalPickleWriter`
