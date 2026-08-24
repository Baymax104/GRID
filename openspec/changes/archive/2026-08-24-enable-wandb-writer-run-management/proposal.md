## Why

> Superseded by `require-wandb-logger-run-for-artifacts`: W&B writers now depend on the `WandbLogger`-owned run and MUST NOT create artifact-only runs.

当前 inference 实验默认不使用 W&B logger，但 `WandbArtifactWriter` 只在已有 `wandb.run` 时才能发布 Artifact。单独运行 inference 时会出现 “No active W&B run exists” 并跳过发布，违背了 W&B writer 不应依赖 W&B logger 的模块边界。

## What Changes

- 让 `WandbArtifactWriter` 自主管理发布所需的 W&B run：已有 active run 时复用，没有 active run 时按自身配置创建 artifact-only run。
- 让 `WandbCheckpointWriter` 在自己的模块中实现同样的 run 管理能力，不通过共享 helper 耦合两个 writer。
- 为两个 writer 增加一致的 run 配置项：`project`、`entity`、`group`、`run_name`、`job_type`、`tags`、`notes`、`mode`、`finish_run`。
- 更新 callback 配置，使训练 checkpoint writer 与推理 artifact writer 都具备独立创建 run 的必要配置。
- 保持 `WandbArtifactLineageCallback` 职责不变：只记录 lineage，不创建或管理 W&B run。

## Capabilities

### New Capabilities

- `wandb-artifact-lineage`: 约束 W&B writer 与 lineage callback 的 run lifecycle 边界。

### Modified Capabilities

- `prediction-output-protocol`: 推理 W&B Artifact writer 不应要求 W&B logger 创建 run，必须能在无 active run 时自行创建 artifact-only run。
- `model-training-components`: checkpoint Artifact writer 不应依赖 W&B logger 或另一个 writer，必须能在无 active run 时自行创建 run 并发布 checkpoint Artifact。

## Impact

- 影响 `src/common/writers/wandb_artifact_writer.py` 和 `src/common/writers/wandb_checkpoint_writer.py`。
- 影响 `configs/callbacks/*.yaml` 中 W&B writer 配置。
- 影响 writer 单元测试与 Hydra compose 验证。
- 不新增依赖，不改变本地 `LocalPickleWriter` 行为，不改变 W&B logger 的 metrics/config/notes 职责。
