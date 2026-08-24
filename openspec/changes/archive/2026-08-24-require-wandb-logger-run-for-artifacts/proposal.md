## Why

当前 W&B writer 仍保留无 active run 时自行 `wandb.init`、按 `fail_on_error` 跳过发布、并自行结束 run 的逻辑；这会让 artifact、lineage、config 和 metrics 分散到不同 run，削弱实验链路的一致性。

现在 inference、training 和 analysis 都已具备 `WandbLogger` 配置，W&B artifact 发布与 lineage 记录应统一依赖 logger-owned run，由 launcher/logger 统一管理 run 生命周期。

## What Changes

- **BREAKING**: `WandbArtifactWriter` 不再创建、结束或配置 W&B run；缺少 logger-owned W&B run 时必须失败。
- **BREAKING**: `WandbCheckpointWriter` 不再创建、结束或配置 W&B run；缺少 logger-owned W&B run 时必须失败。
- **BREAKING**: 删除两个 W&B writer 的 `fail_on_error` 配置和 warning-and-skip 分支；W&B artifact 发布失败即阶段失败。
- 删除 writer 配置中的 run lifecycle 字段，包括 `project`、`entity`、`group`、`run_name`、`job_type`、`tags`、`notes`、`mode`、`finish_run`。
- 保留 `WandbLogger` 作为 W&B run 元信息唯一配置入口：`name`、`save_dir`、`project`、`group`、`job_type`、`notes`。
- 保持 `WandbArtifactLineageCallback` 只记录上游 artifact lineage；它不得创建或结束 W&B run。

## Capabilities

### New Capabilities

- `wandb-artifact-lineage`: 约束 W&B logger-owned run、artifact writer 与 lineage callback 的职责边界。

### Modified Capabilities

- `prediction-output-protocol`: 推理 W&B artifact writer 必须依赖 `WandbLogger` run，且发布失败不得降级为本地-only输出。
- `model-training-components`: checkpoint artifact writer 必须依赖 `WandbLogger` run，且不得自行创建 artifact-only run。

## Impact

- 影响 `src/common/writers/wandb_artifact_writer.py`、`src/common/writers/wandb_checkpoint_writer.py` 和 `src/common/callbacks/wandb_artifact_lineage.py`。
- 影响 `configs/callbacks/*.yaml` 中 W&B writer 配置，删除 writer-level run/failure 配置。
- 影响 writer、lineage 和 Hydra compose 测试；旧的 self-created run 与 `fail_on_error=False` 测试需要删除或反向改写。
- 不新增依赖，不改变 `LocalPickleWriter` 的本地输出协议，不改变 `wandb://` artifact 读取协议。
