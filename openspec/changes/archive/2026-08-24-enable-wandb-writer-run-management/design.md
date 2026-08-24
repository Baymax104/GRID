## Context

> Superseded by `require-wandb-logger-run-for-artifacts`: W&B writers now depend on the `WandbLogger`-owned run and MUST NOT create artifact-only runs.

W&B Artifact 发布已经拆到 `src/common/writers/`，并且 inference callback 配置已包含 `WandbArtifactWriter`。但 inference 实验通常不配置 W&B logger，因此没有 active `wandb.run`。当前 writer 只调用 active run 的 `log_artifact`，导致单独运行 inference 时只能跳过发布。

用户要求保持 writer 与 logger 低耦合，并且不要用共享 helper 将 `WandbArtifactWriter` 和 `WandbCheckpointWriter` 重新耦合。两个 writer 应各自负责发布所需的 W&B run lifecycle：复用已有 run，必要时自行创建，并只关闭自己创建的 run。

## Goals / Non-Goals

**Goals:**
- `WandbArtifactWriter` 在无 W&B logger 的 inference 中也能发布 Artifact。
- `WandbCheckpointWriter` 在无 active run 时也能独立发布 checkpoint Artifact。
- 两个 writer 不互相 import，不共享 publish helper。
- 已有 W&B logger 创建的 run 必须被复用，writer 不重复 `wandb.init`，也不关闭 logger-owned run。
- callback 配置补齐 writer 创建 run 需要的 W&B run 参数。

**Non-Goals:**
- 不让 `WandbArtifactLineageCallback` 创建或管理 W&B run。
- 不改变 `LocalPickleWriter`、`ModelCheckpoint` 或 W&B logger 的职责。
- 不新增全局 run manager callback。
- 不引入新的外部依赖。

## Decisions

### Decision 1: Run lifecycle belongs to each W&B writer

两个 writer 的发布动作都需要 active W&B run，因此由 writer 自身在发布前检查 `wandb.run`。如果存在 active run，则复用；如果不存在，则调用 `wandb.init(...)` 创建 artifact-only run。

备选方案是新增 `WandbRunCallback`。该方案需要每个 artifact writer 配套一个 run callback，容易漏配，也会把 writer 的可用性转移到 callback 顺序上。writer 自主管理 run 更直接。

### Decision 2: 两个 writer 不共享 publish helper

`WandbArtifactWriter` 和 `WandbCheckpointWriter` 分别在自己的模块中实现 `_get_or_create_run`、`_finish_created_run` 和 artifact 发布逻辑。两者可以保持一致参数名，但不通过共享 helper 或互相 import 形成隐式耦合。

备选方案是把 run 管理和发布放在 `src/utils/wandb.py`。该方案减少重复代码，但会让两个 writer 重新依赖同一套 writer-oriented helper，不符合本次要求。

### Decision 3: 只 finish writer 自己创建的 run

writer 应跟踪本次发布是否调用了 `wandb.init`。只有当 writer 自己创建 run 且 `finish_run: true` 时，才调用 `run.finish()`；复用 W&B logger 创建的 run 时不 finish。

### Decision 4: 配置提供 artifact-only run 元数据

训练和推理 callback 配置为 writer 提供 W&B run 所需的 `project`、`group`、`run_name`、`job_type` 和 `finish_run`。训练通常会复用 W&B logger run；推理无 logger 时会创建 artifact-only run。`job_type` 表示生命周期模式，训练为 `train`，推理为 `inference`。

## Risks / Trade-offs

- **Risk:** 两个 writer 中存在少量重复 run 管理代码。→ **Mitigation:** 重复范围限制在各自模块内，测试覆盖两种 writer 的 active-run reuse 与 self-created run finish。
- **Risk:** inference writer 创建的 artifact-only run 不记录完整 metrics。→ **Mitigation:** 这是无 logger inference 的预期行为；metrics/config/notes 仍由 W&B logger 负责。
- **Risk:** W&B 凭据或网络不可用时发布失败。→ **Mitigation:** callback 配置保留 `fail_on_error: false`，不会破坏本地输出。

## Migration Plan

1. 为两个 writer 添加独立 run 配置参数和 run lifecycle 实现。
2. 更新 callback YAML，补齐 writer run 参数。
3. 补充 writer 单元测试：无 active run 时 init、已有 active run 时复用、只 finish 自己创建的 run、两个 writer 不互相 import。
4. 运行 focused tests、Hydra compose、全量 pytest、ruff 和 OpenSpec validate。

## Open Questions

无阻塞未决问题。
