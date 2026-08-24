## Context

GRID 的 W&B 链路已经形成三类组件：`WandbLogger` 负责 run、metrics、resolved config 和 notes；W&B writer 负责发布 output Artifact；`WandbArtifactLineageCallback` 负责对 resolved upstream Artifact 调用 `use_artifact`。此前为了支持无 logger 的 inference，两个 W&B writer 额外实现了 artifact-only run 创建、run 元数据配置、条件 finish 和 `fail_on_error` 降级。

现在官方 train、inference 和 analysis 配置都应配置 `WandbLogger`。继续让 writer 自建 run 会让 config、metrics、lineage 和 output Artifact 分散到不同 W&B run 中。用户也明确要求 W&B writer 的本地 merge 与 artifact 发布不可拆成“本地成功、发布失败可跳过”的两个流程。

## Goals / Non-Goals

**Goals:**

- 让 W&B run 生命周期只由 `WandbLogger` 和 launcher logger finalization 管理。
- 让 `WandbArtifactWriter` 和 `WandbCheckpointWriter` 只发布到 logger-owned run，不再 `wandb.init` 或 `run.finish`。
- 删除 writer-level run 配置和 `fail_on_error` 降级逻辑。
- 让缺少 logger-owned run、Artifact 创建失败、上传失败都明确失败。
- 保持 data-side `wandb://` 解析与 writer 发布职责分离。

**Non-Goals:**

- 不改变 `LocalPickleWriter` 的本地输出协议。
- 不改变 `wandb://` URI 解析、Artifact 下载或 resolved-reference registry 的字段协议。
- 不新增全局 W&B run manager callback。
- 不让 launcher 扫描业务配置或按 W&B artifact 做分支装配。

## Decisions

### Decision 1: Logger-owned run 是唯一发布目标

W&B writer 在发布时应从 Lightning `trainer.loggers` 中找到 `WandbLogger` 并使用其 `experiment` run。找不到 W&B logger run 时直接失败。相比继续读取裸 `wandb.run`，从 `trainer.loggers` 获取 run 更能表达真实依赖：writer 依赖当前 experiment 配置的 logger，而不是进程里碰巧存在的全局 run。

备选方案是保留 `wandb.run` fallback。该方案实现更小，但仍允许外部隐式 run 接管 artifact 发布，不能保证 artifact 与当前 Hydra config/metrics/notes 在同一个 run。

### Decision 2: Writer 不再管理 run lifecycle

删除两个 writer 中的 `_get_or_create_run`、`_finish_created_run` 和所有 `wandb.init` 参数。writer 构造参数只保留 artifact 发布和本地 bundle 生成所需字段，例如 artifact name/type、role、aliases、metadata、selection、output_dir 和 post-processing。

备选方案是保留参数但废弃不用。该方案对旧配置表面兼容，但会让配置继续暗示 writer 能独立创建 run，不利于维护者理解职责边界。

### Decision 3: 删除 `fail_on_error`

`WandbArtifactWriter` 的本地 merge 文件是发布 W&B Artifact 的中间产物，不是对外承诺的本地输出。发布失败后保留该文件不能表示阶段成功，因此不保留 warning-and-skip 分支。`WandbCheckpointWriter` 同理：checkpoint Artifact 发布失败就是 W&B output 失败。

备选方案是保留 `fail_on_error=False` 作为调试出口。该方案会把 artifact 发布失败降级为成功运行，和“统一 W&B logger run 链路”的目标冲突。

### Decision 4: Lineage callback 继续不创建 run

`WandbArtifactLineageCallback` 应只记录 lineage。它可以同样从 `trainer.loggers` 获取 W&B logger run；如果没有 logger-owned run，应按配置 fail 或 warn，但不得创建 run。官方 W&B 实验配置应保留该 callback，使使用 `wandb://` 上游产物时 lineage 进入同一个 logger-owned run。

### Decision 5: 配置职责收敛到 logger

`configs/logger/*.yaml` 是 W&B run 元信息唯一配置面：`name`、`save_dir`、`project`、`group`、`job_type`、`notes`。`configs/callbacks/*.yaml` 中的 W&B writer 只声明 artifact 写入语义，删除 `project`、`group`、`run_name`、`job_type`、`finish_run` 和 `fail_on_error`。

## Risks / Trade-offs

- [Risk] 用户单独启用 W&B writer 但关闭 W&B logger 时会从“跳过发布”变为失败。→ Mitigation: 这是预期破坏性变化；官方 W&B writer 配置必须同时具备 W&B logger defaults，单元测试和 Hydra compose 覆盖该契约。
- [Risk] 从 `trainer.loggers` 获取 run 可能触发 lazy `WandbLogger.experiment` 初始化时机差异。→ Mitigation: `pipeline_launcher` 已在进入训练/推理/分析 loop 前调用 `log_hyperparameters`，通常会提前初始化 logger run；测试中仍覆盖 writer publish 时获取 run 的行为。
- [Risk] 删除 `fail_on_error` 会让短暂 W&B 网络故障直接失败。→ Mitigation: W&B writer 是 W&B output 发布组件；需要本地-only输出时应启用 `LocalPickleWriter` 或关闭 W&B writer，而不是让 W&B writer 降级。
- [Risk] 已完成但未归档的旧 OpenSpec change 仍描述 writer 独立管理 run。→ Mitigation: 本 change 明确反转旧约束，实施时同步更新/归档冲突 specs。

## Migration Plan

1. 删除 W&B writer 的 run lifecycle 参数、`wandb.init` 和 `finish_run` 逻辑。
2. 添加一个小的 logger-run 获取函数或模块内 helper，从 `trainer.loggers` 中获取 W&B logger `experiment`。
3. 修改 W&B writer 和 lineage callback，使缺少 logger-owned run 时按新契约失败或按 lineage 配置 warning。
4. 删除 callback YAML 中 writer-level run/failure 配置，保留 logger YAML 的 run 元信息。
5. 翻转 writer 测试：删除 self-created run 与 `fail_on_error=False` 测试，新增缺少 W&B logger run 时失败、存在 logger run 时发布、不调用 `wandb.init`/`finish` 的测试。
6. 运行 focused writer/lineage tests、Hydra compose、Ruff、pytest 和 `openspec validate require-wandb-logger-run-for-artifacts --strict`。

## Open Questions

无。
