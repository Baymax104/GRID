## 1. Writer Run Lifecycle

- [x] 1.1 为 `WandbArtifactWriter` 增加 run 配置参数，并在模块内实现 active run 复用、自建 run、只 finish 自建 run。
- [x] 1.2 为 `WandbCheckpointWriter` 增加同等 run 配置参数，并在模块内独立实现 run 管理与 Artifact 发布，不 import artifact writer helper。
- [x] 1.3 保持两个 writer 的 `fail_on_error` 行为：发布失败时可按配置跳过，不破坏本地输出。

## 2. 配置接入

- [x] 2.1 为所有 inference `wandb_artifact_writer` 配置补齐 `project`、`group`、`run_name`、`job_type`、`finish_run`。
- [x] 2.2 为所有 training `wandb_checkpoint_writer` 配置补齐同等 run 参数。
- [x] 2.3 确认 `WandbArtifactLineageCallback` 配置和实现不创建 W&B run。

## 3. 测试与验证

- [x] 3.1 为 `WandbArtifactWriter` 添加无 active run 时 `wandb.init`、已有 run 时复用、只 finish 自建 run 的测试。
- [x] 3.2 为 `WandbCheckpointWriter` 添加同等 run lifecycle 测试，并验证不依赖 `wandb_artifact_writer` helper。
- [x] 3.3 运行 focused writer tests、Hydra compose、`uv run ruff check src tests`、`uv run pytest` 和 `openspec validate enable-wandb-writer-run-management --strict`。
