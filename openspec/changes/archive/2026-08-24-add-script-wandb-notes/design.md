## Context

官方训练脚本当前直接拼接 `uv run torchrun ...` 命令，并在四个训练脚本末尾固定附带 `--dry-run`。`tail_sid_diagnosis.sh` 已经使用 Bash 数组构造 Hydra 参数，更适合承载可选参数。W&B 侧无需新增封装：Lightning `WandbLogger` 接受 `**kwargs` 并透传到 `wandb.init`，因此 `logger.wandb.notes` 可以直接进入 W&B run notes。

## Goals / Non-Goals

**Goals:**

- 让写入 W&B 的官方脚本支持 `--notes="..."` 和 `--notes "..."`。
- 让脚本继续支持 `--dry-run`，并保持现有统一入口的 dry-run 语义。
- 让脚本继续接受额外 Hydra override，便于临时调整训练和 diagnosis 参数。
- 使用 Bash 数组传参，保证 notes 中包含空格、逗号或冒号时不被拆分。

**Non-Goals:**

- 不修改 `src/utils/launcher.py`、`src/utils/logging.py` 或 Lightning logger 实例化逻辑。
- 不把 notes 记录为 metric、summary scalar 或 artifact。
- 不改变推理脚本的 logger 策略。
- 不在本变更中新增 W&B tags、Reports 或 workspace 视图。

## Decisions

- **脚本解析 `--notes`，再生成 `logger.wandb.notes=...` override。** 这样 notes 属于运行入口的用户意图，不需要污染 experiment 顶层配置。替代方案是在 Python launcher 中读取环境变量，但会让备注逻辑变成全局隐式行为。
- **logger YAML 显式声明 `notes: null`。** 这样脚本使用普通 Hydra override，不依赖 `+logger.wandb.notes=...` 追加未知字段。替代方案是脚本每次使用 `+`，但配置契约不够清晰。
- **未识别参数原样透传给 Hydra。** 训练脚本仍可接受 `trainer.root.max_steps=...`、`devices=[0,1]` 等现有 override。替代方案是只白名单固定参数，但会降低调参效率。
- **移除训练脚本固定 dry-run，改为显式 `--dry-run`。** `./rkmeans_train.sh --dry-run` 表达 smoke run，`./rkmeans_train.sh` 表达真实训练。替代方案是保留默认 dry-run，但会让正式训练入口继续需要手动删脚本或改文件。

## Risks / Trade-offs

- **默认行为变化** -> 训练脚本不再默认 dry-run；实现时需在脚本和任务说明中明确，并用 dry-run smoke 验证显式参数仍生效。
- **Bash quoting 差异** -> 使用数组保存参数，并支持 `--notes=value` 与 `--notes value` 两种常见形式。
- **空 notes 语义不清** -> `--notes=""` SHALL 不写入 notes override，等价于使用 logger 默认 `null`。
- **重复实现解析逻辑** -> 第一版可在每个脚本内保持小型解析函数；若后续脚本继续增长，再考虑抽成共享 shell helper。
