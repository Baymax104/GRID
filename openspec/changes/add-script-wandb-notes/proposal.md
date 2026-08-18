## Why

当前训练与 diagnosis 运行依赖根目录脚本承载关键路径参数，但脚本缺少统一的运行备注入口。后续做 RKMeans、RVQ、RQ-VAE、TIGER 与 Tail-SID diagnosis 对比时，需要把本次运行的实验意图、配置差异和数据来源直接写入 W&B run notes，避免只靠时间戳和曲线回忆实验背景。

## What Changes

- 为写入 W&B 的官方启动脚本增加 `--notes` 参数，支持 `--notes="..."` 与 `--notes "..."` 两种形式。
- 保留脚本继续接受 `--dry-run`，并将其转交给统一 Hydra 入口的 dry-run 逻辑。
- 允许脚本把未识别参数原样追加到 Hydra 参数列表，用于继续覆盖 trainer、data、model 等配置。
- 在 W&B logger 配置中声明 `notes: null`，使脚本可以通过 `logger.wandb.notes=...` 稳定覆盖。
- 不改变 Python launcher、训练模块、MetricCallback 或 W&B logger 实例化链路。

## Capabilities

### New Capabilities

- None

### Modified Capabilities

- `cli-dry-run`: 扩展官方启动脚本的 CLI 契约，使脚本参数同时支持 dry-run、W&B notes 和额外 Hydra override。

## Impact

- Affected scripts:
  - `rkmeans_train.sh`
  - `rvq_train.sh`
  - `rqvae_train.sh`
  - `tiger_train.sh`
  - `tail_sid_diagnosis.sh`
- Affected configs:
  - `configs/logger/rkmeans_train.yaml`
  - `configs/logger/rvq_train.yaml`
  - `configs/logger/rqvae_train.yaml`
  - `configs/logger/tiger_train.yaml`
  - `configs/logger/tail_sid_diagnosis.yaml`
- Verification should focus on shell argument parsing, Hydra compose behavior, W&B logger construction arguments, and preservation of `--dry-run`.
