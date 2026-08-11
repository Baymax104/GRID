## Why

`tail_sid_diagnosis` 属于离线 analysis 实验，不是 Lightning train/inference 实验。当前统一入口只支持 `run_mode: train|inference`，导致 analysis 只能通过独立 CLI 和根脚本游离在 Hydra experiment/config 体系之外。

## What Changes

- 新增通用 `run_mode: analysis`，用于不需要 Lightning model/trainer 的官方离线分析实验。
- 新增 `src/common/analysis/` runner 契约：analysis runner 由 Hydra 实例化，并通过统一主入口执行 `run()`。
- 新增 `configs/analysis/` 组件分组，用于放置 analysis runner 的构造参数。
- 后续将 `tail_sid_diagnosis` 改为 official analysis experiment，并删除独立 argparse CLI 入口；根脚本改走 `uv run --module src.main experiment=tail_sid_diagnosis ...`。
- 允许 analysis runner 复用 `src/data` 的 dataset/dataloader/reader/helper，但不要求使用 Lightning `BaseDataModule`。
- 顺手修复旧 OpenSpec 中与当前代码不一致的 `cfg.components`、`data_loading` 表述，改为当前生效的 `cfg.data` / `cfg.model` / `cfg.trainer` / `cfg.callbacks` / `cfg.logger` / `cfg.analysis` 分组语义。

## Capabilities

### New Capabilities
- `analysis-runner-contract`: 定义 official analysis 实验的 runner 协议、Hydra 配置入口和 data/common 复用边界。

### Modified Capabilities
- `unified-main-entrypoint`: 将统一入口从 train/inference 扩展到 train/inference/analysis。
- `component-grouped-experiment-configs`: 增加 `analysis` 组件分组，并修正旧的 `data_loading`/`cfg.components` 表述。
- `experiment-config-componentization`: 修正旧的 experiment-local `components` 约束，改为当前 repo-level component config group 约束。
- `tail-sid-resolution-diagnosis`: 将 diagnosis 从独立 CLI 调整为 analysis runner 驱动的 official experiment。

## Impact

- Affected code: `src/main.py`, new `src/common/analysis/`, `src/quantization/tail_sid_diagnosis/`, root scripts.
- Affected configs: new `configs/analysis/`, new/updated `configs/experiment/tail_sid_diagnosis.yaml`, possible `configs/data/tail_sid_diagnosis.yaml`.
- Affected OpenSpec: update existing specs listed above and keep the two completed diagnosis changes as context for migration.
- No new dependencies expected.
