## Context

当前 `src/main.py` 通过 `run_mode` 分发到 train/inference 两条 Lightning 链路。`pipeline_launcher()` 会实例化 datamodule、model、callbacks、loggers 和 trainer，因此适合训练和推理，但不适合 `tail_sid_diagnosis` 这种离线分析：它没有模型权重、没有 batch predict，也不需要 Lightning Trainer。

用户已明确决策：

- 设计成通用 analysis 实验。
- 不保留独立 CLI。
- 可以复用 dataset/dataloader 等 data 模块，但不强制使用 datamodule。
- 使用 `configs/analysis/`。
- 顺手修复旧 spec 中与当前配置结构不一致的内容。

## Goals / Non-Goals

**Goals:**
- 在统一主入口中新增 `run_mode: analysis`。
- 新增通用 analysis runner 契约，runner 由 Hydra `_target_` 实例化，并暴露 `run()`。
- 新增 `configs/analysis/` 组件分组，official analysis experiment 通过 defaults 组合 analysis runner 配置。
- 将 `tail_sid_diagnosis` 迁移为 official analysis experiment。
- 删除或停用独立 argparse CLI 与旧根脚本直接 module 调用。
- 修复旧 OpenSpec 中过时的 `cfg.components`、`data_loading`、experiment-local componentization 约束。

**Non-Goals:**
- 不把 analysis runner 伪装为 LightningModule、Callback 或 DataModule。
- 不引入新的依赖。
- 不在本 change 中实现后续 recommendation correlation 或图表输出。
- 不改变现有 train/inference experiment 的运行行为。

## Decisions

### Decision 1: `run_mode: analysis` is a first-class main dispatch branch

`src/main.py` 增加 `run_analysis(cfg)`。它不调用 `pipeline_launcher()`，而是通过一个轻量 analysis launcher 实例化 `cfg.analysis.runner` 并执行 `runner.run()`。

Alternative considered: 复用 `run_mode: inference`。该方式需要构造没有意义的 Lightning model/trainer/predict callback，长期会污染推理链路。

### Decision 2: Analysis runner lives under `src/common/analysis/`

通用协议和 launcher 放在 `src/common/analysis/`，具体业务 runner 仍放在职责域目录，例如 `src/quantization/tail_sid_diagnosis/`。这样 common 只承载运行契约，不承载诊断业务逻辑。

Alternative considered: 把 analysis launcher 放入 `src/utils/launcher.py`。这会让 `utils` 继续承担实验组件逻辑，不符合当前职责边界。

### Decision 3: `configs/analysis/` owns runner construction

Official analysis experiment 的 entry config 仍在 `configs/experiment/`，但 runner 构造参数放入 `configs/analysis/<experiment>.yaml`。顶层手动输入字段仍由 experiment config 暴露，并被 analysis config 引用。

Alternative considered: 把 runner `_target_` 直接写在 `configs/experiment/tail_sid_diagnosis.yaml`。这对一个实验可行，但不符合现有 component-grouped config 方向。

### Decision 4: Data reuse is helper/dataset level, not mandatory DataModule

Analysis runner 可以复用 `src.data.components.readers.TFRecordReader`、dataset/dataloader、`src.data.utils` 等 data 域能力；只有需要 Lightning stage 生命周期时才使用 `BaseDataModule`。

Alternative considered: analysis 全部强制通过 `BaseDataModule`。这会把非 Lightning 分析绑定到 Trainer stage，不必要且容易制造假 dataloader。

### Decision 5: Remove standalone diagnosis CLI

`tail_sid_diagnosis` 不再保留独立 argparse CLI。根脚本改为统一入口命令模板：

```bash
uv run --module src.main experiment=tail_sid_diagnosis ...
```

Alternative considered: 保留独立 CLI 作为 debug 入口。用户明确要求不保留独立 CLI；保留双入口也会削弱 official experiment 的唯一性。

## Risks / Trade-offs

- [Risk] `extras.print_config_warnings` 当前默认检查 `data` 和 `model`，analysis experiment 可能没有 `model` -> 让 warning 根据 `run_mode` 调整，analysis 只要求自身必需 config。
- [Risk] 旧 spec 与当前代码结构已有偏差 -> 在本 change 中同步修改相关 requirements，避免新提案建立在过时文字上。
- [Risk] Analysis runner 协议过早设计太大 -> 第一版只要求 `_target_` + `run()`，不加入 callback/logger 生命周期。
- [Risk] 删除独立 CLI 可能影响已有临时脚本 -> 根脚本提供统一入口替代路径，并通过 tests 验证 Hydra composition。
