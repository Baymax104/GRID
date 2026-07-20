## Context

当前项目默认使用 Lightning 自带的 progress bar 显示训练进度。Lightning 内部的 `current_epoch` 是 0-based 计数，因此控制台上会看到类似 `Epoch 4/9` 的文本。虽然这与内部状态一致，但对日常阅读不够直观。

用户希望只修改显示层，让日志中的 epoch 从 1 开始显示，而不改变训练内部逻辑、checkpoint 恢复语义或 logger 行为。

## Goals / Non-Goals

**Goals:**
- 将默认 progress bar 中的 epoch 文本从 0-based 改为 1-based。
- 统一应用到 train / validation / test / predict 场景的默认 progress bar 显示。
- 保持内部 `current_epoch`、checkpoint、scheduler 等训练语义不变。

**Non-Goals:**
- 不修改模型内部 epoch 计数。
- 不更改 logger、checkpoint、dry run、step/batch 逻辑。
- 不引入新的训练配置语义，只改显示层。

## Decisions

### 1. 通过自定义 Lightning progress bar callback 实现显示修正
- 决策：新增一个继承自 `TQDMProgressBar` 的自定义 callback，在显示 description 时将 epoch 文本改为 `current_epoch + 1`。
- 原因：这是 Lightning 官方支持的扩展点，影响面局限在显示层。
- 备选方案：修改 trainer 或模型中的 epoch 变量。未采用，因为会污染真实训练语义。

### 2. 保持内部状态 0-based，仅改展示文本
- 决策：只在 progress bar description 中做 `+1` 显示，不触碰 `trainer.current_epoch`、`self.current_epoch` 或 checkpoint 状态。
- 原因：避免影响恢复训练、scheduler 和任何依赖内部 epoch 的逻辑。

### 3. 将自定义 callback 接入默认 callbacks 装配路径
- 决策：通过默认 callbacks 配置替换或补充当前默认 progress bar，使主链路默认采用该显示策略。
- 原因：用户希望统一行为，而不是只对某个实验单独生效。

## Risks / Trade-offs

- [显示值与内部 epoch 值不一致] → 在文档/注释中明确这是显示层 1-based，内部状态仍为 0-based。
- [自定义 progress bar 与现有 unbounded dataset 场景兼容性问题] → 延续当前默认 TQDM 路径，不启用 `RichProgressBar`。
- [仅部分运行场景应用到新 callback] → 实施时检查默认 callbacks 装配路径，确保 train/inference 的默认 progress bar 一致。

## Migration Plan

1. 新增自定义 `TQDMProgressBar` callback。
2. 在默认 callbacks 配置中接入它。
3. 验证控制台 epoch 显示从 1 开始，而内部训练逻辑不变。

## Open Questions

- 当前无阻塞性开放问题。
