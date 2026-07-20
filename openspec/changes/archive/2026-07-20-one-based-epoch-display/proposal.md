## Why

当前 Lightning 默认 progress bar 会将 epoch 以 0-based 方式显示，例如 `Epoch 4/9`。这对框架内部状态是合理的，但对日常阅读训练日志并不直观，容易让用户误读当前处于第几个 epoch。

现在需要仅调整日志显示层，把 epoch 文本改为从 1 开始显示，而不改变任何训练内部状态或 checkpoint 语义。

## What Changes

- 为默认 Lightning progress bar 引入自定义显示逻辑，将 epoch 文本从 0-based 改为 1-based。
- 统一应用于使用默认 progress bar 的 train / validation / test / predict 运行。
- 明确这是纯显示层修改，不改变 `current_epoch` 的内部语义。

## Capabilities

### New Capabilities
- `one-based-epoch-display`: 将控制台 progress bar 中的 epoch 显示统一调整为从 1 开始。

### Modified Capabilities

## Impact

- 受影响代码：默认 callbacks 装配、progress bar callback 配置与可能新增的自定义 callback 实现
- 不涉及依赖变更
- 不改变训练逻辑、checkpoint 恢复、scheduler 或 logger 语义
