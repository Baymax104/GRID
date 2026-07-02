## ADDED Requirements

### Requirement: Progress bar SHALL display epoch numbers starting at 1
系统在控制台 progress bar 中显示 epoch 时，必须从 1 开始，而不是直接暴露内部 0-based epoch 值。

#### Scenario: Training progress displays first epoch
- **WHEN** 训练在内部 `current_epoch == 0` 时开始显示 epoch
- **THEN** 控制台必须显示为 `Epoch 1/...`

#### Scenario: Resumed training displays user-facing epoch number
- **WHEN** 训练从内部 `current_epoch == 4` 的状态恢复
- **THEN** 控制台必须显示为 `Epoch 5/...`

### Requirement: One-based epoch display SHALL not change internal training state
epoch 的 1-based 显示必须只影响日志展示层，不得改变训练内部语义。

#### Scenario: Internal state remains zero-based
- **WHEN** 控制台显示 1-based epoch 文本
- **THEN** 内部 `current_epoch` 语义必须保持不变
- **AND** checkpoint 恢复行为不得受到影响

### Requirement: One-based epoch display SHALL apply consistently across default runs
默认 progress bar 在 train / validation / test / predict 场景中都必须使用同一套 epoch 显示规则。

#### Scenario: Default pipeline uses custom epoch display
- **WHEN** 用户通过默认主链路运行训练或推理
- **THEN** 默认 progress bar 必须使用从 1 开始的 epoch 显示规则
