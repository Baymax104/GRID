# training-log-verbosity-control Specification

## Purpose
TBD - created by archiving change unify-training-log-verbosity. Update Purpose after archive.
## Requirements
### Requirement: Training progress bar SHALL not display metrics
系统在训练阶段的终端 progress bar SHALL NOT 显示任何数值指标。

#### Scenario: Quantization training step does not expose metrics
- **WHEN** `ResidualQuantization` 执行训练 step
- **THEN** 终端 progress bar 不得显示 `train/loss`
- **AND** 不得显示 quantization loss、reconstruction loss、覆盖率、熵或其他 verbose 指标

#### Scenario: Common training module does not expose train metrics
- **WHEN** 其他训练模块执行训练 step
- **THEN** 终端 progress bar 不得显示对应的 `train/loss`

### Requirement: Validation and test progress bar SHALL not display metrics
系统在验证和测试阶段的终端 progress bar SHALL NOT 显示任何数值指标。

#### Scenario: Validation metrics stay out of progress bar
- **WHEN** 模型在验证阶段记录指标
- **THEN** 终端 progress bar 不得显示 `val/loss`
- **AND** 其他验证指标不得显示到 progress bar

#### Scenario: Test metrics stay out of progress bar
- **WHEN** 模型在测试阶段记录指标
- **THEN** 终端 progress bar 不得显示 `test/loss`
- **AND** 其他测试指标不得显示到 progress bar

### Requirement: Non-loss metrics SHALL remain available to loggers
除了终端 progress bar 可见性变化外，非 loss 指标 SHALL 仍保留给 logger 使用。

#### Scenario: Verbose metrics are still emitted to logger
- **WHEN** 训练模块计算非 loss 指标
- **THEN** 这些指标仍必须通过 logger 输出
- **AND** 不得因为终端收敛而删除其日志记录行为

