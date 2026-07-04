## MODIFIED Requirements

### Requirement: Repository SHALL provide a single Python main entrypoint
仓库必须提供单一 Python 运行入口来承载 train / inference 主链路分发。

#### Scenario: User launches training through unified entrypoint
- **WHEN** 用户通过统一入口执行一个 `run_mode: train` 的 experiment
- **THEN** 系统必须执行训练链路
- **AND** 不得再要求额外的 `train` 布尔开关来决定是否执行 `trainer.fit()`
- **AND** 只能通过显式的训练后测试字段决定是否继续执行测试链路

#### Scenario: Training experiment enables post-training test
- **WHEN** 一个训练 experiment 显式声明 `run_test_after_training: true`
- **THEN** 系统必须在训练完成后继续执行 `trainer.test()`
- **AND** 应优先使用训练过程中解析出的最佳 checkpoint 进行测试

#### Scenario: Training experiment disables post-training test
- **WHEN** 一个训练 experiment 显式声明 `run_test_after_training: false`
- **THEN** 系统在训练完成后不得自动执行 `trainer.test()`
