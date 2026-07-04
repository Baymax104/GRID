## Why

当前训练 experiment 已经通过 `run_mode: train` 显式声明自己走训练主链路，但配置中仍继续保留 `train: true|false` 作为第二层开关，造成语义重复。与此同时，`test: true|false` 的真实作用并不是“是否为 test 模式”，而是“训练完成后是否继续执行 `trainer.test()`”；这个命名过短，人工阅读时难以一眼理解其行为。

现在需要进一步收紧训练 experiment 的配置接口：删除冗余的 `train` 开关，并把训练后测试行为改成更直白、可读的显式字段，降低配置阅读和维护成本。

## What Changes

- 删除训练类 experiment 顶层 `train` 布尔开关。
- 将训练类 experiment 顶层 `test` 布尔开关重命名为更清晰的 `run_test_after_training`。
- 调整统一主入口训练链路：`run_mode: train` 必然执行 `trainer.fit()`，只有 `run_test_after_training` 控制是否在训练后继续执行 `trainer.test()`。
- 同步官方训练 experiment 配置、注释和相关文档表述。

## Capabilities

### Modified Capabilities
- `unified-main-entrypoint`: 进一步收敛训练 experiment 的执行语义，移除冗余 `train` 开关，并显式化训练后测试开关。

## Impact

- 受影响代码：`src/main.py`、可能涉及 dry-run 相关训练链路判断的共享装配逻辑
- 受影响配置：所有 `configs/experiment/*_train.yaml`
- 受影响文档：如有描述训练 experiment 顶层运行开关语义的说明
- **BREAKING**：旧字段 `train` 和 `test` 将不再作为官方训练 experiment 的配置接口
