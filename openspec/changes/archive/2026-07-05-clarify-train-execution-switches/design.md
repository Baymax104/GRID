## Context

统一主入口改造后，仓库已经用 `run_mode` 区分 train / inference 两类主链路，但训练 experiment 仍保留两枚历史布尔开关：`train` 和 `test`。从当前实现看，`run_mode: train` 已经决定进入 `run_training(...)`，而 `train` 只是在其中再决定一次是否执行 `trainer.fit()`；这会形成“train experiment 但可以不 train”的重复表达。另一方面，`test` 实际表示“训练结束后是否追加一次 `trainer.test()`”，却容易被误读为测试模式本身。

本次目标不是扩展新的执行模式，而是把训练 experiment 的配置接口收敛到更符合直觉的最小集合。

## Goals / Non-Goals

**Goals:**
- 让 `run_mode: train` 成为训练主链路的唯一入口语义。
- 删除训练 experiment 中冗余的 `train` 布尔开关。
- 用更直观的字段名表达“训练后是否执行测试”。
- 保持现有训练后测试能力，但提升其可读性。

**Non-Goals:**
- 不新增 `eval`、`test_only` 或其他独立执行模式。
- 不改变推理 experiment 的配置接口。
- 不重构 datamodule、callback、logger 的结构。

## Decisions

### 1. `run_mode: train` 隐含执行 `trainer.fit()`
- 决策：统一入口进入训练主链路后，不再额外读取 `train` 布尔开关，而是默认执行 `trainer.fit()`。
- 原因：`run_mode: train` 已经足够表达“这是一个训练 experiment”。保留第二层 `train` 开关只会增加状态空间和阅读负担。

### 2. 将 `test` 重命名为 `run_test_after_training`
- 决策：训练 experiment 顶层使用 `run_test_after_training: true|false` 作为唯一的训后测试开关。
- 原因：该字段真正表达的是“训练完成后继续执行测试”，而不是测试模式本身；新命名应直接暴露这一点。
- 备选方案：`post_train_test`、`should_test_after_fit`。未采用，因为不如 `run_test_after_training` 直白。

### 3. 本次不保留隐式“test-only”组合
- 决策：不再支持通过 `run_mode: train` + `train: false` + `test: true` 这种布尔组合表达只测试。
- 原因：这类能力从未作为清晰的官方接口存在，继续保留只会让训练 experiment 的语义变得暧昧；如果未来确实需要 test-only，应通过独立模式或独立 experiment 显式设计。

### 4. Dry run 仍可关闭训后测试
- 决策：dry run 相关逻辑继续在训练链路中关闭训后测试，但应改为消费新字段名。
- 原因：dry run 目标仍是最小执行，不应因为字段重命名而改变其行为。

## Risks / Trade-offs

- [历史命令覆盖 `test=true/false`] → 需要同步迁移到 `run_test_after_training=true/false`，并在提案中明确 breaking change。
- [少量用户可能依赖隐式 test-only 组合] → 本次主动放弃该模糊接口；若未来有真实需求，再设计显式模式。
- [字段重命名影响 dry-run 或脚本行为] → 需要最小静态验证训练 experiment 与 dry-run 覆写逻辑仍一致。

## Migration Plan

1. 更新统一入口训练链路，移除 `train` 判断并引入 `run_test_after_training`。
2. 更新训练 experiment 配置，删除 `train` / 重命名 `test`。
3. 更新 dry-run 逻辑中的训练后测试覆写。
4. 做最小配置全文检查，确认官方训练 experiments 不再暴露旧字段。

## Open Questions

- 当前无阻塞性开放问题。
