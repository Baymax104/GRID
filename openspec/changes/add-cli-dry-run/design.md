## Context

当前 train / inference 入口完全依赖 Hydra + Lightning 装配。主链路默认会实例化 logger、checkpoint callback 和推理写入 callback，因此即使只是想做一次冒烟验证，也会生成 checkpoint、pickle、tensor、CSV 和 W&B 记录等业务产物。与此同时，Hydra 输出目录、运行日志与 `config_tree.log` 已深度绑定到默认运行流程，不适合作为第一版 dry run 的禁写目标。

用户希望通过命令行参数 `--dry-run` 启动一个最小规模的真实运行：主链路要真实执行，但只跑单 batch / 单 step，并且不写业务结果。

## Goals / Non-Goals

**Goals:**
- 提供 `--dry-run` CLI 入口，适用于 `src.train` 与 `src.inference`。
- 让 dry run 真实进入 Hydra / Lightning / datamodule / model / trainer 主链路。
- 在 dry run 下把训练/推理规模压缩到单 batch / 单 step。
- 在 dry run 下禁止写入 checkpoint、prediction 文件、CSV logger、W&B logger 等业务结果。

**Non-Goals:**
- 不要求完全无文件落盘；Hydra 输出目录与运行日志可以保留。
- 不修改依赖清单。
- 不改变非 dry run 的默认行为。
- 不实现新的复杂配置体系或多种 dry run 模式等级。

## Decisions

### 1. 通过入口预处理支持 `--dry-run`
- 决策：在 `src/train.py` 与 `src/inference.py` 中对 `sys.argv` 做轻量预处理，识别并移除 `--dry-run`，再将其转换为 Hydra 可接受的覆盖项（如 `++dry_run=true`）。
- 原因：Hydra 对未知 CLI flag 敏感，不能直接把 `--dry-run` 留给 Hydra 解析。
- 备选方案：要求用户写 `dry_run=true`。未采用，因为用户已明确要求 `--dry-run` 体验。

### 2. 以统一配置开关驱动 dry run 行为
- 决策：内部统一使用 `cfg.dry_run` 控制主链路行为。
- 原因：便于在装配阶段集中处理 logger/callback/trainer 参数覆盖。
- 备选方案：分别在 train/inference 层分散判断。未采用，因为会让行为分裂。

### 3. dry run 下屏蔽业务结果写入组件
- 决策：在装配阶段过滤或覆盖写入型 callbacks / loggers。
- 训练：禁用 `ModelCheckpoint`、`CSVLogger`、`WandbLogger`
- 推理：禁用 `LocalPickleWriter`
- 原因：这些是当前业务结果写入的主要出口，屏蔽它们可最小改动达成目标。
- 备选方案：保留组件但让它们运行时 no-op。未采用为首选，因为 callback/logger 级过滤更直接。

### 4. dry run 下收缩 trainer 运行规模
- 决策：对 trainer 注入最小 batch/step 覆盖。
- train：`max_steps=1`、`limit_train_batches=1`、`limit_val_batches=0`、`limit_test_batches=0`
- inference：`limit_predict_batches=1`
- 原因：满足“执行一次运行”的 smoke 目标，同时避免额外时间与资源消耗。
- 备选方案：完整流程但不写产物。未采用，因为成本过高。

## Risks / Trade-offs

- [某些实验依赖 logger/callback 副作用] → dry run 明确定位为 smoke 模式，只保证主链路运行，不保证实验产物完整性。
- [Hydra CLI 预处理与现有参数交互出错] → 将逻辑限制为只处理 `--dry-run` 单一 flag，并保持其余参数顺序不变。
- [过滤 callback/logger 后仍有遗漏写入点] → 以当前已识别的主写入点为第一版目标，并在验证中检查默认训练/推理模板。

## Migration Plan

1. 在 `src/train.py` / `src/inference.py` 增加 `--dry-run` 预处理。
2. 在配置或装配层引入 `dry_run` 开关默认值。
3. 在 `launcher_utils.py` / instantiator 层实现 dry run 条件下的 callback/logger 过滤与 trainer 覆盖。
4. 验证训练与推理 dry run 都只跑最小规模且不产出业务结果。

## Open Questions

- 当前无阻塞性开放问题。
