## ADDED Requirements

### Requirement: CLI SHALL support `--dry-run` for train and inference
系统必须允许用户在训练与推理入口通过 `--dry-run` 启动 dry run 模式，并保持与其他 Hydra 参数兼容。

#### Scenario: Train entry receives dry-run flag
- **WHEN** 用户执行 `-m src.train --dry-run ...`
- **THEN** 训练入口必须成功识别该 flag
- **AND** 主链路必须以内部 `dry_run=true` 的语义继续运行

#### Scenario: Inference entry receives dry-run flag
- **WHEN** 用户执行 `-m src.inference --dry-run ...`
- **THEN** 推理入口必须成功识别该 flag
- **AND** 不得因为未知 CLI 参数导致 Hydra 解析失败

### Requirement: Dry run SHALL execute only a minimal smoke-sized run
dry run 必须真实进入主链路，但运行规模必须压缩到单 batch / 单 step 的 smoke 级别。

#### Scenario: Train dry run executes minimal steps
- **WHEN** 训练入口启用 `dry_run=true`
- **THEN** trainer 必须只执行最小训练步数
- **AND** 不得继续执行正常规模的验证或测试

#### Scenario: Inference dry run executes minimal prediction batches
- **WHEN** 推理入口启用 `dry_run=true`
- **THEN** trainer 必须只执行最小预测 batch 数

### Requirement: Dry run SHALL not write business results
dry run 模式下系统不得写入业务结果产物。

#### Scenario: Train dry run disables result-producing components
- **WHEN** 训练 dry run 运行
- **THEN** 系统不得写入 checkpoint
- **AND** 不得写入 CSV logger 或 W&B 结果

#### Scenario: Inference dry run disables prediction outputs
- **WHEN** 推理 dry run 运行
- **THEN** 系统不得写入 prediction pickle 文件
- **AND** 不得写入 `merged_predictions_tensor.pt`

### Requirement: Dry run MAY preserve runtime metadata outputs
dry run 第一版可以保留 Hydra 输出目录和运行元信息写入，只要不写入业务结果。

#### Scenario: Dry run preserves operational logs
- **WHEN** dry run 运行
- **THEN** 允许生成 Hydra 输出目录与普通运行日志
- **AND** 允许保留 `config_tree.log` 与 `tags.log`
