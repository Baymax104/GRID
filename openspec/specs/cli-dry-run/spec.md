# cli-dry-run Specification

## Purpose
TBD - created by archiving change add-cli-dry-run. Update Purpose after archive.
## Requirements
### Requirement: CLI SHALL support `--dry-run` for train and inference
系统 SHALL 允许用户在训练与推理入口通过 `--dry-run` 启动 dry run 模式，并保持与其他 Hydra 参数兼容。

#### Scenario: Train entry receives dry-run flag
- **WHEN** 用户执行 `-m src.train --dry-run ...`
- **THEN** 训练入口必须成功识别该 flag
- **AND** 主链路必须以内部 `dry_run=true` 的语义继续运行

#### Scenario: Inference entry receives dry-run flag
- **WHEN** 用户通过统一入口执行推理 experiment 并附带 `--dry-run ...`
- **THEN** 推理入口必须成功识别该 flag
- **AND** 不得因为未知 CLI 参数导致 Hydra 解析失败

### Requirement: Dry run SHALL execute only a minimal smoke-sized run
dry run SHALL 真实进入主链路，但运行规模 SHALL 压缩到单 batch / 单 step 的 smoke 级别。

#### Scenario: Train dry run executes minimal steps
- **WHEN** 训练入口启用 `dry_run=true`
- **THEN** trainer 必须只执行最小训练步数
- **AND** 不得继续执行正常规模的验证或测试

#### Scenario: Inference dry run executes minimal prediction batches
- **WHEN** 推理入口启用 `dry_run=true`
- **THEN** trainer 必须只执行最小预测 batch 数

### Requirement: Dry run SHALL not write business results
dry run 模式下系统 SHALL NOT 写入业务结果产物。

#### Scenario: Train dry run disables result-producing components
- **WHEN** 训练 dry run 运行
- **THEN** 系统不得写入 checkpoint
- **AND** 不得写入 CSV logger 或 W&B 结果

#### Scenario: Inference dry run disables prediction outputs
- **WHEN** 推理 dry run 运行
- **THEN** 系统不得写入 prediction pickle 文件
- **AND** 不得写入 `merged_predictions_tensor.pt`

### Requirement: Dry run SHALL preserve runtime metadata outputs
dry run 第一版 SHALL 保留 Hydra 输出目录和运行元信息写入，只要不写入业务结果。

#### Scenario: Dry run preserves operational logs
- **WHEN** dry run 运行
- **THEN** 允许生成 Hydra 输出目录与普通运行日志
- **AND** 允许保留 `config_tree.log`
- **AND** 不得要求保留已删除的 `tags.log`

### Requirement: Official W&B scripts SHALL accept run notes
写入 W&B 的官方启动脚本 SHALL 支持用户通过 `--notes` 参数为本次运行提供备注，并将备注透传到 W&B run notes。

#### Scenario: Notes are passed with equals syntax
- **WHEN** 用户执行 `./rvq_train.sh --notes="RVQ 20k aligned with RKMeans"`
- **THEN** 脚本必须向 Hydra 传入 `logger.wandb.notes=RVQ 20k aligned with RKMeans`
- **AND** W&B logger 初始化必须接收该 notes 值

#### Scenario: Notes are passed with separated value syntax
- **WHEN** 用户执行 `./rkmeans_train.sh --notes "SGD 20k baseline"`
- **THEN** 脚本必须向 Hydra 传入 `logger.wandb.notes=SGD 20k baseline`
- **AND** notes 文本中的空格不得被拆分成多个 Hydra 参数

#### Scenario: Empty notes are ignored
- **WHEN** 用户执行 `./tail_sid_diagnosis.sh --notes=""`
- **THEN** 脚本不得向 Hydra 追加非空 notes override
- **AND** W&B logger 必须保留配置中的默认 notes 值

### Requirement: Official W&B scripts SHALL preserve dry-run and Hydra override compatibility
写入 W&B 的官方启动脚本 SHALL 同时支持脚本级 `--dry-run`、`--notes` 和额外 Hydra override，并将它们组合成统一入口可解析的参数列表。

#### Scenario: Dry-run flag is forwarded
- **WHEN** 用户执行 `./rkmeans_train.sh --dry-run --notes="smoke test"`
- **THEN** 脚本必须向统一入口传入 `--dry-run`
- **AND** 系统必须继续以内部 `dry_run=true` 语义运行
- **AND** notes override 必须仍然被传给 W&B logger 配置

#### Scenario: Extra Hydra overrides are preserved
- **WHEN** 用户执行 `./rvq_train.sh --notes="longer layer budget" trainer.root.max_steps=30000 devices=[0,1]`
- **THEN** 脚本必须保留 `trainer.root.max_steps=30000`
- **AND** 脚本必须保留 `devices=[0,1]`
- **AND** 这些额外参数必须追加到默认 experiment 参数之后传给 Hydra

#### Scenario: Missing notes value fails fast
- **WHEN** 用户执行 `./rqvae_train.sh --notes`
- **THEN** 脚本必须以非零状态退出
- **AND** 错误信息必须说明 `--notes` 缺少取值

### Requirement: Training scripts SHALL not default to dry-run
官方训练脚本 SHALL 将 dry-run 作为显式脚本参数，而不是无条件附加到每次运行。

#### Scenario: Train script without dry-run starts normal training config
- **WHEN** 用户执行 `./tiger_train.sh --notes="full train"`
- **THEN** 脚本不得自动追加 `--dry-run`
- **AND** Hydra 配置中的 `dry_run` 必须保持默认 `false`

#### Scenario: Train script with dry-run keeps smoke behavior
- **WHEN** 用户执行 `./tiger_train.sh --dry-run --notes="smoke train"`
- **THEN** 脚本必须追加 `--dry-run`
- **AND** 统一入口必须继续执行 dry-run override 重写逻辑

