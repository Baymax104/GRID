## ADDED Requirements

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
