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

### Requirement: Downstream W&B scripts SHALL require a quantization method group

The official `tiger_train.sh`, `tiger_inference.sh`, and `tail_sid_diagnosis.sh` scripts SHALL require callers to select the quantization-method W&B group and SHALL forward that value to the top-level Hydra `group` field.

#### Scenario: Group is passed with equals syntax
- **WHEN** a user invokes a downstream script with `--group=rkmeans`
- **THEN** the script MUST pass `group=rkmeans` to the unified entrypoint

#### Scenario: Group is passed with separated syntax
- **WHEN** a user invokes a downstream script with `--group rvq`
- **THEN** the script MUST pass `group=rvq` as one Hydra argument

#### Scenario: RQ-VAE downstream run is grouped explicitly
- **WHEN** a user invokes a downstream script with `--group rqvae`
- **THEN** the script MUST pass `group=rqvae` to the unified entrypoint

#### Scenario: Missing group fails fast
- **WHEN** a user invokes a downstream script without `--group` or provides the flag without a value
- **THEN** the script MUST exit with status 2
- **AND** the error MUST state that `--group` requires `rkmeans`, `rvq`, or `rqvae`

#### Scenario: Unsupported group fails fast
- **WHEN** a user invokes a downstream script with a group other than `rkmeans`, `rvq`, or `rqvae`
- **THEN** the script MUST exit with status 2
- **AND** it MUST NOT invoke the unified Python entrypoint

#### Scenario: Group preserves existing script contracts
- **WHEN** a downstream script receives group together with dry-run, notes, checkpoint or artifact paths, and extra Hydra overrides supported by that script
- **THEN** every supported option MUST remain intact
- **AND** extra Hydra overrides MUST still appear after the script's default `group` argument so explicit trailing overrides retain precedence

### Requirement: Quantization scripts SHALL require an explicit embedding input
RKMeans、RVQ 和 RQVAE 的官方训练与推理脚本 SHALL 要求调用者通过 `--embedding-path` 显式提供 embedding 数据源，并将其作为 `embedding_path` Hydra override 传给统一入口。

#### Scenario: W&B embedding is passed with equals syntax
- **WHEN** 用户执行 `./rkmeans_train.sh --embedding-path=wandb://producer123 --notes="train"`
- **THEN** 脚本必须向 Hydra 传入 `embedding_path=wandb://producer123`
- **AND** 不得继续传入任何硬编码的 embedding producer run

#### Scenario: Local embedding is passed with separated value syntax
- **WHEN** 用户执行 `./rvq_inference.sh --embedding-path "outputs/semantic embeddings.pt" --ckpt-path wandb://checkpoint123`
- **THEN** 脚本必须将 `embedding_path=outputs/semantic embeddings.pt` 作为单个参数传给 Hydra
- **AND** 本地路径中的空格不得被拆分

#### Scenario: Missing embedding input fails fast
- **WHEN** 用户执行任一 RKMeans、RVQ 或 RQVAE 官方训练/推理脚本而未提供 `--embedding-path`
- **THEN** 脚本必须以状态码 2 退出
- **AND** 错误信息必须说明需要本地路径或 `wandb://` 引用

#### Scenario: Separated embedding flag without value fails fast
- **WHEN** 用户传入 `--embedding-path` 后未提供取值或紧接另一个 `--` 参数
- **THEN** 脚本必须以状态码 2 退出
- **AND** 错误信息必须说明 `--embedding-path` 缺少取值

### Requirement: Quantization embedding input SHALL preserve existing script composition
显式 embedding 输入 SHALL 与量化脚本现有的 dry-run、notes、checkpoint、设备参数和额外 Hydra override 组合使用，且额外 override SHALL 继续位于脚本默认参数之后。

#### Scenario: Training metadata and dry-run are preserved
- **WHEN** 用户向量化训练脚本同时传入 `--embedding-path`、`--notes` 和 `--dry-run`
- **THEN** 脚本必须转发 embedding 与 W&B notes overrides
- **AND** 脚本必须向统一入口转发 `--dry-run`

#### Scenario: Inference checkpoint is preserved
- **WHEN** 用户向量化推理脚本同时传入 `--embedding-path` 和 `--ckpt-path`
- **THEN** 脚本必须分别传入 `embedding_path=<value>` 与 `ckpt_path=<value>`

#### Scenario: Extra Hydra override remains last
- **WHEN** 用户在正式 `--embedding-path` 参数之后额外传入原始 `embedding_path=<other-value>` Hydra override
- **THEN** 两个参数必须按正式 flag 值在前、额外 override 在后的顺序传给 Hydra
- **AND** Hydra 必须能够按既有规则使用尾部 override 作为最终值

