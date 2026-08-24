# model-training-components Specification

## Purpose
TBD - created by archiving change consolidate-model-training-components. Update Purpose after archive.
## Requirements
### Requirement: Train model configs SHALL group training dependencies
Official train model configs SHALL pass training-only model dependencies through a single `training_model_config` object under `model.root`. The group SHALL include the primary loss function, optimizer factory, optional scheduler factory, and any model-specific auxiliary training loss such as RQVAE reconstruction loss.

#### Scenario: Quantization train configs use training_model_config
- **WHEN** a maintainer inspects `configs/model/rkmeans_train.yaml`, `configs/model/rvq_train.yaml`, or `configs/model/rqvae_train.yaml`
- **THEN** `root` MUST pass `training_model_config: ${model.training_model_config}`
- **THEN** `root` MUST NOT pass separate `loss_function`, `optimizer`, `scheduler`, or `reconstruction_loss_function` fields
- **THEN** `training_model_config` MUST contain the loss and optimizer definitions used by that model

#### Scenario: TIGER train config uses training_model_config
- **WHEN** a maintainer inspects `configs/model/tiger_train.yaml`
- **THEN** `root` MUST pass `training_model_config: ${model.training_model_config}`
- **THEN** `root` MUST NOT pass separate `loss_function`, `optimizer`, or `scheduler` fields
- **THEN** `training_model_config` MUST contain the cross-entropy loss, optimizer factory, and scheduler factory used by TIGER training

#### Scenario: No-scheduler train models preserve null scheduler
- **WHEN** a train model does not use a scheduler
- **THEN** its `training_model_config.scheduler` field MUST be `null`
- **THEN** model optimizer configuration MUST continue returning an optimizer without a Lightning scheduler entry

### Requirement: RVQ train config SHALL declare the standard VQ training dependencies
The official RVQ train model config SHALL declare the VQ commitment loss and optimizer settings used for the standard RVQ experiment through `training_model_config`.

#### Scenario: RVQ train config declares quantization loss
- **WHEN** a maintainer inspects `configs/model/rvq_train.yaml`
- **THEN** `training_model_config.loss_function` MUST target `src.common.loss.beta_quantization_loss.BetaQuantizationLoss`
- **AND** it MUST set `beta: 0.25`
- **AND** it MUST set `reduction: mean`

#### Scenario: RVQ train config declares optimizer
- **WHEN** a maintainer inspects `configs/model/rvq_train.yaml`
- **THEN** `training_model_config.optimizer` MUST target `torch.optim.AdamW`
- **AND** it MUST set `lr: 0.001`
- **AND** it MUST set `weight_decay: 0.0`
- **AND** `training_model_config.scheduler` MUST be `null`

### Requirement: TrainingModelConfig SHALL be a passive runtime container
`TrainingModelConfig` SHALL be a passive runtime container for instantiated training dependencies and factories. It MUST NOT implement training logic, optimizer stepping, scheduler stepping, or model-family-specific behavior.

#### Scenario: Model constructors receive one training dependency object
- **WHEN** a train-capable model is constructed from official config
- **THEN** the model constructor MUST receive a `training_model_config` object
- **THEN** the model constructor MUST NOT require separate `loss_function`, `optimizer`, `scheduler`, or `reconstruction_loss_function` constructor arguments

#### Scenario: Existing training behavior remains model-owned
- **WHEN** a model executes training or `configure_optimizers`
- **THEN** loss computation MUST remain in the model implementation
- **THEN** optimizer and scheduler construction MUST remain coordinated by the model's `configure_optimizers`
- **THEN** `TrainingModelConfig` MUST NOT call model parameters or Lightning trainer state directly

### Requirement: Inference model configs SHALL not gain training_model_config
Official inference model configs SHALL not add `training_model_config`, because inference model construction does not require loss, optimizer, scheduler, or reconstruction loss dependencies.

#### Scenario: Inference configs remain training-dependency free
- **WHEN** a maintainer inspects `configs/model/*_inference.yaml`
- **THEN** those configs MUST NOT add `training_model_config`
- **THEN** those configs MUST NOT add loss, optimizer, scheduler, or reconstruction loss dependencies solely for structural consistency with train configs

### Requirement: Training checkpoint references SHALL support optional W&B run URIs
训练 checkpoint 输入字段 `ckpt_path` SHALL 保持本地路径兼容，并 MAY 使用 `wandb://<run-id>` 引用 producer run 产出的 checkpoint Artifact。该 URI 解析和 Artifact 下载 SHALL 复用 `src/data/components/artifacts.py` 中的 resolver，并 MUST 发生在 Trainer 使用 checkpoint 前。

#### Scenario: Resume or inference checkpoint from W&B run URI
- **WHEN** 训练恢复或推理配置使用 `ckpt_path=wandb://8b61h7ly`
- **THEN** `src/data/components/artifacts.py` 中的 resolver MUST 查找 run `8b61h7ly` 的 checkpoint Artifact
- **THEN** `src/data/components/artifacts.py` 中的 resolver MUST 调用 `src/utils/wandb.py` 下载并解析到具体本地 checkpoint 文件路径
- **THEN** Trainer 接收到的 `ckpt_path` MUST 是可由 Lightning 加载的本地文件路径

#### Scenario: Local checkpoint path remains compatible
- **WHEN** `ckpt_path` 是本地 checkpoint 文件或 checkpoint 目录
- **THEN** 当前本地路径解析和 latest checkpoint 查找行为 MUST 保持兼容
- **THEN** 系统 MUST NOT 为本地 checkpoint 路径调用 W&B Artifact resolver

### Requirement: Checkpoint artifact publishing SHALL not change model training config ownership
checkpoint Artifact 发布 SHALL 作为训练基础设施能力实现，MUST NOT 扩展 `TrainingModelConfig` 的职责，也 MUST NOT 把 Artifact 逻辑放入模型类。

#### Scenario: TrainingModelConfig remains passive
- **WHEN** training 配置启用 checkpoint Artifact 发布
- **THEN** `TrainingModelConfig` MUST 仍只承载 loss、optimizer、scheduler 等训练依赖
- **THEN** `TrainingModelConfig` MUST NOT 调用 W&B API
- **THEN** model implementation MUST NOT 直接发布 checkpoint Artifact

### Requirement: Checkpoint artifact writer SHALL publish ModelCheckpoint outputs
checkpoint Artifact writer SHALL 以 Lightning `ModelCheckpoint` 已写出的本地 checkpoint 为发布源，MUST NOT 自行决定 checkpoint 保存时机或复制 `ModelCheckpoint` 的选择逻辑。

#### Scenario: Publish best ModelCheckpoint output
- **WHEN** training 配置启用 checkpoint Artifact writer
- **AND** 当前 trainer 中存在唯一 `ModelCheckpoint`
- **AND** 该 callback 的 `best_model_path` 指向存在的 `.ckpt` 文件
- **THEN** checkpoint writer MUST 发布该 `.ckpt` 文件
- **THEN** Artifact metadata MUST 记录 `best_model_path`、`monitor`、`mode` 和 `best_model_score`

#### Scenario: ModelCheckpoint configuration determines artifact content
- **WHEN** `ModelCheckpoint` 的 `monitor`、`mode`、`save_top_k` 或 `filename` 配置发生变化
- **THEN** checkpoint writer 发布的 Artifact MUST 跟随 `ModelCheckpoint` 实际写出的 checkpoint
- **THEN** checkpoint writer MUST NOT 用另一套 ranking 或文件命名逻辑覆盖 `ModelCheckpoint`

#### Scenario: Missing or ambiguous ModelCheckpoint is explicit
- **WHEN** checkpoint Artifact writer 启用
- **AND** 当前 trainer 没有 `ModelCheckpoint`，或存在多个未被配置唯一选择的 `ModelCheckpoint`
- **THEN** checkpoint writer MUST 按配置明确 skip 或 fail
- **THEN** checkpoint writer MUST NOT 静默选择任意 checkpoint

### Requirement: W&B checkpoint writer SHALL require a WandbLogger-owned run

`WandbCheckpointWriter` SHALL publish checkpoint Artifacts only to the W&B run owned by the configured Lightning `WandbLogger`. It MUST NOT create, configure, finish, or otherwise manage a W&B run.

#### Scenario: Checkpoint writer publishes through logger-owned run
- **WHEN** a training experiment configures `WandbCheckpointWriter`
- **AND** the trainer has a configured W&B logger whose `experiment` provides a run
- **AND** the selected checkpoint file exists
- **THEN** `WandbCheckpointWriter` MUST publish the checkpoint Artifact to that logger-owned run
- **THEN** the published Artifact metadata MUST include `role: checkpoint`, `task_name`, `local_output_path`, and `bundle_file`

#### Scenario: Missing logger-owned run fails checkpoint publishing
- **WHEN** `WandbCheckpointWriter` reaches artifact publishing
- **AND** the trainer does not expose a configured W&B logger whose `experiment` provides a run
- **THEN** publishing MUST fail with a clear error naming the missing W&B logger run
- **THEN** `WandbCheckpointWriter` MUST NOT call `wandb.init`

#### Scenario: Checkpoint publish failure is not downgraded
- **WHEN** `WandbCheckpointWriter` fails to create or log a W&B Artifact
- **THEN** the exception MUST propagate
- **THEN** `WandbCheckpointWriter` MUST NOT provide a `fail_on_error` option
- **THEN** `WandbCheckpointWriter` MUST NOT log a warning and continue as if W&B output succeeded

#### Scenario: Checkpoint writer configuration excludes run lifecycle fields
- **WHEN** maintainers inspect official training callback configs
- **THEN** `wandb_checkpoint_writer` MUST NOT declare `project`, `entity`, `group`, `run_name`, `job_type`, `tags`, `notes`, `mode`, `finish_run`, or `fail_on_error`
- **THEN** W&B run identity MUST be configured through the experiment's logger config

