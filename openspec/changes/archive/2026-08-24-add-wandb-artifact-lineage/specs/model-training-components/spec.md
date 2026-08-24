## ADDED Requirements

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
