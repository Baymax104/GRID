# wandb-artifact-lineage Specification

## Purpose
TBD - created by archiving change add-wandb-artifact-lineage. Update Purpose after archive.
## Requirements
### Requirement: W&B reference resolver SHALL support run-based artifact references
系统 SHALL 提供独立的 W&B reference resolver，使现有阶段间文件字段能够通过 `wandb://` URI 引用 producer run 的 output Artifact。resolver MUST 只在输入值使用 `wandb://` 协议时触发 W&B API。Short URI defaults MUST come only from explicit resolver arguments supplied by experiment config; resolver MUST NOT infer entity or project from environment variables, active W&B runs, or W&B API account defaults.

#### Scenario: Resolve short run URI
- **WHEN** `semantic_id_path` 的值为 `wandb://1mzveep4`
- **AND** 调用方传入 `default_entity` 和 `default_project`
- **THEN** resolver MUST 在传入的 W&B entity/project 中查找 run `1mzveep4`
- **THEN** resolver MUST 使用字段名推断 Artifact role 为 `semantic_id`
- **THEN** resolver MUST 下载唯一匹配 Artifact 中的目标文件并返回本地文件路径

#### Scenario: Short run URI without explicit identity fails
- **WHEN** `embedding_path`、`semantic_id_path` 或 `ckpt_path` 的值为 `wandb://<run-id>`
- **AND** URI 未包含 entity/project
- **AND** 调用方未传入 `default_entity` 或未传入 `default_project`
- **THEN** resolver MUST raise 并说明 short W&B URI requires experiment `user/project` defaults or a fully-qualified URI
- **THEN** resolver MUST NOT read `WANDB_ENTITY` or `WANDB_PROJECT`
- **THEN** resolver MUST NOT read an active `wandb.run`
- **THEN** resolver MUST NOT call `wandb.Api().default_entity`

#### Scenario: Resolve cross-project run URI
- **WHEN** 输入引用为 `wandb://baymaxam/GRID/1mzveep4?role=semantic_id`
- **THEN** resolver MUST 使用 URI 中的 entity 和 project
- **THEN** resolver MUST NOT 依赖当前运行配置中的默认 entity/project

#### Scenario: Local path bypasses W&B
- **WHEN** `embedding_path`、`semantic_id_path` 或 `ckpt_path` 的值不是 `wandb://` URI
- **THEN** resolver MUST 原样返回该引用或交给现有本地/远端文件读取路径
- **THEN** resolver MUST NOT 初始化 W&B API
- **THEN** resolver MUST NOT 要求 W&B 凭据存在

### Requirement: W&B artifact selection SHALL be deterministic
resolver SHALL 根据字段名、URI 参数、Artifact metadata 和 alias 选择唯一 Artifact。匹配结果为 0 个或多个时 MUST 失败并给出可操作错误。

#### Scenario: Missing artifact fails clearly
- **WHEN** `ckpt_path` 为 `wandb://8b61h7ly`
- **AND** run `8b61h7ly` 没有 role 为 `checkpoint` 的 output Artifact
- **THEN** resolver MUST raise 并说明该 run 缺少 checkpoint Artifact

#### Scenario: Ambiguous artifact fails clearly
- **WHEN** `semantic_id_path` 为 `wandb://1mzveep4`
- **AND** run `1mzveep4` 有多个 role 为 `semantic_id` 且未被 alias 或 file 唯一限定的 output Artifact
- **THEN** resolver MUST raise 并提示用户补充 `alias`、`role` 或 `file`

#### Scenario: Explicit URI parameters override defaults
- **WHEN** `semantic_id_path` 为 `wandb://1mzveep4?role=semantic_id&alias=v3&file=merged_predictions_tensor.pt`
- **THEN** resolver MUST 使用 URI 中的 role、alias 和 file
- **THEN** resolver MUST NOT 使用字段默认 role 覆盖 URI 参数

### Requirement: Consumer runs SHALL record upstream artifact lineage through an explicit callback
当当前运行存在活动 W&B run、resolver 成功解析 W&B Artifact，且 `WandbArtifactLineageCallback` 被显式配置时，系统 SHALL 通过 W&B `use_artifact` 记录 consumer run 对 upstream Artifact 的使用关系。Official consumers that resolve inputs during Lightning execution MUST make resolved references available before the lineage callback's applicable setup/start hook.

#### Scenario: Active W&B run records lineage
- **WHEN** 当前 `tiger_train` run 使用 `semantic_id_path=wandb://1mzveep4`
- **AND** 当前进程存在活动 W&B run
- **AND** 当前实验显式启用了 `WandbArtifactLineageCallback`
- **THEN** 系统 MUST 对解析出的 semantic ID Artifact 调用 `use_artifact`
- **THEN** W&B lineage MUST 能表示 producer run -> Artifact -> current run

#### Scenario: Diagnosis inputs are registered before callback setup
- **WHEN** Tail-SID diagnosis 使用 W&B-backed Semantic ID 或 embedding 输入
- **AND** Lightning 调用 diagnosis DataModule setup
- **THEN** DataModule MUST 在构造 Dataset 前解析这些输入并注册 resolved references
- **AND** `WandbArtifactLineageCallback.setup` MUST 能读取这些 references
- **AND** 当前 logger-owned W&B run MUST 对每个尚未记录的 upstream Artifact 调用 `use_artifact`

#### Scenario: No active W&B run still resolves file
- **WHEN** 输入值为 `wandb://1mzveep4`
- **AND** 当前进程没有活动 W&B run
- **THEN** resolver MAY 下载 Artifact 并返回本地文件路径
- **THEN** resolver MUST NOT 创建隐式 tracking run 仅用于记录 lineage

#### Scenario: Callback disabled does not affect path resolution
- **WHEN** 输入值为 `wandb://1mzveep4`
- **AND** 当前实验未启用 `WandbArtifactLineageCallback`
- **THEN** resolver MAY 下载 Artifact 并返回本地文件路径
- **THEN** 系统 MUST NOT 为了记录 lineage 改变 logger 初始化顺序

### Requirement: Output artifact writing SHALL be optional and role-based
系统 SHALL 提供可选 W&B Artifact writer，用于把训练 checkpoint 或推理 model output bundle 发布为 W&B Artifact。writer MUST 通过配置显式启用，并 MUST 为 Artifact 写入 role metadata。

#### Scenario: Publish inference output artifact through WandbArtifactWriter
- **WHEN** inference 配置启用 W&B Artifact 发布
- **AND** `${paths.output_dir}/pickle/merged_predictions_tensor.pt` 已存在
- **THEN** `WandbArtifactWriter` MUST 将该文件发布为 W&B Artifact
- **THEN** Artifact metadata MUST 包含 `role`、`task_name`、`local_output_path` 和 `bundle_file`

#### Scenario: Publishing disabled has no W&B side effects
- **WHEN** inference 配置未启用 W&B Artifact 发布
- **THEN** 推理完成后 MUST 只保留当前本地输出行为
- **THEN** 系统 MUST NOT 因 W&B writer 未启用而初始化 W&B API

#### Scenario: Checkpoint artifact role
- **WHEN** training 配置启用 checkpoint Artifact 发布
- **THEN** 发布的 checkpoint Artifact metadata MUST 包含 `role: checkpoint`
- **THEN** Artifact MUST 能被 `ckpt_path=wandb://<run-id>` 解析

### Requirement: Resolved references SHALL be auditable
系统 SHALL 将每个成功解析的 W&B 引用记录为结构化 resolved reference，保留原始输入、producer run、Artifact 标识和本地缓存路径。该记录 SHALL 由 `src/data/components/artifacts.py` 提供，不要求 launcher 直接改写 Hydra 主配置。

#### Scenario: Data artifacts component records resolved reference
- **WHEN** `semantic_id_path=wandb://1mzveep4` 成功解析
- **THEN** `src/data/components/artifacts.py` MUST 记录该字段的原始引用
- **THEN** `src/data/components/artifacts.py` MUST 记录 producer run id、Artifact name/version 和本地 resolved path
- **THEN** 显式启用的 `WandbArtifactLineageCallback` 或 config logger MAY 将这些字段写入当前 W&B config

### Requirement: W&B integration SHALL remain modular and opt-in
字段级 W&B reference 解析 SHALL 位于 `src/data/components/artifacts.py`；底层 W&B URI 解析和 Artifact 下载 helper SHALL 位于 `src/utils/wandb.py`；Artifact 发布 SHALL 位于 writer 模块；lineage 记录 SHALL 位于 `src/common/callbacks/WandbArtifactLineageCallback`。主流程、模型和本地 writer SHALL NOT 直接实现 W&B Artifact 选择、上传或下载逻辑。

#### Scenario: Lineage callback is not a logger or writer
- **WHEN** 维护者检查 `WandbArtifactLineageCallback`
- **THEN** 它 MUST 位于 `src/common/callbacks/`
- **THEN** 它 MUST NOT 实现 logger 接口
- **THEN** 它 MUST NOT 写出或发布 output Artifact
- **THEN** 它 MUST 只读取 resolved-reference registry 并在 active W&B run 中调用 `use_artifact`

#### Scenario: No W&B protocol means no W&B module execution
- **WHEN** 所有输入路径都是本地路径且 Artifact 发布配置关闭
- **THEN** 训练、推理和分析流程 MUST NOT 调用 W&B Artifact resolver 或 writer
- **THEN** 本地路径实验 MUST 在无 W&B 凭据环境中保持可运行

#### Scenario: Launcher does not branch on W&B config
- **WHEN** 维护者检查 launcher 装配流程
- **THEN** launcher MUST NOT 扫描 Hydra 配置来判断是否存在 `wandb://`
- **THEN** launcher MUST NOT 因 artifact lineage 提前实例化 W&B logger
- **THEN** launcher MUST NOT 直接调用 W&B Artifact resolver、writer 或 `use_artifact`

#### Scenario: W&B code does not enter model logic
- **WHEN** 维护者检查 train/inference model implementation
- **THEN** 模型类 MUST NOT 直接调用 `wandb.Api`、`wandb.Artifact`、`log_artifact` 或 `use_artifact`
- **THEN** 模型类 MUST NOT 解析 `wandb://` URI

### Requirement: W&B artifact lifecycle SHALL be owned by WandbLogger

Official W&B-backed train, inference, and analysis experiments SHALL use the run exposed by the configured Lightning `WandbLogger.experiment` as the single W&B run used for config logging, metrics logging, Artifact publishing, and upstream Artifact lineage recording.

#### Scenario: Logger config is the run identity source
- **WHEN** maintainers inspect official W&B-backed experiment configs
- **THEN** W&B run identity fields MUST be declared under `configs/logger/*.yaml`
- **THEN** W&B writer callback configs MUST NOT duplicate run identity fields

#### Scenario: Writer does not manage run lifecycle
- **WHEN** maintainers inspect `WandbArtifactWriter` or `WandbCheckpointWriter`
- **THEN** those writers MUST NOT call `wandb.init`
- **THEN** those writers MUST NOT call `wandb.finish` or `run.finish`
- **THEN** those writers MUST publish only through the current logger-owned run

### Requirement: W&B artifact lineage callback SHALL record only on logger-owned runs

`WandbArtifactLineageCallback` SHALL record resolved upstream Artifact usage on the current logger-owned W&B run. It MUST NOT create, configure, finish, or replace a W&B run.

#### Scenario: Lineage records through logger-owned run
- **WHEN** a configured experiment resolves one or more `wandb://` input references
- **AND** `WandbArtifactLineageCallback` is enabled
- **AND** the trainer has a configured W&B logger whose `experiment` provides a run
- **THEN** the callback MUST call `use_artifact` on that logger-owned run for each resolved upstream Artifact

#### Scenario: Missing logger-owned run follows lineage failure policy
- **WHEN** resolved W&B artifact references exist
- **AND** `WandbArtifactLineageCallback` is enabled
- **AND** the trainer does not expose a configured W&B logger whose `experiment` provides a run
- **THEN** the callback MUST NOT call `wandb.init`
- **THEN** the callback MUST raise when `fail_on_missing_run` is true
- **THEN** the callback MUST warn and skip lineage recording when `fail_on_missing_run` is false

### Requirement: W&B writer and lineage behavior SHALL remain explicit

W&B logger configuration SHALL NOT automatically enable W&B artifact writers or lineage recording. Writers and lineage callbacks SHALL remain explicit callbacks, but when configured they MUST use the logger-owned run.

#### Scenario: Logger alone does not publish artifacts
- **WHEN** an experiment configures `WandbLogger`
- **AND** it does not configure a W&B writer callback
- **THEN** the experiment MUST NOT publish output Artifacts solely because the logger exists

#### Scenario: Callback presence requires logger-owned run
- **WHEN** an experiment configures a W&B writer callback or `WandbArtifactLineageCallback`
- **THEN** that callback MUST rely on the configured W&B logger-owned run
- **THEN** lifecycle ownership MUST remain with the logger and launcher finalization path

