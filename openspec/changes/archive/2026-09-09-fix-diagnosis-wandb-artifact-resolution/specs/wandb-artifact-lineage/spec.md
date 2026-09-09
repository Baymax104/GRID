## MODIFIED Requirements

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
