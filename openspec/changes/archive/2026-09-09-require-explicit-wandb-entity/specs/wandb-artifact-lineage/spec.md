## MODIFIED Requirements

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
