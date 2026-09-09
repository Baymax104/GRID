## MODIFIED Requirements

### Requirement: Training checkpoint references SHALL support optional W&B run URIs
训练 checkpoint 输入字段 `ckpt_path` SHALL 保持本地路径兼容，并 MAY 使用 `wandb://<run-id>` 引用 producer run 产出的 checkpoint Artifact。该 URI 解析和 Artifact 下载 SHALL 复用 `src/data/components/artifacts.py` 中的 resolver，并 MUST 发生在 Trainer 使用 checkpoint 前。Short W&B URI resolution MUST use explicit experiment-provided `user/project` defaults.

#### Scenario: Resume or inference checkpoint from W&B run URI
- **WHEN** 训练恢复或推理配置使用 `ckpt_path=wandb://8b61h7ly`
- **AND** experiment config declares top-level `user` and `project`
- **THEN** `src/main.py` MUST pass `default_entity=${user}` and `default_project=${project}` to the checkpoint resolver
- **THEN** `src/data/components/artifacts.py` 中的 resolver MUST 查找 run `8b61h7ly` 的 checkpoint Artifact
- **THEN** `src/data/components/artifacts.py` 中的 resolver MUST 调用 `src/utils/wandb.py` 下载并解析到具体本地 checkpoint 文件路径
- **THEN** Trainer 接收到的 `ckpt_path` MUST 是可由 Lightning 加载的本地文件路径

#### Scenario: Short checkpoint URI without experiment user fails
- **WHEN** training or inference config uses `ckpt_path=wandb://8b61h7ly`
- **AND** the URI does not include entity/project
- **AND** experiment config does not provide top-level `user`
- **THEN** checkpoint resolution MUST fail before Trainer receives `ckpt_path`
- **THEN** checkpoint resolution MUST NOT infer entity from environment variables or W&B account defaults

#### Scenario: Local checkpoint path remains compatible
- **WHEN** `ckpt_path` 是本地 checkpoint 文件或 checkpoint 目录
- **THEN** 当前本地路径解析和 latest checkpoint 查找行为 MUST 保持兼容
- **THEN** 系统 MUST NOT 为本地 checkpoint 路径调用 W&B Artifact resolver
