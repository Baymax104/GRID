## ADDED Requirements

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
