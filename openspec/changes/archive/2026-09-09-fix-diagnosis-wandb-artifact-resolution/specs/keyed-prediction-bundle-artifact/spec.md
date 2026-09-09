## MODIFIED Requirements

### Requirement: keyed prediction bundle references SHALL support optional W&B run URIs
keyed prediction bundle 的引用字段 SHALL 保持本地路径兼容，并 MAY 使用 `wandb://<run-id>` 指向 producer run 产出的 keyed prediction bundle Artifact。该解析 SHALL 发生在 `src/data/components/artifacts.py` 的 bundle 读取入口。该能力 MUST NOT 改变 `merged_predictions_tensor.pt` 的文件内容协议。Short W&B URI resolution MUST use explicit experiment-provided `user/project` defaults passed to artifact loaders.

#### Scenario: semantic ID bundle loaded from W&B run URI
- **WHEN** TIGER 数据 preprocessing 使用 `semantic_id_path=wandb://1mzveep4`
- **AND** artifact loader 配置传入 `wandb_entity: ${user}` 和 `wandb_project: ${project}`
- **THEN** `src/data/components/artifacts.py` MUST 将该引用解析为本地 `merged_predictions_tensor.pt`
- **THEN** `load_model_output` MUST 按既有 keyed bundle 协议加载 `keys` 与 `predictions`
- **THEN** 数据侧 lookup MUST 继续通过 key 查询

#### Scenario: embedding bundle loaded from W&B run URI
- **WHEN** 量化训练或推理配置使用 `embedding_path=wandb://abc1234`
- **AND** artifact loader 配置传入 `wandb_entity: ${user}` 和 `wandb_project: ${project}`
- **THEN** `src/data/components/artifacts.py` MUST 将该引用解析为 semantic embedding bundle Artifact 文件

#### Scenario: Diagnosis resolves bundle roles before dataset loading
- **WHEN** Tail-SID diagnosis 使用短 W&B Semantic ID 引用和可选 embedding 引用
- **AND** diagnosis DataModule 配置接收实验 `user/project`
- **THEN** DataModule MUST 分别使用 `semantic_id_path` 与 `embedding_path` 字段名调用统一 resolver
- **THEN** resolver MUST 分别选择 `semantic_id` 与 `semantic_embedding` Artifact role
- **THEN** Dataset MUST 继续通过解析后的本地路径读取既有 keyed bundle

#### Scenario: Short bundle URI without experiment user fails
- **WHEN** `load_model_output` receives `wandb://abc1234`
- **AND** the URI does not include entity/project
- **AND** artifact loader config does not provide `wandb_entity`
- **THEN** resolution MUST fail before selecting or downloading an Artifact

#### Scenario: Existing bundle loader ownership is moved explicitly
- **WHEN** 实现迁移 `load_model_output` 和 `load_semantic_id_tensor`
- **THEN** 这些函数 SHOULD 位于 `src/data/components/artifacts.py`
- **THEN** 旧调用方 MUST 被更新到新导入路径，或由 `src/data/utils.py` 提供明确的兼容 re-export
- **THEN** bundle 的 `keys` 与 `predictions` 校验规则 MUST 与本地路径一致

#### Scenario: Bundle content protocol remains unchanged
- **WHEN** W&B Artifact 中包含 `merged_predictions_tensor.pt`
- **THEN** 文件内容 MUST 仍为包含 `keys` 和 `predictions` 的单文件 bundle
- **THEN** Artifact metadata MUST NOT 替代 bundle 内部 `keys` 与 `predictions` 数据
