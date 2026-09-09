# Keyed Prediction Bundle Artifact

## Purpose

定义 keyed prediction bundle 的持久化与加载协议，确保推理阶段导出的 item embedding、semantic IDs 等预测结果能够通过业务主键稳定查询，同时为模型侧提供明确的 semantic ID tensor 加载路径。
## Requirements
### Requirement: keyed 预测产物 SHALL 以单文件 bundle 形式保存
凡是由推理阶段导出的 keyed predictions（例如 item embedding、semantic IDs），`merged_predictions_tensor.pt` MUST 保存为单文件 bundle，而非可直接按业务主键索引的裸 tensor。

#### Scenario: bundle 结构统一
- **WHEN** 推理回调将 keyed predictions 落盘为 `merged_predictions_tensor.pt`
- **THEN** 文件内容 MUST 为一个包含 `keys` 与 `predictions` 字段的对象
- **THEN** `keys` 与 `predictions` 的第一维长度 MUST 一致

#### Scenario: 文件名保留但语义升级
- **WHEN** 维护者或脚本引用 `.../pickle/merged_predictions_tensor.pt`
- **THEN** 路径约定 MAY 保持不变
- **BUT** 文件内容 MUST NOT 再被视为“item_id 可直接索引的裸 tensor”

### Requirement: keyed prediction bundle SHALL NOT 假设业务主键等于 tensor 行号
推理产物协议不得依赖业务主键（如 `item_id`）是从 0 开始且连续的数组下标。加载时 SHALL 按 key 值排序 keys 和 predictions，查询时 SHALL 使用 `torch.searchsorted` 二分查找行号，SHALL NOT 使用 Python dict 运行时索引。加载时 SHALL 检测重复 key 并 raise。keyed prediction bundle 的加载与查询函数 SHALL 位于 `src.data.utils`。

#### Scenario: 非连续 item IDs 仍可成功导出
- **WHEN** 推理结果中的 `keys` 包含非连续或大于样本数的 item IDs
- **THEN** 导出 `merged_predictions_tensor.pt` MUST 成功
- **THEN** 导出逻辑 MUST NOT 因"key 超出样本数"而触发越界异常

#### Scenario: 加载时按 key 排序
- **WHEN** `load_model_output` 加载 `.pt` 文件
- **THEN** 返回的 bundle 中 `keys` MUST 按 key 值升序排列
- **THEN** `predictions` 的行顺序 MUST 与排序后的 `keys` 一一对应
- **THEN** `load_model_output` MUST 从 `src.data.utils` 导入

#### Scenario: 查询使用 searchsorted
- **WHEN** `gather_predictions_by_keys` 查找 key 对应的 predictions
- **THEN** MUST 使用 `torch.searchsorted` 在 `keys` tensor 上二分查找行号
- **THEN** MUST NOT 使用 Python dict 或 `key_to_index` 运行时字段
- **THEN** `gather_predictions_by_keys` MUST 从 `src.data.utils` 导入

#### Scenario: 重复 key 检测
- **WHEN** `load_model_output` 加载的 `keys` 包含重复值
- **THEN** MUST raise ValueError

#### Scenario: 查询不存在的 key
- **WHEN** `gather_predictions_by_keys` 的 `lookup_keys` 包含 `keys` 中不存在的值
- **THEN** MUST raise KeyError

### Requirement: keyed semantic ID bundle SHALL provide model-side tensor extraction
当 keyed prediction bundle 保存 semantic IDs 时，系统 SHALL 提供模型侧 tensor extraction 路径，从 bundle 的 `predictions` 中获得 semantic ID tensor，同时保留数据侧按 key 查询的完整 bundle 协议。semantic ID tensor extraction 函数 SHALL 位于 `src.data.utils`。

#### Scenario: 模型侧加载 semantic ID tensor
- **WHEN** TIGER 模型配置从 `semantic_id_path` 加载 prefix 校验数据
- **THEN** 加载结果 MUST 是 semantic ID tensor，而不是完整 `ModelOutput` bundle
- **THEN** tensor 第一维 MUST 对应 item 行，第二维 MUST 对应 semantic ID hierarchy
- **THEN** `load_semantic_id_tensor` MUST 从 `src.data.utils` 导入

#### Scenario: 数据侧仍使用完整 bundle 查询
- **WHEN** TIGER 数据 preprocessing 执行 `item_id -> semantic_id` 映射
- **THEN** 数据侧 MUST 继续使用包含 `keys` 与 `predictions` 的完整 keyed bundle
- **THEN** 数据侧 lookup MUST 保持通过 key 查询而不是假设 item id 等于 tensor 行号

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

### Requirement: W&B artifact downloads SHALL be rank-zero-only when distributed ranks share cache

When torch distributed is initialized, W&B artifact resolution SHALL download artifacts only on rank 0 and SHALL have other ranks resolve files from the same deterministic local download root after synchronization.

#### Scenario: Non-distributed download preserves current behavior
- **WHEN** W&B artifact resolution runs without initialized torch distributed
- **THEN** the resolver MUST call `artifact.download(...)`
- **THEN** the resolver MUST resolve the requested artifact file from the returned local directory

#### Scenario: Rank zero performs distributed download
- **WHEN** W&B artifact resolution runs with initialized torch distributed on rank 0
- **THEN** rank 0 MUST call `artifact.download(...)`
- **THEN** rank 0 MUST synchronize with other ranks before returning the resolved file path

#### Scenario: Non-zero rank waits for downloaded cache
- **WHEN** W&B artifact resolution runs with initialized torch distributed on a non-zero rank
- **THEN** the rank MUST NOT call `artifact.download(...)`
- **THEN** the rank MUST synchronize with rank 0
- **THEN** the rank MUST resolve the requested artifact file from the deterministic download root

#### Scenario: Missing shared cache is reported
- **WHEN** a non-zero rank cannot find the requested file under the deterministic download root after synchronization
- **THEN** the resolver MUST raise a file-not-found error that includes the requested file and local directory

### Requirement: W&B prediction artifacts SHALL use keyed prediction bundles

Prediction artifacts published by `WandbArtifactWriter` SHALL contain the merged keyed prediction bundle generated by that writer. The published file MUST preserve the same `merged_predictions_tensor.pt` bundle contract used by local prediction outputs.

#### Scenario: W&B artifact contains keyed bundle
- **WHEN** `WandbArtifactWriter` publishes an inference artifact
- **THEN** the artifact MUST include a file named `merged_predictions_tensor.pt`
- **THEN** that file MUST contain an object with `keys` and `predictions` fields
- **THEN** `keys` and `predictions` MUST have matching first-dimension lengths

#### Scenario: W&B artifact metadata records writer output
- **WHEN** `WandbArtifactWriter` publishes an inference artifact
- **THEN** artifact metadata MUST include the artifact `role`
- **THEN** artifact metadata MUST include `task_name`
- **THEN** artifact metadata MUST include the W&B writer's local output path
- **THEN** artifact metadata MUST include the bundle file name

