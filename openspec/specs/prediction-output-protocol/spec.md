# prediction-output-protocol Specification

## Purpose
TBD - created by archiving change simplify-prediction-output-protocol. Update Purpose after archive.
## Requirements
### Requirement: ModelOutput SHALL 直接持有 keys + predictions tensor

`ModelOutput` 作为 keyed prediction output 的字段规范层，SHALL 定义于 `src.data.components.data_models`，并直接持有 `keys` 和 `predictions` 两个属性，SHALL NOT 携带 `key_name`/`prediction_name` 等字段命名映射参数。`prediction_step` SHALL 直接构造 `ModelOutput(keys=..., predictions=...)`，使写入数据在实现中一目了然。

#### Scenario: prediction_step 直接构造 ModelOutput
- **WHEN** 任何 inference module 的 `predict_step` 产出预测结果
- **THEN** 返回值 MUST 为 `ModelOutput(keys=<业务主键>, predictions=<预测值>)`
- **THEN** `ModelOutput` MUST NOT 接受 `key_name` 或 `prediction_name` 参数
- **THEN** `ModelOutput` MUST 从 `src.data.components.data_models` 导入

#### Scenario: ModelOutput 不提供行格式转换
- **WHEN** 维护者检查 `ModelOutput` 的接口
- **THEN** `ModelOutput` MUST NOT 提供 `list_of_row_format` 属性或 `_convert_to_list` 方法

### Requirement: Writer SHALL 直接缓存和合并 tensor，不经过行格式中间层

`LocalPickleWriter` SHALL 直接缓存 `ModelOutput` 对象（而非行格式 `list[dict]`），在合并时 SHALL 直接对 `keys` 和 `predictions` 执行 `torch.cat` 产出 keyed prediction bundle。Writer SHALL NOT 持有 `prediction_key_name`/`prediction_name` 配置参数，SHALL NOT 在 `setup()` 中向 module 注入字段名。

#### Scenario: writer 缓存 ModelOutput 对象
- **WHEN** `write_on_batch_end` 接收到 `ModelOutput` 预测结果
- **THEN** writer MUST 将 `ModelOutput` 对象直接追加到缓冲区
- **THEN** writer MUST NOT 调用 `list_of_row_format` 或任何行格式转换

#### Scenario: writer 合并时直接拼接 tensor
- **WHEN** rank 0 在 `on_predict_end` 中执行合并
- **THEN** writer MUST 对所有 `ModelOutput.keys` 执行 `torch.cat` 产出 `keys` tensor
- **THEN** writer MUST 对所有 `ModelOutput.predictions` 执行 `torch.cat` 产出 `predictions` tensor
- **THEN** writer MUST 保存 `{"keys": keys, "predictions": predictions}` 到 `merged_predictions_tensor.pt`

#### Scenario: writer 不产出行格式 pkl
- **WHEN** 推理完成后检查输出目录
- **THEN** 目录中 MUST NOT 存在 `merged_predictions.pkl` 文件

#### Scenario: flush_frequency 保持行数语义
- **WHEN** `handle_batch` 判断是否需要 flush 缓冲区
- **THEN** 判断依据 MUST 为缓冲区中累计的样本数（各 `ModelOutput.keys` 长度之和）
- **THEN** `flush_frequency` MUST NOT 表示 batch 数

### Requirement: Writer SHALL NOT 依赖模块的 prediction_key_name/prediction_name 属性

`BaseBufferedWriter` SHALL NOT 在 `setup()` 中读取或设置 `pl_module.prediction_key_name` 或 `pl_module.prediction_name`。字段命名映射的职责 SHALL 完全由 `prediction_step` 承担。

#### Scenario: writer setup 不注入字段名
- **WHEN** `BaseBufferedWriter.setup()` 被调用
- **THEN** MUST NOT 访问 `pl_module.prediction_key_name` 或 `pl_module.prediction_name`
- **THEN** MUST NOT 向 `pl_module` 设置任何属性

### Requirement: Prediction writer SHALL be batch-only and independent from Lightning batch index inference

`LocalPickleWriter` SHALL operate as a batch-only Lightning callback that consumes `predict_step` outputs directly. It SHALL NOT require Lightning `BasePredictionWriter` interval handling, and SHALL NOT depend on inferred `batch_indices` from the dataloader.

#### Scenario: writer handles prediction batch outputs directly
- **WHEN** Lightning finishes a prediction batch
- **THEN** `LocalPickleWriter` MUST consume the batch `ModelOutput` from `on_predict_batch_end`
- **THEN** `LocalPickleWriter` MUST buffer and flush predictions using the existing sample-count `flush_frequency` semantics

#### Scenario: writer does not request dataloader batch indices
- **WHEN** inference uses an `IterableDataset` through `DataloaderWithIterationRetry`
- **THEN** `LocalPickleWriter` MUST NOT trigger Lightning `BasePredictionWriter` batch index inference
- **THEN** inference MUST NOT emit a writer-induced warning about inability to infer batch indices from `DataloaderWithIterationRetry`

#### Scenario: writer exposes no epoch interval configuration
- **WHEN** maintainers inspect inference callback configs
- **THEN** configs MUST NOT declare `write_interval`
- **THEN** `LocalPickleWriter` MUST NOT expose an epoch or batch-and-epoch writing mode

#### Scenario: prediction completion still merges outputs
- **WHEN** prediction completes
- **THEN** rank-local buffered outputs MUST be flushed
- **THEN** rank 0 MUST merge pickle shards into `merged_predictions_tensor.pt`
- **THEN** configured post-processing functions MUST still run on rank 0 after merge

### Requirement: Prediction artifact writing SHALL be separate from local bundle writing
推理 Artifact 发布 SHALL 由与 `LocalPickleWriter` 平级的 W&B writer 承担，二者 SHOULD 位于 `src/common/writers/`，MUST 分别放在独立 writer 文件中，MUST NOT 保留旧 `src/common/inference/` 兼容导入，MUST NOT 改变 `LocalPickleWriter` 的本地 batch 缓存、flush、merge、post-processing 职责，且 MUST NOT 依赖 `LocalPickleWriter` 产出的本地 bundle。

#### Scenario: Local writer works without W&B writer
- **WHEN** 推理实验使用 `LocalPickleWriter`
- **AND** 未启用 W&B Artifact 发布
- **THEN** writer MUST 继续写入 `${paths.output_dir}/pickle/merged_predictions_tensor.pt`
- **THEN** 推理流程 MUST NOT 要求存在 W&B logger 或 W&B Artifact writer

#### Scenario: WandbArtifactWriter publishes its own merged output
- **WHEN** 推理实验启用 W&B Artifact 发布
- **THEN** `WandbArtifactWriter` MUST 独立 flush、merge、post-process 自己的 prediction shards
- **THEN** `WandbArtifactWriter` MUST 发布自己合并后的 model output bundle

#### Scenario: W&B logger does not imply W&B writer
- **WHEN** 推理或训练实验配置了 W&B logger
- **AND** 未显式启用 W&B Artifact 发布
- **THEN** 系统 MUST NOT 自动发布推理 output Artifact
- **THEN** 系统 MUST NOT 要求使用 W&B-specific writer

#### Scenario: W&B writer is not paired to LocalPickleWriter by type
- **WHEN** `WandbArtifactWriter` 和 `LocalPickleWriter` 同时启用
- **THEN** `WandbArtifactWriter` MUST 使用自己的 `output_dir` 产出待发布 bundle
- **THEN** 它 MUST NOT 通过 Python 类型检查或实例引用强制要求同一实验存在 `LocalPickleWriter`

### Requirement: WandbArtifactWriter SHALL be an independent prediction writer

`WandbArtifactWriter` SHALL consume `ModelOutput` directly during prediction and SHALL independently flush, merge, post-process, and publish prediction outputs as W&B Artifacts. It SHALL NOT require `LocalPickleWriter`, SHALL NOT read a `source_path` produced by another writer, and SHALL NOT expose a `source_path` configuration parameter.

#### Scenario: W&B writer handles prediction batch outputs directly
- **WHEN** Lightning finishes a prediction batch and `WandbArtifactWriter` is enabled
- **THEN** `WandbArtifactWriter` MUST consume the batch `ModelOutput` from `on_predict_batch_end`
- **THEN** `WandbArtifactWriter` MUST buffer and flush predictions using sample-count `flush_frequency` semantics

#### Scenario: W&B writer merges its own shards
- **WHEN** prediction completes
- **THEN** `WandbArtifactWriter` MUST flush rank-local buffered outputs
- **THEN** rank 0 MUST merge only the shard files in the W&B writer's own `output_dir`
- **THEN** rank 0 MUST save `merged_predictions_tensor.pt` in the W&B writer's own `output_dir`

#### Scenario: W&B writer does not depend on local writer output
- **WHEN** `LocalPickleWriter` is disabled and `WandbArtifactWriter` is enabled
- **THEN** prediction completion MUST NOT require `${paths.output_dir}/pickle/merged_predictions_tensor.pt`
- **THEN** W&B publishing MUST use the bundle produced by `WandbArtifactWriter`

#### Scenario: W&B writer and local writer can coexist
- **WHEN** `LocalPickleWriter` and `WandbArtifactWriter` are both enabled
- **THEN** each writer MUST write temporary shards and merged bundles under its own configured `output_dir`
- **THEN** neither writer MUST read, delete, or post-process the other writer's files

### Requirement: W&B prediction artifact writer SHALL require a WandbLogger-owned run

`WandbArtifactWriter` SHALL publish inference output Artifacts only to the W&B run owned by the configured Lightning `WandbLogger`. It MUST NOT create, configure, finish, or otherwise manage a W&B run.

#### Scenario: Inference writer publishes through logger-owned run
- **WHEN** an inference experiment configures `WandbArtifactWriter`
- **AND** the trainer has a configured W&B logger whose `experiment` provides a run
- **THEN** `WandbArtifactWriter` MUST publish its merged model output bundle to that logger-owned run
- **THEN** the published Artifact metadata MUST include `role`, `task_name`, `local_output_path`, and `bundle_file`

#### Scenario: Missing logger-owned run fails
- **WHEN** `WandbArtifactWriter` reaches artifact publishing
- **AND** the trainer does not expose a configured W&B logger whose `experiment` provides a run
- **THEN** publishing MUST fail with a clear error naming the missing W&B logger run
- **THEN** `WandbArtifactWriter` MUST NOT call `wandb.init`

#### Scenario: Publish failure is not downgraded
- **WHEN** `WandbArtifactWriter` fails to create or log a W&B Artifact
- **THEN** the exception MUST propagate
- **THEN** `WandbArtifactWriter` MUST NOT provide a `fail_on_error` option
- **THEN** `WandbArtifactWriter` MUST NOT log a warning and continue as if W&B output succeeded

#### Scenario: Writer configuration excludes run lifecycle fields
- **WHEN** maintainers inspect official inference callback configs
- **THEN** `wandb_artifact_writer` MUST NOT declare `project`, `entity`, `group`, `run_name`, `job_type`, `tags`, `notes`, `mode`, `finish_run`, or `fail_on_error`
- **THEN** W&B run identity MUST be configured through the experiment's logger config

### Requirement: Runtime prediction outputs MAY carry non-persistent auxiliary payloads
`ModelOutput` SHALL allow an optional named auxiliary payload for prediction callbacks while preserving `keys` and `predictions` as the only fields of the standard keyed prediction bundle. Existing callers that provide only keys and predictions MUST remain valid.

#### Scenario: Ordinary prediction output is created
- **WHEN** a model constructs `ModelOutput(keys, predictions)` without auxiliary data
- **THEN** existing writer and consumer behavior MUST remain unchanged

#### Scenario: Trace-enabled prediction output is created
- **WHEN** TIGER attaches a named Prefix Trace tensor payload
- **THEN** a dedicated auxiliary writer MUST be able to select that payload
- **AND** standard prediction writers MUST ignore it when producing `merged_predictions_tensor.pt`

### Requirement: Auxiliary tensor writers SHALL remain domain neutral
Shared auxiliary output writers SHALL operate on named tensor payloads and generic schema metadata without importing TIGER or Tail-SID diagnosis modules.

#### Scenario: Prefix trace payload is persisted
- **WHEN** a shared auxiliary writer receives the configured trace payload name
- **THEN** it MUST merge and persist the generic keyed tensor structure
- **AND** it MUST NOT branch on TIGER-specific metric names
