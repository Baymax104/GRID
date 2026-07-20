# tensorflow-free-tfrecord-loading Specification

## Purpose
TBD - created by archiving change remove-tensorflow-runtime. Update Purpose after archive.
## Requirements
### Requirement: TFRecord loading SHALL not require TensorFlow runtime
系统必须能够在不依赖 TensorFlow 运行时的前提下读取和解析 `.tfrecord.gz` 数据。

#### Scenario: Load compressed TFRecord without TensorFlow
- **WHEN** 官方 experiment 读取 `.tfrecord.gz` 数据文件
- **THEN** 数据读取与 example 解析不得依赖 `tensorflow` 模块

### Requirement: Parsed TFRecord output SHALL remain compatible with existing preprocessing pipeline
新的 TFRecord 解析输出必须保持对现有预处理链路的兼容性。

#### Scenario: Item/text preprocessing receives parsed row
- **WHEN** `sem_embeds_inference_flat` 的预处理函数消费解析后的单条样本
- **THEN** 现有预处理步骤必须仍可生成 `text_tokens`、`text_mask` 与 `item_ids` 所需输入结构

#### Scenario: Sequence preprocessing receives parsed row
- **WHEN** `tiger_train_flat` 的 sequence 预处理函数消费解析后的单条样本
- **THEN** 现有 semantic id 映射、padding、label masking 链路必须继续可用

### Requirement: Runtime code SHALL not import TensorFlow
运行时代码中不得继续保留 `import tensorflow` 依赖。

#### Scenario: Inspect runtime data loading modules
- **WHEN** 开发者查看运行时数据读取与预处理模块
- **THEN** 这些模块中不得出现 `import tensorflow as tf`

### Requirement: Official pipelines SHALL preserve downstream model-input contract
替换 TensorFlow 后，官方 pipeline 必须保持模型前输入契约不变。

#### Scenario: Quantization pipeline consumes mapped embeddings
- **WHEN** `rkmeans_train_flat` 进入模型前阶段
- **THEN** 量化模型仍必须收到与当前链路兼容的 `ItemData.transformed_features["input_embedding"]`

#### Scenario: Recommendation pipeline consumes sequential model input
- **WHEN** `tiger_train_flat` 进入模型前阶段
- **THEN** 推荐模型仍必须收到与当前链路兼容的 `SequentialModelInputData` 与 `SequentialModuleLabelData`

