## MODIFIED Requirements

### Requirement: tiger sequence 实验 SHALL 使用新 data contract
tiger_train、tiger_inference 这类 sequence 级生成式推荐实验 SHALL 使用新的 data contract，包括 reader factory、配置直写 preprocessing chain、precomputed semantic_id 基于 keyed bundle 局部参数注入、新 shuffle 语义，以及推理 id 字段与模型输入序列分离。

#### Scenario: sequence 实验 data_reader 采用 factory 形式
- **WHEN** 维护者查看 tiger_train / tiger_inference 的数据读取配置
- **THEN** `data_reader` MUST 采用 factory / `_partial_` 形式，由 dataset 运行时按 worker 文件列表实例化

#### Scenario: sequence 实验 preprocessing chain 配置直写
- **WHEN** 维护者查看上述实验的 data 配置
- **THEN** preprocessing 的顺序与参数 MUST 直接以 `preprocessing_functions` list 形式声明在配置中，不通过 Hydra resolver 派生

#### Scenario: sequence 实验 semantic_id 注入基于 keyed bundle 局部参数
- **WHEN** 维护者查看上述实验的 semantic_id 注入配置
- **THEN** semantic_id MUST 通过 `map_sparse_id_to_semantic_id(semantic_id_bundle=...)` 的局部 `semantic_id_bundle` 参数注入，而不是通过 dataset config 的 `semantic_id_map` 字段

#### Scenario: sequence 实验使用新 shuffle contract
- **WHEN** 维护者查看上述实验的 shuffle 配置
- **THEN** shuffle 语义 MUST 通过 `shuffle_files`（dataset config）与 `shuffle_rows`（reader factory）表达，而不是继续依赖 dataloader 的 `should_shuffle_rows`

#### Scenario: sequence 实验 dataset config 不携带旧派生字段
- **WHEN** 维护者检查上述实验的 dataset config
- **THEN** 它 MUST 不再携带 `features_to_consider`、`semantic_id_map`、`num_placeholder_tokens_map`、`field_type_map`、`keep_user_id`、`user_id_field`、`min_sequence_length`、`feature_map` 等旧协议字段

#### Scenario: TIGER training data excludes user identity model features
- **WHEN** 维护者检查 `tiger_train` 的 data 和 model 配置
- **THEN** training preprocessing MUST NOT retain `user_id` as a model input feature
- **AND** model input mapping MUST NOT map `user_id` into TIGER forward or generation arguments

#### Scenario: TIGER inference id field is output metadata only
- **WHEN** `collate_fn_inference_for_sequence` receives a field matching `id_field_name`
- **THEN** it MUST store that field in `SequentialModelInputData.user_id_list`
- **AND** it MUST NOT store that field in `SequentialModelInputData.transformed_sequences`
- **AND** attention masks MUST be computed from non-id sequence fields

### Requirement: sequence 链路 config 类 SHALL 精简为运行时字段
`SemanticIDDatasetConfig` 和 `SequenceDataloaderConfig` SHALL 只暴露运行时消费的字段，删除旧架构遗留字段和无消费方的兼容字段。

#### Scenario: SemanticIDDatasetConfig 不保留旧协议字段
- **WHEN** 维护者检查 `SemanticIDDatasetConfig` 定义
- **THEN** 它 MUST 不再保留 `semantic_id_map`、`keep_user_id`、`user_id_field`、`features_to_consider`、`num_placeholder_tokens_map`、`field_type_map`、`min_sequence_length`、`feature_map`、`file_format`，且 `SequenceDatasetConfig` 基类 MUST 被删除

#### Scenario: SequenceDataloaderConfig 不保留旧 shuffle 字段
- **WHEN** 维护者检查 `SequenceDataloaderConfig` 定义
- **THEN** 它 MUST 不再保留 `should_shuffle_rows`，shuffle 语义只通过 `dataset_config.shuffle_files` + reader `shuffle_rows` 表达

### Requirement: map_sparse_id_to_semantic_id SHALL 接收局部 bundle 参数
`map_sparse_id_to_semantic_id` SHALL 接收局部 `semantic_id_bundle` 参数，不接收 `dataset_config`。

#### Scenario: 函数签名不包含 dataset_config
- **WHEN** 维护者检查 `map_sparse_id_to_semantic_id` 的函数签名
- **THEN** 它 MUST 接收 `semantic_id_bundle` 作为局部参数，且 MUST NOT 接收 `dataset_config`

#### Scenario: 函数从局部参数获取 bundle
- **WHEN** 函数执行 semantic_id lookup
- **THEN** 它 MUST 直接使用 `semantic_id_bundle` 参数调用 `lookup_values_in_keyed_prediction_bundle`，不通过 `dataset_config.semantic_id_map` 间接获取
