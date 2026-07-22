## MODIFIED Requirements

### Requirement: tiger sequence 实验 SHALL 使用新 data contract
tiger_train、tiger_inference 这类 sequence 级生成式推荐实验 SHALL 使用新的 data contract，包括 reader factory、配置直写 preprocessing chain、precomputed semantic_id 基于 keyed bundle 局部参数注入、新 shuffle 语义、推理 id 字段与模型输入序列分离、sequence length 在 preprocessing 配置处显式声明、label generator 以 preprocessing 纯函数声明、TIGER 专用 batch contract，以及统一且纯拼装的 TIGER sequence collate entry point。

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
- **WHEN** `collate_fn_sequence` receives a field matching `output_key_field_name`
- **THEN** it MUST store that field in `TigerModelInput.output_keys`
- **AND** it MUST NOT store that field in `TigerModelInput.input_ids`
- **AND** attention masks MUST come from the configured preprocessed attention mask field

#### Scenario: TIGER train/eval sequence length is local to preprocessing
- **WHEN** 维护者检查 `tiger_train` data 配置
- **THEN** `sequence_length` MUST be declared on the `normalize_sequence` preprocessing callable
- **AND** train/val/test collate blocks MUST NOT declare `sequence_length`
- **AND** train/val/test dataloader blocks MUST NOT 作为 collate 参数注入来源继续声明 `sequence_length`

#### Scenario: TIGER inference sequence length is local to preprocessing
- **WHEN** 维护者检查 `tiger_inference` data 配置
- **THEN** inference sequence normalization MUST be declared as `normalize_sequence` preprocessing
- **AND** inference collate blocks MUST NOT declare `sequence_length` or `padding_token`
- **AND** predict dataloader blocks MUST NOT 作为 collate 参数注入来源继续声明这些字段

#### Scenario: TIGER label generators are preprocessing functions
- **WHEN** 维护者检查 `tiger_train` 的 preprocessing 配置
- **THEN** label generation MUST be declared in the preprocessing chain as a Hydra `_partial_` callable
- **AND** collate blocks MUST NOT configure label generator callables

#### Scenario: TIGER input normalization is a preprocessing function
- **WHEN** 维护者检查 `tiger_train` 的 preprocessing 配置
- **THEN** input normalization MUST be declared after label generation as `normalize_sequence`
- **AND** collate blocks MUST NOT configure `sequence_length`

#### Scenario: TIGER sequence collate returns TIGER-specific dataclasses
- **WHEN** TIGER train or inference collate functions produce a batch
- **THEN** they MUST return `TigerModelInput` for model inputs
- **AND** train/eval/test collate MUST return `TigerLabelData` for labels
- **AND** they MUST NOT return legacy generic sequential batch dataclasses

#### Scenario: TIGER train, eval, and inference collate use unified sequence entry point
- **WHEN** 维护者检查 `tiger_train` 或 `tiger_inference` 的 collate 配置
- **THEN** train, eval, and inference collate blocks MUST target `src.data.components.collate.collate_fn_sequence`
- **AND** training augmentation MUST be declared in train preprocessing rather than collate configuration
- **AND** input normalization MUST be declared in train/eval preprocessing rather than collate configuration
