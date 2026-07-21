# item-quantization-data-contract Specification

## Purpose
TBD - created by archiving change migrate-item-chain-experiments-data-pipeline. Update Purpose after archive.
## Requirements
### Requirement: item 级量化实验 SHALL 使用新 data contract
rvq_train、rqvae_train、rkmeans_inference 这类 item 级量化实验 SHALL 使用新的 data contract，包括 reader factory、配置直写 preprocessing chain、precomputed embedding 基于 keyed bundle lookup，以及新 shuffle 语义。rkmeans_train 已先行迁移并作为参照样板。

#### Scenario: 量化实验 data_reader 采用 factory 形式
- **WHEN** 维护者查看 rvq_train / rqvae_train / rkmeans_inference 的数据读取配置
- **THEN** `data_reader` MUST 采用 factory / `_partial_` 形式，由 dataset 运行时按 worker 文件列表实例化

#### Scenario: 量化实验 preprocessing chain 配置直写
- **WHEN** 维护者查看上述实验的 data 配置
- **THEN** preprocessing 的顺序与参数 MUST 直接以 `preprocessing_functions` list 形式声明在配置中，不通过 Hydra resolver 派生

#### Scenario: 量化实验 embedding 注入基于 keyed bundle 局部参数
- **WHEN** 维护者查看上述实验的 embedding 注入配置
- **THEN** embedding MUST 通过 `map_sparse_id_to_embedding(embedding_bundle=...)` 的局部 `embedding_bundle` 参数注入，而不是通过 dataset config 的 `embedding_map` 字段

#### Scenario: 量化实验使用新 shuffle contract
- **WHEN** 维护者查看上述实验的 shuffle 配置
- **THEN** shuffle 语义 MUST 通过 `shuffle_files`（dataset config）与 `shuffle_rows`（reader factory）表达，而不是继续依赖 dataloader 的 `should_shuffle_rows`

#### Scenario: 量化实验 dataset config 不携带旧派生字段
- **WHEN** 维护者检查上述实验的 dataset config
- **THEN** 它 MUST 不再携带 `features_to_consider`、`embedding_map`、`num_placeholder_tokens_map`、`field_type_map` 等旧协议字段，相关参数下沉到 `preprocessing_functions` 各步骤

