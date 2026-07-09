## Why

tiger_train 和 tiger_inference 两个 sequence 链路实验仍使用旧 data 配置模式：`data_reader` 直接实例化（非 factory）、preprocessing 通过分散命名引用 + Hydra resolver 派生拼装、semantic_id 经 `dataset_config.semantic_id_map` 注入、shuffle 经 dataloader 的 `should_shuffle_rows` 控制。`map_sparse_id_to_semantic_id` 仍接收整个 `dataset_config`（违反 preprocessing-minimal-parameter-contract）。此外 `SequenceDatasetConfig` 携带 7 个遗留字段无运行时消费方，`SequenceDataloaderConfig` 残留 `should_shuffle_rows` 旧字段。需要将这两个实验迁移到新 data contract，并完成 sequence 链路 config 类的清理。

## What Changes

- 重构 `map_sparse_id_to_semantic_id`：从接收 `dataset_config` 改为接收局部 `semantic_id_bundle` 参数（类比 `map_sparse_id_to_embedding`），符合 preprocessing-minimal-parameter-contract 与 dataset-owned-preprocessing-assembly。
- 扁平化 `SequenceDatasetConfig` 到 `SemanticIDDatasetConfig`：删除 `SequenceDatasetConfig` 类，`SemanticIDDatasetConfig` 成为独立类，仅保留运行时消费的字段（`data_reader`、`preprocessing_functions`、`shuffle_files`）。删除遗留字段（`user_id_field`、`keep_user_id`、`num_placeholder_tokens_map`、`field_type_map`、`min_sequence_length`、`feature_map`、`features_to_consider`、`file_format`、`semantic_id_map`）。
- 从 `SequenceDataloaderConfig` 删除 `should_shuffle_rows`：shuffle 语义统一为 `shuffle_files`（dataset config）+ `shuffle_rows`（reader factory）。
- 迁移 `configs/data/tiger_train.yaml`：data_reader 改 factory、`preprocessing_functions` 直写 list（semantic_id_bundle 局部注入）、删除 `features`/`dataset`/`preprocessing`/`data_reader` 旧顶层块、dataloader 删 `should_shuffle_rows`、清理 collate 冗余参数。
- 迁移 `configs/data/tiger_inference.yaml`：同上（predict-only 变体），collate 的 `id_field_name` 硬编码 `user_id`。
- `SequenceDataModule._build_collate_fn` 不需改动（已有正确绑定逻辑）。
- collate 函数（`collate_with_sid_causal_duplicate`/`collate_fn_train`/`collate_fn_inference_for_sequence`）不需改动。

## Capabilities

### New Capabilities
- `tiger-sequence-data-contract`: 要求 tiger_train / tiger_inference 使用新 data contract —— reader factory、配置直写 preprocessing chain、precomputed semantic_id 基于 keyed bundle 局部参数注入、新 shuffle 语义、sequence config 类精简为运行时字段。

### Modified Capabilities
<!-- 无。本次变更是将既有通用契约应用到 sequence 链路实验，不改变通用契约本身的 requirement。 -->

## Impact

- **代码文件**：
  - `src/data/components/preprocessing.py`：重构 `map_sparse_id_to_semantic_id` 签名
  - `src/data/components/config_models.py`：删除 `SequenceDatasetConfig`、精简 `SemanticIDDatasetConfig`、删除 `SequenceDataloaderConfig.should_shuffle_rows`
- **配置文件**：
  - `configs/data/tiger_train.yaml`
  - `configs/data/tiger_inference.yaml`
- **不需改动**：`SequenceDataModule._build_collate_fn`、collate 函数、`BaseDataModule`、`SequenceDataset`、`TFRecordReader`、experiment 配置（`configs/experiment/tiger_train.yaml`、`tiger_inference.yaml`）。
- **resolver 保留**：`extract_fields_from_list_of_dicts` / `create_map_from_list_of_dicts` 定义在 `src/utils/custom_hydra_resolvers.py`，迁移后 sequence 链路也不再引用，可考虑在后续变更中统一清理。
- **sequence 链路 dataloader config 保留 sequence 专属字段**：`labels`/`masking_token`/`sequence_length`/`padding_token`/`oov_token` 由 `SequenceDataModule._build_collate_fn` 消费，与新 item 链路不同。
