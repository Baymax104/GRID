## Why

rvq_train、rqvae_train、rkmeans_inference 三个 item 链路实验仍使用旧 data 配置模式，且因 `remove-legacy-item-data-config-fields` 已删除 `ItemDatasetConfig`/`ItemDataloaderConfig` 的旧字段（features_to_consider、embedding_map、num_placeholder_tokens_map、field_type_map、should_shuffle_rows 等），这三个配置当前已无法实例化。需要将它们迁移到与 rkmeans_train 一致的新 data contract，消除坏配置并统一 item 链路架构。

## What Changes

- 将 `configs/data/rvq_train.yaml` 迁移到新 data contract：data_reader 改 factory（`_partial_`）、preprocessing_functions 直写 list、删除 `features`/`dataset`/`preprocessing`/`data_reader` 旧顶层块、dataloader 删除 `should_shuffle_rows`、embedding 通过 `map_sparse_id_to_embedding(embedding_bundle=...)` 局部注入。
- 将 `configs/data/rqvae_train.yaml` 同步迁移（与 rvq_train 结构一致）。
- 将 `configs/data/rkmeans_inference.yaml` 迁移到新 data contract（predict only 变体：predict_dataset_config + predict_dataloader，feature_to_input_name 从 dataloader 内嵌移到顶层）。
- 三个实验均对齐 rkmeans_train 样板：shuffle 语义拆为 `shuffle_files`（dataset config）+ `shuffle_rows`（reader factory），删除 resolver 派生字段引用。
- 不涉及代码改动：`map_sparse_id_to_embedding` 已迁移新协议，`collate_fn_items` 签名不变，resolver（extract_fields_from_list_of_dicts / create_map_from_list_of_dicts）保留（sequence 链路仍在使用）。

## Capabilities

### New Capabilities
- `item-quantization-data-contract`: 要求 item 级量化实验（rvq_train / rqvae_train / rkmeans_inference）使用新 data contract —— reader factory、配置直写 preprocessing chain、precomputed embedding 基于 keyed bundle lookup、新 shuffle 语义。

### Modified Capabilities
<!-- 无。本次变更是将既有通用契约应用到更多实验，不改变通用契约本身的 requirement。 -->

## Impact

- **配置文件**（仅此三处，无代码改动）：
  - `configs/data/rvq_train.yaml`
  - `configs/data/rqvae_train.yaml`
  - `configs/data/rkmeans_inference.yaml`
- **依赖的已迁移代码**（已就绪，不需修改）：`map_sparse_id_to_embedding`（接收 embedding_bundle）、`collate_fn_items`、`ItemDatasetConfig`/`ItemDataloaderConfig`（字段已精简）、`TFRecordReader` factory、`BaseDataModule` shuffle 读取逻辑。
- **resolver 保留**：`extract_fields_from_list_of_dicts` / `create_map_from_list_of_dicts` 定义在 `src/utils/custom_hydra_resolvers.py`，tiger_train/tiger_inference sequence 链路仍引用，不删除。
- **运行脚本**：`configs/experiment/rvq_train.yaml`、`rqvae_train.yaml`、`rkmeans_inference.yaml` 的 experiment 组装不变，但迁移后这三个实验恢复可实例化（当前已坏）。
