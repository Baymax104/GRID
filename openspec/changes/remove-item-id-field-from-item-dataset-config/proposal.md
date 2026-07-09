## Why

`ItemDatasetConfig.item_id_field` 在运行时无任何 Python 代码从 `dataset_config` 实例上读取——`SequenceDataset` 和 `BaseDataModule` 均不消费它。它的唯一消费方是 `collate_fn_items`，后者通过 collate 的 `_partial_` 配置直接接收该值。将 collate 域的配置留在 dataset config 上既造成冗余（collate 已硬编码同名值），也模糊了 dataset config 与 collate config 的职责边界。

## What Changes

- **BREAKING**：从 `ItemDatasetConfig` 删除 `item_id_field` 字段（dataclass 定义 + docstring）。
- 删除 5 个 item 链路配置文件中 dataset_config 块的 `item_id_field: id` 声明（共 9 处）。
- 将 `rkmeans_inference.yaml` collate 中残留的 `${data.predict_dataset_config.item_id_field}` 插值引用改为硬编码 `id`，与其余 4 个配置统一。
- collate 块的 `item_id_field: id` 保留不动（`collate_fn_items` 仍需要该参数）。
- 不涉及 Python 代码改动（`collate_fn_items` 签名不变，只是值的来源从 dataset config 插值变为 collate 配置直写）。

## Capabilities

### New Capabilities
- `item-dataset-config-runtime-fields-only`: 要求 `ItemDatasetConfig` 只保留 dataset 运行时（`SequenceDataset` / `BaseDataModule`）实际消费的字段，collate 域配置不放在 dataset config 上。

### Modified Capabilities
<!-- 无。`item-data-config-minimal-contract`（来自未归档变更 remove-legacy-item-data-config-fields）尚未进入 openspec/specs/，无法作为 modified capability 引用。 -->

## Impact

- **配置文件**（5 个，纯删除/改写，无代码改动）：
  - `configs/data/sem_embeds_inference.yaml`
  - `configs/data/rkmeans_train.yaml`
  - `configs/data/rvq_train.yaml`
  - `configs/data/rqvae_train.yaml`
  - `configs/data/rkmeans_inference.yaml`
- **代码文件**（1 个，仅删 dataclass 字段）：
  - `src/data/components/config_models.py` — `ItemDatasetConfig` 删除 `item_id_field` 字段
- **`collate_fn_items` 签名不变**——仍接收 `item_id_field: str` 参数，值来源改为 collate 配置直写。
- 迁移后 `ItemDatasetConfig` 仅剩 3 个字段：`data_reader`、`preprocessing_functions`、`shuffle_files`——全部由 `SequenceDataset` / `BaseDataModule` 运行时消费。
