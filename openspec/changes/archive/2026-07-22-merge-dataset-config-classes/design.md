## Context

当前 data pipeline 已经统一到 `src.data.datasets.SequenceDataset` 与 `src.data.data_module.BaseDataModule`。dataset 运行时只读取 `dataset_config.data_reader`、`dataset_config.preprocessing_functions`、`dataset_config.shuffle_files` 三个字段。

`src/data/components/config_models.py` 仍保留 `SemanticIDDatasetConfig` 与 `ItemDatasetConfig` 两个类，但它们字段完全一致，默认值也一致。item 与 sequence 的差异已经移动到 preprocessing、collate、dataloader 配置中；dataset config 类名不再承载不同运行时语义。

## Goals / Non-Goals

**Goals:**

- 用唯一 `DatasetConfig` 表达 dataset runtime config。
- 移除 `SemanticIDDatasetConfig` / `ItemDatasetConfig` 两个重复类。
- 统一所有 data YAML 的 dataset config `_target_`。
- 同步 living specs，避免文档继续要求旧类名。

**Non-Goals:**

- 不改变 `SequenceDataset` 的读取、preprocessing、shuffle 行为。
- 不改变 `SequenceDataloaderConfig` / `ItemDataloaderConfig` 的业务差异。
- 不改变 collate function、preprocessing function 或 TFRecord reader 行为。
- 不保留旧类名 alias；本次是清理重复公开配置类。

## Decisions

### D1: 新类命名为 `DatasetConfig`

选择使用中性名称 `DatasetConfig`，而不是保留 `ItemDatasetConfig` 或 `SemanticIDDatasetConfig` 作为别名。

理由：当前字段集合和运行时消费者完全通用，类名不应继续暗示 item-only 或 semantic-id-only 语义。

### D2: dataloader config 保留两类，但 `dataset_config` 类型统一

`SequenceDataloaderConfig` 与 `ItemDataloaderConfig` 仍有不同字段（如 item collate 的 `feature_to_input_name` / `limit_files`），因此不在本变更合并。二者的 `dataset_config` 类型注解统一改为 `DatasetConfig`。

### D3: 所有 Hydra `_target_` 一次性迁移

所有 `configs/data/*.yaml` 中 dataset config block 都改为 `src.data.components.config_models.DatasetConfig`。这样 Hydra instantiate smoke 能覆盖全部官方 experiments，避免部分链路继续依赖旧类名。

### D4: Living specs 改为统一 contract

所有非 archive 的 OpenSpec living specs 中，涉及 `SemanticIDDatasetConfig` / `ItemDatasetConfig` 的规范性要求都改为 `DatasetConfig`。归档历史保持不可变，不纳入修改范围。

## Risks / Trade-offs

- **旧代码直接 import 旧类名会失败** → 这是预期 BREAKING 行为；通过 grep 和 compileall 验证仓库内不再引用旧类名。
- **OpenSpec 文档与实现不一致** → 同步更新 living specs，并运行 `openspec validate --specs --no-interactive`。
- **Hydra target 漏改导致某实验无法 instantiate** → 对 7 个官方 experiments 做 compose / datamodule instantiate smoke。
