## Why

`SemanticIDDatasetConfig` 与 `ItemDatasetConfig` 当前字段完全一致，且运行时 `SequenceDataset` / `BaseDataModule` 只消费同一组 dataset 字段。继续保留两个类会制造不必要的 item/sequence 分支概念，使配置模型与已经统一的数据加载实现不一致。

## What Changes

- 合并 `SemanticIDDatasetConfig` 与 `ItemDatasetConfig` 为统一的 `DatasetConfig`。
- 更新 `SequenceDataloaderConfig.dataset_config` 与 `ItemDataloaderConfig.dataset_config` 的类型注解为 `DatasetConfig`。
- 将所有 `configs/data/*.yaml` 中 dataset config `_target_` 统一指向 `src.data.components.config_models.DatasetConfig`。
- 更新 living specs，使 dataset config contract 只描述统一 `DatasetConfig`。
- **BREAKING**：不再支持通过 Hydra 或 Python import 使用 `SemanticIDDatasetConfig` / `ItemDatasetConfig` 作为公开配置类名。

## Capabilities

### New Capabilities

- `unified-dataset-config-contract`: 统一 dataset config 类名、字段集合与 Hydra target contract。

### Modified Capabilities

- `data-model-role-separation`: 配置模型模块中的 dataset config 类清单从 item/semantic 专用类改为统一 `DatasetConfig`。
- `data-config-class-convention`: dataclass 约定不再要求 `ItemDatasetConfig` 存在，改为统一 `DatasetConfig`。
- `tiger-sequence-data-contract`: TIGER sequence 链路 dataset config contract 改为使用统一 `DatasetConfig`。
- `item-dataset-config-runtime-fields-only`: item 链路 dataset config runtime-only contract 迁移到统一 `DatasetConfig`。
- `data-reader-factory-contract`: reader factory contract 的适用对象改为统一 `DatasetConfig`。
- `data-reader-component-contract`: 原始数据源组件命名 contract 的检查对象改为统一 `DatasetConfig`。
- `item-data-config-minimal-contract`: item dataset 配置最小字段 contract 改为统一 `DatasetConfig`。

## Impact

- `src/data/components/config_models.py`：删除两个重复 dataset config 类，新增统一 `DatasetConfig`，更新 dataloader config 类型注解。
- `configs/data/*.yaml`：更新 10 个 dataset config block 的 `_target_`。
- `openspec/specs/`：更新相关 living spec 的类名与 contract。
- 验证重点：7 个 experiment 的 Hydra compose / datamodule instantiate，`compileall`、`ruff`、`openspec validate --specs --no-interactive`。
