## Why

当前 data 模块正处于从旧的 `iterator instance + dataset mutation` 模式向新的 `data_reader factory + constructor-driven dataset` 模式迁移过程中。`rkmeans_train` 是目前推进最深的实验，但其 data 链路仍存在若干新旧混杂点：

- `BaseDataModule` 与 `SequenceDataset` 的构造参数尚未完全对齐；
- `rkmeans_train.yaml` 仍残留 `data.dataset` / `should_shuffle_rows` 等旧配置痕迹；
- `ItemDatasetConfig` / `SequenceDatasetConfig` 对 `data_reader` 的类型契约尚未稳定到 factory 模式；
- shuffle 语义尚未彻底收敛到 `shuffle_files` 与 `shuffle_rows` 两层。

如果不先把 `rkmeans_train` 的 data 链路稳定下来，后续向 `rqvae_train`、`rvq_train`、`tiger_train` 等实验迁移时会反复返工核心接口。因此本次需要把 `rkmeans_train` 打造成第一个稳定的新 data pipeline 模板。

## What Changes

- 稳定 `rkmeans_train` 的 data 链路，使其完整采用 `data_reader` factory 模式
- 统一 `SequenceDataset`、`BaseDataModule`、`config_models.py` 对 `data_reader` 的 contract
- 废弃 `should_shuffle_rows`，明确使用：
  - `dataset_config.shuffle_files`
  - `data_reader.shuffle_rows`
- 清理 `rkmeans_train.yaml` 中残留的旧 config 命名与引用
- 为后续其他实验迁移沉淀一个最小稳定模板

## Capabilities

### New Capabilities
- `data-reader-factory-contract`: 规定 dataset config 中的 `data_reader` 必须以 factory / partial 形式提供，由 dataset 在运行时按需实例化
- `data-shuffle-contract`: 规定新 data pipeline 只通过 `shuffle_files` 与 `shuffle_rows` 表达 shuffle 语义

### Modified Capabilities
- `data-reader-component-contract`: 从“仅统一 reader 命名”进一步收敛为“统一 reader factory 使用方式”

## Impact

- 受影响代码：`src/data/components/config_models.py`、`src/data/components/datasets.py`、`src/data/datamodules/base.py`、必要时 `src/data/utils.py`
- 受影响配置：`configs/data/rkmeans_train.yaml`
- 受影响 scope：以 `rkmeans_train` 为主，但会影响 data 核心 contract，为后续迁移其他实验铺路
- 不要求本次同时迁完全部 experiments；目标是先稳定 `rkmeans_train` 及其依赖的最小公共 contract
