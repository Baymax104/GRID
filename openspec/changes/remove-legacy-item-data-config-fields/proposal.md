## Why

在 `rkmeans_train` 与 `sem_embeds_inference` 两条 item 链路完成迁移后，当前主链路已经稳定采用了新的 data contract：

- `data_reader` 使用 factory / `_partial_`
- preprocessing chain 在配置中直写
- preprocessing 不再接收 `dataset_config`
- shuffle 语义收敛为 `shuffle_files` 与 `shuffle_rows`

但 item 相关的共享配置模型和 datamodule 代码中仍残留旧协议字段与兼容逻辑，例如：

- `ItemDatasetConfig` 中保留大量仅服务旧派生配置模式的字段
- `ItemDataloaderConfig` 中仍保留 `preprocessing_functions`、`should_shuffle_rows`、`oov_token` 等旧字段
- `BaseDataModule` 仍会 fallback 到 `should_shuffle_rows`

如果继续在这些兼容层上推进其他实验迁移，后续很容易出现“迁移完成但旧协议仍未真正消失”的问题。因此需要先对**已迁移的 item 链路**直接清理旧协议字段。

## What Changes

- 清理 `ItemDatasetConfig` 中不再属于 item 新协议主 contract 的旧字段
- 清理 `ItemDataloaderConfig` 中不再属于 item 新协议主 contract 的旧字段
- 清理 `BaseDataModule` 中针对 item 新协议已不再需要的旧 shuffle fallback
- 以 `rkmeans_train` 与 `sem_embeds_inference` 两条已迁移链路作为验证样板

## Capabilities

### New Capabilities
- `item-data-config-minimal-contract`: 规定已迁移 item 链路只暴露新 data contract 所需的最小配置字段

### Modified Capabilities
- `data-reader-factory-contract`: 对 item 链路去除旧字段兼容层，进一步强化新 contract
- `config-declared-preprocessing-contract`: 对 item 链路明确只保留配置直写所需字段

## Impact

- 受影响代码：`src/data/components/config_models.py`、`src/data/datamodules/base.py`
- 验证对象：`configs/data/rkmeans_train.yaml`、`configs/data/sem_embeds_inference.yaml`
- 本次不触达 sequence / tiger / semantic-id 相关配置模型与实验
