## Why

`sem_embeds_inference` 是整个推荐 pipeline 的第一个实验，输入链路简单、验证反馈快，因此非常适合作为下一个 data 重构迁移对象。

当前它仍停留在旧 data 配置模式：

- 使用 `dataset` 旧节点命名；
- preprocessing 由分散的 `preprocessing.*` 节点再拼接成 `dataset.preprocessing_functions`；
- `dataset_config` 中仍包含通过 resolver 派生的 preprocessing 专用字段；
- dataloader 仍保留 `should_shuffle_rows`；
- `data_reader` 仍不是新 contract 下的 factory 形态。

在 `rkmeans_train` 已经形成较稳定模板的前提下，需要把 `sem_embeds_inference` 迁到同一套新 data contract，以尽快验证“item + text preprocessing + inference-only”场景下的迁移可行性。

## What Changes

- 将 `sem_embeds_inference` 的 dataset config 命名收敛为 `predict_dataset_config`
- 将 preprocessing chain 改为在配置文件中直接显式声明的 `preprocessing_functions`
- 清理仅服务 preprocessing 的 resolver 派生字段
- 将 `data_reader` 改为 factory 形态，并统一 `shuffle_files` / `shuffle_rows` contract
- 暂不把删除 `features` 作为硬目标，迁移完成后再判断是否还需要保留

## Capabilities

### New Capabilities
- `item-text-inference-data-contract`: 规定 item-level text inference 场景下的数据链路也使用新的 data reader / preprocessing contract

### Modified Capabilities
- `config-declared-preprocessing-contract`: 将该 contract 从 `rkmeans_train` 扩展到 `sem_embeds_inference`
- `data-reader-factory-contract`: 将 reader factory contract 应用于 `sem_embeds_inference`

## Impact

- 受影响配置：`configs/data/sem_embeds_inference.yaml`
- 受影响代码：主要是 data config 装配路径，必要时少量 data 核心兼容逻辑
- 预期收益：快速验证 inference-only 场景能否平滑采用新 data contract，并为后续 `rkmeans_inference` / `tiger_inference` 迁移提供模板
