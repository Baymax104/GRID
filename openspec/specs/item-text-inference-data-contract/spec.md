# item-text-inference-data-contract Specification

## Purpose
TBD - created by archiving change migrate-sem-embeds-inference-data-pipeline. Update Purpose after archive.
## Requirements
### Requirement: item-level text inference data pipeline SHALL use the new data contract
`sem_embeds_inference` 这类 item-level text inference 实验必须使用新的 data contract，包括 reader factory、row-only preprocessing 和配置直写 preprocessing chain。

#### Scenario: predict dataset config uses explicit preprocessing chain
- **WHEN** 维护者查看 `sem_embeds_inference` 的 data 配置
- **THEN** 必须能直接看到 preprocessing 的顺序与参数

#### Scenario: predict pipeline uses reader factory contract
- **WHEN** 维护者查看该实验的数据读取配置
- **THEN** `data_reader` MUST 采用 factory / `_partial_` 形式

#### Scenario: predict pipeline uses new shuffle contract
- **WHEN** 维护者查看该实验的 shuffle 配置
- **THEN** shuffle 语义 MUST 通过 `shuffle_files` 与 `shuffle_rows` 表达，而不是继续依赖 `should_shuffle_rows`

