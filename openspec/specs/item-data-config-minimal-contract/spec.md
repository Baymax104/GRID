# item-data-config-minimal-contract Specification

## Purpose
TBD - created by archiving change remove-legacy-item-data-config-fields. Update Purpose after archive.
## Requirements
### Requirement: migrated item data pipelines SHALL expose only the minimal new config contract
对于已经迁移完成的 item data pipelines，共享配置模型 SHALL 只暴露新协议所需的最小字段集合，不再保留仅服务旧协议的兼容字段。

#### Scenario: ItemDatasetConfig 不再暴露旧派生字段容器
- **WHEN** 维护者检查 item dataset 配置模型
- **THEN** 它 MUST 不再保留仅服务旧 preprocessing 派生模式的字段

#### Scenario: ItemDataloaderConfig 不再暴露旧 shuffle / preprocessing 兼容字段
- **WHEN** 维护者检查 item dataloader 配置模型
- **THEN** 它 MUST 不再保留已迁移 item 链路不再使用的旧协议字段

#### Scenario: item 链路只依赖新 shuffle contract
- **WHEN** 维护者检查已迁移 item 链路的文件级 shuffle 语义
- **THEN** 它 MUST 只通过 `dataset_config.shuffle_files` 表达，而不是继续依赖 dataloader 旧字段

