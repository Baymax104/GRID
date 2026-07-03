## ADDED Requirements

### Requirement: Official data pipeline SHALL use row-based iteration only
官方数据链路必须只支持逐样本迭代，不得继续暴露 batch iterator 模式。

#### Scenario: Dataset setup selects iterator
- **WHEN** dataset 完成 setup 并准备创建底层数据迭代器
- **THEN** 必须始终使用逐样本迭代路径
- **AND** 不得再根据配置切换到 batch iterator

### Requirement: Iterator API SHALL not expose batch iteration
官方 iterator 抽象与实现不得继续暴露 `iter_batches()` 能力。

#### Scenario: Raw iterator interface
- **WHEN** 开发者查看 `RawDataIterator` 抽象接口
- **THEN** 接口中不得再包含 batch iteration 方法

#### Scenario: TFRecord and Parquet iterators
- **WHEN** 开发者查看 TFRecord 或 Parquet iterator 实现
- **THEN** 实现中不得再包含官方 batch iteration 入口

### Requirement: Dataloader assembly SHALL assume row-based batching
datamodule 在组装 DataLoader 时必须直接按 row-based 模式的语义设置批处理参数。

#### Scenario: Build dataloader for official experiments
- **WHEN** datamodule 为训练、验证、测试或预测阶段构建 DataLoader
- **THEN** `batch_size` 必须直接使用配置的每设备 batch 大小
- **AND** 不得再基于 `iterate_per_row` 改写 `batch_size` 或 `drop_last`

### Requirement: Official configs SHALL not declare iterate_per_row
官方 experiment 配置不得继续声明 `iterate_per_row` 参数。

#### Scenario: Inspect official experiment configs
- **WHEN** 开发者查看 `configs/experiment/*.yaml`
- **THEN** 不得再看到 `iterate_per_row` 配置项
