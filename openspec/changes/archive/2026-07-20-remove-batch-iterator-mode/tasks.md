## 1. 迭代器与 dataset 收缩

- [x] 1.1 从 `RawDataIterator` 中删除 `iter_batches()` 抽象方法
- [x] 1.2 删除 `TFRecordIterator` 与 `ParquetDataIterator` 中的 batch 迭代实现
- [x] 1.3 删除 `UnboundedSequenceIterable.setup()` 中基于 `iterate_per_row` 的条件分支，固定使用 `iterrows()`

## 2. datamodule 与配置语义清理

- [x] 2.1 删除 dataloader 组装中围绕 `iterate_per_row` 的 `batch_size` / `drop_last` 分支
- [x] 2.2 从 interfaces/dataclass 中删除 `iterate_per_row` 字段与相关双模式注释
- [x] 2.3 从官方 experiment 配置中移除 `iterate_per_row` 参数

## 3. 验证

- [x] 3.1 做全文搜索，确认代码与配置中不再残留 `iterate_per_row` 与 `iter_batches` 的官方入口语义
- [x] 3.2 做最小静态检查，确认数据加载相关 Python 文件仍可解析
- [x] 3.3 复核官方 experiment 的数据链路仍与 row-based 语义一致
