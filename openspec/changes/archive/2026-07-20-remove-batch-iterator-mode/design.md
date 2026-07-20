## Context

当前数据加载代码仍保留双模式设计：dataset 可以逐条样本迭代，也可以直接逐 batch 迭代。但实际已落地的官方 experiment 全部统一使用 row-based 模式，即先逐条样本经过预处理，再由 collate 函数组装 batch。未使用的 batch mode 使 iterator 抽象、dataset setup、dataloader 参数和配置语义变得更复杂，降低了主链路可读性。

用户已经明确接受这是一次接口收缩：不仅移除主链路中的 `iterate_per_row=False` 分支，也要连 `iter_batches()` 抽象与具体实现一起彻底删除。

## Goals / Non-Goals

**Goals:**
- 官方数据链路只保留 row-based 迭代模式。
- 删除 `iterate_per_row` 配置语义及相关条件分支。
- 删除 `RawDataIterator.iter_batches()` 抽象以及 TFRecord / Parquet 的 batch 迭代实现。
- 保持现有官方 experiment 与默认 train/inference 装配行为不变。

**Non-Goals:**
- 不尝试保留对 batch iterator 模式的向后兼容。
- 不对现有 preprocessing 或 collate 逻辑做语义重写。
- 不引入新的数据加载模式替代 batch iterator。

## Decisions

### 1. 将 row-based iteration 定义为唯一官方模式
- 决策：删除所有围绕 `iterate_per_row` 的运行时条件分支，dataset 默认永远使用 `iterrows()`。
- 原因：所有现有 experiment 均已使用该模式，继续保留双模只会制造认知负担。
- 备选方案：先 deprecated 再删除。未采用，因为用户已明确要求本次做硬删除。

### 2. 连同 iterator 抽象一起收缩
- 决策：从 `RawDataIterator` 中移除 `iter_batches()` 抽象，并删除 `TFRecordIterator.iter_batches()` / `ParquetDataIterator.iter_batches()`。
- 原因：若仅删除主链路分支而保留底层 batch API，会留下“看起来还支持但实际上不可达”的伪能力。
- 备选方案：保留实现但不接线。未采用，因为与“彻底删除”目标不符。

### 3. 清理 dataloader 中与双模式相关的参数分支
- 决策：`batch_size` 统一使用 `curr_config.batch_size_per_device`，`drop_last` 统一使用配置值，不再根据 `iterate_per_row` 变换为 `None` / `False`。
- 原因：一旦 dataset 只吐 row，DataLoader batch 语义就固定了。

### 4. 清理配置与类型定义中的双模表述
- 决策：删除 dataclass 中的 `iterate_per_row` 字段，并从 experiment 配置中移除该参数。
- 原因：让配置语义与实际能力保持一致，避免未来维护者误认为仍可切换模式。

## Risks / Trade-offs

- [未来若想恢复 batch iterator，会需要重新设计] → 接受这是一次明确的能力收缩，以换取当前主链路简化。
- [文档/注释可能残留旧术语] → 在实现后做全文搜索，优先清理代码与配置中的核心语义。
- [少量工具函数仍兼容 batch 输入，造成误导] → 本轮重点收缩主链路与 iterator API；剩余兼容代码可保留但不再有官方入口。

## Migration Plan

1. 删除 iterator 抽象与实现中的 `iter_batches()`。
2. 删除 dataset setup 中的 `iterate_per_row` 条件分支，固定使用 `iterrows()`。
3. 删除 dataloader 中依赖 `iterate_per_row` 的 `batch_size` / `drop_last` 分支。
4. 删除 interfaces/dataclass 与 experiment 配置中的 `iterate_per_row` 字段。
5. 做全文搜索和最小静态检查，确认官方 experiment 不再声明或依赖该能力。

## Open Questions

- 当前无阻塞性开放问题。
