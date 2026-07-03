## Why

当前数据加载框架同时保留了按 row 迭代和按 batch 迭代两种模式，但仓库内所有官方 experiment 都已经统一使用 row-based 模式。未被使用的 batch 模式分支增加了 dataloader、iterator、config 和注释层面的复杂度，使主链路更难理解和维护。

现在需要正式收缩接口，只保留 row-based 数据迭代能力，并彻底删除 batch iterator 模式相关代码，使数据链路更单一、更易读。

## What Changes

- **BREAKING** 删除 `iterate_per_row` 配置语义，官方链路统一为 row-based 数据迭代。
- **BREAKING** 删除 datamodule / dataset 中围绕 `iterate_per_row=False` 的运行分支。
- **BREAKING** 删除 `RawDataIterator.iter_batches()` 抽象及其在 TFRecord / Parquet iterator 中的实现。
- 更新相关 dataclass 默认值、注释和文档，使其只表达 row-based 模式。
- 保证现有官方 experiment 与默认 train/inference 主链路在收缩后仍能正常装配。

## Capabilities

### New Capabilities
- `row-based-data-iteration`: 官方数据链路只支持逐样本迭代，并移除未使用的 batch iterator 模式。

### Modified Capabilities

## Impact

- 受影响代码：`src/data/loading/components/iterators.py`、`src/data/loading/components/dataloading.py`、`src/data/loading/components/interfaces.py`、`src/data/loading/datamodules/sequence_datamodule.py`
- 受影响配置：所有仍声明 `iterate_per_row` 的 experiment 配置
- 不涉及新增依赖，但属于明确的接口收缩与破坏性变更
