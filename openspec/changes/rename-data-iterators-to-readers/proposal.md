## Why

当前 `src/data/components/iterators.py` 及其相关命名把原始数据读取组件称为 `iterator`，但这些对象的职责已经明显超出 Python 语义上的“迭代器”：它们还承担文件路径管理、文件格式知识、按文件/按行读取策略、shuffle 控制等职责。

在当前实现里，这类对象更接近 **reader / data reader**：

- `SequenceDataset` / `ItemDataset` 持有它们作为原始数据来源；
- `DataModule` 通过它们判断文件后缀与读取方式；
- Hydra 配置把它们作为一个独立组件注入到 dataset config 中。

因此继续沿用 `data_iterator`、`BaseIterator`、`TFRecordIterator`、`iterators.py` 这套命名，会让职责边界显得偏窄，也不利于新维护者快速理解“这是原始数据读取器，而不是普通 Python iterator 对象”。

另一个背景是：历史变更 `flatten-data-loading-and-rename-components` 曾明确保留 `iterators.py` 命名。本次若采纳 reader 语义，属于一次**有意识地修正先前命名决策**，需要通过新的 OpenSpec 提案显式记录。

## What Changes

- 将 `src/data/components/iterators.py` 重命名为 `src/data/components/readers.py`
- 将 `BaseIterator`、`TFRecordIterator`、`ParquetDataIterator` 统一重命名为 reader 语义的类名
- 将配置 dataclass、dataset 实现、datamodule、Hydra 配置中的 `data_iterator` 命名统一改为 `data_reader`
- 更新所有 `_target_` 路径、Python imports、文档与注释中的 iterator 旧命名
- 更新 OpenSpec 中此前“`iterators.py` 保留不动”的相关规格，使其改为 reader 语义

## Capabilities

### Modified Capabilities
- `data-loading-package-layout`: 将原始文件数据读取组件的模块/类型命名从 iterator 语义收敛为 reader 语义

### New Capabilities
- `data-reader-component-contract`: 规定 dataset config 与 data pipeline 对外暴露的原始数据源组件统一采用 `data_reader` 命名

## Impact

- 受影响代码：`src/data/components/readers.py`（原 `iterators.py`）、`config_models.py`、`datasets.py`、`src/data/datamodules/base.py` 及相关 imports
- 受影响配置：全量 `configs/data/*.yaml` 中的 `data_iterator` key 与 `_target_` 路径
- 受影响文档：`src/data/README.md`、`src/embedding/README.md`、`AGENTS.md` 及其他提及 iterator 旧命名的文档
- 受影响 spec：需要显式修订 `flatten-data-loading-and-rename-components` 相关规格中的 `iterators.py` 命名决策
- 预期行为：纯命名收敛，不改变读取逻辑、shuffle 语义、文件切分策略或 dataset 产出格式
