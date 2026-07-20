## 1. 重命名 reader 模块与类型

- [x] 1.1 将 `src/data/components/iterators.py` 重命名为 `src/data/components/readers.py`
- [x] 1.2 将 `BaseIterator`、`TFRecordIterator`、`ParquetDataIterator` 统一改为 reader 语义命名
- [x] 1.3 更新所有 Python import 到新的模块与类型名

## 2. 统一配置与调用侧命名

- [x] 2.1 将 `SequenceDatasetConfig` / `ItemDatasetConfig` 中的 `data_iterator` 字段改为 `data_reader`
- [x] 2.2 更新 `SequenceDataset`、相关 datamodule 与其他调用侧对该字段的访问命名
- [x] 2.3 更新全量 `configs/data/*.yaml` 中的顶层 key、插值引用与 `_target_` 路径

## 3. 同步文档与 spec

- [x] 3.1 更新 `src/data/README.md`、`src/embedding/README.md`、`AGENTS.md` 等文档中的 iterator 旧命名
- [x] 3.2 更新本次 change 的 specs，显式修订 `data-loading-package-layout` 中对 `iterators.py` 的旧要求
- [x] 3.3 新增/更新 reader contract 规格，明确配置对外术语统一为 `data_reader`

## 4. 验证收尾

- [x] 4.1 grep 验证 `src/`、`configs/`、`openspec/` 中不再残留本次需要迁移的 `data_iterator` / `iterators.py` / `*Iterator` 旧命名（允许历史归档文档除外）
- [x] 4.2 import smoke check：相关 data pipeline 模块能成功导入
- [x] 4.3 Hydra compose 最小验证：受影响的 data experiments 仍能解析新的 `data_reader` 配置与 `_target_`
