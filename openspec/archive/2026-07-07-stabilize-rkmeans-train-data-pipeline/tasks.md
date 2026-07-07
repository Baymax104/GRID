## 1. 收紧 data_reader factory contract

- [x] 1.1 更新 `src/data/components/config_models.py`，统一 `SequenceDatasetConfig` / `ItemDatasetConfig` 中 `data_reader` 的类型语义为 factory / partial
- [x] 1.2 为 dataset config 补齐 `shuffle_files` 等新 contract 所需字段
- [x] 1.3 清理仍按 instance 语义书写 `data_reader` 的注释或类型残留

## 2. 对齐 rkmeans_train 依赖的 data 核心实现

- [x] 2.1 对齐 `BaseDataModule` 与 `SequenceDataset` 的构造参数与调用方式
- [x] 2.2 更新 file suffix 获取逻辑，使其兼容 `data_reader` factory 模式
- [x] 2.3 清理 `should_shuffle_rows` 在 `rkmeans_train` 主链路中的依赖，改为新 shuffle contract
- [x] 2.4 必要时更新 `src/data/utils.py` 或相关辅助逻辑，使文件分配语义与 `shuffle_files` 对齐

## 3. 稳定 rkmeans_train 配置链路

- [x] 3.1 清理 `configs/data/rkmeans_train.yaml` 中残留的旧节点引用（如 `${data.dataset}`）
- [x] 3.2 确保 `collate`、`dataset_config`、`train/val/test_dataloader` 都统一引用新的 config 节点
- [x] 3.3 确保 `rkmeans_train` 只使用 `shuffle_files` 与 `shuffle_rows` 表达 shuffle 语义

## 4. 验证收尾

- [x] 4.1 grep 验证 `rkmeans_train` 主链路不再依赖 `should_shuffle_rows` 作为目标 contract
- [x] 4.2 import / instantiate smoke check：`BaseDataModule`、`SequenceDataset`、`ItemDatasetConfig`、`TFRecordReader` 可按新 contract 协同工作
- [x] 4.3 Hydra compose 最小验证：`experiment=rkmeans_train` 的 data 配置能够成功解析并对齐新 contract
- [x] 4.4 总结该模板对其他 experiments 迁移的可复用规则
