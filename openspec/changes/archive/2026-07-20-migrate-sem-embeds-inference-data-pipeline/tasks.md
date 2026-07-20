## 1. 收敛 sem_embeds_inference 配置命名

- [x] 1.1 将 `configs/data/sem_embeds_inference.yaml` 中的 `dataset` 收敛为 `predict_dataset_config`
- [x] 1.2 更新 `collate`、`predict_dataloader` 等相关引用到新节点名

## 2. 恢复配置直写 preprocessing chain

- [x] 2.1 将 `preprocessing.*` 分散节点收敛为顶层公共 `preprocessing_functions`
- [x] 2.2 让 `predict_dataset_config.preprocessing_functions` 直接引用该公共块

## 3. 清理 preprocessing 专用派生字段

- [x] 3.1 去掉 `predict_dataset_config` 中仅服务 preprocessing 的 resolver 派生字段（如 `features_to_consider`、`field_type_map`）
- [x] 3.2 在 preprocessing 配置中直接显式写入对应最小参数

## 4. 对齐 reader / shuffle contract

- [x] 4.1 将 `data_reader` 改为 `_partial_` factory 形式
- [x] 4.2 为 `predict_dataset_config` 补齐 `shuffle_files`
- [x] 4.3 为 reader 显式配置 `shuffle_rows`
- [x] 4.4 清理 `predict_dataloader` 中对 `should_shuffle_rows` 的目标依赖

## 5. 验证收尾

- [x] 5.1 compose / instantiate smoke check：`sem_embeds_inference` 的 data 配置能成功解析
- [x] 5.2 row-chain 验证：单条 item row 经过 preprocessing 后得到预期的 id / text token / text_mask 形态
- [x] 5.3 迁移完成后判断 `features` 是否仍有保留价值，并记录结论
