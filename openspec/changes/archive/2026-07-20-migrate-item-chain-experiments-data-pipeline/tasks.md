## 1. 迁移 rvq_train / rqvae_train（train/val/test 链路）

- [x] 1.1 改写 `configs/data/rvq_train.yaml`：删除 `data_reader`、`preprocessing`、`dataset`、`features` 旧顶层块
- [x] 1.2 新增 `preprocessing_functions` 直写 list（4 步：filter_features_to_consider([id]) → convert_to_dense_numpy_array → convert_fields_to_tensors(field_type_map={id: torch.__dict__.get int32}) → map_sparse_id_to_embedding(embedding_bundle=load_keyed_prediction_bundle(${embedding_path}), sparse_id_field=id, embedding_field_to_add=embedding)）
- [x] 1.3 新增 `train_dataset_config` / `eval_dataset_config`（ItemDatasetConfig：item_id_field=id, keep_item_id=true, data_reader factory 内嵌 shuffle_rows, shuffle_files, preprocessing_functions=${data.preprocessing_functions}）
- [x] 1.4 更新 train/val/test_dataloader：删除 `should_shuffle_rows`，`dataset_config` 引用改 `${data.train_dataset_config}` / `${data.eval_dataset_config}`
- [x] 1.5 更新 collate 的 `item_id_field` 引用为 `${data.train_dataset_config.item_id_field}`，`feature_to_input_name` 保留顶层
- [x] 1.6 将相同改动同步到 `configs/data/rqvae_train.yaml`（逐字一致）

## 2. 迁移 rkmeans_inference（predict only 链路）

- [x] 2.1 改写 `configs/data/rkmeans_inference.yaml`：删除 `data_reader`、`preprocessing`、`dataset`、`features` 旧顶层块
- [x] 2.2 新增 `preprocessing_functions` 直写 list（同 rvq_train 4 步）
- [x] 2.3 新增 `predict_dataset_config`（ItemDatasetConfig：item_id_field=id, keep_item_id=true, shuffle_files=false, data_reader factory shuffle_rows=false, preprocessing_functions=${data.preprocessing_functions}）
- [x] 2.4 更新 predict_dataloader：删除 `should_shuffle_rows`，`dataset_config` 引用改 `${data.predict_dataset_config}`
- [x] 2.5 将 `feature_to_input_name` 从 predict_dataloader 内嵌提升到顶层，collate 引用改 `${data.feature_to_input_name}`，`item_id_field` 引用改 `${data.predict_dataset_config.item_id_field}`

## 3. 验证

- [x] 3.1 对 rvq_train 执行 dry-run smoke check，确认配置可实例化、无 Hydra 解析错误（用户手动验证完成，可运行）
- [x] 3.2 对 rqvae_train 执行同样 dry-run smoke check（用户手动验证完成，可运行）
- [x] 3.3 对 rkmeans_inference 执行 dry-run smoke check（predict 模式，需 ckpt_path）（用户手动验证完成，可运行）
- [x] 3.4 grep 确认 item 链路三个配置不再引用 `extract_fields_from_list_of_dicts` / `create_map_from_list_of_dicts` / `should_shuffle_rows` / `embedding_map` / `features_to_consider`（仅 `filter_features_to_consider` 函数名 + `features_to_consider:` 参数合法匹配，与样板一致）
- [x] 3.5 对比 rkmeans_train 样板，确认三个迁移后配置的结构与字段引用一致（rvq_train/rqvae_train 与 rkmeans_train 逐字一致；rkmeans_inference 为 predict-only 变体结构对齐）
