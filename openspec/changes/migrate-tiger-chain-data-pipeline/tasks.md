## 1. 重构 `map_sparse_id_to_semantic_id`（代码）

- [x] 1.1 修改 `src/data/components/preprocessing.py` 中 `map_sparse_id_to_semantic_id` 签名：将 `dataset_config: DictConfig` 参数替换为 `semantic_id_bundle: dict[str, Any] | None = None`
- [x] 1.2 修改函数体：删除 `id_bundle = dataset_config.semantic_id_map.get(k, None)` 逻辑，改为直接使用 `semantic_id_bundle` 参数调用 `lookup_values_in_keyed_prediction_bundle(semantic_id_bundle, v)`；删除 `id_bundle is not None` 分支判断（bundle 为 None 时直接 raise ValueError，类比 `map_sparse_id_to_embedding`）

## 2. 精简 config 类（代码）

- [x] 2.1 删除 `SequenceDatasetConfig` 类定义，`SemanticIDDatasetConfig` 去除继承改为独立 `@dataclass`，仅保留 3 个字段：`data_reader`、`preprocessing_functions`（default_factory=list）、`shuffle_files`（default=False）
- [x] 2.2 更新 `SequenceDataloaderConfig.dataset_config` 类型注解从 `SequenceDatasetConfig` 改为 `SemanticIDDatasetConfig`
- [x] 2.3 从 `SequenceDataloaderConfig` 删除 `should_shuffle_rows` 字段

## 3. 迁移 tiger_train.yaml（配置）

- [x] 3.1 删除 `data_reader`、`preprocessing`、`features`、`dataset` 旧顶层块
- [x] 3.2 新增 `preprocessing_functions` 直写 list（4 步：filter_features_to_consider([sequence_data, user_id]) → convert_to_dense_numpy_array → convert_fields_to_tensors(field_type_map={sequence_data: int32, user_id: int32}) → map_sparse_id_to_semantic_id(semantic_id_bundle=load_keyed_prediction_bundle(${semantic_id_path}), features_to_apply=[sequence_data], num_hierarchies=${model.root.num_hierarchies})）
- [x] 3.3 新增 `train_dataset_config` / `eval_dataset_config`（SemanticIDDatasetConfig：data_reader factory 内嵌 shuffle_rows, shuffle_files, preprocessing_functions=${data.preprocessing_functions}）
- [x] 3.4 更新 train/val/test_dataloader：删除 `should_shuffle_rows`，`dataset_config` 引用改 `${data.train_dataset_config}` / `${data.eval_dataset_config}`
- [x] 3.5 清理 collate 配置：`train_collate` 删除冗余的 `sequence_length`/`padding_token`（保留 `sequence_field_name`/`sid_hierarchy`/`max_batch_size`）；`eval_collate` 删除冗余的 `sequence_length`/`padding_token`

## 4. 迁移 tiger_inference.yaml（配置）

- [x] 4.1 删除 `data_reader`、`preprocessing`、`features`、`dataset` 旧顶层块
- [x] 4.2 新增 `preprocessing_functions` 直写 list（同 tiger_train 4 步）
- [x] 4.3 新增 `predict_dataset_config`（SemanticIDDatasetConfig：data_reader factory shuffle_rows=false, shuffle_files=false, preprocessing_functions=${data.preprocessing_functions}）
- [x] 4.4 更新 predict_dataloader：删除 `should_shuffle_rows`，`dataset_config` 引用改 `${data.predict_dataset_config}`
- [x] 4.5 collate 的 `id_field_name` 从 `${data.dataset.user_id_field}` 改为硬编码 `user_id`；删除冗余的 `sequence_length`/`padding_token`

## 5. 验证

- [x] 5.1 grep 确认 tiger 链路配置不再引用 `should_shuffle_rows` / `semantic_id_map` / `extract_fields_from_list_of_dicts` / `create_map_from_list_of_dicts` / `keep_user_id` / `user_id_field` / `features_to_consider` / `num_placeholder_tokens_map`
- [x] 5.2 grep 确认 `map_sparse_id_to_semantic_id` 签名中无 `dataset_config`，且 Python 代码中无其他调用方传入 `dataset_config`
- [x] 5.3 grep 确认 `SequenceDatasetConfig` 类定义已删除，全仓无残留引用
- [x] 5.4 对 tiger_train 执行 Hydra `--cfg job` 验证 config composition（所有 ${...} 引用解析成功）
- [x] 5.5 对 tiger_inference 执行 Hydra `--cfg job` 验证 config composition
- [x] 5.6 对 tiger_train 执行 dry-run smoke check，确认 datamodule 实例化通过（datamodule 实例化成功；model 实例化失败为预存问题：SemanticIDEncoderDecoder 缺 embedding_dim/num_embeddings_per_hierarchy 参数，与 data 迁移无关）

## 6. 修复预存 bug

- [x] 6.1 修复 `TokenizerConfig` 已从 `config_models.py` 删除但 `preprocessing.py` / `utils.py` 仍 import 导致的 ImportError（移除 import，类型注解改为 `Any`）
