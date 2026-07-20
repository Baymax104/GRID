## 1. 统一 collate 配置

- [x] 1.1 将 `configs/data/rkmeans_inference.yaml` collate 块的 `item_id_field: ${data.predict_dataset_config.item_id_field}` 改为硬编码 `item_id_field: id`

## 2. 删除 ItemDatasetConfig 的 item_id_field 字段

- [x] 2.1 从 `src/data/components/config_models.py` 的 `ItemDatasetConfig` 删除 `item_id_field` 字段定义及 docstring 条目

## 3. 删除配置文件中 dataset_config 块的 item_id_field 声明

- [x] 3.1 `configs/data/sem_embeds_inference.yaml`：删除 `predict_dataset_config` 中的 `item_id_field: id`（1 处）
- [x] 3.2 `configs/data/rkmeans_train.yaml`：删除 `train_dataset_config` 和 `eval_dataset_config` 中的 `item_id_field: id`（2 处）
- [x] 3.3 `configs/data/rvq_train.yaml`：同上（2 处）
- [x] 3.4 `configs/data/rqvae_train.yaml`：同上（2 处）
- [x] 3.5 `configs/data/rkmeans_inference.yaml`：删除 `predict_dataset_config` 中的 `item_id_field: id`（1 处）

## 4. 验证

- [x] 4.1 `uv run python -c "from src.data.components.config_models import ItemDatasetConfig; import inspect; print(inspect.signature(ItemDatasetConfig))"` 确认签名无 `item_id_field`
- [x] 4.2 grep 确认 `${data.*_dataset_config.item_id_field}` 插值引用为零
- [x] 4.3 grep 确认 dataset_config 块中无 `item_id_field`（collate 块的 `item_id_field: id` 保留）
- [x] 4.4 对 rkmeans_train 执行 Hydra compose smoke check（`--cfg job`），确认配置可组合、无解析错误
