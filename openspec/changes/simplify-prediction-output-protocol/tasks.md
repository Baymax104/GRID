## 1. 重写 ModelOutput

- [ ] 1.1 重写 `src/common/components/model_output.py`：`ModelOutput.__init__(self, keys, predictions)`，删除 `key_name`/`prediction_name`/`list_of_row_format`/`_convert_to_list`

## 2. 重写 prediction_writers.py

- [ ] 2.1 `BaseBufferedWriter.__init__` 去除 `prediction_key_name`/`prediction_name` 参数及赋值
- [ ] 2.2 `BaseBufferedWriter.setup` 删除字段名注入逻辑（`hasattr` 检查和赋值）
- [ ] 2.3 `handle_batch` 改为直接缓存 `ModelOutput` 对象，用 `len(model_output.keys)` 判断 flush 时机
- [ ] 2.4 `_flush_buffer` 改为直接 `pickle.dump(self.buffer)`（缓存 `list[ModelOutput]`）
- [ ] 2.5 `_merge_files` 改为遍历分片收集 `ModelOutput`，`torch.cat` 合并 `keys`（统一 `torch.long`）和 `predictions`，保存 `merged_predictions_tensor.pt`，不产出 `merged_predictions.pkl`
- [ ] 2.6 `LocalPickleWriter.__init__` 去除 `should_merge_list_of_keyed_tensors_to_single_tensor` 参数
- [ ] 2.7 删除 `from src.utils.tensor_utils import merge_list_of_keyed_tensors_to_single_tensor` 导入

## 3. 简化 4 个 predict_step

- [ ] 3.1 `semantic_embedding_inference_module.py`：`ModelOutput(keys=item_ids, predictions=semantic_embeddings)`，去除 `key_name`/`prediction_name`
- [ ] 3.2 `residual_kmeans.py`：`ModelOutput(keys=item_ids, predictions=cluster_ids)`，去除 `key_name`/`prediction_name`
- [ ] 3.3 `residual_quantization_vae.py`：同上
- [ ] 3.4 `residual_vector_quantization.py`：同上
- [ ] 3.5 `tiger_generation_model.py` predict_step：`ModelOutput(keys=ids, predictions=generated_sids)`，去除 `key_name`/`prediction_name`

## 4. 清理模块属性

- [ ] 4.1 `TransformerBaseModule` 删除 `_prediction_key_name`/`_prediction_name` 属性、property 和 setter
- [ ] 4.2 `SemanticIDEncoderDecoder` 删除 `prediction_key_name`/`prediction_value_name` 构造参数及赋值

## 5. 删除 merge_list_of_keyed_tensors_to_single_tensor

- [ ] 5.1 从 `src/utils/tensor_utils.py` 删除 `merge_list_of_keyed_tensors_to_single_tensor` 函数

## 6. 更新配置文件

- [ ] 6.1 `configs/callbacks/sem_embeds_inference.yaml` 去除 `prediction_key_name`/`prediction_name`/`should_merge_list_of_keyed_tensors_to_single_tensor`
- [ ] 6.2 `configs/callbacks/rkmeans_inference.yaml` 去除 `prediction_key_name`/`prediction_name`/`should_merge_list_of_keyed_tensors_to_single_tensor`
- [ ] 6.3 `configs/callbacks/tiger_inference.yaml` 去除 `prediction_key_name`/`prediction_name`

## 7. 更新文档

- [ ] 7.1 `AGENTS.md` 移除 `merged_predictions.pkl` 描述，更新 `merged_predictions_tensor.pt` 描述
- [ ] 7.2 `src/embedding/README.md` 更新 ModelOutput 描述（如有引用旧字段）

## 8. 验证

- [ ] 8.1 `ruff check src/` 通过
- [ ] 8.2 grep 确认零残留：`prediction_key_name`/`prediction_name`/`prediction_value_name`/`list_of_row_format`/`merge_list_of_keyed_tensors_to_single_tensor`/`should_merge_list_of_keyed_tensors_to_single_tensor`/`merged_predictions.pkl`
- [ ] 8.3 确认 `merged_predictions_tensor.pt` 的保存逻辑产出 `{"keys": tensor, "predictions": tensor}` 格式
