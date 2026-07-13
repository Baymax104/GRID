## Why

推理结果保存存在一次无意义的 tensor→dict→tensor 往返转换：`prediction_step` 产出 `(keys_tensor, predictions_tensor)`，`ModelOutput.list_of_row_format` 通过 `key_name`/`prediction_name` 展开为 `list[dict]`，writer 再通过 `merge_list_of_keyed_tensors_to_single_tensor` 从 dict 中提取回 `(keys_tensor, predictions_tensor)`。映射职责分散在 writer、module、ModelOutput 三处，导致 `prediction_step` 实现中无法直接看出写入了什么数据（需追到 YAML 配置才知道字段名），且 tiger 的 `prediction_value_name` 与 writer 注入的 `prediction_name` 独立设置，存在静默失效隐患。

## What Changes

- **BREAKING** `ModelOutput` 简化为直接持有 `keys` + `predictions` 两个 tensor，去除 `key_name`/`prediction_name`/`list_of_row_format`/`_convert_to_list`
- **BREAKING** `LocalPickleWriter` 直接缓存 `ModelOutput` 对象并 `torch.cat` 合并，不再经过行格式中间层；去除 `prediction_key_name`/`prediction_name`/`should_merge_list_of_keyed_tensors_to_single_tensor` 配置参数
- **BREAKING** `BaseBufferedWriter` 去除 `prediction_key_name`/`prediction_name` 参数及 `setup()` 中的注入逻辑
- 删除 `merge_list_of_keyed_tensors_to_single_tensor`（唯一消费者是 writer）
- 不再产出 `merged_predictions.pkl`（行格式中间产物，无代码消费者）
- `TransformerBaseModule` 去除 `prediction_key_name`/`prediction_name` 属性及 setter
- `SemanticIDEncoderDecoder` 去除 `prediction_key_name`/`prediction_value_name` 构造参数
- 3 个 callback YAML 去除 `prediction_key_name`/`prediction_name`/`should_merge_list_of_keyed_tensors_to_single_tensor` 字段
- 4 个 `predict_step` 实现去除 `key_name=`/`prediction_name=` 参数

## Capabilities

### New Capabilities
- `prediction-output-protocol`: 定义 `ModelOutput` 作为推理结果的字段规范层，直接持有 `keys` + `predictions` tensor；writer 直接缓存和合并 tensor，不经过行格式中间层

### Modified Capabilities
（无 — `keyed-prediction-bundle-artifact` 的最终产物格式要求不变，`{"keys": tensor, "predictions": tensor}` 结构保持一致）

## Impact

- `src/common/components/model_output.py` — 重写 `ModelOutput` 类
- `src/common/components/prediction_writers.py` — 重写 `handle_batch`/`_flush_buffer`/`_merge_files`/`setup`/`__init__`
- `src/utils/tensor_utils.py` — 删除 `merge_list_of_keyed_tensors_to_single_tensor`
- `src/common/modules/transformer_base_module.py` — 删除 `prediction_key_name`/`prediction_name` 属性
- `src/recommendation/tiger_generation_model.py` — 删除 `prediction_key_name`/`prediction_value_name` 参数，简化 `predict_step`
- `src/embedding/semantic_embedding_inference_module.py` — 简化 `predict_step`
- `src/quantization/residual_kmeans.py` — 简化 `predict_step`
- `src/quantization/residual_quantization_vae.py` — 简化 `predict_step`
- `src/quantization/residual_vector_quantization.py` — 简化 `predict_step`
- `configs/callbacks/sem_embeds_inference.yaml` — 去除 3 个字段
- `configs/callbacks/rkmeans_inference.yaml` — 去除 3 个字段
- `configs/callbacks/tiger_inference.yaml` — 去除 2 个字段
- `AGENTS.md` — 更新 `merged_predictions.pkl` 描述（不再产出）
