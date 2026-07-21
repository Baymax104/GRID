## Why

TIGER 当前确认不使用 `user_id` 参与训练，但训练和推理模型配置仍把 `user_id` 声明为模型输入，collate 也会把 `user_id` 当作序列特征放入 `transformed_sequences`。这使“用户身份用于输出归属”和“用户身份作为模型特征”两个概念混在一起，增加误用风险。

## What Changes

- 从 TIGER 训练链路中移除 `user_id` 模型输入：训练 data/model 配置不再把 `user_id` 传入模型。
- 保留推理链路中的 `user_id` 作为预测输出 key：`user_id` 仍用于 `ModelOutput.keys`，以便把生成结果映射回原始用户。
- 调整推理 collate 行为，使 id 字段只进入 `SequentialModelInputData.user_id_list`，不再作为普通序列特征进入模型输入。
- 删除 TIGER 模型中的 user embedding / `num_user_bins` / `user_id` encoder 分支等未启用训练路径。
- **BREAKING**: TIGER 模型不再支持通过 `user_id` / `num_user_bins` 启用用户 embedding 作为训练或推理特征。

## Capabilities

### New Capabilities
- `tiger-user-identity-output-contract`: 明确 TIGER 推理中的用户身份只用于输出归属，不作为模型生成特征。

### Modified Capabilities
- `tiger-sequence-data-contract`: sequence 数据链路需要区分序列特征与 id 字段，推理 id 字段不得进入模型输入序列。

## Impact

- Affected configs:
  - `configs/data/tiger_train.yaml`
  - `configs/data/tiger_inference.yaml`
  - `configs/model/tiger_train.yaml`
  - `configs/model/tiger_inference.yaml`
- Affected code:
  - `src/data/components/collate.py`
  - `src/recommendation/tiger_generation_model.py`
- Validation focus:
  - TIGER train/inference Hydra compose + instantiate smoke。
  - `collate_fn_inference_for_sequence` 输出仍包含 `user_id_list`，但 `transformed_sequences` 不包含 `user_id`。
  - `predict_step` 输出 `ModelOutput.keys == batch.user_id_list`。
  - 训练/推理模型配置不再暴露 `num_user_bins` 或 `feature_to_model_input_map.user_id`。
