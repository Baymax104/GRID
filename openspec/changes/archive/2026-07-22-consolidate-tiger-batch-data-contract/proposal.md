## Why

当前 sequence batch dataclass 仍保留通用 masked-token 训练遗留语义：`transformed_sequences`、`user_id_list`、`label_location`、label `attention_mask` 和 flatten labels 都不能直接表达 TIGER 的真实输入/标签。TIGER 现在只有固定的 semantic ID 序列输入、encoder attention mask、推理输出 key，以及下一个 item 的 semantic ID 标签，继续保留通用协议会增加理解和维护成本。

## What Changes

- 将 `SequentialModelInputData`、`SequentialModuleLabelData`、`LabelFunctionOutput` 收敛为 TIGER 专用 batch contract：`TigerModelInput`、`TigerLabelData`、`GeneratedLabels`。
- TIGER label output 直接返回 `input_ids` 和 `target_ids`，其中 `target_ids` 使用 `(batch_size, num_hierarchies)`，不再使用 flatten labels、`label_location` 或 label attention mask。
- TIGER train/inference collate 直接构造 `TigerModelInput(input_ids, attention_mask, output_keys)` 与 `TigerLabelData(target_ids)`。
- TIGER model 不再通过 `feature_to_model_input_map` 间接映射 `sequence_data -> input_ids`；训练和推理均直接读取 `TigerModelInput.input_ids` / `attention_mask`。
- 删除未使用的 `identity_label`，只保留当前 TIGER 需要的 `next_k_token_masking` label callable。
- **BREAKING**：不再支持旧 `SequentialModelInputData` / `SequentialModuleLabelData` / `LabelFunctionOutput` 类名、旧字段名、`feature_to_model_input_map` 配置项，以及 `identity_label` 入口。

## Capabilities

### New Capabilities
- `tiger-specific-batch-contract`: 定义 TIGER 专用 batch / label / label-output 数据模型及其字段语义。

### Modified Capabilities
- `pure-label-function-contract`: label callable 返回 TIGER 专用 `GeneratedLabels`，并且 `next_k_token_masking` 直接产出二维 `target_ids`。
- `tiger-sequence-data-contract`: TIGER sequence collate、推理 output key 和模型输入 contract 改为固定 `input_ids` / `attention_mask` / `output_keys`。
- `data-model-role-separation`: runtime batch dataclass 名称从通用 sequential 模型收敛为 TIGER 专用模型。
- `self-contained-tiger-generation-model`: 自包含 TIGER 模型不再接受 `feature_to_model_input_map`，直接消费 TIGER batch contract。

## Impact

- 代码：`src/data/components/data_models.py`、`src/data/components/label_functions.py`、`src/data/components/collate.py`、`src/recommendation/tiger_generation_model.py`。
- 配置：`configs/model/tiger_train.yaml`、`configs/model/tiger_inference.yaml`，以及可能涉及 `configs/data/tiger_train.yaml` label callable 的返回 contract。
- Specs：新增 TIGER batch contract，更新 label function、sequence data、data model role separation 和 self-contained TIGER model 相关 living specs。
- 验证重点：旧类名/字段/配置项 grep 清零、compile、ruff、train/inference collate smoke、TIGER model_step/predict_step smoke、Hydra compose/instantiate、OpenSpec strict 和全量 specs validate。
