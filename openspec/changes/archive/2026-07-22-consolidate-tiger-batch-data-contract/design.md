## Context

TIGER sequence 链路已经从通用推荐/embedding retrieval 训练壳收敛为自包含生成式推荐模型，但 data model 仍保留通用 masked-token 协议的命名和字段：`SequentialModelInputData.transformed_sequences`、`mask`、`user_id_list`，以及 `SequentialModuleLabelData.labels`、`label_location`、`attention_mask`。这些字段中只有 `sequence_data`、encoder attention mask 和推理 output key 被 TIGER 实际消费；`label_location` 与 label attention mask 不再参与模型计算。

同时，`LabelFunctionOutput` 仍以 `sequence` / flatten `labels` / `label_location` 表示输出，导致 TIGER 在 `model_step()` 中需要把 `(batch_size * num_hierarchies,)` 的 labels reshape 回 `(batch_size, num_hierarchies)`。这暴露了旧通用协议，而不是 TIGER 的真实训练目标：给定历史 semantic ID 序列，生成下一个 item 的完整 semantic ID。

本变更基于用户确认：允许删除 `feature_to_model_input_map`，也允许删除未使用的 `identity_label`。

## Goals / Non-Goals

**Goals:**
- 用 TIGER 专用 dataclass 明确表达模型输入、标签和 label function 输出。
- 让 `target_ids` 原生使用 `(batch_size, num_hierarchies)`，删除 flatten label 与 `label_location` 遗留协议。
- 让 TIGER 模型直接消费 `input_ids` / `attention_mask` / `target_ids`，删除 `feature_to_model_input_map` 间接映射。
- 保持当前 TIGER semantic ID masking 语义、训练 loss 语义、推理输出 keyed bundle 语义不变。

**Non-Goals:**
- 不改变 item embedding、rkmeans、rvq、rqvae 等 item-level batch contract。
- 不改变 TFRecord reader、dataset preprocessing、semantic ID lookup 或 causal duplicate augmentation 的核心行为。
- 不引入兼容旧 `Sequential*` / `LabelFunctionOutput` 类名的 alias。

## Decisions

1. **以 TIGER 命名替代通用 Sequential 命名**

   在 `src/data/components/data_models.py` 中定义：

   ```python
   @dataclass
   GeneratedLabels = namedtuple("GeneratedLabels", ["input_ids", "target_ids"])

   @dataclass
   class TigerModelInput:
       input_ids: torch.Tensor
       attention_mask: torch.Tensor
       output_keys: torch.Tensor | list[str] | None = None

   @dataclass
   class TigerLabelData:
       target_ids: torch.Tensor
   ```

   这些名称直接对应 TIGER 运行时概念：generated label tensors、encoder input、encoder attention mask、prediction output keys、decoder target semantic IDs。

2. **`next_k_token_masking` 返回二维 `target_ids`**

   `next_k_token_masking` 保持“最后 `next_k` 个 token 是下一个 item semantic ID”的 masking 语义，但返回 `GeneratedLabels(input_ids=masked_sequence, target_ids=target_ids)`，其中 `target_ids.shape == (batch_size, next_k)`。这样 TIGER 不再需要根据 batch size reshape flatten labels。

3. **训练 collate 只支持 TIGER 单序列 label 字段**

   当前 TIGER sequence 链路只有 `sequence_data` 一个模型序列字段。`collate_fn_train` 应直接构造 `TigerModelInput` 和 `TigerLabelData`；如配置出现多个 label 字段或没有 label 字段，应显式报错，而不是继续构造 dict-of-labels。

4. **推理 id 字段只进入 `output_keys`**

   `collate_fn_inference_for_sequence` 对 `id_field_name` 字段只设置 `TigerModelInput.output_keys`，不进入 `input_ids`，也不参与 attention mask。推理结果仍由 `predict_step()` 写成 `ModelOutput(keys=output_keys, predictions=generated_sids)`。

5. **删除 `feature_to_model_input_map`**

   TIGER 模型只有固定输入 `TigerModelInput.input_ids`。模型配置和 `SemanticIDEncoderDecoder.__init__` 不再接受 `feature_to_model_input_map`；`model_step()` 直接调用 `generate(..., input_ids=model_input.input_ids)` 或 `forward(..., input_ids=model_input.input_ids, future_ids=label_data.target_ids)`。

6. **删除 `identity_label`**

   `identity_label` 当前没有官方配置使用，且在 TIGER 专用 label output contract 下没有明确业务含义。删除它可以避免暴露未使用的 label API。

## Risks / Trade-offs

- **风险：旧类名或字段残留导致运行时 AttributeError** → 使用 grep 覆盖 `SequentialModelInputData`、`SequentialModuleLabelData`、`LabelFunctionOutput`、`transformed_sequences`、`user_id_list`、`label_location`、`feature_to_model_input_map`、`identity_label`。
- **风险：label shape 改为二维后 loss/evaluator 输入不一致** → 增加 train collate smoke 和 `SemanticIDEncoderDecoder.model_step()` smoke，验证 `target_ids.shape == (batch_size, num_hierarchies)` 且 loss 可计算。
- **风险：推理 output key 丢失** → 增加 inference collate + `predict_step()` smoke，验证 `ModelOutput.keys` 来自 `TigerModelInput.output_keys`。
- **风险：Hydra 配置仍传旧参数** → 对 `tiger_train` / `tiger_inference` 执行 compose + instantiate，验证模型配置不再包含 `feature_to_model_input_map`。
