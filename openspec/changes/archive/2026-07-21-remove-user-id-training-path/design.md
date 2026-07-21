## Overview

TIGER 不再把 `user_id` 当作可训练模型特征。用户身份在当前 pipeline 中有两个不同语义：

1. **序列生成上下文**：由 `sequence_data` 转换后的 semantic ID 序列提供。
2. **输出归属 key**：由原始 `user_id` 标识，用于 prediction writer 合并和下游消费。

本变更将这两条路径拆开：训练路径只保留 semantic ID 序列；推理路径保留 `user_id_list` 作为输出 key，但不再把 `user_id` 注入 `generate()` / `encoder_forward_pass()`。

## Design Decisions

### 1. 训练配置不再读取或映射 `user_id`

`tiger_train` 的 data preprocessing 只需要训练序列，因此训练配置 SHALL 不再保留 `user_id` 作为 `features_to_consider` 或模型输入映射字段。训练 batch 中的模型输入应只包含 label 相关的 `sequence_data`。

### 2. 推理 collate 将 id 字段视为输出元数据

`collate_fn_inference_for_sequence` 当前对所有字段统一 padding/trim，并把 id 字段也写入 `transformed_sequences`。实现时应改为：

- 当字段是 `id_field_name` 时，保存到 `model_input_data.user_id_list`；
- 不对 id 字段执行序列 padding/trim；
- 不将 id 字段写入 `model_input_data.transformed_sequences`；
- mask 仍由第一个非 id 序列字段产生。

这样 `predict_step` 仍可输出 `ModelOutput(keys=user_id_list, predictions=generated_sids)`，但模型不会收到 `user_id` 参数。

### 3. TIGER 模型删除 user embedding 分支

既然不再支持 `user_id` 训练，模型构造参数和内部逻辑应移除：

- `num_user_bins` 参数；
- `self.user_embedding`；
- `generate(..., user_id=...)` 参数；
- `forward(..., user_id=...)` 参数；
- `encoder_forward_pass(..., user_id=...)` 参数；
- 通过 user embedding prepend encoder token 的逻辑。

保留代码注释时应避免继续描述“+1 because we have user_id token”等不再成立的 shape 假设。

### 4. 配置删除 user_id 模型输入映射

`feature_to_model_input_map` 应只表达真正传给模型的特征。TIGER train/inference 配置应删除：

```yaml
user_id: user_id
```

并删除不再消费的：

```yaml
num_user_bins: null
```

推理 data 配置仍可读取 `user_id`，但只供 collate 写入 `user_id_list`。

## Risks

- 推理输出 key 不能丢失：必须用最小 smoke 验证 `predict_step` 返回的 `ModelOutput.keys` 与输入 `user_id_list` 一致。
- collate 行为改变后，依赖 `transformed_sequences["user_id"]` 的旧实验会断开；这是本变更的有意 breaking change，范围限定在 TIGER。
- 若后续重新引入用户特征，应作为新的显式 capability 设计，而不是复用当前 id 输出路径。

## Non-Goals

- 不改变 semantic ID 生成算法、beam search、prefix check 或 evaluator 逻辑。
- 不改变 prediction writer 的输出 bundle 协议。
- 不重新设计通用 non-sequential feature collate contract；本次只处理 TIGER 当前不使用的 `user_id` 模型输入路径。
