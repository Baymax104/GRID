# Embedding 实验数据链路说明

本文记录 `experiment=sem_embeds_inference` 中，数据从原始 TFRecord 到进入 embedding 模型前的数据处理过程。

## 对应入口

- 配置：`configs/experiment/sem_embeds_inference.yaml`
- datamodule：`src.data.datamodules.item.ItemDataModule`
- collate：`src.data.components.collate.collate_fn_items`
- 模型：`src.embedding.semantic_embedding_inference_module.SemanticEmbeddingInferenceModule`

## 原始字段

该实验在配置中定义的核心输入字段为：

- `id`
- `text`

其中：

- `id` 用作 item 标识
- `text` 是原始文本特征

底层文件由 `TFRecordReader` 读取，样本解析后首先表现为 `dict[str, tf.Tensor]`，其中常见是 `SparseTensor` / `RaggedTensor` 一类的 TensorFlow 结构。

## 预处理顺序

`sem_embeds_inference.yaml` 中的 `preprocessing_functions` 顺序如下：

1. `filter_features_to_consider`
2. `convert_to_dense_numpy_array`
3. `squeeze_tensor_in_place`
4. `convert_fields_to_tensors`
5. `convert_bytes_to_string`
6. `tokenize_text_features`
7. `squeeze_tensor_in_place`

## 字段形状变化

### 1. 原始 TFRecord 解析后

单条样本可以近似理解为：

```python
{
    "id": tf.SparseTensor | tf.RaggedTensor,
    "text": tf.SparseTensor | tf.RaggedTensor,
}
```

### 2. `convert_to_dense_numpy_array`

将 TensorFlow 稀疏结构转成 dense numpy：

```python
{
    "id": np.ndarray,    # 常见近似 shape: (1,)
    "text": np.ndarray,  # 常见近似 shape: (1,)
}
```

### 3. 第一次 `squeeze_tensor_in_place`

只处理 `text`，去掉无意义的额外维度。通常仍可近似视为：

```python
"text": np.ndarray  # shape 接近 (1,) 或单值包装
```

### 4. `convert_fields_to_tensors`

只处理 `id`，并按配置中的 `torch.int32` 转成 PyTorch Tensor：

```python
"id": torch.Tensor  # shape: (1,), dtype=torch.int32
```

### 5. `convert_bytes_to_string`

只处理 `text`，将 bytes 转成字符串数组：

```python
"text": np.ndarray[str]  # 常见近似 shape: (1,)
```

### 6. `tokenize_text_features`

该步骤最关键，会把 `text` 转成 tokenizer 输出，并新增 `text_mask`。

配置中 tokenizer 关键参数：

- `max_length: 128`
- `padding: max_length`
- `truncation: true`

因此单条样本在这一步后通常变成：

```python
{
    "id": torch.Tensor([item_id]),      # shape: (1,)
    "text": torch.Tensor(...),          # shape: (128,)
    "text_mask": torch.Tensor(...),     # shape: (128,)
}
```

这里的 `text` 已不再是原始字符串，而是 tokenized 后的 `input_ids`。

### 7. 第二次 `squeeze_tensor_in_place`

处理：

- `text`
- `text_mask`

由于此时两者通常已经是一维 `(128,)`，这一步一般不会再改变形状。

## Collate 后的数据结构

batch 进入 `collate_fn_items` 后，会被包装成 `ItemData`：

```python
ItemData(
    item_ids=...,
    transformed_features={...}
)
```

同时配置中定义了字段映射：

```yaml
feature_to_input_name:
  id: item_ids
  text: text_tokens
  text_mask: text_mask
  embedding: input_embedding
```

所以 batch size 为 `B` 时，进入模型前通常为：

```python
ItemData(
    item_ids: torch.Tensor,              # shape: (B, 1) 或 (B,)
    transformed_features={
        "text_tokens": torch.Tensor,   # shape: (B, 128)
        "text_mask": torch.Tensor,     # shape: (B, 128)
    },
)
```

## 模型实际消费的输入

`SemanticEmbeddingInferenceModule` 会根据配置里的映射：

```yaml
semantic_embedding_model_input_map:
  input_ids: text_tokens
  attention_mask: text_mask
```

把 `ItemData.transformed_features` 映射成 Hugging Face T5 encoder 所需输入：

```python
input_ids = model_input.transformed_features["text_tokens"]
attention_mask = model_input.transformed_features["text_mask"]
```

因此模型前最终张量形状通常为：

```python
input_ids: torch.Tensor        # (B, 128)
attention_mask: torch.Tensor   # (B, 128)
```

## 输出形状

`SemanticEmbeddingInferenceModule.predict_step()` 输出语义向量，并包装成 `OneKeyPerPredictionOutput`。

常见输出可近似理解为：

```python
predictions: torch.Tensor  # shape: (B, hidden_dim)
```

对于当前配置中的 `flan-t5-base`，`hidden_dim` 常见为 `768`，因此通常可近似看作：

```python
(B, 768)
```

## 一句话总结

`sem_embeds_inference` 的链路可以简化为：

```text
TFRecord(id, text)
-> dense numpy
-> id 转 tensor / text 转 string
-> text tokenize 成 (text_tokens, text_mask)
-> collate 成 ItemData
-> 模型消费 (B, 128) 的 input_ids / attention_mask
-> 输出 (B, 768) 的语义向量
```
