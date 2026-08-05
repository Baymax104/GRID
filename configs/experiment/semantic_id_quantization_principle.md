# Semantic ID 量化核心原理

本文简要说明 residual quantization 中 `embedding`、`codebook`、`codebook_size`、`n_layers` 与 semantic ID 的关系。

## Codebook 的形状

在量化模型中，每一层都有一个 codebook。单层 codebook 可以理解为一张向量表：

```text
codebook.shape = (codebook_size, embedding_dim)
               = (n_clusters, n_features)
```

其中：

- `codebook_size` / `n_clusters`：该层有多少个可选 code，也就是多少个 centroid。
- `embedding_dim` / `n_features`：每个 centroid 的向量维度。
- `n_layers` / `num_hierarchies`：有多少层 residual quantization。

因此，多层时可以概念化理解为：

```text
(n_layers, codebook_size, embedding_dim)
```

实际代码中使用的是每层一个 centroid 参数表，而不是必须合并成三维 tensor。

## Embedding 与 Codebook 的关系

输入 embedding 是连续向量：

```text
embedding.shape = (embedding_dim,)
```

单层量化时，会把该 embedding 与当前层 codebook 中所有 centroid 比较距离，并选择最近的 centroid：

```text
code_id = argmin(distance(embedding, codebook[i]))
```

因此，一个连续 embedding 会被映射成一个离散整数 `code_id`：

```text
code_id ∈ [0, codebook_size)
```

## Residual Quantization 如何生成 Semantic ID

Residual quantization 会逐层量化剩余残差：

```text
原始 embedding
  │
  ├─ 第 0 层：选择最近 centroid，得到 code_id_0
  │          residual_1 = embedding - codebook_0[code_id_0]
  │
  ├─ 第 1 层：量化 residual_1，得到 code_id_1
  │          residual_2 = residual_1 - codebook_1[code_id_1]
  │
  └─ 第 N 层：继续量化上一层 residual
```

最终，一个 embedding 得到的 semantic ID 形状是：

```text
semantic_id.shape = (n_layers,)
```

其中每个位置都是对应层 codebook 的编号：

```text
semantic_id[i] ∈ [0, codebook_size)
```

例如：

```text
n_layers = 4
codebook_size = 256

semantic_id = [12, 87, 203, 5]
```

含义是：

```text
第 0 层选择 codebook_0[12]
第 1 层选择 codebook_1[87]
第 2 层选择 codebook_2[203]
第 3 层选择 codebook_3[5]
```

## Batch 情况

如果输入是一批 embedding：

```text
embeddings.shape = (batch_size, embedding_dim)
```

则输出 semantic IDs 为：

```text
semantic_ids.shape = (batch_size, n_layers)
```

每一行对应一个 item 的 semantic ID。

## 一句话总结

`codebook` 是离散 code 到连续向量的映射表；residual quantization 会用多层 codebook 逐步逼近原始 embedding，并把每层选中的 code 编号组成 semantic ID。
