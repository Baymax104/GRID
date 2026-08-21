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

## RVQ 与 RQ-VAE 已知问题记录

本节记录 2026-08-21 对当前实现与论文官方仓库 `snap-research/GRID` 的对照审计结论，用于后续逐步展开修复。对照参考的官方仓库提交为 `2fe3475b2d369580234093f35d52b1a2f54d0472`。

### 当前观察到的训练现象

- `RKMeans` 的量化训练表现相对正常：`train/quantization_loss` 下降，`train/mse` 下降，`frac_unique_ids` 和分层 entropy 上升。
- `RVQ` 的 `train/loss` 等同于 `train/quantization_loss`，当前曲线呈明显上升趋势；但 `train/mse`、`val/mse` 仍下降，下游 TIGER 效果最好。
- `RQ-VAE` 的 `train/quantization_loss` 上升，`train/reconstruction_loss` 下降；由于 reconstruction loss 数值更大且权重同为 1，`train/loss` 总趋势下降。`val/loss` 与 `train/loss` 当前不是同一语义。

### RVQ 问题

当前 `RVQ` 主训练链路与官方实现基本一致：逐层训练、残差归一化、STE 量化、`BetaQuantizationLoss(beta=0.25)`、SGD 优化。需要重点处理的是指标口径和 loss 尺度，而不是优先重写训练算法。

已确认问题：

- `BetaQuantizationLoss` 默认 `reduction: sum`，当前 `configs/model/rvq_train.yaml` 未显式指定 reduction。该标量受 batch size、embedding 维度和当前训练层 residual 尺度影响，不能直接作为跨层、跨阶段的健康度指标。
- RVQ 是逐层训练，`train/quantization_loss` 会在不同 `current_layer` 上复用同一个指标名。跨 layer 直接观察单条曲线会混合不同目标。
- 当 `normalize_residuals: true` 时，residual norm ratio 类指标不再等价于原始 embedding 空间的重构误差。它仍可做趋势参考，但不能直接解释为“原始向量重构质量”。

对现有结果的解释：

- RVQ 下游 TIGER 表现最好，不证明 RVQ 量化训练最健康。更可能是 RVQ 生成的 semantic ID 更低熵、更压缩，短期内更容易被 TIGER 学习。
- `train/quantization_loss` 上升说明当前 codebook 优化信号不稳定，至少不能用该 raw loss 证明训练正常。
- `mse` 下降与 raw quantization loss 上升同时存在，说明当前日志混合了“归一化 residual 逼近效果”和“STE codebook loss 标量”两个不同概念。

后续修复方向：

- 显式记录 normalized quantization loss，例如 per-element / mean reduction loss；是否将训练 loss 本身改成 `reduction: mean` 需要单独实验验证。
- 在训练 payload 中记录 `current_layer`、当前层 loss、当前层是否 initialized，避免把不同层目标画成一条无法解释的曲线。
- 对 residual stats 增加说明或重命名，明确其在 `normalize_residuals: true` 下不是原始空间重构误差。

### RQ-VAE 问题

`RQ-VAE` 的问题比 RVQ 更严重，包含训练策略偏差和指标语义偏差。

已确认问题：

- 当前 `training_step` 的 `train/loss` 是 `quantization_loss_weight * quantization_loss + reconstruction_loss_weight * reconstruction_loss`。
- 当前 `eval_step` 的 `val/loss` 只来自 `forward(encoded_embeddings)` 返回的 quantization loss，没有计算 reconstruction loss，也没有返回完整 eval loss。因此 `train/loss` 与 `val/loss` 不是同一目标。
- 当前 `_compute_output_stats` 使用 encoded 空间的 residual，但传入原始 768 维 input embedding 作为 norm denominator。对于 RQ-VAE，encoder 输出维度为 64，因此 residual ratio / mse 的空间语义与 RKMeans、RVQ 不一致。
- 当前本地实现采用 progressive joint unlocking：上一层初始化后下一层参与训练，所有层初始化后 reconstruction loss 才进入总目标。该策略与官方配置中 `train_layer_wise: true` 及 reconstruction stage 的意图不一致。
- 官方实现本身也存在 eval 和 reconstruction stage 语义不清的问题，因此后续修复不应盲目逐行回退到官方代码，而应明确目标语义后再实现。

对现有结果的解释：

- reconstruction loss 下降说明 encoder/decoder 正在更好地重构 normalized input embedding，但不代表 codebook 质量提升。
- quantization loss 上升说明 encoded 空间中的 residual-to-codebook 匹配在变差，或者至少被 reconstruction 目标主导后没有得到稳定优化。
- `train/loss` 下降主要由较大的 reconstruction loss 下降驱动，掩盖了 quantization loss 的恶化。
- 当前 `val/loss` 实际更接近 validation quantization loss，不是 validation total loss，因此不能直接与 `train/loss` 对比。
- 当前 RQ-VAE 的 `mse` / residual ratio 不能直接横向比较 RKMeans 和 RVQ；它混合了 encoded residual 与原始 embedding norm。

后续修复方向：

- 先修指标语义：让 RQ-VAE eval 同时返回 `loss`、`quantization_loss`、`reconstruction_loss`，并在配置中挂载对应 `val/*` 和 `test/*` 指标。
- 拆分 RQ-VAE 统计空间：encoded 空间指标使用 encoded embedding 作为 denominator，命名为 `encoded_*`；reconstruction 空间指标通过 decoder 输出与 normalized input 比较，命名为 `reconstruction_*`。
- 再修训练阶段：引入显式阶段机，例如 reconstruction warmup、逐层 codebook 训练、joint finetune。每个阶段应有独立日志字段，避免不同目标混在同一个 loss 曲线中。
- 修复后必须重新跑 tokenizer 训练，并用 TIGER 下游 Recall/NDCG 与 Tail-SID diagnosis 共同验证，不能只看 tokenizer loss。

### 推荐展开顺序

1. 仅修日志和指标口径，不改变训练行为。
2. 用短实验确认 RVQ/RQ-VAE 曲线是否变得可解释。
3. 单独评估 RVQ `reduction: mean` 或 normalized loss 对训练和下游结果的影响。
4. 单独实现并评估 RQ-VAE 显式阶段机。
5. 最后用下游 TIGER 指标和 Tail-SID diagnosis 判断 semantic ID 是否真正改善。
