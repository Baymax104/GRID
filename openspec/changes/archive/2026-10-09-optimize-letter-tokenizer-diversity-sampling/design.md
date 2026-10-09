## Context

证据见 docs/letter-tokenizer-speed-diagnosis-20261009.md。原 diversity_loss 对每个 id 执行 int(labels[index])，labels 位于 CUDA。诊断已用真实 checkpoint 证明批量搬运采样输入可以消除主要同步成本。

## Goals / Non-Goals

目标：消除逐样本标量搬运，保留正样本及随机状态、loss、梯度、optimizer 轨迹。
范围外：修改预算、分组算法/频率、并行参数、精度、Sinkhorn、模型结构或启动完整训练。

## Decisions

在 positives=None 分支每次读取一次 labels.tolist() 和 ids.tolist()，按 codebook 索引递增顺序构建 group 成员列表。仍过滤 self，再逐样本调用 random.choice。构造同设备的 positives tensor 后沿用原检查和 loss。

不新增 group 缓存：当前256行批量输入足够小，缓存增加 update_groups/checkpoint load 的失效复杂度。group 字典插入次序不参与采样，成员次序必须保持原 nonzero 的递增顺序。显式 positives 分支不访问 CPU 采样数据，也不消耗 Python RNG。

## Risks / Trade-offs

- 随机序列可能漂移 → 与旧函数对照 sampled loss、Python RNG state、梯度及多步 AdamW 轨迹。
- 仍有固定批量搬运与错误检查同步 → 保留契约，真实 CUDA 有限测量验证收益。
- 有限 batch 不代表完整实验 → 记录 GPU 负载、batch、checkpoint 和测量范围。

## Migration Plan

聚焦测试通过后使用 mutagen_sync.ps1 flush，同步 node1；strict 加载旧 checkpoint，使用原函数作为差分 oracle。检查完成后保留验证记录，不自动启动正式实验。
