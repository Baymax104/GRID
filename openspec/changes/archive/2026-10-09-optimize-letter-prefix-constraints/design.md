## Context

诊断见 docs/letter-speed-diagnosis-20261009.md。HF PrefixConstrainedLogitsProcessor 对每个 beam 调用 CPU trie 回调，CUDA 前缀读取和逐行索引引入同步。T5 encoder 执行一次，decoder cache 正常，瓶颈不在重复 encoder 或关闭 cache。

## Goals / Non-Goals

目标：消除逐 beam 的设备往返，并与原 HF 约束逐位等价。
范围外：更换 beam search、修改训练 loss、减少验证范围、重启正式实验。

## Decisions

实现 LETTER 专属 LogitsProcessor：input_ids.tolist() 每步调用一次，CPU 查现有 trie，收集所有合法行列索引后批量写入加性 mask。保持 scores + mask，非法词为负无穷；不对合法词重新归一化。保留原 allowed_tokens 回调作为兼容与差分参照。generate 用 logits_processor 替换 prefix_allowed_tokens_fn，其他参数不变。

GPU trie 有更大的维护与兼容成本；现有批量 CPU 原型已有约 40 倍有限样本收益，先应用已验证的最小改动。处理器仅引用派生 Python trie，不改变 state_dict 或 checkpoint identity。

## Risks / Trade-offs

- 仍有每步一次设备往返 → 有限 checkpoint 核验耗时，后续是否继续优化以实测为准。
- EOS 后 padding 与非法前缀行为可能漂移 → 对 HF 原处理器及完整生成做差分测试。
- 有限样本不代表全量 DDP 验证 → 记录样本数和硬件环境，不声明正式实验结果。

## Migration Plan

聚焦测试通过后使用 mutagen_sync.ps1 flush，同步后读取真实 checkpoint 做有限、无 W&B 发布的验证。正式训练保持停止。回退只需恢复 generate 的原回调装配。
