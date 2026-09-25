## Context

参考 facebookresearch/liger commit b6ccc37af5ee623ddc1d1ead3490c31aaeaf4524，训练与预测遵守 GRID 的统一入口和 bundle 协议。

## Goals / Non-Goals

依赖 add-liger-content-model，实现官方全词表生成、合法 SID 解析、冷启动商品补入及纯 cosine 重排；安全拒绝非法 SID 映射为商品0。

目标为忠实的 LIGER 方法在 GRID 数据协议中的适配，不声称复现论文数值，不改变既有 TIGER/CGBS 行为。

## Decisions

- 使用独立模块而非 CGBS 子类，避免继承混合概率与额外约束。
- 保留官方共享投影、全词表 T5 CE、无 EOS 目标的固定长度 SID 生成；GRID 的完整 SID 含去重位。
- 原始内容维度从 bundle 获取，不做 PCA；训练商品集合仅从 training 数据取得，冷启动为目录减训练商品集合。
- 明确记录数据划分、历史窗口与官方差异，官方代码的非法候选 argmax 映射和重复候选以安全去重处理。

## Risks / Trade-offs

- [GRID 数据并非官方数据协议] → 文档显式标为 GRID-adapted baseline。
- [重复 SID 使内容 lookup 不唯一] → fail closed，要求完整去重 SID。
- [全目录 CE 成本] → 可配置目录投影分块；先做有界 dry run，不自动全量训练。
- [checkpoint 输入错配] → 保存目录身份与训练集合并严格核验。
