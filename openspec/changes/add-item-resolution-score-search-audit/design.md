## Context

保留的六条训练已核对；主比较使用 MIR/固定第二层的 best checkpoint。现有训练有目标 NLL，但没有全目录排名或搜索漏失，无法据此选择下一结构。

## Goals / Non-Goals

目标：一次有界、不重训的证据采集，分离模型分布与预算搜索。不是新的全量效果实验，不据128用户的稀疏Hit指标声称显著增益。

## Decisions

- 使用统一 inference/predict 入口加载已有 checkpoint，新增独立子类和 evidence bundle；这是 checkpoint 推理取证，不另设离线 runner。
- 对 evaluation 全部用户按 SHA256(seed,user_key) 选最小128个，排序固定，抽样不读取标签。数据仅CPU扫描一次，禁止多worker/多卡造成不同样本。
- 每用户枚举全部目录 SID，分块128调用现有精确边缘目标，保存完整目录分数、精确rank、目标责任、全局出口质量及质量归一化检查。真实标签只在分布计算后取指标，不能注入候选。
- 对同一用户运行Q=64/128/256，S固定4096；记录近似Top-K、目标rank（未命中=-1）、证书、剩余质量、对精确Top-K的重合及返回物品的概率保留比例。S饱和时不能将Q不改善解释为评分无效。
- 默认每模型128用户、单卡、batch1，共4次串行；精确chunk只控制显存，不改变分布。保存user、输入hash、label、catalog/checkpoint指纹便于跨模型严格配对。
- 目标排名按概率降序、并列item key升序。精确搜索为诊断上界，不作为部署方案或效率baseline。

## Risks / Trade-offs

- 精确枚举仍耗时 → 固定128用户、chunk128、4个checkpoint；不追加seed，不许无限扩展矩阵。真实GPU耗时未知。
- 小样本随机波动 → 优先报告配对目标rank/logprob、完整分布与预算误差；Hit/NDCG只作为解释，不作论文胜负。
- checkpoint/输入错配 → 延用模型contract，新增输入hash及样本协议；对比必须检查keys、labels和输入hash。
- 运行失败少于128用户 → bundle记录实际keys；正式决策要求每个run包含128用户，同一数据集的两个run用户完全相同，dry-run不能计为有效。

## Migration Plan

不修改或恢复旧队列。用户只启动新审计队列，完成后依据分布与搜索两类证据决定去留。

## Open Questions

无实施阻塞；方法如何修改取决于取证结果。
