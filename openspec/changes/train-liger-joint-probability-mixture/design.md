## Context

当前 Liger.losses 优化完整词表 SID CE 与商品内容 CE；混合概率只用于解码或冻结概率缓存上的门控学习。Beauty/seed42 四臂 testing 支持整体方法和相对固定0.5的权重校准增量，但 constant 相对 content-only 的区间跨零；动态门控和学习排序均没有建立各自预设优势。旧阶段已结题，新阶段解决训练目标是否能增强混合候选生成，不恢复旧预算。

核心可反驳假设：让主模型在训练中直接优化推理使用的混合条件概率，能在匹配50k预算下超过原始LIGER，并相对仅解码混合提供额外收益。现有结果是正面动机而非本假设的证明。定位为LIGER上的训练目标与内容引导解码对齐，不声称概率混合组件本身首创；精确文献贡献比较不在本工程提案中完成。

## Goals / Non-Goals

目标：准备一次从零联合训练，给出可审计的效果和机制判断。推荐模型的Transformer、内容投影和位置等参数全部训练，增加一个全局可学习bias，alpha=sigmoid(bias)，初始0.5。

边界：SID与预计算内容向量固定；不训练语义编码器/量化器，不增加动态网络或独立排序器，不自动启动GPU完整实验。超过LIGER是结果验收标准，不保证正向结果。

## Decisions

### 训练目标

本次采用 L_total=L_sid+L_content+L_mix，三个系数均为1，不搜索权重。L_sid/L_content保持原LIGER定义；新增L_mix为四个真实SID前缀目标的混合NLL均值，与SID CE使用相同token平均尺度。保留原分支监督以避免仅用混合NLL导致强分支吸收梯度；这是联合目标版本，不能称为仅优化混合NLL。

Pmix=(1-alpha)Pg+alpha Pc。Pg在全目录合法下一SID分支上归一化；Pc由当前可训练内容分数在子树内logsumexp聚合后条件归一化。混合分支使用完整目录，与解码支持集一致；原内容CE仍保留原训练seen mask，必须使用独立张量，不能把mask后的logits传入混合分支。仅训练样本提供目标，cold目录信息的可用性与原LIGER一致，不能把此设计称为完全归纳设置。

teacher forcing时只用目标前缀，推理不使用目标；共享概率定义但不宣称已对齐实际beam排序损失。提取无detach/no_grad的共享分布函数；不要将逐步生成接口本身用于训练。每次训练forward仅计算一次全库内容logits，复用同一dropout realization；聚合索引可缓存，带梯度分数不能跨step缓存。主模型checkpoint保存标量，不另用冻结门控checkpoint覆盖它。

### 匹配基线与一次实验范围

首个实例为Beauty/seed42、50k optimizer steps、warmup2500、batch256、fp32、每500步dense验证，以val/ndcg@10选checkpoint；继承既有50k LIGER的模型、数据、优化器、scheduler及验证机会。初始化共享参数须与相同seed的基线一致，新标量用确定性初始化，不能消耗额外随机数改变基座初始化。输入SID/embedding保持Artifact ID/digest一致。

复用50k基线3mntjejz及其原hybrid testing yyzwh5xl，但实现交付前必须核对resolved config和来源；不将30k结果当匹配基线。新训练不恢复best49k。testing已查看，本次是后续探索，不称全新独立确认。

预算上限1次完整训练+3次prediction：新模型混合推理、既有50k基线固定0.5混合推理、新模型alpha1推理。后两者分别提供无新增训练的解码控制与生成概率移除消融；固定0.5对照不完全隔离“整体训练vs最优两阶段校准”，alpha1消融也不是独立训练的纯内容模型。所有候选为beam20并cold，内容终排；报告全部@5/@10、覆盖、有效候选数和新增/损失命中。

训练实际成本未知，既有50k基线约6449秒（107.5分钟，包含验证），只能作为量级参考；混合聚合及反向额外成本通过轻量检查报告step时间/显存，不能按GPU利用率推断。推理历史约80–90秒/次含trace，3次约4–5分钟仅为参考。显存或运行失败时记录失败，不自动缩batch、减目录或增加完整训练次数。

### 决策与停止

主验收为新模型混合−原始50k LIGER hybrid的NDCG10配对95% CI下界>0，Recall10点估计不下降。采用用户配对bootstrap1000次、NumPy PCG64 seed42。超过baseline必须是同split/用户的比较，不混用dense/evaluation/testing。

辅助比较新模型混合−基线固定0.5混合用于判断相对简单解码改动是否有增量；若主比较通过但辅助不明确，只支持整体效果，不声称联合训练必要。新模型混合−自身alpha1若不明确，不声称生成分支必要。

最多三个结论：主门槛通过则保留整体方法并按辅助证据限定机制；主门槛未通过/不确定则结题，不为叙事调参；出现真实实现或来源缺陷则判实验无效，先纠错并记录协议修订，不能作为机制反证。预算不因失败重置。

## Risks / Trade-offs

- 混合梯度偏向强内容分支：保留原分支损失、记录alpha及各损失和有限梯度检查；不保证互补出现。
- 新增目标和合法支持集共同作用：本次验证整体方案，不将所有增益归因于单一因素。
- 全库可微聚合增加显存/时间：先最小合成检查，不宣称检索效率优势。
- 单seed/已查看testing：只作限定范围的探索结论，不升级为跨seed稳定性。

## Migration Plan

使用独立experiment及显式训练模式，默认LIGER不变；通过统一src/main.py进入，脚本支持dry-run、notes及额外override。先实现和验证，再按Mutagen约定交付运行代码；完整运行由用户手动开始。

## Open Questions

2026-09-24实现、测试与node1同步已完成，详见../../../../research/docs/grid-experiments/2026-09-24-liger-joint-mixture-implementation.md。合成微基准已记录，真实目录的额外成本仍未知；当前没有已发现的阻断实现疑点。62项不同聚焦测试及CPU/GPU合成smoke只证明工程契约，不证明推荐效果。完整训练及结果审计仍待用户手动运行。
