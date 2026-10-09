# v5.4 结题后的完整目标缺口

2026-10-06，基于当前研究状态、已闭合账本和五组实际训练／Validation主审计，重新审核原目标：单一推荐模型随机初始化，全部模块共同连续训练0→50000，相对匹配的LIGER dense取得Recall@10、NDCG@10各至少8%的可复现收益。整体目标仍未完成，保留v5.2部分收益与v5.4停止决定。

## 已证明与未完成的要求

| 要求 | 当前证据结论 |
| --- | --- |
| 单模型、随机初始化、共同连续50k，无推荐checkpoint初始化／teacher／CP pool／阶段重启 | 五次实际训练均已证明 |
| 与native匹配训练预算 | 均50k更新、global256、1280万sample presentations；不证明等FLOPs、等搜索成本或已收敛 |
| R10、N10各至少+8%，两项配对绝对差CI下界为正 | 五个固定候选均未通过；不能按NDCG单项或跨方案平均补齐 |
| 同一冻结方法的成对seed复现 | 未运行，当前未分配seed43 pair额度 |
| 冻结选择后的完整Testing | 当前scratch阶段未运行；旧续训／pool结果不能替代 |
| 多卡训练、单卡完整推理 | 实际双local rank训练、单local0完整Validation均已证明 |

v5.2相对native为R10+4.6587%、N10+12.9686%，两项CI正，但Recall未达8%。v5.3点估计为+7.0572%／+13.8794%，仍未达Recall门槛；其相对v5.2的增量CI跨0，保留部分证据，不改写为已证实优于v5.2。v5.4负结果停止的是固定catalog-only结构，不否定全部residual路线。

## 一个可以排除的狭窄改进方向

Validation共有22363个用户，native命中2168个。Recall相对+8%的点估计门槛至少需要2342个命中。v5.2现有2269个，缺73个净命中。

cold-target用户只有51个，v5.2目前未命中这些目标。即使假设51个全部命中、所有seen目标结果保持，最多也只有2320个，仍缺22个。因此，从v5.2出发，只改善cold-target结果不足以达标，后续机制还必须改善seen目标的命中取舍。

这是基于既有结果的上限计算，不是预期恢复量，也不证明训练可达该上限或配对CI通过。它没有排除“减少错误cold占位以改善seen排序”的方向；现有输出没有证明cold占位造成seen损失，不能把这一关联直接变成新增模块或实验预算。

## 当前执行边界

最新累计实际成本为5train／250000更新／5完整Validation，全部闭合；新增训练、完整Validation、Testing、seed43和扫描的当前可用额度均为0。原阶段Testing cap不是当前无条件授权，active goal也不重置额度。

本次没有发现阻断既有取舍的实现疑点，不追加checkpoint扫描、重复bootstrap或远端模型评分。继续完整验收需要有正面证据支持的固定方案与明确新增成本授权；本次没有自动提出或启动第六个实验。目标保持active，不标为完成。

## 可复核来源

- [当前完整目标与额度审核](evidence/copmrec-unified-catalog-only-residual-50k-20261006/goal-gap-and-allocation-audit-after-v54.json)
- [五次固定实验的取舍](evidence/copmrec-unified-catalog-only-residual-50k-20261006/stage-decision.json)
- [当前闭合账本](evidence/copmrec-unified-catalog-only-residual-50k-20261006/cumulative-budget-latest.json)
- [v5.4执行与bad-case结果](copmrec-v5-4-execution-20261006.md)
