## Context

用户授权长时自主scratch50k研究和node1正式运行。v5 firstVal的强NDCG/Top5正向被保留，原双8 gate未过；已有bucket矩阵及100点raw训练轨迹完成有限bad case，不存在生成候选遗漏瓶颈，末段尾部桶亦未崩塌。没有证据归因alpha、residual或某项loss。

实际证据为`docs/evidence/copmrec-unified-50k-20261005/{training-candidate42,inference-val-candidate42,candidate42-badcase-summary,training-trajectory-head-tail}.json`。v5 source390/f67ca2d4...、训练m50dan21实际50k、ownbest41k；独立Val8w893ra3有22363同policy用户。native42仍为35ig0tz6/rawbest45k/Valwdms8w77。

## Goals / Non-Goals

**Goals:** 一次固定机制检验，判断共享query的双目录评分是否增加Top10净命中，同时保持v5头部收益。最终目标仍为相对同seed同预算native的R10和N10各至少8%、paired CI下界正且成对复现。

**Non-Goals:** 不开展alpha/weight/temperature/CP扫描，不续训或使用外部teacher，不新增第二T5或额外阶段，不宣称当前已达到双8或双seed验收，不使用Testing筛选。

## Decisions

1. 记投影为p_i、seen残差为r_i、唯一history query为q。分数为normalize(q)点乘[0.5normalize(p_i)+0.5normalize(p_i+r_i)]再除以当前temperature。平均向量不再normalize，以严格等于两路cosine logits的平均。
2. 内容目录只执行一次同样含dropout的projection，两个视图复用。用父类item_content_residual保留seen/cold语义；history encoding完全沿用v5。它不是独立native LIGER视图，因为query读取history residual。
3. dense CE与legal-prefix mixture NLL共同读取同一个final logits；unit SID CE和learnedalpha不变。不存在单独双head CE、第二query、第二组参数或查询T5计算。zero-residual初始化与v5数值/RNG同；catalog残差一阶梯度约减半，history路径不变，成功也不能单独归因视图互补。
4. 新类派生UnifiedScratchCoPMRec，版本v5.1且contract包含catalog scoring规则。保留strict随机起点、同50k日程/optimizer/cold/full-state检查，拒绝其他版本和wrong scoring；不修改已冻结旧模型。
5. 训练与选择完全沿用v5：seed42、DDP2每卡128/global256、FP32、peak主干.0003/residual.002、wd.035、warm2500/cos50000/min0、clip1；每500 raw ValN10选ownbest。完整Val仅单卡，同history资格/标签不参与排名/稳定ties。
6. 最小对照仅已冻结v5与native42输出。一次v5.1 scratch50k和一次完整Val；不生成新旧checkpoint组合。成功目标沿用双8+双CI正；若Recall/净命中不改善或通过明显头部下降换覆盖，双目录假设不获支持，即使保留任何实际部分正向结果也不得移门槛。
7. 初始研究累计上限仍3train/150k/3fullVal/3newTest；当前已用1train/50k和1completeVal（2startup，1零输出失败）。本修改将第2个训练和第2个Val指定给v5.1，最终最多3train不变，余下1train保留未分配。本阶段不自动训练43或进入Test。只有取得新的真实正面证据后再决定复现所需的具体预算，不能将原剩余1槽描述为2个43模型。
8. source由原390文件逐字节保持加新运行文件组成；官方Mutagen flush成功后，真实CPU起点和DDP2一步smoke验证；仅root编排唯一正式job/预算/PID/W&B/source。不得停止其他node1进程。

## Related Work and Evidence Boundary

LIGER已融合内容与SID，不能将普通内容投影或ID embedding声称首创。旧两模型logitpool的不同query/训练轨迹与本共享query不同；该已保留正向证据加v5真实头部/覆盖不平衡支持有界检验，不是因果证明。ComiRec多兴趣与本两种目录评分不同，不能直接迁移效果。[LIGER](https://arxiv.org/abs/2411.18814)、[ComiRec](https://arxiv.org/abs/2005.09347)

不同时新增EMA/SWA：非线性参数平均不等于输出平均；SWA原constant/cyclic SGD轨迹与当前AdamW单调cos不同，旧pool不足以直接支持它。[SWA](https://arxiv.org/abs/1803.05407)

## Risks / Trade-offs

固定平均可能压低已有正确高分；共享query可能不能保留原native的命中，两目录亦可能趋同。新增参数0、T5 forward次数不变，但向量操作/activation不同，须实测资源，不称同FLOPs。v5.1对v5的变化同时涉及catalog gradient与几何，无单因素因果主张。

两seed完整验收目前未授权到具体新分配，旧3run预算会因这次有依据迭代少一个配对槽；goal active与预算未完成都须如实报告，不能标complete或借该版本名重置账本。
