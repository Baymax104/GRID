## Context

用户当前目标是单模型随机初始化、每个最终模型固定 50000 更新，相对同预算 LIGER dense 至少双指标 8% 并可复现。原约 8% 来自双方64k生产预算及双checkpoint pool，不证明50k单模型效果。源码确认 CollaborativeResidualCoPMRec 可不提供预训练，零残差不耗RNG、同表同时服务历史与目录、三loss均向seen残差传梯度；旧scratch-ranking的teacher/calibration不属于当前方案。

## Goals / Non-Goals

**Goals:** 连续0→50000训练单成员模型，明确初始参数、完整日程、loss/推理与checkpoint契约，比较同seed/native50k，冻结方案后成对第二seed复现。

**Non-Goals:** 不将早期CP或50k后续训作为最终scratch证据，不做teacher、阶段优化器重启、pool、alpha/bias/mixed/historyCE扫描，不扩大数据集或更换上游SID/content特征。

## Decisions

1. 模型复用v0 learned-alpha三loss，seen商品残差零初始化、scale1，并在历史与目录投影后共享；cold残差严格为0。新薄派生类仅增加连续预算、来源、checkpoint与raw-validation/eligible-deployment语义，不添加另一训练模块。
2. 每个模型单次50k，主干LR3e-4、item peak .002（multiplier20/3），AdamW wd.035、warm2500/cos50000/min0、global256/seed42、FP32/clip1。item LR沿用已保留正向残差的绝对尺度，避免将warmstart multiplier20直接乘scratch主干而变成.006；只检验此一固定值。
3. 训练内checkpoint选择与原native相同：完整目录raw dense Val NDCG@10，每500步、各自best；完整promotion Validation与最终Testing在独立单卡predict中共享history资格规则。新class可override短eval_step调用原Liger.retrieve得到rawdense，公开retrieve使用同history exclusion，标签不进入排序。最终只有一个best，不平均不同训练时刻。
4. 首个正式训练直接scratch。实时v0 producer inventory仅发现best48000；早期35k作为用户允许的可选快速探针未启用。跨架构v0→residual optimizer需要moment/name迁移，不为省时引入额外阶段或声称weights-only恢复等于连续训练。
5. seed42主baseline为原LIGER35ig0tz6完整50k、自己raw Valbest45000，同history完整Val wdms8w77、Test vnhmag7v。64k pool仅辅助参考。首个完整Val双8且paired绝对CI下界正后冻结代码/超参数、执行nativeLIGER43与相同candidate43各完整50k，不能用42 baseline作为43分母。
6. 新stage累计上限3正式训练/150k，每个生产链≤50k，完整Val≤3，新Test≤3；3个实际训练为candidate42、native43、candidate43，不开启optional warm probe。首个Val未过时，剩余额度登记保留而不自动消耗或重置；先做有限已保存输出bad case，不扫参数或进入Testing。第二seed结果必须全部报告，不以其表现决定隐藏seed。
7. 两个seed的checkpoint均只按固定Val选择；第一次Testing前冻结两seed模型及baseline。Testing最多candidate42/native43/candidate43三次、全部单卡，原native42输出复用；各seed双R/N≥8%且paired绝对CI下界>0为效果复现要求，不用均值补某seed未达、不用Testing改模型。
8. 用户已明确授权node1正式训练与推理。训练优先物理5,6→CUDA_VISIBLE_DEVICES=5,6→local[0,1]、每卡128；推理物理5→[0]。实际启动前检查空闲，不碰其他进程。官方Mutagen flush后归档真实runtime bytes、dirty/untracked、W&B resolved config/notes/checkpoint/output来源。

## Risks / Trade-offs

- scratch初期残差可能主导未训练projection → 零残差、保留v0初始RNG、三loss真实梯度与cold0检查，效果仅由完整50k/Val/Test证明。
- 不能从CP本身证明实际已花费更新 → driver唯一job/counter/terminal audit，禁止finishedrun或best权重重新fit；中断恢复只能same-run完整optimizer/scheduler/globalstep恢复，并如实计入已消耗或重放更新。
- 方法多1,548,928个残差参数 → 报告参数与更新预算，不能声称同参数量/FLOPs/HPO。
- 原seed42训练source archive历史缺口 → 继续披露，不由新source backfill；新native43与candidate43用同runtime归档成对检验。
- Beauty及既有反复开发split → 精确保存scope，bootstrap条件于固定模型/用户，第二seed支持有限训练复现，不能称未触碰独立数据或普遍收益。

## Migration Plan

新增薄模块/config/script，不修改旧模型/协议。先聚焦CPU tests/compose/Bash/source审计，再官方flush、真实CPU初始化/双卡一步smoke，明确0formal效应；通过后唯一scratch42完整50k。失败修复实现证据与训练效果分开，旧报告/outputs保持字节不变。

## Open Questions

50k单模型是否能保留至少8%，及第二seed是否可复现，尚未测得。若否定，应保留真实正反证据与剩余额度，在同累计成本与原目标下重新判断，不以旧pool结果补验收。
