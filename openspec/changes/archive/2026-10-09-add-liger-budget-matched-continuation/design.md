## Context

用户于预算审计后明确要求开始固定对照。当前数据/seed/初始50k相同，CoPMRec后增14k并三次optimizer重启、两checkpoint独立融合。64k完整成本与selected权重祖先长度分别记录，原10.77%数值不追改成预算匹配结论。

## Goals / Non-Goals

**Goals:** 新LIGER完整生产链50k+6k+6k+2k=64k、global256；阶段选择、freshoptimizer、history资格和双成员机会匹配；完整源字节、真实CP及独立raw输出验证。

**Non-Goals:** 不调CoPMRec、不扫LR/pair/weight、不证明绝对收敛或同FLOPs、不回填历史源字节，不改变旧研究caps。

## Decisions

1. `LigerBudgetContinuation`继承native Liger，参数phase、expected_pretraining_reference/sha256、可选公共raw checkpoint对象。A仅接受原baseline native45000；B仅接受nativeA Valbest；C仅接受nativeB Valbest。正常nativehook和完整strictstate验证后只复制权重；pretraining metadata记录actual SHA/URI/step与递归祖先，stage checkpoint使用独立liger_budget_* keys，绝不伪装CoPMRec。
2. 每段用统一Trainer.fit且ckpt_path=null，重建AdamW1e-4/wd.035/warm300/horizon6000/min0，DDP2/global256/FP32/clip1/seed42、val1000。A/B选原dense nohistory best，C选historyexcluded best；保留SID/content两loss不变。完整steps与源CPbest来源分开审核，dry-run不计正式成本。
3. `HistoryExcludedLigerBudgetPool`恢复真实A/C nativeCP，校验C经B祖先指向相同A；each独立encode/fullcatalog dense logits后固定half平均，sharedknownhistorymask/stablecatalogrow Top10。singleC normalnativehook同policy推理。只单进程/单GPU推理，不允许poolfit或wrappercheckpoint替代native成员。
4. model/component配置承载参数，薄experiment承载阶段元信息和manualrefs；root脚本支持dry-run、notes两形式及extraoverrides。所有正式runtime字节留source snapshot，代码仅officialMutagen同步，cache不发布。
5. stageA→审计actualbest→stageB→审计actualbest→stageC顺序执行，不能placeholder代替前驱产物。2完整单卡Val（singleC/poolAC）与既有原LIGER同policyVal按N→R→简单方案选唯一baseline；最多1新Test并复用已固定CoPMRec ws2原output。每次SSHlaunch先记录唯一job/pid，传输失败只查原job不盲重试。

## Risks / Trade-offs

- 原LIGERbest45k、v0best48k导致selected祖先步数不同 → 各自遵循同选择机会，完整实际budget各64k，另报weightlineage，不能偷换checkpoint。
- 原train source archive缺口 → 重用已验证CPbytes与actual配置；当前新runtime源字节留档，不回填旧历史。
- 相同updates和双成员不等于FLOPs/HPO → 主张限定更新与样本呈现，记录实际运行耗时且不虚称compute等价。
- A/B和C选择资格不同 → 逐段匹配真实CoPMRec政策；最终所有输出统一history资格，跨阶段scalar不充当纯续训增量。
- 更强baseline可能抹去部分10% → 正向保留同budget真实增益；负向收缩10%主张；CI跨0保留不确定，不用Testing再调参。

## Migration Plan

仅新增文件，旧方法路径不改。focused CPU tests/Hydra/bash→officialflush和源hash→node1真实CP CPUpreflight→DDP2 smoke→顺序3正式训练→2单卡Val→冻结baseline→必要1Test→独立输出及来源审计→记录结果。出现实现故障修复相同阶段，不增加研究方法扫描；已消费正式步骤计数保留。
