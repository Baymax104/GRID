# CoPMRec 独立商品打分截距阶段

本阶段已结题：两臂6000步训练及完整Evaluation配对审计均完成，独立bias增量未获支持；两臂均未满足同split LIGER的Recall10/NDCG10双10%晋级门槛，未启动Testing。阶段2次/12000步、全线程5次/30000步训练预算已实际耗尽，不追加bias/temperature/LR扫描。全线程10%目标仍未完成。下方预注册与执行记录保留当时状态，最新结果见文末。

## 决策与授权

用户已明确授权本自主目标新增模块/方法、SSH启动双卡训练和单卡推理，目标始终为固定真实LIGER dense Testing的Recall10/NDCG10均≥10%相对提升。此文登记新问题，原v4自主残差阶段3×6000步已关闭且不追加运行；当前Testing仍只消费1/3，后续确认沿用全线程剩余2次，不另开3次。

## 完整路线门禁

1. **原假设。** 商品身份协同自由度共享历史/目录能改善判别，不只是额外续训。此前v3/v3.1候选修复与alpha固定对照未产生稳定推荐增益，不能只把候选遗漏减少作为当前目标。
2. **支持。** residual与zero-residual同6000步续训六个等步数均正向，各自best +2.47%R/+2.02%N；首个dense Testing +5.26%R/+5.64%N、两项paired CI正。保留这一有限正收益，明确dense部署不声称生成候选贡献。
3. **反证及门槛。** 固定mixed同候选终排净损40命中、两CI负，关闭该规则及前提竞争loss。lr20固定5000标量正向，新best6000对旧v4点估计+.94%R/+1.38%N但两CI跨0，不能证明额外LR收益；相对LIGER Validation +6.92%/+7.15%，仍非原两10%门槛。原残差机制未被否定，目标仍未完成。
4. **关键疑点。** 全量原始标签、输入/输出/checkpoint/source349字节及匹配control已核验，没有会阻断残差阶段结题的实现疑点。新best6000错误Top10的最高频20件商品仍占18.5093%，两固定key子组18.5250%/18.5162%、各自Top20重合19/20，前6名相同；这20件商品的训练causal期望target只占2.9374%。稳定排序现象支持检验独立截距，不能推出概率校准误差或bias缺失是唯一根因。
5. **原预算与取舍。** 原3臂/18000步全部实际完成，剩余训练0，不继续LR/mixed/候选修复；新best6000只作为Validation选出的开发起点，Testing未验证。新问题有当前checkpoint跨组的正面依据，下面明确追加唯一matched pair，不包装为原阶段续接。

完整数字、raw频率规则和固定子组来源见[evidence目录](evidence/copmrec-v4-autonomous-20261005/score-bias-proposal.md)、[新Validation](evidence/copmrec-v4-autonomous-20261005/validation-residual-lr20-audit.json)、[曝光审计](evidence/copmrec-v4-autonomous-20261005/validation-item-exposure-audit.json)。频率依据现有连续子序列target+max32有放回采样后去重规则；审计一轮不是6k实际训练曝光。

## 新问题、定位与机制

**学习独立商品logit截距，是否能在同预算续训之外改善稳定的商品误排和总体dense推荐？** 当前代码没有独立商品score截距；normalized cosine仍可间接表达共同偏好，不作表达不可能主张。

商品bias为成熟组件，[Koren/Bell/Volinsky 2009原论文](https://chrisvolinsky.com/files/publications/ieeecomputer2009.pdf)Eq3–5含商品截距，不能宣称首创。该工作针对显式评分平方误差，本实验针对序列next-item content CE与合法SID-prefix mixture NLL；[LIGER官方实现](https://raw.githubusercontent.com/facebookresearch/liger/main/src/evaluation.py)的归一化内积及ID+text设置为定位依据，不推定本干预收益。

`logits_i=cos(q,z_i)/temperature+seen_i*b_i`，b全零初始化，共12101标量；cold33行值/梯度为0，不进入历史encoder。v4.1只做dense/content部署，保持v0三loss各1和learned alpha；bias直接由content CE及mixture NLL监督，SID CE无直接bias梯度。不用Validation频率拟合bias或指定inverse-prior惩罚，不重试mixed、alpha、temperature或LR扫描。

## 一次性预算与具体对照

本节保留两个训练启动前的预算登记；其中“当前”指登记时状态，不覆盖文末实际结题数量。

两臂均从 `wandb://baymaxam/GRID/nj9elah1?role=checkpoint&alias=v0&file=checkpoint_epoch=000_step=006000.ckpt` weights-only开始，SHA7c52228a665b593b50e45e8aba039f3f4aa4a10ca134d9345c59c59fb142f055。bias scale1可学习对比scale0全零冻结，共同基础参数和已有协同残差继续训练，optimizer/scheduler/step重置。

每臂6000步，seed42、FP32、global batch256（GPU2,4→Trainer[0,1]，128/卡）、clip1、warmup300/cosine6000、val1000、dense NDCG10选best。主干LR1e-4，residual和新增bias共用item参数组LR.002、wd.035，总两组相同，不增独立biasLR旋钮。预先固定step6000匹配标量比较；各自best配对比较明确包含checkpoint选择。

新阶段最多2正式训练/12000步、每臂一次单卡全量Evaluation。wholethread累计最多5正式训练/30000步；当前实际仍3/18000。额外Testing沿全线程累计3次上限：现1次，只有Validation选择结果达到原两10%门槛才消耗下一次。若matched control自身达到目标，可明确报告续训收益并验证它，不强留bias叙事。

两个split分开判定：Validation固定LIGER run5azn5vm0的R10=.08979117291955462/N10=.04647035630072118，+10%晋级阈值R10≥.09877029021151008（至少2209命中）、N10≥.0511173919307933。Testing仍用042139al，验收R10≥.07855386128873586（至少1757命中）、N10≥.04002978116676896。禁止把较低Testing绝对阈值拿来判断Validation晋级。

## 可反驳预测及各结果决定

预测bias应在匹配zero-bias之外提高整体NDCG10且不损害Recall10，并使固定offender涉及的净排序损益更有利；曝光下降、loss下降或bias范数不算推荐成功。不要求所有商品偏置符号相同或每类样本均改善。

整体效果与组件归因分别判断，以下规则在两个新训练启动前固定：两个臂中达到同split LIGER两项10%门槛者，按Validation NDCG10选择唯一checkpoint单卡Testing；若两臂均达到门槛，以更高NDCG10者为winner。再独立复算真实用户/末商品标签、合法完整SID、lineage/source/paired CI。bias相对control的正向增量另行报告；即使整体winner达标，配对增量证据不足也不声称已证明bias有效。精确NDCG10相同则选择冻结bias对照，此tie规则在control启动及两个完整Validation结果产生前补充登记。

正向但整体不足10%：保留有限增量结论，目标仍未完成，该两臂预算不自动增加。负向或配对增量证据不足：关闭bias机制迭代，不扫bias/temperature/LR，不将不显著称等价；若两臂均未达到整体门槛，保留原v4。cold排序可能随seen分数及共享训练变化，如实报告。

## 执行记录（按当时状态保留）

实现与独立审阅完成；173项聚焦CPU回归通过、Ruff和OpenSpec strict通过。Mutagen三Watching无冲突；61个关键文件及355个运行源字节与远端匹配。两个实际Hydra模型加载同nj9elah1 best6000，旧权重/RNG相同、bias全零，新optimizer为空且initial_lr=.0001/.002。production batch128/卡双卡dry-run含一次有效更新，退出0。

学习bias的正式6000步训练已于2026-10-04T22:01:36Z启动，唯一job为`logs/autonomous/copmrec_v4_1_bias_20261004T220136Z`，PID3812332，实际W&B run为[5ke7nt98](https://wandb.ai/baymaxam/GRID/runs/5ke7nt98)。运行Config的模型/数据与准备配置完全匹配，source record为355文件、SHA2632dd55f2e2eb2b45e59ae958143fba5659b94096fea9e9a9b9124d20eaa762，world2/物理GPU2,4。新阶段已启动1/2、全线程4/5；实际完成步数仍仅旧阶段18000，不把启动等同完成。此后串行固定0 bias对照及两次单卡完整Validation；尚无v4.1效果结论。

随后5ke7nt98实际完成6000步并零退出，六点Validation和唯一best6000 Artifact、权重及optimizer/source355文件均独立审计通过。训练内best R10=.09810848534107208、N10=.05050303786993027，两项均未到晋级门槛；尚不构成配对效果结论。固定0 bias对照已于2026-10-04T22:15:05Z启动，job`logs/autonomous/copmrec_v4_1_bias_control_20261004T221505Z`，PID3861598。新阶段实际启动2/2、全线程5/5，训练槽余0；已完成步数新6000/全线程24000，control仍运行。两次单卡完整Validation尚未启动，Testing仍1/3。

control run[l3zyr91b](https://wandb.ai/baymaxam/GRID/runs/l3zyr91b)随后同样零退出并完成6000步，best6000/六点/来源及全零bias审计通过。其训练内best R10=.09877923130989075/N10=.05042651668190956，Recall达到门槛、NDCG不足；两组均无整体晋级资格。新阶段实际完成12000步，全线程30000步，训练槽及该累计步数预算用完。学习bias的单卡完整Validation于2026-10-04T22:28:54Z启动（GPU2、PID3912263）；两个原始输出的配对证据尚待审计，Testing仍1/3。

## 2026-10-05：完整配对审计与结题

学习bias训练[5ke7nt98](https://wandb.ai/baymaxam/GRID/runs/5ke7nt98)与冻结bias对照[l3zyr91b](https://wandb.ai/baymaxam/GRID/runs/l3zyr91b)均完成6000步并零退出；六次Validation按相同NDCG10规则选出的best均恰为固定step6000。训练内固定6000步与各自best的标量比较一致，不将其当作独立用户配对证据。真实单卡完整Evaluation分别为[o7ycqky2](https://wandb.ai/baymaxam/GRID/runs/o7ycqky2)与[yqsmsdt1](https://wandb.ai/baymaxam/GRID/runs/yqsmsdt1)，均finished、exit0、物理GPU2→local0、NPROC1，dense/content部署且候选trace关闭。

两臂独立读取175个原始Evaluation文件，22363个用户及末商品标签逐值相同，Top10完整SID合法且每用户唯一；公共bundle读取、SID/embedding Artifact、四个catalog buffer、真实validation-selected checkpoint的producer/digest/bytes/MD5/SHA以及每个训练/推理的355个source archive文件和61个关键hash均核验通过。新运行source SHA为`2632dd55f2e2eb2b45e59ae958143fba5659b94096fea9e9a9b9124d20eaa762`。原始标签SHA为`efe624caa09a26812661a9930c29724a5c0d23670c304fe5c2e48de56471a72f`，canonical int64用户SHA为`5c14775a3aec5907a79e94c4cc0b2a00be6344c0e72cb2e40cca75e9c34744d9`。LIGER指标来自固定5azn5vm0的真实dense trace，不使用hybrid summary代替。

| 模型 / 完整Evaluation run | Top10命中 | Recall10 | NDCG10 | 相对同split LIGER R10 / N10 |
| --- | ---: | ---: | ---: | ---: |
| 固定LIGER dense / 5azn5vm0 | 2008 | 0.08979117291955462 | 0.04647035630072118 | baseline |
| 固定v4 best6000 / mq8hhof5 | 2147 | 0.09600679694137638 | 0.049793932567196456 | +6.9223% / +7.1520% |
| 学习bias / o7ycqky2 | 2194 | 0.0981084827617046 | 0.05050305197307684 | +9.2629% / +8.6780% |
| 冻结bias续训对照 / yqsmsdt1 | 2209 | 0.09877923355542638 | 0.05042652891207879 | +10.0100% / +8.5133% |

完整原始指标及lineage见[学习bias审计](evidence/copmrec-v4-autonomous-20261005/validation-bias-validation-audit.json)、[冻结bias审计](evidence/copmrec-v4-autonomous-20261005/validation-bias-control-validation-audit.json)、[训练对照](evidence/copmrec-v4-autonomous-20261005/bias-training-comparison.json)和[完整配对分析](evidence/copmrec-v4-autonomous-20261005/bias-validation-pair-analysis.json)。独立复算采用float64用户贡献，因而与训练内float32标量存在微小尾数差异，门禁使用独立复算值。

### 组件增量未获支持

学习bias相对同预算冻结bias对照，Recall10相对−0.6790%、NDCG10相对+0.1518%；新增27命中、损失42、净损15，共同命中261升位、214降位。其预注册“提高NDCG10且不损Recall10”的点预测未通过，两个绝对Δ配对95%区间均跨0：

| 比较 | Recall10绝对Δ及95% CI | NDCG10绝对Δ及95% CI |
| --- | --- | --- |
| 学习bias − 冻结bias | −0.0006707507937217726；[−0.0013862183070249966, 0.0000011179179895301872] | +0.00007652306099805003；[−0.00023114619319761478, 0.0003895703624965395] |
| 学习bias − 固定v4 best6000 | +0.0021016858203282206；[0, 0.004249206278227423] | +0.0007091194058803838；[−0.0003217825922760872, 0.0017420395042755056] |
| 冻结bias − 固定v4 best6000 | +0.0027724366140499932；[0.000715467513303224, 0.004874122434378214] | +0.0006325963448823339；[−0.0003886775191897871, 0.0016495466173171555] |

两臂相对LIGER的Recall10/NDCG10绝对Δ区间均严格正向，但不能将整体收益归因于新增bias。上述CI为2000次、seed42、PCG64用户配对bootstrap，仅条件于固定选出的checkpoint，不覆盖训练seed或checkpoint选择不确定性；跨0不代表等价。

固定offender仍为旧v4 best6000预先确定的20件商品，没有从新结果重新选取。其错误Top10曝光数从源v4的40995降到学习bias的25343（占错误曝光11.4448%）和冻结对照的25575（11.5504%）；bias相对对照只额外减少232次曝光，两固定key子组分别减少107/125次。两个臂都明显减少旧offender曝光，而整体NDCG与bias配对增量未形成所需支持，不能把曝光下降解释为已证明校准修复。固定offender目标子群639人，bias相对对照净命中0、R/N区间均跨0；51个cold目标中命中相同且4人升位，小样本结果不扩展为cold机制主张。

### 按预注册门禁关闭，未消费Testing

本次同split Validation阈值始终为至少2209命中、Recall10≥0.09877029021151008且NDCG10≥0.0511173919307933。学习bias两项均不足；冻结对照仅Recall达标，NDCG不足。因此eligible arms为空，唯一Testing winner为null，两个臂均未进行Testing，不使用Testing绝对阈值替代Validation门槛。

本阶段2次正式训练/12000步及2次完整Validation全部实际完成；全线程5次正式训练/30000步已耗尽，训练槽余0。全线程Testing仍1/3、余2，本阶段消费0；该余量不自动授权新路线或训练。按预注册负向/不确定规则关闭独立bias机制迭代，不追加bias/temperature/LR扫描，也不把当前不足10%的结果包装为目标达成。原v4开发起点与此前已验证的部署结果保留；本阶段两个新best6000没有Testing确认，整体目标仍active。

学习bias best6000 URI为`wandb://baymaxam/GRID/5ke7nt98?role=checkpoint&alias=v0&file=checkpoint_epoch=000_step=006000.ckpt`，SHA为`2fce039ee9a8000cfb76426e4d74d48edd4c090cc14f4079dda62e0bd055d129`；冻结对照best6000 URI为`wandb://baymaxam/GRID/l3zyr91b?role=checkpoint&alias=v0&file=checkpoint_epoch=000_step=006000.ckpt`，SHA为`4bea4d5771c4f056ad6e0cd69d1204a3220b639868803cbc19aa3332ecf0a3c9`。两者记录用于追溯，未选择新的Testing checkpoint。结题摘要与证据文件SHA见[score-bias-stage-closure.json](evidence/copmrec-v4-autonomous-20261005/score-bias-stage-closure.json)。
