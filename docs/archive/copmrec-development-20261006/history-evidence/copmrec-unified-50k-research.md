# CoPMRec v5：50k 从头整体训练与复现

## 当前目标与状态

用户接受预算匹配下 R10 +7.896221%／N10 +8.121248% 的旧结果，授权长时自主任务及 node1 训练、推理。本阶段目标是推荐模型随机初始化后全部模块从第 0 步一起训练，单次连续 50000 optimizer updates，单 checkpoint 部署；相对相同预算、同 seed、同 history 资格的 LIGER dense，Recall@10 和 NDCG@10 各至少 +8%，并用第二 seed 成对复现。训练优先双卡，完整推理固定单卡。

当前实现、91项聚焦测试、独立审阅、官方同步、真实CPU预检和双卡一步smoke均通过。首个candidate42随机初始化连续50k正式训练及ownbest41k的完整单卡Validation均完成，预算/source/输入/用户/输出独立审计通过。相对同预算native42，R10+3.60%（CI跨零）、N10+11.29%（CI为正），原双8%门槛未通过。按用户允许保留明显部分增益的要求，保留NDCG正向结果，继续分析已有bad case；不自动启动43或Testing，剩余2次／100k暂未消费。旧 64k、双 checkpoint pool 的结果是方案依据，不是新目标的效果证据。OpenSpec 为 `add-copmrec-unified-50k`。

## 问题证据与路线判断

1. 核心可反驳假设：共享的 seen 商品协同残差能与 v0 的内容／SID／混合概率目标从头共同学习，在一个连续 50k 日程内保留实质推荐收益；既有改善不必依赖 50k 后的 optimizer 重启和 checkpoint 融合。
2. 正向证据：旧残差／scale0 匹配续训的六个等步验证点均正向，各自 best 的 R10／N10 增量约 2.47%／2.02%；旧单 residual dense Testing 相对当时 native dense 约 +5.26%／+5.64%。最终双方各 64k、同 history、同双成员 pool 时仍约 +8%，但该结果同时包含训练与部署因素。
3. 负向或未确认：固定 alpha=0.8 的从头训练对 learned-alpha v0 为 −3.05%／−2.60%；固定 mixed 终排对同候选 content 的配对 CI 为负；history content-CE 干预对匹配控制未确证增量。v3 首层恢复解决局部漏召回却没有净推荐收益。这些设置不加入本模型。旧 LR20 相对早期 residual 的额外增量 CI 跨零，不宣称其独立显著或最优。
4. 关键未知是单次 scratch 的效果，而非新的候选恢复规则。直接实现零残差初始化，检查真实三 loss 梯度、cold 安全、完整 optimizer 日程与 checkpoint 语义，再用完整 50k／Validation 鉴别；不用旧效果或 smoke 代替。
5. 本阶段显式最多三个正式训练：candidate42、晋级后 native43、相同 candidate43，每个 50k；累计最多 150k 新更新、3 次完整单卡 Validation、3 次新单卡 Testing。旧所有研究额度封存，不重置。可选 35k warm probe 未启用，不另消耗训练槽。

上述具体效果来源见 [残差阶段](copmrec-v4-autonomous-research.md)、[固定 alpha 负向结果](copmrec-v0-fixed-alpha08-x70hphts-result.md)、[history CE 阶段](copmrec-v4-4-eligible-content-continuation-research.md) 和 [64k 预算匹配报告](liger-budget-matched-continuation-20261005.md)。证据目录保存实际文件指纹，不用旧报告中的阶段开始快照覆盖其最终结论。

## 方法与定位

商品表示为 `z_i = projection(content_i) + seen_i * r_i`，`r_i` 为全零初始化的 12101×128 参数表，历史输入和完整目录 cosine 评分共享同一表示，cold 残差及其梯度为零。保持 v0 unit SID CE、unit full-catalog content CE、unit legal-prefix mass mixture NLL，alpha 由 mixture NLL 学习；不加入 teacher、额外 ranker、固定 alpha、bias、mixed 终排、训练 history CE 改动或 pooling。

LIGER 本身联合使用语义 ID 和内容信息；其论文中的 dense 对照也讨论 ID 与文本表示相加。因此这里的贡献不能表述为首创 ID embedding 或普通 content CE，而是检验当前共享表示和概率联合目标的整体训练机制。单模型完整目录 dense 部署的效果不能归因为生成候选的召回收益。[LIGER 原论文](https://arxiv.org/html/2411.18814v2)

## 固定训练配置

| 项目 | 固定值 |
|---|---|
| 推荐参数初始化 | 随机主干／projection，gate bias=0，residual=0；无推荐 checkpoint |
| 训练预算 | 0→50000 连续 optimizer 更新，无阶段 optimizer 重建 |
| optimizer | AdamW，weight decay 0.035 |
| peak LR | 主干 0.0003，residual 0.002（multiplier 20/3） |
| scheduler | warmup 2500，cosine horizon 50000，min ratio 0 |
| batch／精度 | 每卡 128、DDP2、global 256、FP32、clip 1 |
| 数据 | 同 Beauty SID／content Artifact；causal max32、history20、sequence80 |
| seed | 首个 42；冻结后成对 43 |
| checkpoint 选择 | 每 500 更新，原 native raw dense Val NDCG@10 own best |
| 完整 Val／Test | 单卡、单 checkpoint、full catalog dense、共同有效 history 排除、stable catalog row ties |

残差 peak .002 沿用已保留正向设置的绝对尺度，不将旧 warmstart 主干 .0001 上的 multiplier20 直接搬到 scratch 主干 .0003 而变为 .006。新增 1,548,928 参数须披露，不声称同参数量或同 FLOPs。

## 对照、门槛与执行顺序

seed42 主对照是 native `35ig0tz6` 实际完整50k、自己 raw Val best45000。相同 history 资格的完整 Validation `wdms8w77`：R10=0.09694584805258687、N10=0.05404889855718787；首个 candidate 晋级需 R10≥0.10470151589679383（至少2342命中）、N10≥0.0583728104417629，两个配对绝对差 CI95 下界均正。

原同口径 Testing `vnhmag7v` 的 R10=0.07722577471716675、N10=0.04281558436299254。只在首个完整 Val 晋级后冻结方法和全部超参数，再各训练完整50k的 native43／candidate43，分别按同 raw Val 规则选 own best并做完整 Val。第一次新 Testing 前冻结两 seed 的全部模型引用；最多三次新 Testing 为 candidate42、native43、candidate43，复用 native42 固定输出。最终两个 seed 分别满足双8%及双配对 CI 下界正才视为效果可复现，不用跨 seed 平均抵消失败、不隐藏第二 seed 或换 seed。

首个 Val 未过时，如实登记已消费 1／50k 与剩余 2／100k，不自动消费余量或扫描。对已保存输出作有限 bad case 分析后按同问题、真实新依据作下一决策；不降低目标。原 Beauty split 已参与历史开发，不能称为未触碰独立数据；原 native42 source archive 的历史缺口不由新归档回填。

## 来源、资源与预算记录

正式运行前必须通过核心测试、Hydra／Bash 检查、实际 CPU 初始化、官方 Mutagen 三会话 flush 和真实双卡一步 smoke。smoke 明确不是效果实验，不建立正式 W&B run。实际 runtime bytes（含 dirty／untracked）归档，上游 Artifact、checkpoint producer／digest／文件 SHA、raw split／labels／keys与独立指标须核验。

卡位按实时资源选择并记录物理→local 映射；原计划物理5,6被其他任务占用，当前实际使用0,2。不得停止其他进程；每个正式 job 保存唯一 PID／日志／run ID，观测超时重新核同一句柄。故障恢复只允许同一未结束 run 的完整 optimizer／scheduler／globalstep，不以 best 权重重开训练或隐匿重放成本。

新阶段账本见 `docs/evidence/copmrec-unified-50k-20261005/` 以及 research-state 顶部 `copmrec_unified_50k_20261005`。每个生产模型50k与全研究新消耗分别登记，旧64k对照保持封口。

## 首个正式训练句柄

2026-10-05 UTC12:24:28启动candidate42，PID3561177，job目录`logs/autonomous/copmrec_unified50k_train_candidate42_20261005T122428887229Z`。物理GPU0,2→本地[0,1]、每卡128；这两卡既有常驻进程占用约65GiB，多次观测计算空闲且各剩约15GiB显存。标准batch双卡一步smoke实际exit0后才启动正式任务，不停止其他进程。卡位共享与真实资源观测写入job/resource-allocation.json，不据墙钟时间声称同GPU计算成本。

实际runtime为390文件，SHA256 `f67ca2d4abdb914926eec5428f2e00b776160c7aceb7239657c6efdeed8393a4`，旧383运行文件逐字节不变。真实CPU初始optimizer为空、165个state tensor、11031809个trainable参数、seen12068/cold33、推荐预训练来源null；源与两上游文件SHA匹配。新训练已启动1/3、承诺50k/150k，actualterminal updates尚待证实；新fullVal/Test均0，目标保持active。

UTC12:27:28核验正式run `m50dan21` 为running、PID仍存活、W&B summary训练step999；actualruntime源390文件及全部预备SHA、source Artifact producer/digest、two上游身份与resolved protocol核对一致，未消费checkpoint。首次早期raw Val R10约.01713/N10约.00913，只用于跟踪训练，不能作为最终效果或改日程依据。

UTC12:47:05同一PID／run仍在训练，W&B summary step10499。训练审计、单卡推理driver和raw输出审计已就绪，root重跑必要纯本地检查通过；训练审计已用该run实际配置验证，推理审计36项纯检查及独立审阅通过。审计工具仅位于docs，390文件冻结runtime未变。尚未执行实际terminal训练审计或完整推理，所有早期Val指标均不作为晋级／效果结论。工具指纹及检查范围见`auditor-readiness.json`。

UTC12:51:21最后一次观测仍为同一PID／run running，summary step12499，无exit marker，未出现第二正式run。等待完整50k结束，继续使用唯一句柄；尚未启动第二seed或新Val/Test。

## 首轮训练完成与保存状态边界

UTC14:20:36核验同一job进程已结束、exit0、run `m50dan21` finished、summary step49999。实际训练审计核100次Val对应500→50000更新、max_steps=50000正常终止日志、唯一run与命令，证实消费50000更新／global256／12800000扩展样本呈现。初始optimizer为空、无推荐checkpoint输入、两个上游身份与390文件实际源码归档均通过。

自己的raw Val N10最优为41000：`wandb://baymaxam/GRID/m50dan21?role=checkpoint&alias=v0&file=checkpoint_epoch=000_step=041000.ckpt`，SHA256 `15eab447c21db40f7e47401e8146c75436b82fa394a6ceb56d40d13c7fff1dba`。其参数、完整optimizer moments、scheduler及cold安全均按真实41000步验证。实际Lightning2.6.5仅在该次Val保存新topk时更新save_last，因此last也停在41000；没有50000步完整状态文件，不补造、不重标，也不将last替代best。预算证明与保存状态范围分别见`training-candidate42.json`的budget_proof/saved_last_checkpoint。

此时新训练已完成1/3、50000/150000，新完整Val/Test尚未启动。物理GPU0当前有计算负载，单卡Val准备改用当时计算空闲且余15234MiB的GPU2→本地[0]；官方三会话flush/status正常。只读CPU恢复首次遇W&B service响应超时，记录原错误并重试检查，不启动重复正式job。raw训练Val指标不代替完整eligible Val或效果晋级。

UTC14:41:26首次正式单卡Val启动，PID4051124、job `logs/autonomous/copmrec_unified50k_val_candidate42_20261005T144126009878Z`、物理GPU0→本地[0]。GPU2及之后GPU6在CPU预检期间被其他任务占用，启动资源检查均在创建job前拒绝；这两次没有正式run或模型forward。最终GPU0再次通过CPU恢复与启动前空闲／显存检查，只有一个完整Val任务，新fullVal1/3、Test0。所选CP、数据、单成员与history规则未改变，等待实际raw产物审计。

## 首轮Validation启动故障与有界恢复

UTC14:46/14:49只读进程与目录检查证实PID4051124已退出、exit1、失败run `tz88ztdn`。异常为Trainer.predict setup内WandbArtifactLineageCallback登记来源时，W&B Internal API未在默认20秒内获得core回复；尚未进入预测，无LOCAL_RANK/恢复完成/预测进度，原目录只有source/wandb文件，没有推荐bundle。本次是一次正式启动失败，不作为效果负例或完整Val结果。

SDK0.28.1实际源码和纯env检查确认WANDB_HTTP_TIMEOUT在import时控制该Internal API的GraphQL HTTP及core回复等待；仅服务启动等待参数不适用。一次重试设置120秒，并将infra-only差异写入config/metadata，不跳过lineage、不改模型/data/seed/41k CP或选择规则。原失败job/prep/ready/日志观测按字节保存于`startup-attempts/val-candidate42-tz88ztdn`，远端旧输出目录保留；新输出目录独立为`logs/unified-50k-val-candidate42-startup-retry1`，新CPU恢复仍真实执行。尚不能断言超时根因或保证提高timeout能解决故障。

成本单独登记：训练仍1次／50000更新；固定Val评价机会为candidate42一项，当前完整结果0、失败正式startup1、Test0。所有启动attempt与失败保留，不重置训练预算，也不把零预测失败作为额外的效果选择；后续完整Val上限仍3项固定模型。重试编排已独立审查通过，模型runtime仍冻结390文件。

同一评价的首次startup重试已通过真实CPU严格恢复，并于UTC15:10:52启动，PID4158778，job为`logs/autonomous/copmrec_unified50k_val_candidate42_20261005T151052114075Z`。物理GPU2映射本地[0]，单进程；共享算力，启动时可用显存12189MiB，最低守卫8000MiB，未停止或改变其他进程。正式startup累计2，失败1；本次预测输出与评价结果仍待审计。

## 首个完整Validation与保留边界

重试run`8w893ra3`实际finished／exit0；独立审计175个原始文件、22363用户、labels／keys／catalog及history排除、两上游与own CP、390文件runtime和最终COMMITTED输出Artifact。最初三次只读审计中前两次被W&B的整数浮点编码差异拒绝，原错误分别保留；实际7个config叶、2个Artifact叶数值完全一致。仅这7个已证明路径作整值float/int精确规范化，bool／微小数值变化／其他metadata均仍拒绝；62项纯检查和独立实际fixture通过。随后只增加已有排名桶转移描述，71项纯检查与同bundle再审计通过，指标／CI／原gate逐值未变。模型和实际输出未重跑、未改变。

| 指标 | 50k native42 | 50k v5 | 相对增量 | 配对绝对差CI95 |
|---|---:|---:|---:|---|
| Recall@5 | 0.0651522604 | 0.0723963690 | +11.1187% | [0.00389035, 0.01046371] |
| NDCG@5 | 0.0438338046 | 0.0511383306 | +16.6641% | [0.00496272, 0.00960595] |
| Recall@10 | 0.0969458481 | 0.1004337522 | +3.5978% | [-0.00026830, 0.00733354] |
| NDCG@10 | 0.0540488986 | 0.0601508468 | +11.2897% | [0.00377530, 0.00840819] |

固定双8%＋双CI正门槛为false，不改成功标准或以Top5替代目标。NDCG10正向已确认，Recall10只有正点估计且CI跨零；第二seed与Testing未运行，不能称可复现目标达成。按用户“超过3%的明显收益可以保留”的偏好保留本v5及全部源／checkpoint／输出，未自动消费余下2次／100k。

已有bad case：Top10新命中919、丢失841、净增78；共同命中位次提升553、下降404。NDCG10的新增命中贡献+0.02141329、丢失命中贡献−0.01794538、共同位次贡献+0.00263404；前两项抵消后仍+0.00346791，共同位次占总增量约43.17%。Top5总命中从1457升到1619（+162），第6–10位从711降到627（−84）；这只是位置桶总量变化，尚不能把84直接解释为漏掉84个原目标。

两固定半组NDCG10分别+12.9753%／+9.7460%，配对CI均正；Recall10各CI跨零。seen目标22312用户占主要量，cold目标仅51用户，其中v5新增6命中；不能将小cold组结果泛化或将seen/cold差异当残差因果对照。当前模型使用完整目录dense，没有生成候选阶段漏召回。余下不确定性是前排排序改善同时伴随大量命中交换，不能据此断言alpha、residual尺度或某项loss导致丢失。

排名桶转移如下，行是固定native42，列是v5；未命中仅指不在保存的Top10，不猜测完整目录中的名次。

| native42目标位次 | v5未命中 | v5第1–5位 | v5第6–10位 | 合计 |
|---|---:|---:|---:|---:|
| 未命中Top10 | 19276 | 553 | 366 | 20195 |
| 第1–5位 | 456 | 854 | 147 | 1457 |
| 第6–10位 | 385 | 212 | 114 | 711 |
| 合计 | 20117 | 1619 | 627 | 22363 |

841个丢失中456原位于Top5、385原位于6–10；后者丢失率54.15%，但其中212又升入Top5，不能把尾部总量下降只归结为目标掉出。基准Top5的456掉出对应31.30%的该桶用户。当前正向结果与损失都应保留，后续预测应检验能否同时保留前排NDCG并降低真实命中丢失，而非仅抬高某个桶的数量。

全部100点raw训练Val按固定10k窗口观察：30–40k至40–50k的mean raw R5从0.06231275到0.06358270，R10−R5从0.02882216到0.02887806，6–10桶基本持平；41k之后没有新的raw N10最佳。它不显示末段尾部指标崩塌，也不证明全局收敛或某个loss冲突，不能据此安排额外续训。raw口径与独立history eligible口径不同，不混用绝对值。描述记录为`training-trajectory-head-tail.json`，保留决定／完整成本为`candidate42-badcase-summary.json`。
