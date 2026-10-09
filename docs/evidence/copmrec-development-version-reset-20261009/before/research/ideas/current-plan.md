## 2026-10-09：历史排除的论文定位建议

建议核心学习机制比较采用双方均不排除历史的dense结果，历史排除单独作为可选部署策略报告。此项仅为[叙事建议与证据边界](../docs/copmrec-history-exclusion-narrative-20261009.md)，不改变冻结v5.3、主指标、主矩阵或M3完成事实。无排除@5有当前单元方向证据，NDCG@10正点差但区间含零，Recall@10接近不等于等价。建议新增Training/Testing均0；无协议切换或额外运行授权。

## 2026-10-09：CoPMRec与LIGER均关闭历史排除的对照完成

用户请求的BMX-149完成LIGER单卡Testing `lu8oct42`，新增0训练/0独立Validation/1Testing；CoPMRec关闭排除hc8oct43复用。相同完整目录dense且两者均不排除历史，原值和配对区间见[实证比较](../docs/copmrec-liger-both-history-off-20261008.md)。同时报告固定LIGER排除开关差值。原hybrid基线、原M3五臂250k、开启dense1/9范围均保留；没有其他条件启动授权。

## 2026-10-08：最终历史商品排除的固定Full推理消融完成

用户授权的BMX-148完成一次Beauty/seed42单卡Testing `hc8oct43`，0训练/0独立Validation。固定Full step47500，仅取消最终历史mask，历史编码及残差保持不变。实证原值、配对区间、历史重叠和目标在历史中的数量见[核验记录](../docs/copmrec-final-history-exclusion-control-20261008.md)。此项独立登记，不改原M3五臂250k、原LIGER hybrid或BMX-147联合部署干预的范围；没有追加其他条件授权。

# 当前计划：CoPMRec v5.3 正式版本与新实验

## 2026-10-08：BMX-147单卡LIGER dense内部对照已完成

用户授权“进行该单卡test”，已完成Beauty/seed42唯一条件：复用35ig0tz6 own-best step45000，run `ldi54f1o`，物理GPU1→local[0]，独立tmux；0训练、0独立Validation、1次完整Testing。共同全目录与输入历史资格下，LIGER dense NDCG@10=0.04281558，Full=0.04443310；Full−dense=+0.00161752，95% pointwise CI[-0.00043920,+0.00359993]，四个总体CI均包含0。来源、合法唯一输出、历史排除、相同22363用户及独立四指标复算通过。当前建议不追加Training/Testing，收缩同dense协议学习机制优势主张并完成稿件；不自动展开剩余8个条件。详见[对照结果与判断](../docs/copmrec-liger-dense-control-20261008.md)及[完整GRID证据](../../GRID/docs/evidence/liger-dense-control-20261008/README.md)。以下先前建议及授权边界保留为历史过程，M3和正式hybrid主矩阵完成事实不变。

## 2026-10-08：用户请求后的消融与机制评估

在完整实证验收后，用户明确要求评估消融效果、机制分析并给出下一步建议。研究判断单独记录于[消融与机制评估](../docs/copmrec-m3-assessment-20261008.md)，原BMX-117纯实证issue与以下完成事实保留。当前建议为：保持冻结v5.3及完整方法主比较，收缩mixture/native的独立净增益和beam部署机制主张，优先将稿件对齐实际dense部署并整合已有证据；**新增训练建议为0**。固定Full残差依赖与A2重训增量分别解释，用户bootstrap不当作等价或跨seed证据。只有坚持同dense协议下学习优势的强主张时，另议Beauty/seed42复用正式LIGER own-best的1次单卡内部Testing；这只是最小候选建议，未授权运行，也不自动扩为9个条件。五臂/250k累计预算已耗尽，不更换正式版本或按A3的Testing点估计选新Full。

## 2026-10-08：BMX-117 有界消融与机制实证已完成核验

用户先授权各实验在 node1 独立 tmux 启动，随后确认五个训练完成并授权继续 BMX-117。冻结的八个 issue 已形成完整证据：**5/5 scratch训练、250,000实际updates、0额外独立Validation、5/5单卡Testing**；M2残差干预及Full/A1/A4 prefix共 **4/4 diagnosis任务**，含初次序列化失败的工程attempt共5。M1六份固定预测bundle分析另计1次，**0 model-forward**。每个完成项均核对来源与产物并独立复算，完成含义与差值方向或区间位置无关。

唯一有效范围仍为 **Beauty / seed42**。每训练双卡、50k/DDP2/global256/FP32，每500步训练内raw dense Validation，共100个选点；各自首次最高val/ndcg@10的own-best完成精确身份核验，再做一次完整单卡Testing。Full复用BMX-120的gshpyn49/vosmuihm，不新训练或Testing，也不作变体初始化。group仍为`paper_ablation_copmrec_beauty`和`paper_mechanism_copmrec_beauty`，训练预算未扩大。

| Issue | Training / own-best step | Testing | Physical GPUs：train / test |
| -- | -- | -- | -- |
| BMX-122 | m3frgiim / 48500 | m3veerfx | 5,6 → [0,1] / 0 → [0] |
| BMX-123 | m364gahb / 47000 | m3bk48w2 | 5,6 → [0,1] / 2 → [0] |
| BMX-129 | m3td2jxc / 41000 | m3xm4hcb | 6,7 → [0,1] / 3 → [0] |
| BMX-142 | m3odfrrh / 46000 | m3sfptwc | 6,7 → [0,1] / 4 → [0] |
| BMX-143 | m3kq1tuc / 43000 | m3hycech | 5,7 → [0,1] / 5 → [0] |

五训练均完成50k更新，选中的best步数如上；没有保留50k terminal完整checkpoint，选点后的训练不能由best权重代替。完整训练曲线、停止日志、optimizer/scheduler及冻结参数契约已核验，best不是last.ckpt。训练tmux/端口与原命令保留在[原启动规格](../../GRID/docs/evidence/copmrec-m3-launch-20261008/training-launch-specs.json)，精确checkpoint URI/SHA及实际单卡命令见[训练终验](../../GRID/docs/evidence/copmrec-m3-completion-20261008/training-verified.json)和[Testing规格](../../GRID/docs/evidence/copmrec-m3-completion-20261008/testing-launch-specs.json)。

| 实验 | Issue | 已冻结定义与完成证据 |
| -- | -- | -- |
| A1 | BMX-122 | S+J+N，alpha冻结0.5；own-best + Testing核验完成 |
| A2 | BMX-123 | 历史/目录残差恒0，S+J+M+N保留；J=N重复项保留；own-best + Testing核验完成 |
| A3 | BMX-129 | S+J+M；own-best + Testing核验完成 |
| A4 | BMX-142 | S+J+G+N，G按用户×4层平均，alpha冻结0.5；own-best + Testing核验完成 |
| A5 | BMX-143 | S+2J+M，第二个J复用同一次joint logits/投影/dropout；own-best + Testing核验完成 |
| M1 | BMX-144 | `m3hitc8p`；6臂、134178 user-variant、114 slice、608配对CI、132确定性案例、14项Artifact文件核验；零model-forward |
| M2 | BMX-145 | `717vgnkn`；固定Full残差2×2，N=22363、76 slice、304 CI、12项Artifact身份核验；V11逐key精确复现Full |
| M3 | BMX-146 | Full `j2rworuj` / A1 `5murlou7` / A4 `pltnvpjf`，3/3 own-best来源完成；每源22363用户×4层=89452记录，各8项Artifact文件核验 |

S/J/M/N/G定义与解释边界见[M3统一协议](https://linear.app/baymax104/document/copmrec-v53m3-消融与机制实证协议2026-10-08-643847ec8183)，protocol=`copmrec-m3-v53-20261008-v1`。六份预测的用户及目标SID一致，排序后user-label SHA为`55e2602110f989565aa6c5fc00bef99107a17d3d11222c06c46874b9df50e016`；Top10合法、唯一、排除有效输入历史，cold目录资格核验通过。五份Testing四指标与W&B误差均小于1e-8，固定数据与完整URI/Artifact version/digest/MD5/SHA已登记。

| 来源 | Recall@5 | NDCG@5 | Recall@10 | NDCG@10 |
| -- | --: | --: | --: | --: |
| Full | 0.05334705 | 0.03659647 | 0.07776238 | 0.04443310 |
| A1 NoMixture | 0.05249743 | 0.03596258 | 0.07731521 | 0.04394472 |
| A2 NoResidual | 0.05187139 | 0.03453895 | 0.07981934 | 0.04348076 |
| A3 NoNative | 0.05477798 | 0.03755917 | 0.07905916 | 0.04542925 |
| A4 LegalGenReplace | 0.05146894 | 0.03516278 | 0.07646559 | 0.04324066 |
| A5 JointCEReplace | 0.05263158 | 0.03653958 | 0.07825426 | 0.04485631 |

数值单位为原始比例；以上表仅显示八位小数，完整精度见[Testing独立审计](../../GRID/docs/evidence/copmrec-m3-completion-20261008/testing-audit-summary.json)与[M1独立审计](../../GRID/docs/evidence/copmrec-m3-completion-20261008/m1-independent-audit.json)。所有指标的带符号差值与CI已核验；NDCG@10摘录如下。

| 比较（左减右） | ΔNDCG@10 | 95% pointwise CI | N |
| -- | --: | -- | --: |
| A1 NoMixture − Full | -0.00048838 | [-0.00122374, +0.00024884] | 22363 |
| A2 NoResidual − Full | -0.00095234 | [-0.00306273, +0.00104712] | 22363 |
| A3 NoNative − Full | +0.00099615 | [-0.00026333, +0.00237374] | 22363 |
| A4 LegalGenReplace − Full | -0.00119244 | [-0.00199778, -0.00036154] | 22363 |
| A5 JointCEReplace − Full | +0.00042321 | [-0.00087547, +0.00175918] | 22363 |
| A4 LegalGenReplace − A1 NoMixture | -0.00070405 | [-0.00145167, +0.00006976] | 22363 |
| A5 JointCEReplace − A3 NoNative | -0.00057294 | [-0.00154451, +0.00039879] | 22363 |

配对bootstrap固定numpy PCG64/seed42/2000次/95% pointwise、不作多重比较校正。M1按固定seen/cold、training频次和历史长度组成19切片；包括五臂相对Full以及A4−A1、A5−A3。NDCG差值按variant_only、full_only、both三项带符号贡献相加，full_only已经带负号，原值与分解见[最终实证汇总](../../GRID/docs/evidence/copmrec-m3-completion-20261008/m3-empirical-summary.json)。单seed用户bootstrap不代表跨训练seed稳定性，区间包含0不证明等价；只交付观测、差值、样本数、区间与来源，不给好坏或积极/消极标签，不据方向扩预算。

BMX-145初次`ls14h0dw`已完成测量但nested DictConfig的JSON序列化失败，缺少完整manifest，保留失败attempt且不计完成。两处metadata普通容器转换修复通过19项测试/Ruff/OpenSpec strict/三会话flush；retry开销独立于正式训练预算。五训练原始runtime source SHA为`42f0d7c7dce0a09961e0c8c23f9ba35982abfb796d13952c3f17eaeba17c9f2e`，304文件字节及各自code Artifact保留；Testing及diagnosis源码为`5e95cf7e87b28b730976b5a6d279acdc7aca3a270a08dd5e4519ce1fc6624867`，origin verified，差异仅两处序列化文件，不回填为训练来源。见[源码差异审计](../../GRID/docs/evidence/copmrec-m3-launch-20261008/serialization-fix-source-audit.json)。

M2是固定Full checkpoint评分干预，不能等同A2重训；native query仍包含历史残差。M3为真目标前缀teacher forcing，测量概率/NLL/rank/entropy/JS与均值/分位数；只有Full记录mix/alpha，A1/A4相关字段为null。Full checkpoint全局alpha=0.9857481122016907仅为该来源的标量。三源同用户/目标核验及原始数值见[前缀汇总](../../GRID/docs/evidence/copmrec-m3-completion-20261008/prefix-three-source-summary.json)，条件化概率不代表自由beam恢复或dense因果中介。

资源只记真实W&B原值：五训练runtime为24136/24132/24027/24025/23047秒，乘两张分配卡所得设备小时估算为13.4089/13.4067/13.3483/13.3472/12.8039；这些不是测得的active GPU时间或独占用量，peak VRAM未记录。五Testing为23/22/23/22/23秒；M2为98秒，Full/A1/A4 prefix为56/50/49秒，M1为148秒。资源边界见[原始资源观测](../../GRID/docs/evidence/copmrec-m3-completion-20261008/resource-observations.json)，不为效率主张追加实验。

机器状态及八issue来源保持可追溯：[research-state.yaml](../research-state.yaml)、[最终实证汇总](../../GRID/docs/evidence/copmrec-m3-completion-20261008/m3-empirical-summary.json)、[训练终验](../../GRID/docs/evidence/copmrec-m3-completion-20261008/training-audit-summary.json)、[Testing终验](../../GRID/docs/evidence/copmrec-m3-completion-20261008/testing-audit-summary.json)、[M1终验](../../GRID/docs/evidence/copmrec-m3-completion-20261008/m1-independent-audit.json)。主矩阵和baseline范围沿用以下既有记录。

## 已完成主矩阵与历史启动快照

2026-10-08 BMX-116正式主矩阵已核验完成：9/9新CoPMRec训练、9/9Testing，9/9原LIGER hybrid配对输出独立核对，原baseline保持复用。已完成checkpoint lineage、testing用户/标签、合法唯一Top10、CoPMRec历史排除与四指标独立复算；九个配对bootstrap及三seed均值/标准差已归档。原LIGER历史政策与CoPMRec不同，保留完整方法比较边界。详见GRID docs/evidence/copmrec-main-completion-20261008/；dense内部对照与消融不计本父任务完成。

| Dataset | CoPMRec NDCG@10 mean +/- std | LIGER hybrid NDCG@10 | Gain | CoPMRec Recall@10 mean +/- std | LIGER hybrid Recall@10 | Gain |
| -- | -- | -- | -- | -- | -- | -- |
| beauty | 0.04355895 +/- 0.00109371 | 0.02992889 | 45.54% | 0.07583956 +/- 0.00269898 | 0.05570213 | 36.15% |
| sports | 0.02398604 +/- 0.00054744 | 0.01574162 | 52.37% | 0.04356986 +/- 0.00067478 | 0.02864393 | 52.11% |
| toys | 0.04668824 +/- 0.00031526 | 0.02629864 | 77.53% | 0.07699705 +/- 0.00066238 | 0.04837214 | 59.18% |


以下各次运行快照按当时事实保留，已由上面的9/9完成核验记录替代，不作为当前待执行清单。

2026-10-08 seed2026三个训练已finished并通过固定50k与best审计，Testing命令已补齐，未启动推理。累计训练完成9/9、Testing执行完成6/9；seed2026三个Testing待执行，已有六个Testing独立核验待完成。回执：GRID docs/evidence/copmrec-seed2026-audit-20261008/。

2026-10-08 六个seed42/200正式Testing全部finished、退出码0，运行中0；用户授权的六条test命令已执行完毕。训练完成6/9、Testing执行完成6/9；输出与用户集合、指标独立复算待核验，完整配对暂不计Done。seed2026未启动。回执：GRID docs/evidence/copmrec-testing-launch-20261008/。

2026-10-08按用户授权启动六个单卡Testing，使用各自审计best v0文件/SHA256，统一group、testing split和独立tmux。6个任务均已验证checkpoint加载、local GPU映射和runtime源码快照。当前W&B running 1 / finished 5，完整单元有效性与独立复算待完成；seed2026未启动。

| Issue | Dataset | Seed | GPU | tmux | Testing run | State |
| -- | -- | -- | -- | -- | -- | -- |
| BMX-120 | beauty | 42 | 7 | copmrec_test_beauty_s42_20261008 | vosmuihm | finished |
| BMX-119 | sports | 42 | 1 | copmrec_test_sports_s42_20261008 | tl55l87o | finished |
| BMX-17 | toys | 42 | 3 | copmrec_test_toys_s42_20261008 | pxh4ffi6 | finished |
| BMX-121 | beauty | 200 | 5 | copmrec_test_beauty_s200_20261008 | 95rx50ma | finished |
| BMX-15 | sports | 200 | 6 | copmrec_test_sports_s200_20261008 | 5dx507jd | running |
| BMX-18 | toys | 200 | 0 | copmrec_test_toys_s200_20261008 | jlod7okq | finished |


最新训练审计：seed42/200共6个训练finished并通过固定50k及best核对，running 0；Testing尚未启动，seed2026尚未启动。六个独立Testing命令已填入各子issue的精确best v0 URI及SHA256并验证脚本参数/Hydra配置。详情见GRID docs/evidence/copmrec-training-audit-20261007/。此前运行快照保留作历史。

2026-10-07 最新运行状态：seed42三个训练均finished、退出码0，best审计与Testing待完成；本轮按用户明确授权启动seed200三个双卡tmux训练，并允许共享显存足够的GPU。累计训练started 6 / finished 3 / running 3，Testing启动与完成仍0；seed2026待后续启动。启动回执：GRID docs/evidence/copmrec-tmux-seed200-20261007/。

## 2026-10-07：明确 hybrid 基线角色，继续冻结版本后的新实验

用户确定 **v5.3 为 CoPMRec 正式版本**；正式 baseline 为 **LIGER hybrid**，**LIGER dense 仅为内部对照**。v5.2 等其他版本的做法、效果和当时决定保留在历史，不取代主比较。

2026-10-07用户明确授权在node1以tmux启动Beauty/Sports/Toys seed42三个双卡训练，并允许共享显存充足的GPU。新CoPMRec训练为 **started 3 / completed 0 / running 3**，Testing仍0启动/完成；其余六单元不在本次启动范围。训练内validation选点保留，不额外Validation。LIGER原正式结果保持复用。运行记录：BMX-120/gshpyn49、BMX-119/jk4zk19n、BMX-17/hs72xkan，详情见GRID docs/evidence/copmrec-tmux-train-launch-20261007/。

## 固定范围与预算

- Beauty / Sports / Toys × seed 42 / 200 / 2026；核心配对新增CoPMRec的9次50k训练、450k更新、0额外Validation、9次单卡Testing；LIGER复用既有9组正式结果。
- 主实验group统一为`paper_main_copmrec_<dataset>`，训练与Testing、同数据集各seed共用；版本与正式身份继续用config/notes/tags表达，issue命令与配置默认值一致。
- 每训练两卡、global256、FP32、AdamW主lr0.0003/残差lr0.002、wd0.035、warmup2500/cosine50000；每500步raw dense Validation选首次最佳NDCG10。
- v5.3保持四loss权重1、共享历史/目录残差、cold残差0、learned全局alpha、全目录联合dense部署与有效输入历史排除；不扫描alpha或修改主排序。
- LIGER hybrid保持原正式run的original策略、生成20与全部cold并集、内容终排及SID/content目标。既有Testing使用原Liger，不能声称已使用新HistoryExcludedHybridLiger；主表接入前核对历史资格和split/keys，必要重评分复用原best并另计成本，不自动重训或重置原issue。
- 既有正式LIGER的同一best做9个dense内部Testing条件，不另训练、不增加Val、不替代正式baseline；phase为internal，9次Testing独立于9主Test计数，命令另行准备。
- 五方法主表仍为CoPMRec、LIGER hybrid、TIGER、LETTER、SASRec，共45个评价槽位。TIGER/LETTER/SASRec既有有效正式issue状态保留，接入新主表前协议审计，不计入本轮新配对完成数。
- M3旧三臂×三数据集的建议清单已被2026-10-08协议替代：Beauty/seed42五变体、5训练/250k/0额外Val/5Testing，训练内各自选点保留；实现已验证，本次有界启动已获用户授权，成本独立于完成的主矩阵。

## 执行与验收

先审计各数据集既定SID/内容及数据manifest，检查实际输入身份、历史资格和源代码快照，然后从各数据集seed42配对开始手动执行，继而完成200/2026。沿用上游技术依赖：Beauty内容/SID为3jtt9mpa/dq77e3wo，Sports为psec3u5i/jcyr5l3p，Toys为d1q00dco/3dycz43g；URI已登记，运行前核对digest。保留上游技术依赖不表示推荐实验完成。

Milestone 2的BMX-116及9个CoPMRec子issue已完成正式训练、own-best审计、Testing和独立复算；M4的LIGER原Done/有效保持。已有LIGER训练、best、Testing和candidate trace可复用；旧开发CoPMRec机制run不能完成新M3任务。BMX-117既定五个训练对照和三个机制证据包已完整核验；实际成本与观测独立登记，不要求额外独立Validation。

报告全体有效单元的Recall/NDCG@5/@10、每seed与均值±标准差、相对LIGER hybrid效应量及配对CI。正、负、不确定结果均进入报告；不因不利结果换模型、换seed、追加扫描或剔除单元。dense收益不直接归因为恢复beam候选，组件增量与完整方法贡献分别评价。

- [正式版本定义](../docs/copmrec-formal-version-20261006.md)
- [正式实验计划、矩阵、成本及完成门禁](../docs/copmrec-formal-experiment-plan-20261006.md)
- [机器状态与实证结果登记](../research-state.yaml)

## 历史边界

更新前研究状态与计划已按原字节完整归档，旧开发成本、正负结果、失败attempt和原blocked目标没有重写。新的正式重跑计划是用户选定v5.3后的独立记录，不表示历史阶段尚未执行，也不复用旧预算剩余数。

- [历史研究状态](../docs/archive/copmrec-development-20261006/research-state-before-formal-v5.3.yaml)
- [历史计划原文](../docs/archive/copmrec-development-20261006/current-plan-before-formal-v5.3.md)
- [CoPMRec版本历史](../../GRID/docs/copmrec-versions.md)

原自主研究goal的blocked状态保留于归档，本次不创建或恢复自动研究goal。2026-10-08用户对BMX-117的明确有界启动授权已单独登记，不恢复旧路线或旧预算。
