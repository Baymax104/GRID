# 当前计划：CoPMRec 论文主方法与证据完整性

## 2026-10-06：v5.4 固定验证负向，保留 v5.2

用户批准的唯一 scratch50k 训练 `cv5wwhck` 和单卡完整 Validation `y9earebw` 已正常结束。训练终态主审计与附加独立复核通过：实际50000更新、100个raw Validation点、own-best46500；保存的完整参数／optimizer／scheduler状态止于46500步，没有50000步终态full-state checkpoint。训练物理GPU5、6→local[0,1]，推理物理GPU5→local[0]，同一414文件runtime source已核。

主审计对175文件／22363用户的原始预测重算：相对固定native42，R10 **−6.78%**、N10 **−6.79%**，两配对绝对差95%CI均负；相对v5.2，分别 **−10.93%／−17.49%**，两CI也负。新增558／丢失806个Top10命中，净−248；共同命中商品位置贡献NDCG−0.004044，Top5命中净−298。取消history显式residual的固定改善预测被本次结果反驳，停止v5.4 catalog-only结构，保留v5.2主方法、v5.3部分证据及停止固定辅助CE的既有决定。具体优化与表示原因仍未知，不将负结果泛化为整个residual路线无效，不拼造三方法四象限或Top11恢复量。

完整Validation的附加本地独立复核与已有预测的bounded bad-case分解均通过，远端字节与bootstrap结论明确复用SHA绑定的主审计。新增1train／50000更新／1完整Val已审计登记完成，累计5train／250000更新／5完整Val，[当前账本](../../GRID/docs/evidence/copmrec-unified-catalog-only-residual-50k-20261006/cumulative-budget-latest.json)已闭合。Validation启动attempt累计6次，其中旧阶段1次产生预测前失败，单独保留。新Testing／seed43／扫描均0，剩余授权实验为0；不自动追加新方法或预算。native双8%及配对复现／Testing目标未降低、尚未完成，整体goal保持active，仅闭合这次固定验证阶段。

[完整结果、五次路线取舍与证据边界](../../GRID/docs/copmrec-v5-4-execution-20261006.md)。以下保留各阶段发生时的历史快照。

## 2026-10-06：用户批准v5.4，唯一从头50k训练已启动

用户明确批准“批准运行 v5.4”，新增1次随机起点连续50000更新＋1次单卡完整Validation，累计上限5train／250k／5Val；原4train／200k／4Val封口及原字节证据保持，新Testing／seed43／扫描均0。v5.4仅在v5.2上取消history显式residual，保留catalog residual、原三loss、learned alpha与同50k配方。

实际本地64核心＋80配置及114编排／审计／登记检查、独立审阅、官方同步、生产CPU空optimizer与双rank一步smoke均通过。唯一正式job已启动：物理GPU5、6→local[0,1]，PID87017，source414／ae36e28d86d5a4c47c63d4f57b7a7753e2fe483168e9e1be62c414b47c6687c6；旧408运行文件不变。现在只计启动和50000预算承诺，尚未计实际完成或推荐收益，当前保留主方案仍为v5.2。后续核验实际50k与own-best，再执行已批准的单卡完整Validation，按固定native双8／paired CI及v5.2增量规则决定保留或停止。

[实际运行记录](../../GRID/docs/copmrec-v5-4-execution-20261006.md)及[启动回执](../../GRID/docs/evidence/copmrec-unified-catalog-only-residual-50k-20261006/progress-training-start-receipt.json)已归档。以下保留各阶段发生时的历史快照。

## 2026-10-06：v5.4本地实现已完成，新正式额度仍待答复

2026-10-06，本地可逆实现和聚焦检查已完成，独立静态审阅通过；推荐效果尚未验证。当前主方案仍为 v5.2，v5.3 的部分正向证据和停止固定辅助 CE 的决定保持。

已批准的累计额度为 4 次训练／200000 更新／4 次完整 Validation，已全部用完。v5.4 新增 1 次随机起点 50000 更新＋1 次单卡完整 Validation 的问题仍待用户答复，未分配、保留或启动额度；本次没有远端同步、生产 CPU probe、DDP smoke、正式训练或完整 Validation。新 Testing／seed43 pair／扫描均为 0。

[v5.4实现与本地验证](../../GRID/docs/copmrec-unified-catalog-only-residual-50k-research.md)：64核心＋80配置检查，独立静态review通过；414文件source ae36e28d86d5a4c47c63d4f57b7a7753e2fe483168e9e1be62c414b47c6687c6，旧408字节保持。以下保留发生时的历史快照。


## 2026-10-06：仅目录侧residual固定方案已审阅，新增额度待用户决定

四次scratch证据的路线五问和完整目标审核已完成，未发现阻断既有取舍的实现错误；v5.2主方案、v5.3 partial证据及停止固定辅助CE均保持。已形成[唯一下一固定proposal](../../GRID/docs/copmrec-v5-4-catalog-only-residual-proposal-20261006.md)，architecture与evidence独立只读review通过。它只在v5.2上取消history侧显式residual，保留catalog residual、单query、原三loss和同50k配方；不新增CE、参数、teacher或checkpoint续训。query仍经SID embedding与共享目标学习交互信息，不能称为恢复native排序，也没有保证收益。

现有四臂均共享history+catalog residual，v5.2整体部分收益及925新增／824丢失支持在同一seen覆盖取舍问题内检验该固定结构；它不是对所有损失根因的穷尽排查。当前方案未实施、未分配、未启动，实际账本仍4train／200k／4完整Val，剩余0。只请求新增1fresh50k训练＋1单卡完整Val，若明确授权累计才为5train／250k／5Val，新Testing／43／扫描均0；没有明确增量则停止该固定结构，不自动追加下一臂。完整双8与配对复现／Testing目标没有降低。以下保留发生时快照。

## 2026-10-06：v5.3唯一固定验证完成，停止辅助CE并保留v5.2主方案

随机初始化联合连续50000更新训练 `hqw189d2` 已正常exit0，100个raw Validation点、own-best45000、真实408文件source／输入／完整状态边界的主审计和独立复核通过。完整参数、154份optimizer moments与scheduler保存至45000，未取得50000终态完整状态，实际50k预算由全历史和终止日志另行证明。单物理GPU6→local[0]完整Validation `6u6g62hk` 已正常exit0，175文件／22363用户原始输出、附加独立复核与固定三份Top10四组分析均通过。

v5.3相对固定同预算native42：R10=0.10378750614854894（+7.05719557%），N10=0.06155058765023162（+13.87944860%），两个paired绝对CI下界为正；Recall未达+8%，双8门禁false。相对冻结v5.2仅+2.29175848%／+0.80628911%，两CI均跨0，不能称为确定的辅助CE增量。按实施前proposal保留v5.3相对native的明确partial证据，停止该固定辅助CE，当前主方案仍为v5.2；无扫描、新训练、seed43或Testing分配，整体可复现目标仍未完成。

对v5.2的427新增命中=B组149恢复＋D组278新覆盖，375丢失=A组95＋C组280，净增52；共同命中位置变化贡献N10为负−0.0001540930。该分解描述覆盖恢复与既有覆盖损失的取舍，不单独归因辅助CE、共享query或目录残差。初始内容监督加倍、额外计算与不同own-best边界保留。

新增1train50k＋1完整Val额度已全部使用；累计实际4train／200000、4完整Val、attempt5／预测前失败1次，新Testing0／seed43 pair0。终态登记先归档原state／ledger，旧闭合尾部、父账本及注册哈希保持。详见[本次结题](../../GRID/docs/copmrec-v5-3-stage-decision-20261006.md)和[实际登记](../../GRID/docs/evidence/copmrec-unified-native-view-ce-50k-20261006/progress-validation-complete-receipt.json)。以下小节保留发生时快照。

## 2026-10-06：用户新增固定验证，v5.3已启动唯一从头50k训练

用户明确批准新增1次随机起点50k训练＋1次单卡完整Validation；旧3train／150k与3完整Val封口证据保持，累计上限4train／200k与4完整Val，新增Test／43／扫描均为0。v5.3只在v5.2加入权重1的共享query无目录残差视图CE，部署仍使用联合分数；不独立隔离额外视图与初始content监督加倍，等更新数也不表示等FLOPs。

实际74核心＋40配置检查、独立review、官方同步、CPU空optimizer及双rank一步smoke／W&B0已通过。真实408文件source `d6c50005…b22a0`，旧402字节保持。唯一正式job已启动，物理GPU3、6→本地[0,1]，PID3147507；实际50k终态、ownbest与完整推荐效果仍待审计，当前保留部分方法仍为v5.2。其后只运行已授权的1次单卡完整Val，相对固定native42检验双8和两paired CI，并描述v5.2增量。以下小节保留发生时快照。

## 2026-10-06：v5.2独立复核与bad case完成，原150k阶段封口

实际50k／ownbest47k训练与单卡完整Validation均已核；附加独立复核通过。v5.2相对同预算native R10+4.658672%、N10+12.968595%，两个paired CI下界为正，保留部分正向；原双8门禁未通过，复现／新Testing未运行，goal保持active。

false cold槽从v5的6638降至44，共同2032个seen命中用户的目标前cold槽175→0；51个cold-target用户命中6→0。seen目标新增237、丢失208，仍有用户条件排序取舍，不能将占位下降等同Recall损失根因。原3train／150000更新、3完整Val均用完，attempt4含1次预测前失败、Test0，不重置上限。[完整五项结题](../../GRID/docs/copmrec-v5-2-stage-decision-20261006.md)。

[下一固定辅助视图CE proposal](../../GRID/docs/copmrec-v5-3-native-view-ce-proposal-20261006.md)拟1train50k＋1完整Val、0Test；当前仅形成方案，代码未实施、新额度未分配、无新正式启动。以下较早段落保留发生时快照。


## 2026-10-06：v5.2实际训练与完整Validation已审计，保留部分正向证据

最后槽run `rha8mrvs`已finished／exit0，100次raw Val／终止日志／globalstep共同证明随机起点连续50000更新、global256。训练主审计及独立终态复核通过；own raw dense N10首个最大值选中47000（checkpoint SHA `a82c2dc1…c764c`），best／last完整参数、moments和scheduler均只保存并核验到47000，未取得50000终态完整状态，不重标或替代checkpoint。实际402文件source `16142f5a…3b9e0`的archive／manifest及旧396字节边界由主审计与额外独立source复核确认，原历史native source缺口未回填。

唯一完整Val `d7lcftto`／PID2995739已finished／exit0，物理GPU7→本地[0]／单进程；原175文件、22363用户、标签／keys／目录、合法唯一零history输出与真实checkpoint／source均核验。R10=0.10146223673031346、N10=0.061058281374572636，相对同政策native42分别+4.6586715867%／+12.9685951139%，paired绝对CI95分别[0.0007590663148951393,0.008227876402987076]／[0.004809319752127361,0.009381575180557149]，均为正。当前部分保留方法升级为v5.2；R10仍未过+8%，双8门禁false、goal active／target false，不开启原seed43配对或Testing。

对冻结v5的R10／N10仅+1.024043%／+1.508598%；R10 CI跨0，N10 CI下界仅略高于0，不能将微增宣称为已证明的物质CE收益或cold因果机制。附加独立推理复核及预固定cold占位分析仍pending，由真实产物再限定解释，不产生新分数／列表或新增正式评价。

累计训练started3／completed3、actual／committed150000；完整Val started3／completed3，正式启动attempt4含保留的1次预测前失败；in-progress0、reserved0、Test0。原3训练／150k、3完整Val及新Test上限保留，没有新预算或43分配。登记前ledger／state原字节已归档，旧closed tail逐字节不变。详情见[v5.2实际报告](../../GRID/docs/copmrec-unified-full-catalog-ce-50k-research.md)与[终态登记](../../GRID/docs/evidence/copmrec-unified-full-catalog-ce-50k-20261006/training-completion-and-validation-registration.json)。以下小节保留发生时的历史快照。

## 2026-10-06：v5.2最后槽唯一连续50k已实际启动

完整目录content CE是从v5派生的唯一改动；原SID／mixture／learnedalpha／单query单scorer／cold residual0不变。正式run [rha8mrvs](https://wandb.ai/baymaxam/GRID/runs/rha8mrvs)，PID766535，物理GPU6、7→本地[0,1]，随机初始化、全部模块共同0→50000/global256；先前396运行字节不变，新402文件source16142f5a…3b9e0。已通过新33＋旧45／27核心覆盖、40配置、59编排拒绝检查、54／86纯auditor自检、官方flush/status、实际CPU空optimizer和双rank一步smoke／W&B0。

原150k账本已启动3／3、已承诺150000，实际已核完成仍2／100000；完整Val2／3、attempt3／failed0prediction1、Test0。唯一job先观察至终态，再核100raw点／ownbest／实际源码及输入，后续只做其一个完整单卡Val；不能提前报v5.2收益。双8%主门禁、两pairedCI下界正及成对复现目标保持，seed43／新Test未分配。[v5.2方法与实际证据](../../GRID/docs/copmrec-unified-full-catalog-ce-50k-research.md)。下面保留此前发生时快照。


## 2026-10-06：最后50k槽登记给v5.2完整目录CE，正式尚未启动

v5.1固定双目录平均已负向结题，保留v5已确认的NDCG／Top5部分收益。已保存输出的cold占位只读核查及现行content CE支持集差异，为唯一v5派生v5.2提供有界鉴别依据；case频率关联不证明丢失原因，更不估算Top11恢复。旧v5／v5.1训练CE把cold logits替换为−100；v5.2仅将训练content CE改为完整目录分母，seen训练目标要求、原SID／mixture损失、learned alpha、单query／单目录和历史排除部署规则保持。

原累计上限3train／150000更新、3完整Val、3新Test不重置。实际仍为已完成并启动2train／100000更新、2完整Val、attempt3（其中1次预测前失败）、新Test0；最后1train／50000与1完整Val已reserved，尚未启动，不提前增加started／committed。未分配训练槽归零，未启动的保留槽仍为1；新43成对复现和新Testing额度均为0，整体双8%及复现目标仍active。

v5.2采用`UnifiedFullCatalogCECoPMRec`，exact5 `dense_ce_support`同步模型／CP／配置与writer；完整继承随机初始化连续50k、DDP2 global256、FP32、warm2500、主干peak0.0003／residual0.002、原SID／content输入。含strict bool CP guard的新33个核心例通过，旧45＋27个父模型／双目录例此前已通过且旧字节保持，累计聚焦覆盖105，配置／脚本40项通过，独立review与OpenSpec strict已通过。checkpoint bool guard已收尾，真实Bash argv／完整Hydra准备及实际CPU随机起点预检已绑定402文件/source16142f5a…3b9e0，旧396字节逐项保持。官方Mutagen flush／status退出0且三个session均Watching；CPU预检optimizer state0、model forward0。新增auditor训练54／推理86纯检查与来源绑定通过。实际双卡smoke已exit0并核两local rank／1step／W&B run0，physical GPU6,7→local[0,1]，source相同；正式运行0，计数仍不提前增加。

主门禁仍相对native42 `wdms8w77`的R10≥0.10470151589679383、N10≥0.0583728104417629，两个paired绝对差CI下界为正；冻结v5 `8w893ra3`用于本次增量。只完成本次单seed不能宣布整体成对复现目标完成，负／不确定不自动扫参或扩预算。见[固定方案](../../GRID/docs/copmrec-unified-full-catalog-ce-50k-research.md)、[实际登记](../../GRID/docs/evidence/copmrec-unified-full-catalog-ce-50k-20261006/stage-registration.json)与[v5.1 closure后的最终分配](../../GRID/docs/copmrec-v5-1-stage-decision-20261006.md)。

## 2026-10-06：v5.1 完整结果已核，停止双目录干预并保留v5

训练 `cm0i584p` 已 finished／exit0，独立审计证明完整50000更新、100个raw Validation点，own NDCG@10最佳checkpoint为46000；完整状态已核至46000，不能把last文件当作终态50000。实际396源文件/source200a1239…af77c2与两个输入已核，旧390字节保持。固定0.5／0.5双目录评分、单query、原三loss和整个训练配置不变。训练审计首次本地实施凭据字段名错误已保留并严格修正，未更改runtime或重跑训练。

ownbest完整单卡Validation `w19eutzj` 已finished／exit0；唯一PID600977，物理GPU7→本地[0]，实际strict CPU恢复后启动。完整22363用户输出、175原文件、标签、历史资格、CP/source和配对统计均已核。相对native42，R10−1.614391%、N10+3.324260%，两CI跨0；相对冻结v5，R10−5.031167%、N10−7.157376%，两CI均负。停止固定0.5／0.5双目录机制与权重扫描，保留v5已确认NDCG／Top5收益和v5.1负结果，不将该干预否证泛化为整个残差路线失效。

累计训练已完成2／3、100000／150000；完整Val已独立核完2／3，失败startup仍单独保留1次、累计启动attempt3；Test0，最后一个50k训练槽未分配。现有输出只读bad case检查冷商品错误占位6638→11105及其用户分布，0新增模型／评价，不推断Top11或丢失的原因。原整体双8%与成对复现目标保持；[五项路线门禁与成本](../../GRID/docs/copmrec-v5-1-stage-decision-20261006.md)已记录。以下较早小节保留其发生时的快照。

## 2026-10-06：v5.1 唯一连续50k训练已启动

原150k累计账本中的第2槽已经启动：PID182288，job `logs/autonomous/copmrec_unified_dualview50k_train_candidate42_20261005T161444299962Z`，物理GPU4,7→本地[0,1]。固定共享query双目录0.5/0.5评分，随机初始化、全部模块共同0→50000、原三loss/learnedalpha/global256/FP32/AdamW/.0003+.002/warm2500cos50000；按own rawdense ValN10选点，后续完整Val固定单卡。已有72核心（新27+旧45）与40配置测试、独立核阅、官方flush、实际CPU空optimizer、双rank一步smoke通过；真实396文件/source200a1239…af77c2，旧390逐字节不变。正式源码归档和终态实际更新、完整推荐效果待审计，不提前称50k完成。详情见[v5.1研究记录](../../GRID/docs/copmrec-unified-dualview-50k-research.md)。

当前新训练已启动2/3，已完成1/3、实际已核50000/150000、已承诺100000/150000；完整Val已完成1/3、Test0。最后1个50k槽仍未分配，原43成对验收条件与双8目标保持；本次只运行v5.1 seed42及其一个完整Val，不扫权重、CP、温度或alpha，不自动43/Test。以下10月5日小节保留其发生时的阶段快照，机器状态以research-state顶部为准。


## 2026-10-05：用户恢复50k从头整体训练自主目标

旧64k预算匹配约8%的结论和关闭成本保留。用户新授权把方法改为一个随机初始化、全部模块从step0共同训练、单次连续50k与单checkpoint部署的模型；允许node1正式训练/推理，多卡训练、单卡推理。当前goal为active，旧结果不证明新scratch效果。

固定CoPMRec v5统一模型：v0三loss各1/learnedalpha，零初始化seen商品残差共享历史与目录、cold0，无teacher/外部推荐checkpoint/optimizer阶段重启/pool；主干peak.0003/itempeak.002、AdamW wd.035、warm2500/cos50000/min0、DDP2每卡128/global256/FP32/clip1。训练内每500步原native rawdense ValN10选ownbest，独立full Val/Test同有效history资格、稳定catalog rowties。

显式新累计上限3正式train/150k、3 fullVal、3新单卡Test；每模型50k，旧额度不重置。先candidate42完整50k与原native42同policyVal比较双8及两paired绝对CI下界正；晋级才冻结同方法，并用native43/candidate43成对各50k复现。首次新Test前冻结所有引用，各seed对同seed分母独立通过，不能用均值补失败。首个Val未过不自动消费剩余预算或扫参数，不缩小整个goal。

现已完成核心45＋配置46项测试、独立审阅、官方同步、真实CPU和DDP2一步smoke。candidate42 run m50dan21正常结束，物理GPU0,2→本地[0,1]，实际50000更新/global256；100次Val及max_steps终止日志证明预算，真实390文件源码归档和两个上游Artifact通过，未消费推荐checkpoint。own raw Val最优41k的完整参数/moments/scheduler已审；Lightning2.6.5的last保存行为使last也为41k，没有50k终态完整状态，预算和状态证据分别披露。该CP的单卡fullVal run8w893ra3已finished/exit0，独立原始输出审计通过，22363用户同history资格；R10+3.5978%且CI跨0，N10+11.2897%且CI为正，双8门槛false。按用户保留明显部分收益的要求保留v5，当前新train1/3、50k/150k，完整Val1/3、Test0、剩余2次/100k未消费；不自动开始原晋级后的43 pair或Testing。相关工作定位、正反证据、门槛及来源边界见[50k研究记录](../../GRID/docs/copmrec-unified-50k-research.md)，OpenSpec为add-copmrec-unified-50k，当前机器状态以research-state顶部copmrec_unified_50k_20261005为准。

首轮Val启动实际在Trainer.predict setup的W&B lineage登记处超时退出（tz88ztdn/exit1/PID4051124），未开始预测且没有bundle，不作为效果结论。失败job及原始记录按字节归档、远端旧目录保留；仅一次同41k CP/model/data的infra-only重试通过真实CPU预检，WANDB_HTTP_TIMEOUT由SDK默认20改120并记录在config/metadata。重试PID4158778，物理GPU2映射本地[0]，单进程，启动要求至少8000MiB可用显存，不停止其他进程，最终正常结束。完整Val结果1、失败startup1、正式启动attempt2、固定评价机会candidate42一项；全部attempt保留，训练预算不重置。只读审计因实际整数/浮点编码差异修复，精确数值比较与metadata/recovery约束不放宽；没有重跑推荐模型。

有限bad case显示Top10新增919／丢失841／净78，共同命中上移553／下移404；Top5净162、6–10桶总量净−84，两半组NDCG均正且CI为正。下一检查仅已有排名的桶转移及现有100点raw轨迹，区分头部提升与真实命中丢失，不将这些相关描述归因于alpha或某个loss，也不扫描新CP或超参。目标仍active，尚无双seed/Test完整验收。

有限检查完成后，v5.1只检验同一history query对内容目录／内容加残差目录的固定0.5/0.5评分平均；projection一次、平均向量不再normalize、单matmul，final logits共同用于三loss与部署，无新增参数/第二T5。它不是独立native视图，也可能仅改变catalog梯度/几何；不宣称新机制已有效。新OpenSpec add-copmrec-unified-dualview-50k已strict通过、正在实现。原150k账本内第2个50k槽及第2个完整Val指定给唯一v5.1 seed42；第3槽未分配，新Test0，原43配对无法在剩1槽完成的边界明确保留，不重置上限或降低双8／配对验收标准。v5.1未开始正式训练，需源码/配置/CPU/DDP2 smoke与来源预检后唯一启动。

## 2026-10-05：LIGER 预算匹配对照完成，当前剩余收益约 8%

用户要求的完整有界对照已完成：三段原生 LIGER 6000/6000/2000 训练（6an0pxdy/eqlhnpgt/034h3uvv）、两次完整单卡 Validation（d649k83e/3or6e3p0）、一次唯一新 Testing（a6moio4n），实际新增 3 train/14000 更新、2 Val、1 Test，达到固定阶段预算并封口。旧 CoPMRec 7/34k 及旧 Testing3/3 保留，未重置或追加扫描。

双方实际生产与选择消耗均为 **64000 optimizer 更新、global256、16384000 次扩展后样本呈现**，最终都使用两个 checkpoint 固定0.5/0.5 logits pool。LIGER 从自己的 Val best45000 只加载权重，各段 fresh AdamW/LR 重启；保留原生 SID/content CE。A/B 按原 dense 选 best，C 按 history-excluded dense 选 best；仅按 Val N→R→更简单方案选中 A/C pool，在 Testing 前冻结引用与权重。复用固定 CoPMRec ws2fx4oi，不按 Test 改选。

预算匹配 LIGER Testing R10=0.07928274381791352、N10=0.04386710732694347；固定 CoPMRec R10=0.08554308455931672、N10=0.04742966406441521。CoPMRec 真实剩余收益 **R10+7.896221%、N10+8.121248%**，配对绝对差值95% CI分别 [0.0038009211644233778,0.00863032687922014]、[0.002318436123870346,0.004845236495791467]，下界均正。保留当前方法与正向效果，**收缩“相同预算下双指标超过10%”主张**；旧50k baseline双10.77%仅保留为历史口径。

LIGER续训加pool后对原baseline的Test提升R10+2.663578%、N10+2.455935%，不能宣称原模型完全收敛；本次采用用户第二种证明，即匹配实际更新／样本呈现预算。被选权重祖先长度仍为LIGER51k/59k、CoPMRec54k/62k，分别披露；不等同于实际消耗，不宣称同FLOPs或全部HPO成本。Beauty单seed及既有开发选择的统计边界保留。

106项聚焦测试、CPU/DDP2一步smoke通过；所有新实际CP、full Val/Test的175原文件／22363 users／真实labels／keys／合法唯一零history输出，以及383文件runtime归档均审计通过，原374文件逐字节不变。训练物理GPU5,6→本地[0,1]，推理物理5→[0]。本阶段不追加新方法、训练或Testing。

最终记录见 [LIGER预算匹配完整报告](/E:/projects/GRID/docs/liger-budget-matched-continuation-20261005.md) 与 [机器可读最终比较](/E:/projects/GRID/docs/evidence/liger-budget-matched-20261005/final-comparison.json)。以下同日旧节保留为其发生时的历史快照；当前预算与收益结论以本节及研究状态顶部 liger_budget_matched_continuation_20261005 为准。

## 2026-10-05：训练预算公平性已审计，双10%的同预算证明尚缺

用户要求排除“baseline未收敛而只因续训提高”的解释。实时W&B、完整100点Val历史、实际配置及既有checkpoint来源核验确认：原LIGER与v0初始均50k/global256，但当前最终CoPMRec共享生产链是50k+6k+6k+2k=64k，比LIGER多28%，部署也由单成员变为两个checkpoint固定融合。旧7次/34k仅是新增搜索预算，不能用它替代最终方案生产链预算。

原LIGER best45000之后到50k没有新高，支持原cosine日程末5k验证平台；末期LR接近0，而CoPMRec每段重新设置非零LR和optimizer，因此尚不能证明LIGER重启后也不提高。此前同history Testing双10%的数值结果保留，**尚不支持“相同训练预算下双10%”或“纯续训贡献双10%”**。

推荐一个明确有界的公平对照：LIGER从自身Val best45000只加载权重，按6000、6000、2000三段重建optimizer/scheduler；同global256/seed42/FP32/causal32/有效历史20及LR1e-4、WD.035、warm300、每段horizon6000。A/B按原未排history dense Val选自己的best，C按history-excluded dense Val选best；最终固定A/C各.5 pool与C single各一次同政策Validation，并与已有原LIGER比较，仅按Val N→R→简单方案冻结唯一baseline后最多1次单卡Testing，复用当前CoPMRec Testing输出。完整实际生产更新预算双方均64k；selected权重祖先长度另报，不为凑步数替换best，不声称同FLOPs或同超参数搜索成本。

本轮仅完成只读审计及记录，新增正式训练/推理均0；对照尚未实施，新增3阶段/14k及2Val/最多1Test为具体设计，旧阶段7/34k和Testing3/3原样封存，不自动新增或重置预算。若匹配后的baseline使10%不再成立，报告真实剩余收益并收缩主张。证据与曲线见[训练预算公平性审计](../../GRID/docs/copmrec-training-budget-fairness-audit-20261005.md)。

## 2026-10-05：固定 control pool 的匹配 Testing 已验证双指标超过 10%

最终两次完整单卡 Testing `vnhmag7v`（同history LIGER）与 `ws2fx4oi`（固定control pool）均正常退出，175原始Testing文件、22363 users、真实末商品标签/keys/catalog、合法唯一零history输出、CP与374个实际source字节独立审计通过。R10为0.077225774717→0.085543084559（+10.770122%），N10为0.042815584363→0.047429664064（+10.776636%）；配对差值CI分别[0.005767339,0.010733131]与[0.003278716,0.005964255]，下界均正。两个预设SHA用户组也均双10且双CI下界正，warm同样双10；cold只有138用户，不能推广。

部署在Testing之前已按两臂完整Validation的N→R→treated规则冻结为control：固定nj9elah1原v4 best6000与hbgj80bh原生v4.4 best2000，各自query后0.5/0.5 dense logits、原有效输入history资格、bias0和learned alpha。treated对control的mask增量CI跨0，未支持history CE收益，部署CE保留原义；控制续训对旧v4.3增量较小，不能单独声称≥3%物质效应，但整体预承诺双10目标真实通过。原v4.3明确有效层继续保留。

累计实际训练7次/34000步，旧5/30000封存加本次2/4000，不重置；Testing全线程3/3完成，不追加。当前结论限定Beauty/seed42、固定Validation所选checkpoint及GRID适配LIGER、匹配history资格；基线旧training source archive缺口不回填。原始证据见[最终Testing审计](../../GRID/docs/evidence/copmrec-v4-autonomous-20261005/eligible-winner-testing-audit.json)与[阶段报告](../../GRID/docs/copmrec-v4-4-eligible-content-continuation-research.md)。交叉检查、OpenSpec严格验证与记录均已完成，本次自主目标已经验证并结题；该结果不代表多seed、多数据集或论文投稿证据已完整。


## 2026-10-05：保留 v4.3 后，检验 dense CE 训练资格是否需要对齐

实现与准备已通过：49项模型测试、52项配置脚本测试、4项旧配置风险检查，374个运行文件实际同步及node1 CPU恢复核验，双卡标准smoke两rank/max1/exit0且0正式W&B run。实际续训：eligible-treated `x4ge0y88` (finished)、eligible-control `hbgj80bh` (finished)。本阶段已启动2/2训练，完成2/2；累计7/7已启动，已完成步数34000/34000。完整pool推荐结果尚未验证。

历史排除的自身R10+8.57%/N10+17.37%已确认并保留。实际训练源码中，dense CE仅将cold列填-100，仍把当前输入有效历史当作负例；推理则排除这些历史。这是新训练假设的具体依据，不能把推理收益视为训练收益，也不能把401 lost中的高频目标关联解释为因果。本次只改dense CE的历史支持集，SID CE、mixture NLL、learned alpha、cold-100、三loss权重、历史编码和.5/.5推理规则都沿用原义。

root依据已获用户自主方法及SSH授权，显式追加两臂各2000步的匹配续训成本。旧5训练/30000步原样封存，不重置；新累计上限7训练/34000步。两臂同从`l3zyr91b best6000`正常原v4.1 hook/strict weights-only恢复，零bias继续冻结，固定nj9 source，训练GPU2,4→本地[0,1]、每卡128、FP32、seed42、主干LR.0001/residual.002、wd.035、warmup300和cosine horizon6000不变，最多2000步、每1000步验证。按各自history-excluded dense valN10选best，然后仅两次完整单卡pool Validation，与彼此、保留v4.3及固定同政策LIGER `wdms8w77`比较。

规格先完成再实施。预先部署选择为双10%公平门槛、配对CI和原阈值均通过的臂中取Validation N10最高者（相同再R10、treated）；只有这唯一部署可进入两次matched Testing，仍计入全线程3次上限，当前已1。masked对control的增量与续训总体收益分开；明显>=3%的自身收益可保留，负向、不确定或均未达门槛则结束这一有界问题，不扫规则/日程/LR/权重，不丢弃已有效的历史层。[新阶段门禁与规格](../../GRID/docs/copmrec-v4-4-eligible-content-continuation-research.md)。

## 2026-10-05：固定历史资格规则，公平匹配 LIGER 后再判断收益

**已完成并保留：** 单卡完整Validation `wdms8w77`（同政策LIGER）和`iy3o3z3q`（v4.3）均零退出，22363用户原始标签、输出、checkpoint、367个实际运行源字节和配对统计独立审计通过。v4.3 R10=.10651522604301748、N10=.05960712488411068；相对冻结旧pool分别+8.568824%/+17.367342%，188新增/0丢失、1095共享命中排名上移/0下移，两增量CI下界均正。按用户最新标准保留这一层，整体不足10%不否决已确认的组件收益。

公平主比较对同样排除历史的LIGER（R10=.09694584805258687/N10=.05404889855718787）为+9.870849%/+10.283700%，两差值CI下界正，但Recall2382命中未达2385，双10%晋级门槛未通过。本阶段2/2Validation完成，0新增Testing，全线程Testing仍1/3；不使用较低旧基线替代主分母，也不因接近门槛扫描成员/权重/规则。接下来对保留的v4.3做只读bad case分析，检查训练目标与推理资格/排序的关系；尚未确定下一方法或追加训练成本。[实际pool独立审计](../../GRID/docs/evidence/copmrec-v4-autonomous-20261005/history-pool-validation-audit.json)。

只读关系聚合未支持新增共现打分：control 错误Top10的共现/有向关系覆盖比真实目标更高，主要训练频次匹配组也为负。关闭该分析方向，不以关系存在或文献补充自动立新模块。另一项直接事实来自实际22363用户：训练序列无重复，Evaluation目标均不在本人training或有效历史中；pool和LIGER错误Top10分别有19662/15642次有效历史商品占位，1095/905个已有true hit前有历史商品。这里只算已命中位置损失，没有推断Top11或新增命中。[只读占位证据](../../GRID/docs/evidence/copmrec-v4-autonomous-20261005/validation-history-occupancy-analysis.json)。

五项门禁已明确登记新问题：冻结现有pool两个checkpoint及0.5规则，仅在完整目录排序前排除当前输入有效完整SID历史；LIGER固定35ig0tz6 best45000采用完全相同规则。原dense路径与官方未排历史，不能称复现bug，也不能只改CoPM而对较低旧baseline声称10%。新阶段只安排两次完整单卡Validation，若对同policy LIGER的R10/N10均至少+10%、两配对CI下界正且旧固定阈值也通过，再进行两次matched Testing（包含baseline，全线程1→3上限不变）。

用户在本阶段正式运行前明确补充：**不要求每一层立即达到10%；约3%以上较明显的正向改动可以保留，继续bad case分析或作为累计贡献改进其它部分。** 因此组件保留与最终10%验收分开：与自身冻结旧pool相比，R/N任一相对增益至少3%且另一无点退化可保留，配对CI与固定分组分别约束效果主张；整体不足10%不会自动丢弃有效组件。无明显增量或负向则收缩相应主张，不自动扫描窗口/成员/权重或重置预算；后续投入仍须正面依据与具体有界决策。

原5训练/30000步完整保留且不追加，新阶段0优化训练、0训练数据结构/索引构建；这里显式追加有限推理验证成本，不重置原额度。规格完成后才实施，实施验证与正式效果分开。[本阶段门禁与规格](../../GRID/docs/copmrec-v4-3-history-exclusion-research.md)。

## 2026-10-05：固定训练阶段分数融合已结题，目标仍未达

唯一正式Validation `i4xwruok` 在物理GPU2单进程完成并零退出。169项聚焦回归、实际两原checkpoint CPU严格恢复、360个运行源字节、单卡统一入口dry-run，以及全量原始175 shards/22363用户和配对审计全部通过。R10=.0981084827617046、N10=.050786806656135254，相对同split LIGER +9.263%/+9.289%，两项CI正但双10%点门槛均未过，不新增Testing。

对冻结control为145新增/160丢失、净损15命中，R10−.679%/N10+.714%，两项增量CI跨0，N提升且R不降的点预测未满足。固定639真实offender目标恢复+24，其他21724净损39；两固定key组净损8/7。关闭这一个固定pool问题，禁止pair/weight/temperature/normalization/checkpoint扫描，保留局部恢复与整体取舍。累计训练5/30000用尽，Testing仍1/3；目标active未达。下一步仅只读分析训练内用户历史与目标关系，须先有不同的实际正面问题证据才讨论方法与投入，不因接近10%而自动追加。[独立结题及证据](../../GRID/docs/copmrec-v4-2-fixed-logit-pool-research.md)。

以下保留本阶段启动前的固定计划，不能当作当前仍待运行的状态。

原v4及bias阶段已结题，5训练30000步预算全部用完。新配对证据支持一个明确不同的推理问题：source `nj9elah1 best6000` 保留control `l3zyr91b best6000`丢失的258个命中，control新增320个；两个固定key组均Top5下降/Top10上升。冻结20件offender的639个真实target用户中source→control Top5 280→236、Top10 352→304，其余用户则1130→1145、1795→1905。它支持检查不同训练阶段的真实target取舍，不能把命中并集当作融合效果。

明确冻结唯一pair上述两个checkpoint及其SHA，完整目录logits各自由自己的history query计算，再按0.5/0.5平均。追加训练为0，仅一次完整单卡Evaluation；不扫weight、temperature、normalization、checkpoint或pair。可反驳预测是融合N10高于最强单模型且R10不退化，整体晋级仍要求相对同split固定LIGER的R10/N10均至少+10%。若通过才追加一次固定Testing（全线程仍最多3次，当前1次）；整体未达门槛则关闭该固定融合问题，不自动新预算。整体效果与融合增量归因分开：增量CI跨0不能声称确认组件收益，也不触发扫描；若整体达标，仍按原规则Testing。当前仅规格/实现准备，尚未产生融合效果。[完整五项门禁及预承诺](../../GRID/docs/copmrec-v4-2-fixed-logit-pool-research.md)。

## 2026-10-05：独立商品打分截距阶段关闭

最新执行：学习bias和固定0 bias对照均完成6000步，分别选择best6000，实际checkpoint、optimizer、Config和source355文件审计通过。单卡全量Validation o7ycqky2/yqsmsdt1及真实原始标签独立复算完成：bias R10=.0981084827617046/N10=.05050305197307684，相对同split LIGER +9.263%/+8.678%；control R10=.09877923355542638/N10=.05042652891207879，相对+10.010%/+8.513%。两臂均未同时满足原双10%门槛，无新增Testing。

bias对control净少15个Top10命中（new27/lost42），R10相对−.679%、N10相对+.152%，两项配对CI均跨0，预承诺增量预测未满足。关闭该bias迭代，不扫bias/temperature/LR。新阶段2训练12000步和全线程5训练30000步已全部完成，训练余0；Testing仍1/3。保留有限续训收益及负向/不确定结果，目标仍active未达。当前只读评估冻结source/control跨训练阶段互补证据，新问题须独立登记，不能重置原预算。[完整结题及真实配对证据](../../GRID/docs/copmrec-v4-1-score-bias-research.md)。

原v4残差阶段3×6000/18000步已关闭，不继续mixed或LR序列。当前selected nj9elah1 best6000经mq8hhof5独立Validation核验，对LIGER +6.92%R/+7.15%N，新旧v4增量两CI跨0；未做该best的Testing，已验证deploy仍旧best5000的+5.26%/+5.64%，目标仍未完成。

本阶段启动时的依据与预注册保留：best6000错误Top10集中于Top20商品（18.51%、固定子组约18.52%、各组Top20重合19/20），支持一次独立商品截距的匹配续训检验，不证明校准根因。两臂均从同v4 best6000 weights-only开始，零bias学习对照固定0，主干LR.0001、residual/bias共用item LR.002，保留三loss和learned alpha。实现173项聚焦CPU回归、source及production batch双卡更新均通过；完整运行与结题按上述最新结果解释，不能把启动前无结论状态当作当前状态。

## 2026-10-05：v4 自主残差阶段关闭，保留正向证据且全线程目标未达

当前 v4 阶段状态为 `closed_budget_exhausted_positive_residual_below_goal`。3次训练均已完成、共18000步，训练余0；正式推理4次（Validation3、Testing1），尚未消费的2个Testing槽不构成继续运行本阶段的授权。停止追加残差训练、LR序列、固定mixed排序和mixed竞争loss，不重置预算；全线程“相对固定真实LIGER dense的Recall@10与NDCG@10均至少+10%”目标保持active，未宣称完成。

协同残差的正向证据保留：原两臂相同6000步的残差/scale0对照六个等步点均正向，各自best的验证R10/N10相对+2.4709%/+2.0153%。已验证deploy结果仅为z9envq1n best5000的单卡dense Testing run1edkvgk5，完整22363用户R10=0.07516880561641998、N10=0.03844148315032377，相对LIGER dense分别+5.259862241703184%/+5.6354300044485495%，两项配对CI均为正，仍低于1757命中与N10≥0.04002978116676896的10%门槛。

固定mixed终排相对同候选content的两项配对CI均为负，已关闭。第三训练槽lr20（残差LR0.002、主干0.0001）由首次Testing前的Validation方案预承诺，真实dense-validation best为nj9elah1 best6000。新完整evaluation run mq8hhof5 的独立R10=0.09600679694137638、N10=0.049793932567196456，相对真实LIGER dense+6.9223107569721165%/+7.152035256557077%。相对旧v4 best5000 dense+0.9402914903620108%/+1.3832382019506984%，两项配对CI均跨0；固定step5000两点也为正，但仅是标量验证，无配对或纯LR原因主张，且新旧best选择规则差异保留。

nj9elah1 best6000仅保留为当前开发起点，准确URI为 `wandb://baymaxam/GRID/nj9elah1?role=checkpoint&alias=v0&file=checkpoint_epoch=000_step=006000.ckpt`。其Testing尚未运行，不能与上述已验证deploy结果混用；因最新Validation两项均未达10%，不消费第二Testing槽。完整源码/输入/checkpoint字节、原始用户标签、合法唯一输出与配对统计已审计，详见[自主改进阶段与关闭结论](../../GRID/docs/copmrec-v4-autonomous-research.md)。

本阶段关闭时的下一步是独立score bias问题的只读证据及五项路线门禁；其后已有当前checkpoint正面依据并按上节明确登记新阶段和预算。这里保留原阶段关闭边界，不能包装成已关闭残差阶段的自然追加。以下按时间保留历史授权和当时准备状态，不作为当前运行计划。

## 2026-10-05历史记录：用户授权v0派生长时自主改进

用户明确授权node1 SSH训练和推理，以v0为起点、优先双卡checkpoint续训，推理必须单卡，目标为真实LIGER dense的Recall10与NDCG10均至少+10%。旧零预算/手动运行记录保持；新阶段允许agent启动。首个有依据的改动是zero-init seen-item协同残差，共享历史输入和目录/混合评分，保留v0三loss与learned alpha；新增约1.55M参数并安排相同6000步的无残差续训对照。先最多3训练槽/18000步，阶段不逐轮重置，Testing不用于选择。未达10%不缩小目标或宣称任务完成。

基线042139al dense：1597命中、R10=0.071412601172、N10=0.036390710152；验收至少1757命中、N10≥0.040029781167。v0当前1608命中，无损恢复97个高内容遗漏仍不足目标，因此转向商品判别表示有正面依据。完整门禁、相关工作、损益和累计预算见 [自主改进阶段](../../GRID/docs/copmrec-v4-autonomous-research.md)。

两臂各6000步续训均已完成：v4 z9envq1n验证best5000（N10 .04882854968、R10 .09457585961），scale0纯v0续训5rlfx2np验证best6000（N10 .04786393046、R10 .09229531139）；匹配预算各自best提升2.0153%/2.4709%，六个等步数均正向。checkpoint和全零对照均逐字节核验，训练消费2/3，Testing0次，10%目标尚未达到。用户明确排序与其他模块同样在改进范围，下一步对新增残差checkpoint做一次固定完整SID混合概率终排evaluation验证，同候选保存content参考，另以固定LIGER checkpoint重算dense验证基线；单卡，不扫alpha或用Testing选规则。

## 2026-10-05：固定0.8训练Testing完成，负向结题

x70hphts完整22363用户独立核验通过，固定训练Recall10=0.069713366、NDCG10=0.035501181，相对学习alpha v0分别−3.05%/−2.60%，固定checkpoint用户配对95%区间均低于零。新增184、损失233、净−49；dense命中减少46个，hybrid相对自身dense的净差再减少3个。NDCG差−0.000947133中dense差−0.000971662、候选相对dense差改善+0.000024529，主要下降体现于dense评价，不宣称唯一训练根因。

自身dense目标候选遗漏97→102、额外hybrid命中90→92，两个模型dense目标集合已不同；新模型遗漏按层[71,22,3,6]，全层Mass未消除首层问题。233个最终损失中100例候选不覆盖、133例仍覆盖，218例新模型全目录dense排名超过10。训练和Testing额度已消费，剩余0；固定0.8训练不晋级，保留学习默认和v3.1基座，不自动扫alpha、重训或实施content排序优化。源码342文件、checkpoint/输入/原始标签/输出/trace/独立指标全部通过，本轮无新运行/外部写入。完整报告见 [固定0.8训练Testing](../../GRID/docs/copmrec-v0-fixed-alpha08-x70hphts-result.md)。

## 2026-10-05：固定0.8训练ncyol9n0完成，best44500待单卡Testing

用户已完成一次正式50k训练，配置与准备一致，1000条训练alpha均0.800000011920929；验证均值为FP32聚合舍入，checkpoint固定策略和bias正确。best44500的dense val/ndcg@10=0.046339214，与100次验证history最大值一致，比学习alpha v0 best的0.047430176低约2.30%。尚不能据此认定hybrid Testing退化或提高，当前开发基座仍为v3.1。

实际源码归档342个文件及16个准备文件SHA、输入digests、checkpoint文件MD5/策略均核验。last.ckpt实际step44500不是末步恢复状态，50k完成由run output终止日志证明；推理固定验证best44500。完整单卡命令、Hydra、URI、164个state tensor严格加载及冻结gate通过；Mutagen三会话正常、12个相关推理文件一致。训练额度消费后剩余0，准备一次用户手动Testing，agent未启动推理，不追加训练或alpha扫描，取消的0.813推理仍剩余0。命令见 [ncyol9n0单卡Testing](../../GRID/docs/copmrec-v0-fixed-alpha08-ncyol9n0-testing.md)。下方保留准备时记录。

## 2026-10-05：取消v0固定推理，改为固定alpha=0.8从头训练

用户明确取消尚未启动的v0固定0.813推理，旧额度剩余归零；新增一次v0固定0.8从头训练，由用户手动运行，无扫描预算，旧阶段预算不重置，v3.1开发基座保留。问题从“覆盖推理权重”改为“固定全局权重下的联合表示学习能否改善推荐”；过去v3.1推理结果不能代替新训练验证。

使用全层Mass和三项等权损失，随机初始化，冻结gate，其余真实v0配置相同。121项CPU概率/梯度/固定参数更新/恢复/trace/Hydra/脚本及旧版本回归检查通过；实时W&B v0数据和optimizer/scheduler完全一致，trainer仅输出目录不同，模型仅新增固定策略及显式false的历史缺省path_trace。Mutagen三会话正常、远端16个文件SHA一致，双卡零学习率1-step装配验证通过，日志alpha0.8000；该验证每卡batch2、禁用W&B及checkpoint，不证明正式batch128显存或效果。完整训练未启动，剩余用户手动训练额度1。

主要效果标准沿用Testing NDCG10及Recall10，训练仍以dense val/ndcg@10选best。正向支持有限固定训练消融证据；负向/不确定保持学习默认，不自动追加训练/扫描。训练完成审核run后再准备Testing。见 [固定0.8训练命令](../../GRID/docs/copmrec-v0-fixed-alpha08-training.md)。

## 2026-10-04：用户指定v0固定alpha=0.813的单次推理对照

本节为当时准备记录；2026-10-05用户已取消此完整推理，未运行，额度剩余0，改为上述固定0.8从头训练。

复用真实v0的7y54j4m6 best48000，已有 `inference_mixture_alpha` 字段固定0.813，对照6dspa7e3；全层Mass、beam20、content终排、输入与参数保持，只增加已有path trace观测。原学习值0.813255608与固定值非常接近，本次只能判断这点推理变化，不能评价固定权重训练或推出0.813最优。

一次新增用户手动完整Testing、零训练/扫描，不重置旧预算、不改变当前v3.1开发基座。若最终净改善则保留该设置为该checkpoint对照证据；负向或不确定保持学习值，不自动追加。9项聚焦测试、命令透传/Hydra与shell语法检查通过；Mutagen三会话正常、远端12个相关文件摘要一致。真实best48000的164个state tensor严格加载且覆盖前后相同；与6dspa7e3模型配置唯一差异为推理alpha，数据配置相同。未进行真实模型前向、候选搜索或完整推理，新增额度仍为1，效果待用户运行。见 [v0固定alpha命令](../../GRID/docs/copmrec-v0-fixed-alpha-0813-testing.md)。

## 2026-10-04：固定alpha0.813完成Testing，减少遗漏但无最终净命中

用户run3lb22b0r与zznw291t同best43500，仅推理alpha改变。完整22363用户、预测/路径、实际源码与输入独立核验通过，dense Top10及目标dense rank逐项相同。高内容遗漏81→52，原遗漏恢复36、新增7；原81个超dense增量保留48、损失33，新增4，当前52，净损失29恰抵消高内容净恢复29。全体新增40、损失40，Recall10=0.070384117不变，NDCG10=0.035855580、相对+0.0775%，区间[−0.000237653,+0.000308928]跨零。

33个增量损失只有5例目标不在候选，28例仍覆盖而跌出Top10；其中25例伴随新的高内容候选入场。共同命中12升位/92降位，位置贡献负；原首层47个候选全部保留，原29个命中保留28。支持提高alpha保护高内容候选及与content终排的取舍，未证明终排是唯一原因或0.813最优。

保留v3.1及学习alpha默认值，固定0.813不晋级，不自动扫描、重训或实施完成上界。此次一次用户推理已消费、剩余0，旧阶段预算不重置，分析新增运行/外部写入0。完整结果见 [固定alpha结果](../../GRID/docs/copmrec-v3-1-alpha0813-testing-3lb22b0r-result.md)。下方准备记录保留当时状态。

## 2026-10-04：用户指定v3.1固定alpha=0.813的单次推理对照

保留v3.1，用户新授权一次同rbha00vx best43500的固定推理权重对照；不训练、不扫参、不实施完成上界候选。通过 `model.root.inference_mixture_alpha=0.813` 同时覆盖首层Max、后层Mass及root包络混合，checkpoint学习gate保持约0.668786，默认仍使用学习值。

对照zznw291t检验较高内容引导是否减少遗漏并取得最终净收益。首层规则保持但frontier允许变化；dense评分/排名、模型参数与用户输入必须一致。主要评价NDCG10，联合看Recall10、81个遗漏、81个当前增量命中及原首层恢复的损益。正向支持固定设置，负向回退学习值，不确定不自动扩扫描；Testing开发使用边界保持。

实现已通过63项聚焦检查和OpenSpec严格验证；Mutagen三会话同步完成、无冲突，远端7个运行文件摘要及真实checkpoint的164个state tensor严格加载一致，命令透传/Hydra与shell语法核验通过。完整推理由用户手动启动，当前新增完整运行0。原阶段预算不重置，本次新增推理额度1、训练/扫描额度0。命令与门禁见 [固定alpha单卡对照](../../GRID/docs/copmrec-v3-1-fixed-alpha-0813-testing.md)。

## 2026-10-04：用户保留v3.1，后续聚焦前缀完成概率

用户明确保留v3.1并继续bad case探索，当前基座不撤回，整体稳定收益未建立的证据边界保持。新增只读统计：81个高内容遗漏中65例父节点仍在、31例所有保留兄弟最佳内容更弱、53例局部分支rank≤3；24例目标为单商品前缀、保留兄弟全部为多商品前缀，涉及18个商品。原43个增量命中损失只有10例候选漏召回，33例仍在候选且32例伴随高内容候选入场；共同命中降位109例中81个高内容目标的rank变化可由候选成员变化精确解释。

首选待实施方向：保持v3.1首层/root包络和局部混合概率，在长度2/3前缀使用最佳剩余混合概率上界B。边上界为(1-alpha)+alpha*q_content，逆序max递推，搜索增量增加B(child)-B(parent)，完整SID时B=0，固定完整路径分数与v3.1一致。无需新权重、配额、head、生成前向或训练；只调整中间前缀比较，数学上界不保证有限beam推荐收益。

300随机树/31950路径/36936前缀的上界及势差恒等式通过，仅为公式验证；24例不是预期恢复数。若后续获准，最小建议一次同checkpoint推理与zznw291t比较，检验最终损益与81个当前增量命中的保留，而非只看dense召回。当前授权仅探索，剩余实际推理预算0、新训练/调参预算0，本轮模型/配置/脚本修改0、模型前向/实际搜索/新实验/外部写入0，旧关闭预算不重置。

完整方案、候选取舍、相关工作定位和结果门禁见 [后续bad case报告](../../GRID/docs/copmrec-v3-1-bad-case-directions.md)。

## 2026-10-04：v3.1 Testing已核验，候选局部修复但整体未晋级

用户推理 zznw291t 使用与 v3 相同的 rbha00vx best43500。独立核验完整22363名Testing用户、实际输入/产物/源码、指标及一次路径补正；首层 frontier 逐项一致，dense Top10及目标dense rank相同。原高内容遗漏114个恢复58个、新增遗漏25个，净减少到81个，分层为0/60/11/10。原v3首层恢复的47个候选与29个Top10命中全部保留。

全体相对v3新增命中79、损失68、净增11；Recall10=0.070384、NDCG10=0.035828，较v3分别+0.704%/+0.354%，两项用户配对区间均跨零。原103个超出dense Top10的增量命中保留60、丢失43，共同命中30人升位、109人降位；自身dense遗漏与新增增量各81个，Recall恰与dense相等。相对真实v0和LIGER dense的两项@10点估计仍较低，区间跨零；Testing开发边界及固定checkpoint区间范围保持。

相同局部分支rank下58个目标恢复，支持跨root路径优先级补正的局部机制；不证明稳定最终收益或唯一根因。保留首层Max，v3.1保留为开发候选decoder，当前完整方法不正式晋级；约定一次推理已消费、剩余推理预算0，新增训练/调参预算0，不自动重训或扫描下一机制。原关闭阶段保持。本轮只读复算，运行代码/模型前向/搜索/新实验/外部写入均0。

完整损益、区间和案例见 [v3.1结果报告](../../GRID/docs/copmrec-v3-1-testing-zznw291t-result.md)。下方准备及探索记录保留当时状态。

## 2026-10-04：v3.1推理实现完成，单卡命令已准备

用户授权实施上一轮 root 包络修正并提供命令。新增仅推理 decoder，复用 rbha00vx best43500，训练版本仍为 v3、解码版本为 v3.1；首层原 Max 选择，第二层一次 root 先验补正，后续 Mass 与内容终排不变，单 beam20 加全部 cold。独立路径 schema 将原条件概率、root 先验/补正和累计搜索分数分开记录。

52 项聚焦检查通过，含真实四层 HF transition 分数核对、首层 frontier、标签不影响搜索、旧 v3 回归和 writer 校验。node1 的真实 checkpoint MD5、164 个 state tensor 的严格加载、7 个改动运行文件 SHA256 和最终命令替身/Hydra 均核验；Mutagen 三会话已同步且无冲突。

下一步仅一次用户手动 single-GPU full Testing，physical GPU2→logical0、seed42、predict batch32；本轮完整推理/训练启动均0，新增训练和调参预算0，原关闭预算不重置。对照既有 zy8946l2，评估114个遗漏、103个增量命中和全体最终损益；Testing开发使用边界及上一轮正/负/不确定结果门禁继续沿用。尚无v3.1完整效果结果。

可直接复制 [单卡推理命令](../../GRID/docs/copmrec-v3-1-rbha00vx-testing.md)。下方探索记录按当时状态保留。

## 2026-10-04：保留v3首层Max，完成v3.1后续候选探索

用户明确保留首层恢复组件，本轮只分析与探索。新增只读统计：114个dense Top10遗漏中，第二层73例只有10例整个root消失、63例仍保留其他分支；全层94例父前缀仍在，46例保留兄弟的最佳内容后代均弱于目标。纯content极限下Max根加Mass后续会留下root相关的LSE−max项；21个可从已有trace精确计算的失败案例中位数约1.83。该性质不是实现bug，也未被证明解释全部失败。

首选v3.1候选是解码修订：第1层仍用原Max混合分数选相同beam20；第二层对选中root只补正一次，后续路径先验取Max-root与Mass-root混合分数的max包络再统一归一化，继续原局部Mass混合。保留Max分数的相对优先级，复用best43500、alpha及内容参数，不新增head/loss/配额/权重。它仍可能挤掉原命中，不能提前宣称恢复或收益。

建议最小下一验证仅1次同checkpoint推理，与已有zy8946l2配对比较，第1层frontier与dense排名必须相同；同时报告114个遗漏的恢复、103个现有增量命中及全体损益。该推理尚未授权/实施，新增训练建议0，旧关闭预算保持，本次模型/配置/脚本修改0、前向/搜索/运行0。数学恒等式检查通过不构成性能证据，Testing开发使用边界继续披露。

完整方案、备选取舍与路线门禁见 [v3.1探索报告](../../GRID/docs/copmrec-v3-1-exploration.md)。下方“v3整体未晋级”保留整体效果判断，不取消此次用户确认的首层组件。

## 2026-10-04：v3 Testing独立核验，首层改善但整体未晋级

用户单卡推理run zy8946l2消费rbha00vx best43500；独立验证22363名Testing用户、真实目标、同输入/catalog、合法预测、checkpoint及实际源码。Recall10=0.069892、NDCG10=0.035701，相对真实v0分别−2.80%/−2.05%，相对LIGER dense−2.13%/−1.89%；两项用户配对95%区间均跨零。@5点估计提高，但@10没有建立净收益，不认定稳定退化或多seed效果。

首层局部预测获得观测支持：v3自身dense Top10目标首层遗漏0，对照v0为67；原67例首层失败恢复47例候选、29例Top10。不过v3第2–4层仍遗漏73/29/12，114例局部分支rank均≤20、51例合法分支总数≤20，指向跨父前缀累计分数的全局beam竞争。跨模型覆盖新增387、损失449；命中新增264、损失309。两个checkpoint都dense Top10的1325人中，搜索恢复32、丢失53。

NDCG差值账目−0.000746855约由dense变化−0.000744453和候选转换差−0.000002402构成，相关配对区间均跨零；内容/生成参数及alpha同时变化，不确认唯一原因。保留首层改善与后层失利，当前v3暂不晋级；已关闭v2及旧预算保持，本次只读产物复算，无模型修改、前向、搜索、训练/推理启动或新增预算。第二层全局竞争记录为未决候选问题，尚未选择新机制或追加实验。

完整统计与案例见 [v3结果报告](../../GRID/docs/copmrec-v3-testing-zy8946l2-result.md)。testing开发使用边界继续披露，下方待Testing记录按当时状态保留。

## 2026-10-04：v3训练rbha00vx完成，固定best待Testing

用户报告run rbha00vx完成。实时核验finished、实际训练配置与交付v3一致，训练源码指纹及25个相关运行文件匹配；输入SID/embedding lineage正确。完整100个validation记录、best Artifact与node1 checkpoint共同确认best43500，记录NDCG@10=0.04560798406600952、Recall@10=0.08981499820947647，保存alpha=0.668786346912384及[max,mass,mass,mass]契约。这些为validation结果，尚未构成testing收益。

[固定best单卡Testing命令](../../GRID/docs/copmrec-v3-rbha00vx-testing.md) 已按用户要求改为物理GPU2、逻辑[0]的单卡推理，并通过node1 Bash替身与Hydra核验，沿用完整testing、beam20加全部cold及content终排，额外保留candidate/path trace用于候选恢复/损失与剪枝层分析。Mutagen flush成功、三个session均Watching且无conflict；此次agent未启动前向、推理、续训或新增预算。

下一步由用户手动执行Testing，返回inference run后独立复算并与真实v0/LIGER dense作匹配比较，再判断候选改动净收益。没有因训练完成而晋级方法或追加实验；下方交付时的“尚待训练”按历史阶段保留。

## 2026-10-04：v3候选改动已实施，由用户手动开始同v0配置训练

用户授权将“第一层Max、后续Mass”实现为从真实v0派生的v3，当前重点转向候选搜索。根层遗漏有直接路径证据，完整Max对照能恢复部分v0遗漏；首层变化会改变后续frontier，不能把局部优势相加。此前v1/v1.1排序尝试未建立完整净收益，v2testing负向且迭代保持关闭，旧content分析及关闭预算保留。

v3训练混合NLL与候选搜索使用同一逐层聚合，三项基础loss等权、全参数从零、无新增head或teacher；直接继承真实v0的完整training/evaluation/testing、50k更新、每卡128、累积1、500验证间隔、dense NDCG选best及content终排。双卡物理GPU2/3映射逻辑[0,1]。独立入口、恢复契约与 [手动训练命令](../../GRID/docs/copmrec-v3.md) 已交付，完整实验仍由用户开始。

本地120项聚焦检查通过、1项LinuxGloo在Windows跳过；node1另13项通过，含两进程连续更新。v0/v1/v1.1/v2八个resolved行为配置不变，25个运行文件hash一致，Mutagen三会话Watching、无conflict。实现与命令核验不构成效果；本轮完整训练/推理新增0，未启动、停止或续训既有run，agent不追加调参或重置旧预算。

下一决策基于各自validation-selected best在相同testing上的最终Recall/NDCG及候选新增/损失/位置变化。只改善coverage而净收益未改善时，不据此晋级或自动扩预算；testing已经用于开发分析，后续确认明确保留该边界。此前方向2选择及“模型尚未实施”均为历史阶段记录。

## 2026-10-04：v3选择方向2，真实v0 content bad case分析完成

用户选择在当前content排序上优化，方向1不作为本阶段方案。只读核验真实v0 `7y54j4m6 → 6dspa7e3` best48000及22363 testing用户，连接全部原始历史/商品文本、实际1024维content_bank和training出现次数；完整保存853个已覆盖漏排案例与全量正面参照。719个warm漏排中406个处于候选11–15；134个cold漏排分开记录，97个dense Top10缺候选不归为content错误。

具体案例揭示历史商品占位、相近商品区分失败以及输入语义支持未转为模型排名。22363目标均不在最近20件历史，v0 Top1却有3108次为历史商品。固定评分/候选，排除历史的严格排名界至少恢复127、最多167命中，Recall范围0.077584–0.079372，NDCG范围0.042814–0.043390；这是数学界而非新推理或置信区间。相同规则也至少恢复LIGER dense120和v0 dense143，不能与未处理baseline比较宣称v3收益，正式协议未改变。

warm漏排与命中历史长度/类别多样性接近，最大输入相似度也不低；同类竞争未显示错误特有集中。频次差及同目标186商品对照只支持描述性线索，不能确认热门偏差、多兴趣压缩或某层根因。先明确历史竞争的公平比较口径，再聚焦剩余warm竞争判别；暂不新增head/loss，不因cold或搜索组扩大问题。

完整报告：[v3 content具体bad case](../../GRID/docs/copmrec-v3-content-bad-cases.md)。验证：源码/hash与全部用户/案例/元数据契约通过，5000组完整随机排名验证数学上下界通过。本轮运行代码改动0、模型前向/搜索0、训练/推理run0、预算新增0、外部写入0，旧关闭预算保持；testing开发使用明确披露，模型/loss未实施。

## 2026-10-04：用户结束v2迭代，v3先完成收益转化问题分析

v3与v1/v2平级，基于真实v0；推荐参数全部可训练，允许预生成embedding/SID，不使用冻结推荐teacher。当前只完成问题分析，模型/loss未定案，无v3运行入口；v2负向证据和入口保留，不追加v2.x、权重扫描或续训。既有关闭阶段及2000更新校准预算不转移、不重置。

本轮结合正式矩阵、旧冻结保护/高位/边界bad case、v1有界校准及v2 testing，并只读重算同v0/v2 trace的逐用户排名。v0已较好修复dense可识别目标的候选遗漏，却未建立超出内容排序的稳定增量能力；v2恢复239个同候选content漏排目标、损失243个，命中交换对NDCG略正，但共同命中位置损失更大。原content第1名262人中152人降位，11人跌出Top10；v2有211个dense Top10目标已入候选却被head丢失，而v0此项为0。

v3的问题收敛为：保持基础检索与候选能力，学习可靠的增量竞争判断，并完整评价已有正确关系的损失。取消冻结、增加teacher、降低loss或限制分数幅度均不能单独保证收益；d_ui信息不足、lambda是v2主因等尚未证实，不据此再加模块或扫参。testing已用于本轮事后开发分析，不能在未来包装为完全独立确认。

完整分析与证据等级见 [v3卡点报告](../../GRID/docs/copmrec-v3-conversion-bottlenecks.md)，新统计见 [只读排名分解](../../GRID/docs/evidence/copmrec-v3-bottlenecks-20261004/rank-analysis.json)。本轮无运行代码改动、无模型前向/搜索、训练/推理新增0、预算新增0。下方各阶段记录保留其历史状态。

## 2026-10-04：v2 testing已独立核验，当前组件不晋级

用户提供推理run1gu2dsa1，消费beawrjef validation-selected best42500。独立核验完整22363名testing用户/真实标签、相同SID/embedding与catalog、输出合法性及checkpoint/Artifact摘要；Recall10=0.069087、NDCG10=0.033285，与W&B一致。相对真实v0 run6dspa7e3，分别下降3.92%/8.68%，两项用户配对区间均负；相对LIGER dense run042139al的NDCG下降8.53%、区间全负，Recall区间上界为0。

同v2 checkpoint和自然候选，content-only NDCG10=0.035615，加入相关性后降到0.033285；Δ−0.002330区间全负。净Top10命中仅−4，但共同命中用户651人位置下降、256人提升。覆盖净少3人且dense变化区间跨零，不能将当前退化主要解释为新增漏召回，或确认ranking loss已显著破坏内容表示。修正上界合法，但最终Top10中46.34%接近各自允许幅度边界；只是诊断现象，不单独确认根因。

当前v2未建立推荐收益，不晋级该实现；保留负证据，不否定整个d_ui概念或论文路线。此次只读复算，没有启动训练/推理、没有新增或重置预算，不自动扫描权重/追加实验。完整统计、机制边界与阶段判断见 [结果报告](../../GRID/docs/copmrec-v2-testing-1gu2dsa1-result.md)。下方testing待核验为登记时历史状态。

## 2026-10-04：用户登记v2训练run beawrjef

用户请求推理命令用于后续效果评估。已核验beawrjef的validation-selected best为step42500，Artifact不可变版本v0、producer和best/max元信息、checkpoint内ModelCheckpoint记录及node1本地MD5均一致。对应验证NDCG@10=0.04209396615624428、Recall@10=0.08737646788358688；这是validation结果。固定best的 [testing命令](../../GRID/docs/copmrec-v2-beawrjef-testing.md) 通过node1 Bash替身与Hydra核验，使用full testing并发布预测和候选trace；本次未启动推理。独立testing复算及与真实v0/LIGER dense的匹配比较仍待用户运行后进行，下段“尚未核验best”为登记时历史状态。

用户明确当前v2训练run为 [beawrjef](https://wandb.ai/baymaxam/GRID/runs/beawrjef)。2026-10-04 13:27:41（Asia/Shanghai）实时读取W&B，状态finished，最后记录global_step49999。实际配置符合统一后的v2：从零初始化、全evaluation/testing、每500微批验证、lambda0.01/beta0.5、双卡每卡batch128与累积1、50k更新。

已保存run完整配置与summary快照，登记为v2当前效果研究对象；此次只核验配置身份，尚未核验validation-selected best Artifact或独立testing，不将summary标量视为best结果或推荐收益结论。不自动启动推理、对照或追加训练，本次agent启动run及新增训练预算均0。

## 2026-10-04：v0就是真实基础方法，统一v0/v2配置

用户明确BMX-116已有的基础方法就是v0；原版本化入口不应改变基础方法协议。已以run7y54j4m6实际完整配置恢复v0：FileDataModule全evaluation dense验证与选点、testing hybrid推理、500验证间隔；数据、模型和基础训练参数与原run一致，只保留输出目录、版本产物名称和默认关闭trace等非行为差异。

用户确认v2也使用同一evaluation/testing数据链路及500记录间隔，保留融合hybrid选点、固定lambda0.01/beta0.5、全部参数联合训练和DDP安全设置。v1/v1.1已结束迭代，其selection/audit与2500频率显式配置，四个resolved行为装配不变；不把旧开发曲线当作新数据协议对照。

本地62项聚焦检查、Ruff及相关OpenSpec strict通过；node1八个组合及八个Bash入口替身核验通过，33个运行文件指纹一致，Mutagen三个session均Watching、无conflict。v2最新从零命令及完整差异见 [统一交付](../../GRID/docs/copmrec-v0-config-unification.md)。本次未启动/停止正式训练、未改写历史run、未新增或重置预算；最终以各自evaluation选出的best在相同testing上比较Recall/NDCG。

下方v2初次实现及频率调整记录按当时协议保留，selection/audit叙述已由本条统一决定取代。

## 2026-10-04：独立实现v2，固定lambda0.01/beta0.5

用户确认v1迭代结束于v1.1，v2与v1平级并基于v0，后续为v2.x。已实现d-only相关性head：自然beam+cold集合的content population方差加epsilon²后开方、仅统计量detach，以0.5*sigma*tanh(head(d))修正content；训练正例/content Top20不改变统计集合。基础三项loss加固定0.01融合CE，单一optimizer、全部推荐参数共同训练，默认从零。

按用户要求，v2验证频率由2500调整为500微批，使用独立trainer，对齐历史v0 run7y54j4m6实际配置和499/999起始记录步数。11项配置/脚本检查通过，node1默认配置及文档命令替身核验通过，19个运行文件指纹一致、Mutagen三个session均Watching且无conflict。保持默认累积1、50k更新、selection/hybrid口径及共享指标记录；本次频率调整不会将历史v0 full/dense标量变成匹配效果证据，也不修改v1/v1.1配置或启动/停止训练。

本地98项检查通过、1项Linux DDP在Windows跳过；node1另20项CPU检查通过，含两进程连续联合更新和参数同步。Mutagen三个session均Watching、无conflict，18个运行文件指纹一致，v1/v1.1九个运行文件保持原字节；文档中的从零双卡命令通过替身及配置核验，没有实际启动训练。独立checkpoint/trace、尺度及饱和诊断、从零双卡命令见 [v2交付](../../GRID/docs/copmrec-v2.md)。本次完整实验预算/run均0，无训练启动或停止，旧2000更新校准预算保持关闭。lambda/beta是用户指定值，不宣称最优或新校准；推荐收益待匹配v0与独立评价，不能用实现检查或loss下降替代Recall/NDCG。

## 2026-10-04：删除v1.2，退回v1.1

用户撤回简化head版本。v1.2组件、配置、脚本与专属测试已删除，原规格移入GRID/docs/archive；共享代码恢复四路head和v1/v1.1 trace契约。当前方法保持v1.1批量候选评分和校准排序权重0.05726763550972437，不回退为等权。原实现与核验保留作历史，不作效果否证；运行产物与已有预算保持，本次不启动或停止训练。有效命令见 [v1.1说明](../../GRID/docs/copmrec-v1-1.md)，撤回记录见 [说明](../../GRID/docs/copmrec-v1-2.md)。

本地68项回归通过、2项Linux DDP在Windows跳过；Ruff/OpenSpec strict通过。Mutagen三Watching无conflict，node1核实6个运行文件删除、活动源码无v1.2引用、10个v1.1文件指纹一致，双卡训练/推理参数与配置均通过替身核验。

## 2026-10-04：用户授权head简化为v1.2

> 本节为历史实现记录，已被上方用户撤回决定取代，v1.2入口不再可用。

校准从零run57hzkcol已由用户启动，快照时running/global_step29999；最近已记录27.5k selection验证Recall0.044003/NDCG0.018914，较22.5k回落，但仍强于等权同步数。旧v0全量dense曲线仅作不同口径参考，尚不能证明head根因。短程校准两臂已完成且预算保持关闭。

按用户授权，v1.2 head只使用读取完整SID后的decoder末状态，512→128输入缩为128维，仍用content+残差和0.05726763550972437排序权重；此系数继承以隔离结构变化，不宣称适配新head最优。全部推荐参数持续训练，保留批量评分、候选及固定selection/audit协议，独立版本与checkpoint/trace契约。

本地87项检查通过，3项Linux DDP在Windows跳过；node1另外10项检查通过，含v1.2两进程连续更新与参数同步。文档命令仅使用uv替身核验，无实际训练；Ruff/OpenSpec strict通过，Mutagen三Watching且9个运行文件指纹一致。完整命令及证据边界见 [v1.2交付](../../GRID/docs/copmrec-v1-2.md)。本次agent完整训练/新增训练预算为0，未停止已有run，无自动补对照或权重扫描。用户手动运行后以同协议最终Recall/NDCG判断是否保留简化候选，不把实现通过当作收益。

## 2026-10-04：采纳校准权重，交付从零训练命令

用户采纳排序权重0.05726763550972437，明确要求从头初始化、禁止上次checkpoint续训。v1.1组件默认值已更新；命令显式关闭ckpt恢复及v0预训练，保留原双卡batch128、累积1、50k更新、beam20和固定selection/audit。W&B使用 `copmrec_v1_1_calibrated_scratch` group区分等权run。

完整训练由用户手动开始，本次未启动或停止任何完整训练。下一次完整训练用于检查短程恢复能否转化为从零全程推荐收益；当前尚未证明该收益，已有短程阶段预算与结果保持。见 [完整命令](../../GRID/docs/copmrec-v1-1.md)。

## 2026-10-04：验证 v1.1 排序 loss 等权失衡

用户授权进行有界验证。固定 f91njtjx 的17.5k validation最优checkpoint，training-only 16对微批校准权重0.0572676，另16对保留样本的ranking/基础梯度比中位数6.46→0.37、合成方向余弦0.187→0.937。两臂各1000更新、相同模型/AdamW/scheduler和数据seed，只改变ranking权重；全部推荐参数可训练、无冻结teacher，通过现有统一入口执行，原run继续运行。

两臂已完成（等权68h3e3eq、校准mg0nrz9a），固定2048用户：dense Recall0.008789→0.014648，自然覆盖1.5625%→2.1973%，最终hybrid Recall0.006348→0.012695、NDCG0.003617→0.006439；两路Recall配对区间全正，最终NDCG区间仍跨零。支持等权是当前退化的重要因素，保留校准低权重候选；未证明完整从零训练或v0/LIGER优势，不宣称解释全部差距。

固定新阶段2/2臂、2000/2000更新已完成，完整训练预算0；不转移或重置旧关闭阶段，不自动追加权重扫描、第三臂或完整训练。完整证据与边界见 [验证记录](../../GRID/docs/copmrec-loss-weight-verification.md)。

## 2026-10-03：用户采纳，新组件已实现为 v1，基础版本记为 v0

CoPMRec v1 已实现当前候选条件相关性 head 与候选列表 CE，从第一步与基础三项目标训练全部推荐参数，支持显式 v0 weights-only 预训练后共同训练；没有teacher或冻结推荐模块。基础方法（BMX-116）记为 v0，旧正式入口及旧 checkpoint 保持兼容，两版新增开发入口共用固定selection/audit和各自最终hybrid NDCG10选点。

本地120项聚焦测试通过、2项Linux Gloo检查在Windows跳过；node1的13项检查通过，包含v1两进程连续更新和非等长分片指标。真实Beauty生产模型batch128零更新探针通过，157组推荐参数梯度有限，峰值allocated约3.103GiB；不含optimizer状态/全DDP，不能作为完整吞吐或效果证据。OpenSpec strict、Mutagen三Watching与20运行文件指纹通过；版本、默认协议及手动命令见 [实现交付](../../GRID/docs/copmrec-versions.md)。

当前完成实现与验证，完整训练、最终Recall/NDCG净收益仍待用户手动运行和核验。新增完整预算/运行均为0，旧阶段不重置、旧scratch额度不转移。下方“未实施”是设计时历史状态，已由本条更新。

## 2026-10-03：旧入口已清理，完整链路仍缺推荐收益转化

用户明确：BMX-116 已有基础 CoPMRec 的有效训练与推理结果，但相对 LIGER dense 未建立明显优势；双分支保护和高位修正是冻结基础模型后的探索，需清理入口并重新设计模型内组件。上一轮“回到基础 content 终排即完成方法”的建议由本条取代。

已归档并退出 10 个根脚本、9 个 experiment 和 4 份启动测试；基础 CoPMRec/LIGER 入口、算法、结果保留。108 项聚焦测试、OpenSpec strict、Mutagen 三会话 Watching 和远端 19 入口缺失核验通过。详情见 [GRID 清理记录](../../GRID/docs/copmrec-conversion-entrypoint-retirement.md)。下文相关旧启动命令均为追溯，不能直接运行。

新设计为共享模型内的候选条件相关性 head：读取当前用户、内容和完整 SID decoder 表示，以 content 分数加可学习相关性残差终排；候选 softmax 标签监督与基础三项 loss 联合训练全部推荐参数。保留原 mass beam20 与 cold 候选，不使用冻结 teacher、单独 ranker 训练或 covered-only 人群。双分支保护/高位经验落实为减少错误替换和关注前排竞争，不原样搬入旧冻结流程。

设计、累计结果、可检验预测及固定决策协议见 [完整链路与组件设计](../docs/2026-10-03-copmrec-conversion-redesign.md)。组件尚未实施，最终 Recall/NDCG 收益未知。最小开发比较拟议 2 训练 + 2 audit 共 4 run，尚未分配预算；本轮新增完整预算与运行均为 0，旧阶段不重置，旧 scratch 额度不转移。后续先实施并验证训练/标签/输出契约，完整实验仍由用户手动开始。

## 2026-10-03：整体方法重梳理，推荐模型全参数持续联合训练

> 下节为此前建议，已被上方用户澄清与新设计取代；其中 scratch 入口已退役。

用户明确要求从零训练一个完整模型，或预训练部分后进入全参数联合训练；已确认允许预生成 embedding 与固定 SID，推荐模型不保留冻结模块或 teacher。完成[整体方法、公式、组件取舍与证据边界](../docs/2026-10-03-copmrec-unified-method.md)。建议收敛为共享 encoder、内容/SID 两个直接标签监督 head、合法子树内容概率质量和逐层概率混合，从第一步统一训练 SID CE + content CE + mixed NLL，最终由同一 content head 排序。

双分支保护、capped NDCG、边界辅助、独立 ranker 与 decoder 单独微调移出建议主方法；既有正负结果和已关闭阶段预算保留。旧 `copmrec_scratch_train_ddp2` 仍会在 25k 后复制冻结 teacher，不能作为这次无冻结定义的运行入口。当前已完成方法梳理，未迁移生产代码/配置、未停止或替换实际运行、未新增完整实验预算或运行；旧 scratch 的预算不自动转移。下文交付及命令按其原协议保留。

## 2026-10-03：按用户要求改为 GPU 2、3 双卡执行

新增 `copmrec_scratch_train_ddp2`，每卡batch64、累积2、每卡排序4例；有效batch256、每更新排序16例和50k更新预算不变。校准跨卡汇总、验证无重复分片，仍沿用同一研究问题及阶段预算。下方单卡交付为历史记录，当前命令以交付文档双卡补充为准。

## 2026-10-03：用户采纳全模型随机初始化，高位联合训练已交付

新增同一次训练内25k基础联合训练、training校准冻结teacher/cap、后半程全模型高位排序；总50k更新、单GPU有效batch256，在线短列表每微批8个样本，不限旧3454 covered。固定selection/audit划分与旧集合一致。45测试、OpenSpec strict、真实数据零更新探针、Mutagen三Watching及13文件hash通过，尚未完整训练。既有baseline与旧阶段结论保持，上轮高位梯度重分配暂缓。固定新阶段训练+audit共2run，0开始，用户手动启动。见[完整协议、边界与命令](../docs/grid-experiments/2026-10-03-copmrec-scratch-delivery.md)。

## 2026-10-03：回到高位退化，下一候选先做参数鉴别

复用四臂与3454 training covered分数。cap相对单保护仍有9个rank1退化，占共同负向NDCG的59.5%；边界辅助修复其中6个却全体净丢12命中，保留其负向关闭决定。九例十条越过关系6条已有teacher支持；cap不改变同用户pair份额，最高分负例平均只得9.66%的主负例梯度。

下一候选限定为当前Top3训练样本的竞争负例重分配，保留逐用户主正例梯度总量，boundary关闭；原training rank1最高负例份额5.44%→12.52%，rank2–3的12.28%→29.89%，其余2365人的直接分数梯度不变。尚未参数空间验证或实施，不宣称泛化无损。后续优先初始化和已训练状态的小批参数鉴别，当前无新增完整预算，既有阶段预算保持关闭。见[案例复查与方案](../docs/grid-experiments/2026-10-03-copmrec-head-revisit-analysis.md)。

## 2026-10-03：边界辅助audit负向，固定实例关闭

p9czcxbg有效audit11182用户，NDCG10=0.04733686、Recall10=0.09059202，1013命中。相对7wnteuw0新增5/丢17净−12，Recall配对区间全负；约99.76% NDCG下降来自命中交换，旧81损失仅修复1并新增11损失。阶段2/2剩余0，不晋级、不自动扫权重或续训。详见[结果及案例](../docs/grid-experiments/2026-10-03-copmrec-boundary-audit-result.md)。

## 2026-10-03：边界辅助训练完成，等待固定best audit

msut1s0r有效完成20epoch/2160更新，best epoch11/step1296。selection NDCG0.04722317、Recall0.09176281：相对szuw834d NDCG略降、约多2命中，收益未确认。目标校准、冻结hash、cache/history及293源码文件核验通过；阶段1/2，剩余一次手动audit，命令已验证。见[训练结果和audit命令](../docs/grid-experiments/2026-10-03-copmrec-boundary-training-result.md)。

## 2026-10-03：边界辅助项已实现并交付

32 training用户参数探针通过，生产目标及training-only校准lambda=0.2967149913完成；36项测试、严格OpenSpec、node1生产loss/冻结/恢复探针和5运行文件hash通过。完整训练未启动，新固定阶段2run、0开始，旧阶段不重置。见[交付与启动命令](../docs/grid-experiments/2026-10-03-copmrec-boundary-delivery.md)。

## 2026-10-03：边界case分析与辅助目标设计

原training与audit都显示边界负例主NDCG负例梯度份额约5.3%，但正例已从其他pair获得监督，不能断言梯度遗漏。81个content丢失的边界关系63已teacher支持/18未支持。提出保留现有目标并补充第10负例softplus辅助项，training-only校准初始25%正例分数梯度预算（lambda约0.296715）；需先32 training用户参数探针，尚未实施/运行。旧阶段2/2关闭，新预算未授权。见[分析与方案](../docs/grid-experiments/2026-10-03-copmrec-boundary-analysis-proposal.md)。

## 2026-10-02：分母上限audit完成，固定阶段关闭

7wnteuw0独立复算11182用户：NDCG10=0.04765447、Recall10=0.09166518，1025命中。相对双保护新增3/丢1净+2，NDCG点增但CI跨零；相对content仍净−10。9个重点高位退化仅修复1个，生成独有命中59→59；局部改善不足以晋级，阶段2/2剩余0，不追加训练。保留source origin mismatch（LETTER及依赖文件变化，CoPMRec执行源码未变）。见[完整audit及case结果](../docs/grid-experiments/2026-10-02-copmrec-head-risk-audit-result.md)。

## 2026-10-02：分母上限训练完成，等待固定best audit

szuw834d训练有效完成20epoch/2160更新；best epoch7/step864，selection NDCG0.04729218408、Recall0.09158393741。相对klqyxo8m近乎持平（NDCG相对+0.0172%、约少1命中），不能认定推荐增益。checkpoint/训练缓存C/冻结hash/262文件source核验通过；阶段1/2，剩余一次用户手动audit，命令配置解析通过。见[训练核验与audit命令](../docs/grid-experiments/2026-10-02-copmrec-head-risk-training-result.md)。

## 2026-10-02：NDCG分母上限已交付，等待用户手动训练

用户采纳后实现training_mean_cap，C仅从原training covered缓存拟合并冻结（6.8165602684021/3454用户），checkpoint及audit严格重验；保留dual teacher weight1和既有训练协议。32项测试、OpenSpec、node1 32用户生产实现loss/梯度/冻结/元数据往返及5文件hash通过，完整训练未启动。

新固定阶段训练+audit共2run，0开始/剩余2，匹配klqyxo8m/pvhjwp9q的原初始化和2160更新，不续训。旧阶段结论和预算保持，性能未知；预测为高位退化减少且保留互补恢复。见[实现与训练命令](../docs/grid-experiments/2026-10-02-copmrec-head-risk-delivery.md)。

## 2026-10-02：双分支case分析完成，优先补强高位排序监督

复用pvhjwp9q/nwxzx1qs形成213条案例。4个新恢复全部原generation Top10；64个共同退化用户的70条越过关系中49条已双teacher覆盖，继续扩teacher不是首选。9个第1名退化贡献共同命中全部负向NDCG的60.8%。实际NDCG loss按每用户pair权重和归一化，training第1名分母均值15.72、未命中4.54，提供高位风险缩放的具体鉴别依据。

下一候选保留双teacher weight1，仅将主NDCG分母设training均值上限C=6.81656。保存分数鉴别显示rank1梯度约×2.306、未命中直接梯度不变；32 training用户0更新参数梯度探针通过，仍有轻微共享参数恢复冲突，效果未知。生产代码未改、新完整预算未授权，旧阶段2/2及不晋级决定保持。见[案例、目标诊断与最小后续验证](../docs/grid-experiments/2026-10-02-copmrec-dual-preservation-bad-case-analysis.md)。

## 2026-10-02：双分支audit有效但未晋级，固定实例结束

pvhjwp9q固定11182用户、best/源码262文件、原候选历史、主输出及独立指标/区间核验通过。相对content新增71/损失83、净少12，ΔNDCG10=−0.00039819187、CI[−0.00145550534,+0.00068799365]；相对单content保护净多4命中，但共同命中位置贡献−0.00015098753抵消交换收益，ΔNDCG10=−0.00004758390且CI跨零。generation独有组命中56→59，提供局部关系保护支持；整体收益未实现。

固定双teacher weight1阶段2/2完成、剩余0，不晋级，保留content终排，不自动加权重/epoch/模块或budget。机制主张收缩为局部改善互补命中传递，不能宣称稳定推荐提升或整个生成分支无用；下一方案需新增正面依据，当前无新增实验。见[核验、收益分解与路线判断](../docs/grid-experiments/2026-10-02-copmrec-dual-preservation-audit-result.md)。下文保留历史记录。

## 2026-10-02：双分支训练klqyxo8m有效，交付固定best audit

20epoch/2160更新完成，源码262文件、best Artifact与实际checkpoint、80个decoder更新、冻结及缓存历史契约核验通过。best epoch7/step864 v0，selection NDCG10=0.04728404060、Recall10=0.09167337418；相对单content保护分别+0.00015286729/+0.00017887354（约多2命中），是小幅描述性正向信号，未做配对区间，不认定收益。后续epoch未超过best，不自动延长训练。

本阶段已1/2、剩余1个固定best audit，显式content_generation/weight1，继续主比较content和机制匹配nwxzx1qs/zy940m5u，不按audit调参。见[核验与audit命令](../docs/grid-experiments/2026-10-02-copmrec-dual-preservation-training-result.md)。

## 2026-10-02：双分支正确关系保护已实现，等待手动训练

用户采纳后实现content/generation正确pair最大参照并集，保留竞争集合与weight1，仅改变teacher来源。28项测试、OpenSpec、node1 32 training用户梯度探针和5文件hash通过；8/8 generation独有命中产生保护梯度，均未命中8/8直接保护梯度为零，共享参数存在轻微冲突，未证明推荐收益。新固定阶段2个run（训练+best audit），0开始/剩余2，旧阶段结束及不晋级决定保留，不自动搜索。沿用原step48000初始化及20epoch/2160更新，匹配7q2njodh/nwxzx1qs。见[实现与命令](../docs/grid-experiments/2026-10-02-copmrec-dual-preservation-delivery.md)。下文为历史记录。

## 2026-10-02：保护失败样本分析完成，双分支正确关系保护待鉴别

复用nwxzx1qs/zy940m5u形成201条变化案例及保护loss/分数梯度诊断。83个持续损失中65个保护loss下降、54个第10名分差改善，6个修复全部只到第10名；当前多数改善不足以形成命中。45个共同命中退化的50条越过目标关系中41条未被content teacher支持，26条却由原generation支持；丢失的3个旧恢复均原generation Top10。training另有434个content未命中而generation命中的目标，目前不获额外保护。

下一优先候选是在同一CoPMRec目标内，用training标签支持的content/generation正确pair并集作为保护参照，重复pair不隐性加倍，不按evaluation真值设推理规则。该候选只能针对部分盲区，不能声称解决多数83个持续损失（仅9个原generation Top10）。尚未实施或做新loss参数梯度探针，未授权新完整预算；旧阶段2/2、剩余0及不晋级决定保持。见[详细案例、定位与最小鉴别](../docs/grid-experiments/2026-10-02-copmrec-preservation-bad-case-analysis.md)。

## 2026-10-02：内容保护audit有效但未晋级，固定实例结束

nwxzx1qs完成固定11182用户audit，源码/真实best/数据历史/主输出及独立完整SID指标和配对区间核验通过。相对content恢复69、损失85，ΔNDCG10=−0.000350608，95%CI[−0.001419518,+0.000724668]，未通过既定门槛。

相对competitive恢复旧损失6个、新增损失2个，旧恢复保留67/70并新增2个；净多3命中，共同命中位置贡献+0.000141940，ΔNDCG10=+0.000219492但CI[−0.000028924,+0.000481540]仍跨零。保护/恢复权衡有小幅描述性改善，尚不能称可靠收益或稳定机制验证。阶段2/2、剩余0，结束固定weight1保护实例，保留正式content终排，不自动扩权重/epoch/模块/预算。详见[完整audit结果与判断](../docs/grid-experiments/2026-10-02-copmrec-content-preservation-audit-result.md)。下文保留交付历史。

## 2026-10-02：保护训练7q2njodh有效，交付固定best audit

20epoch/2160更新完成，保护weight1与competitive/SID目标、best文件、源码262文件、80个decoder更新及冻结/数据历史契约核验通过。best为epoch9/step1080 v0，selection NDCG10=0.04713117331、Recall10=0.09149450064。相对competitive best NDCG微升0.00004933774、Recall少约5命中；相对content点正，但没有完整audit/配对CI，不认定保护收益。最后轮Recall更高但NDCG较低，仍严格用预定NDCG best。

阶段已1/2、剩余1个用户手动audit；命令明确v0及保护weight1，另复用zy940m5u同用户trace进行匹配机制比较，不更换checkpoint或追加搜索。见[训练核验与audit命令](../docs/grid-experiments/2026-10-02-copmrec-content-preservation-training-result.md)。

## 2026-10-02：内容正确关系保护已交付，等待用户手动训练

用户采纳后实现固定rank-discount单侧间隔保护，默认关闭，新实例weight1叠加competitive NDCG与SID CE1；只训练既有decoder，原初始化/候选/2160更新不变。26项测试及OpenSpec通过，node1 32 training用户梯度检查、冻结契约与命令解析通过，5个运行文件hash一致、3个同步会话正常。14/16 content命中用户产生保护梯度，16个未命中的直接分数梯度不变；共享参数上有轻微恢复梯度冲突，不能称保护与恢复完全解耦。

新固定阶段2个完整run（训练+audit），已开始0、剩余2，完整运行用户手动开始；此前各阶段结束决定保留。匹配对照29o9dx80/zy940m5u，主比较仍content与既定收益门槛，不按audit调权重或轮次。详见[实现、探针与启动交付](../docs/grid-experiments/2026-10-02-copmrec-content-preservation-delivery.md)。

## 2026-10-02：competitive失败样本分析完成，提出有条件的内容排序保护

复用zy940m5u/ri2813bi同11182用户产物及原始历史，形成456条分组案例。旧损失修复11、持续75、新增损失14；旧恢复丢失7、新恢复14。共同命中排名改善169、恶化166，但对应NDCG贡献+0.001258395/−0.001489058，抵消净增4命中的收益。51/75持续损失及103/166共同退化的目标分数相对旧臂上涨，仍被竞争商品涨幅超过；历史相似度和热门度描述不支持单阈值安全融合。

下一优先候选是在现有NDCG+SID CE上增加冻结content正确竞争关系的单侧保护：仅training目标原content命中时提供排序参照，未命中保留恢复监督，推理不依赖真值。当前SID preservation只是监督CE，未约束content排序；这提供具体正面试验依据，但teacher机制尚无效果结论，也可能与现有标签监督重合或抑制恢复。先限于training小批量梯度/契约鉴别，固定一个实例后再判断是否投入；拟议最小2个完整run目前未授权。旧阶段及剩余0不变，未修改代码或启动运行。见[完整分析与停止条件](../docs/grid-experiments/2026-10-02-copmrec-competitive-bad-case-analysis.md)。

## 2026-10-02：competitive audit有效但未晋级，阶段结束

zy940m5u的11182用户、原候选/标签/历史、best文件/目标契约、源码262文件、主输出键对齐、完整SID指标及配对bootstrap均核验通过。相对content新增70/损失89/净少19，ΔNDCG10=−0.000570100、CI[−0.001668975,+0.000544686]。相对旧NDCG净增4命中，但共同命中位置贡献−0.000230664，ΔNDCG10=−0.000132289且CI跨零。

旧丢失案例修复11个，同时新增14个原content命中的损失，总损失86→89；未兑现减少错误替换的预测。有效pair阶段2/2、剩余0，结束固定Top20 pair归一化实例，保留正式内容终排，不自动扩大k/权重/epoch/网络/预算。初始分母稀释现象不是已验证性能根因，不据此否定候选机制或全部NDCG学习。完整核验见[competitive audit结果](../docs/grid-experiments/2026-10-02-copmrec-competitive-audit-result.md)。下文为此前交付记录。

## 2026-10-02：competitive训练有效，交付精确best的audit

29o9dx80完成20epoch/2160更新，实际competitive/k20及其他配置匹配旧ndcg；source262文件、真实best文件/来源、80个decoder更新与冻结hash、缓存/原历史签名均核验通过。best为epoch13/step1512 v1，selection NDCG10=0.047081836、Recall10=0.091941684，高于content和旧ndcg点值；尚无audit或独立全部selection重评分，不称收益成立。

有效pair阶段已1/2、剩余1个统一audit。命令明确competitive目标契约及新v1/旧NLL v0；之后复用ri2813bi按用户键比较旧ndcg，保留原最终收益门槛。报告和命令见[competitive训练核验](../docs/grid-experiments/2026-10-02-copmrec-competitive-training-result.md)。

## 2026-10-02：有效竞争pair实现，待用户手动训练

用户采纳bad case分析的调整并授权实施。已完成competitive pair：原content/current mixed各Top20并集，同mask归一化；默认all保持旧目标，完整候选/推理/冻结边界不变。21项聚焦测试、Ruff、配置/脚本与OpenSpec strict通过；node1 8个training用户的有限检查梯度有限、冻结哈希不变、0 optimizer step/0新run。

单独冻结新有效pair阶段2run（训练1+audit1），当前0/2；旧阶段4/4和小head3/3仍结束。原step48000初始化、20epoch/2160更新及其他超参数匹配旧ndcg，旧ndcg 3r954xoo是主要训练对照，复用ri2813bi固定trace按用户键配对。完整实验用户手动开始；目标效果未知，不自动扩预算。交付与命令见[有效竞争pair实现](../docs/grid-experiments/2026-10-02-copmrec-competitive-pair-delivery.md)。

## 2026-10-02：具体bad case与loss归一化分析

按用户连接已有trace、原历史/商品文本与embedding，分析全部86个损失和63个恢复。72个损失在原mixed已存在，46个恢复原mixed已命中；损失样本中的181个Top10挤入商品全部warm，60个损失目标训练后混合分数反而提高。具体案例同时包含同类发夹混淆、护肤目标被口红/沐浴露挤出，以及美甲工具恢复，不能简化为cold或热门替代。

训练初始分数诊断发现，原content已命中样本的cold负例占pairwise权重分母中位数70.26%，负例分数梯度贡献中位数0.776%；未命中组对应中位数均0。提示已有命中竞争监督的相对稀释，尚非性能根因证明。优先待检验思路是保留NDCG目标及完整推理候选，调整有效竞争pair集合及分子/分母同mask归一化；不新增gate、删cold候选或扫alpha/轮数。用户只授权本次分析；未实现、未训练、阶段预算4/4和剩余0不变。详细证据、四个确定性选例及最小鉴别条件见[bad case分析](../docs/grid-experiments/2026-10-02-copmrec-decoder-bad-case-analysis.md)。

## 2026-10-02：decoder双checkpoint audit结束，补强未晋级

`ri2813bi`通过两份best的实际来源/哈希、固定11182用户/原标签与候选、五路trace、按用户键对齐的主输出、源码262文件、完整SID独立指标与配对bootstrap2000复算核验。排序臂−content ΔNDCG10=−0.000437812，95%CI[−0.001523426,+0.000659815]；ΔRecall10=−0.002056877，新增63/损失86/净少23。未达到推荐收益门槛，正式CoPMRec保留内容终排。

排序臂相对匹配NLL ΔNDCG10=+0.000224866，未校正点对点95%CI[+0.000010062,+0.000436208]；Recall10不变，新增2/损失2，收益全部来自共同命中目标位置改善（50改善/32恶化）。这为本固定开发实例的目标相关排名改善提供有边界证据，没有解决最终误替换损益；相对原mixed ΔNDCG10仅+0.000001324且区间跨零。

阶段实际尝试4/4（1中断+两次完整训练+统一audit），完整有效3，剩余0。结束本固定decoder训练补强，不自动扩loss、alpha、epoch、参数或seed；旧校准/小head阶段3/3保持关闭。保留候选相关性传递机制与原内容终排，不把此次结果解释为所有生成概率无效，也不新增dense优越性主张。详细结果和归因边界见[decoder audit核验与决定](../docs/grid-experiments/2026-10-02-copmrec-decoder-audit-result.md)。下文为此前交付快照。

## 2026-10-02：排序臂完成，交付统一双checkpoint audit

`3r954xoo`排序训练通过配置/来源262文件/checkpoint与best选择/原初始化/缓存与历史/冻结参数核验，20epoch/2160更新，约1063秒。与NLL `hkokjad0`逐字段匹配初始化、训练参数、数据/候选和源码指纹，只有目标臂与输出元信息不同。两臂best均epoch1/step216。排序selection日志NDCG10/Recall10=0.046988860/0.091136754，较content点增0.000427416/0.000447191，较NLL点增0.000252601/0.000268318（约净增5/3命中）；不称显著收益，未做完整best selection独立重评分或全audit。

双checkpoint共享8个固定audit用户的入口smoke通过，五路trace和主输出一致，未创建完整run。现在仅交付原计划一次11182用户统一audit，报告两臂相对content及ndcg相对nll/mixed的配对区间和命中损益；不追加轮次或选择其它epoch。实际尝试3/4（包含中断xdcm511f），完整有效2，剩余1个audit，旧条件阶段3/3仍关闭。见[排序训练核验与audit命令](../docs/grid-experiments/2026-10-02-copmrec-decoder-ndcg-training-result.md)。

## 2026-10-02：NLL重训完成，交付匹配排序臂

用户明确选择不续训、从原step48000重启，`hkokjad0`已完成20epoch/2160更新（最后日志step2159），W&B运行约996秒。配置、实际初始化、used Artifact、源码快照262文件、checkpoint Artifact完整性、缓存/原历史完整SID、冻结内容/alpha与80个decoder张量实际更新核验通过。best为epoch1/step216的v0 checkpoint；不能用最后summary替代best。training loss下降而selection NDCG未超过第2轮，尚未建立继续NLL微调的可靠推荐收益。

best selection日志NDCG10=0.046736259、Recall10=0.090868436，相对同池content缓存复算点增0.000174815/0.000178873（约净增2命中）；不是audit收益或显著性证明。此次未执行decoder best全selection独立重评分或audit，原缓存参照与来源/冻结契约已独立核验。下一步保持同初始化/内容/候选/20epoch预算，手动运行ndcg排序臂，不从NLL best继续训练，随后一次统一双checkpoint audit。

原xdcm511f中断尝试保留；用户授权重启增加1个修复尝试，阶段实际物理上限从原3调整为4（1中断+NLL重训+排序训练+统一audit），已启动2、完整有效1、剩余2。旧条件阶段3/3仍关闭，没有新超参数搜索。完整证据和排序臂命令见[NLL训练核验](../docs/grid-experiments/2026-10-02-copmrec-decoder-nll-training-result.md)。

## 2026-10-02：decoder效率修复，保留中断来源

用户报告训练慢与GPU低利用率。node1本地日志确认NLL `xdcm511f`被KeyboardInterrupt中断，原checkpoint epoch1/step216与optimizer state保留，不视为完整有效训练。实现NLL仅评分原池覆盖目标、chunk4→64、冻结目录投影内存复用、显式CPU线程4；batch8/累积4/FP32/损失/学习率/总20epoch不变，不改baseline。

24项聚焦测试及Ruff/OpenSpec strict通过；真实GPU0有限小批量前向+反向约0.449→0.064秒（NLL）及0.448→0.063秒（排序），约7倍，不代表完整Trainer全程倍数或GPU利用率保证。未启动完整续训、排序训练或audit。原3run预算已有1个中断尝试；若加入一个NLL修复续训，实际物理尝试将为4，须如实记录，不能隐去或重置旧阶段3/3。详细证据、精度边界和续训说明见[效率修复记录](../docs/grid-experiments/2026-10-02-copmrec-decoder-efficiency.md)。下文保留前日交付快照。

## 2026-10-01：按用户要求调整混合排序训练方向

用户要求继续探索混合排序增益，已收敛为直接训练生成混合路径分数的decoder，冻结内容、alpha与首阶段候选池，以Top10指标相关候选竞争监督替代仅做57参数分数修正。匹配的原似然微调作为CoPMRec内部对照，不修改baseline。正面依据是原mixed/generation存在互补命中但损失更大；新方向检验是否能学习收益更好的替换，不预设原因已确证或效果必正。

用户已采纳并授权实现：直接decoder混合排序、匹配NLL对照、共享候选双checkpoint audit均已实现，57项聚焦测试与OpenSpec strict通过。冻结每臂20epoch、batch8、累积4、AdamW lr1e-5、SID保持权重1，预计2160次更新；新增预算固定2训练+1统一audit共3run，当前0/3，完整实验仍由用户手动开始。node1已恢复连接，Mutagen flush成功，三个会话Watching无冲突；真实历史/目标对齐及training batch8初始评分、两臂有限decoder梯度核验通过。可训练参数3,934,784，冻结内容按字节哈希不变；原batch32缓存与当前batch8存在medium精度微差，分数核验容差明确为atol1e-3/rtol1e-4。

旧条件阶段3/3结题及负结果保留。固定候选上的新decoder评分收益不能直接代表重新搜索的端到端收益。见[实现、冻结协议与命令](../docs/grid-experiments/2026-10-01-copmrec-decoder-ranking-delivery.md)和[训练方向调整与鉴别方案](../docs/grid-experiments/2026-10-01-copmrec-ranking-training-direction-revision.md)。

## 2026-10-01：固定校准失败，按条件授权交付排序学习

本阶段已结题：audit `7n6n6jta`完成并通过best checkpoint、来源契约、哈希固定用户集合、原候选/三路缓存逐项相等、四路trace/主输出对齐、Artifact及source完整性和指标独立复算。learned−content ΔNDCG10=−0.000120249，95%CI[−0.000292195,+0.000029002]；ΔRecall10=−0.000536577，新增1/损失7，未晋级。唯一新增目标从第13进入第10；7个原第10目标跌至11–13，误替换大于恢复收益。Top5正点但区间跨零，不替换门槛。完整预算3/3，剩余0；结束固定beta与小型pairwise排序补强，保留正式CoPMRec内容终排、候选机制证据和baseline，不自动扩模型/损失/参数/实验。见[最终audit核验与路线判断](../docs/grid-experiments/2026-10-01-copmrec-pairwise-audit-result.md)。以下为此前阶段快照。

排序器训练`sw8llh9g`已完成并通过配置、缓存契约、源码快照、checkpoint完整性、best选择和selection复算核验。10epoch/140step，best epoch1/step28；selection ΔNDCG10仅+0.000008099，Recall净增3个命中，后期validation下降，尚无可靠推荐收益结论。按冻结协议交付精确`v0` best checkpoint的audit11182用户命令，不改超参数或baseline。累计2/3，audit未运行；见[训练核验与audit命令](../docs/grid-experiments/2026-10-01-copmrec-pairwise-training-result.md)。下段为缓存交付快照。

后续缓存run `i05boy01` 已完成并通过实际来源、源码快照/Artifact完整性、22363个原始training最后目标的完整SID标签逐项对齐、三路候选/指标复算与训练契约核验。已覆盖3454/22363（15.44515%），作为固定pairwise实例的训练样本；evaluation selection11181、audit11182。累计完整运行1/3，训练和audit复验未运行。训练侧mixed点估计小幅正向但区间跨零，且为主模型训练数据，不当作泛化收益。已交付实际manifest路径的CPU 10epoch训练命令；见[训练缓存核验与下一命令](../docs/grid-experiments/2026-10-01-copmrec-training-ranking-cache-result.md)。下段为此前交付快照。

用户要求先做混合修正强度校准，若效果不行再尝试排序学习。复用hpfflo20完成一次固定beta网格分析：selection选beta0.5，audit ΔNDCG10=−0.00023345，95%CI[−0.00091192,+0.00043692]；ΔRecall10=−0.00125201，新增25/损失39，未通过门槛，不扩大网格。见[校准结果](../docs/grid-experiments/2026-10-01-copmrec-beta-calibration-result.md)。

已实现并同步一个CoPMRec内部5→8→1零残差起点pairwise修正器，冻结原checkpoint与mass20+cold，training最后目标拟合，evaluation selection选best、audit复验。46项聚焦测试、Ruff、OpenSpec strict通过；三个Mutagen会话Watching无冲突，19个运行文件哈希一致，node1三个入口配置核验通过。真实收益未知、完整运行0/3；现在仅交付第一个训练缓存手动命令，不自动运行。开发evaluation已有探索消费，不能称独立确认；baseline与正式方法未替换，旧实例预算与负结果保留。见[冻结实现与三阶段交付](../docs/grid-experiments/2026-10-01-copmrec-pairwise-ranking-delivery.md)。

## 2026-10-01：混合概率延续到最终排序，作为 CoPMRec 内部探索

开发验证已结题：`hpfflo20`（Beauty/42 evaluation，22363用户）完成并通过实际checkpoint/input lineage、Artifact文件完整性、source快照、三路排序与主输出对齐和summary复算。mixed−content NDCG10=−0.00023781，95%CI[−0.00094671,+0.00041688]；Recall10=−0.00089433，区间跨零。新增116/损失136/净损20；Top5点估计改善但区间同样跨零，不能替换主门槛。依预定规则，结束直接按既有混合路径概率终排的具体实例，保留原CoPMRec内容终排；不否定完整方法，也不宣称统计显著劣势。0训练1完整evaluation推理，剩余预算0，不自动增加权重搜索、网络或确认矩阵。见[结果与判断](../docs/grid-experiments/2026-10-01-copmrec-mixture-ranking-result.md)。以下保留探索与交付历史状态。

后续实现授权已完成：新增 CoPMRec 专用推理子类、默认 evaluation 配置与根目录入口，一次 mass 搜索共享候选、统一 teacher forcing 生成三种排序，主输出 mixed。43项聚焦检查、Ruff、OpenSpec strict通过；Mutagen flush成功，三个session Watching且无冲突，7个运行文件本地/node1哈希一致，远端统一入口配置核验通过。Beauty/42 的 step48000 best checkpoint及evaluation目录已核验；完整运行尚未开始，手动命令及指标说明见[推理交付](../docs/grid-experiments/2026-10-01-copmrec-mixture-ranking-inference.md)。下段为先前探索完成时状态，实际运行仍为0。

用户明确要求：排序改进用于把混合概率优势转成最终推荐收益，属于 CoPMRec 当前问题，不新开研究问题、不改 baseline。本轮已核查训练/搜索使用混合概率而 hybrid 终排只用内容分数的实现衔接点。优先候选是在原 mass20 与全部 cold 并集上，使用同一 checkpoint 的完整 SID 混合路径对数概率终排；不加独立网络、不调 alpha、不扩大候选。

既有结果证明恢复与竞争损益，尚未证明混合分数能够更准确地区分候选。现有 trace 没有全部候选的路径分数，不能伪称已离线验证排序效果。最小验证建议为1个 Beauty/seed42 evaluation run，0训练，单次共享候选搜索与统一 teacher forcing 打分，对照原内容/混合/纯合法生成次序；完整运行由用户手动启动，本轮实际新增运行0、未实现代码、未替换正式方法。具体公式、cold公平处理、晋级/停止规则及测试集使用边界见[混合概率排序探索](../docs/grid-experiments/2026-10-01-copmrec-mixture-ranking-exploration.md)。

## 2026-10-01：复用正式矩阵收敛问题域，定位建议待作者审阅

完成24条正式推理run的candidate trace及9份path trace只读复算，覆盖9个主比较单元与三个seed42的Legal/Mass/Max机制单元。CoPMRec相对LIGER hybrid的九个单元NDCG10/Recall10配对区间均为正；相对原LIGER dense的三数据集汇总区间均跨零。同checkpoint的Mass相对Legal在三个数据集均有独立净收益，主要恢复内容模型已能识别的已见商品，同时减少dense Top10之外的候选内晋升命中；Mass相对Max仅Toys/seed42有逐项正区间，存在根层保留与后续路径存活的取舍。

建议问题域为“内容增强SID生成候选中的商品可识别性与候选可达性缺口”，贡献围绕合法子树内容条件概率、前缀候选准入与完整命中损益解释；不将事后目标dense rank分组作为部署选择规则，不主张稳定超过dense、cold-start、长尾、多兴趣或效率优势。具体见[问题域与贡献研究记录](../docs/grid-experiments/2026-10-01-copmrec-problem-scope-and-contribution.md)。这是结果驱动的定位建议，尚待作者审阅；保留原协议、全部正负结果与五方法正式矩阵，不改写论文正文，不重置门槛，本轮新增训练/预测/实验预算均为0。

## 2026-09-29：五方法与核心机制范围已确认并同步 Linear

当前排期以[精简实验总表](../docs/2026-09-29-copmrec-experiment-overview.md)及 [Linear 项目](https://linear.app/baymax104/project/copmrec-论文-4e4c9e46577f)为准。五方法仅 CoPMRec、LIGER、TIGER、LETTER、SASRec，范围为 Beauty/Sports/Toys × 42/200/2026；核心机制保留各数据集 seed42 的合法生成对照、mass/max 内部变体及路径/覆盖/命中损益分析。

取消新增 alpha=1、learned-vs-fixed alpha、alpha 轨迹专项、独立效率/成本实验；COBRA、ETEGRec、EAGER、SpecGR 不进入本轮。已有结果及负结果保留；dense 仅复用已有输出作补充。可学习 alpha 是设计选择，不作为独立自适应收益主张；mass 胜过 max 不是论文成立前提。

本次授权为实验排期及项目组织，完整实验仍由用户手动开始，既有匹配协议与停止规则保持。LETTER/SASRec 新槽位等待各自适配与协议任务完成。下文 2026-09-25 矩阵数量及执行状态是历史快照，不能用作当前剩余运行数；本轮未重新核验 W&B。

## 2026-09-25：论文实验矩阵 v1 已整理

进入冻结方法后的补证据规划，见[论文实验矩阵](../docs/2026-09-25-copmrec-experiment-matrix.md)。seed修订为Beauty/Sports/Toys × 42/200/2026，不再新增43/44。W&B已确认Beauty/200 CoPMRec训练`83djht0e`完成、best step45500但尚无testing；Beauty/2026 CoPMRec训练`i2kcjzh5`也已定位，best step46500且无testing，但它是单GPU历史协议，不满足当前双卡矩阵协议，不能直接作为当前有效槽位。其他seed200/2026结果仍待来源审计。TIGER/COBRA条件槽位仍为18训练/18预测，成本测量另列6条件。

本次只修订论文矩阵并只读审计W&B，不新增完整运行授权；完整实验仍由用户手动开始。Sports/Toys产物与测试消费史未核验。原LIGER默认warmup10000须显式覆盖为匹配50k协议的2500。既有方法、关闭路线及负结果保持，预算不因CI跨0或结果不利自动增加。

## 2026-09-25：论文机制解释已补齐并验证

正文已补齐四项机制缺口：叶节点内容相关性与前缀beam决策的粒度错位、推荐场景采用子树总质量的理由、1147个可分析用户的首次路径分歧诊断，以及PAG best-descendant与CoPMRec set-level probability allocation的语义边界。摘要、方法、实验、限制和结论已形成“候选遗漏—结构原因—mass设计—路径存活—候选覆盖—最终效果边界”的完整链条。未新增训练或推理实验；沿用既有`5wfpsg9a`/`sjf8qcgs`证据。LaTeX编译通过，8页PDF逐页检查无版面缺陷。继续保留mass独立最终NDCG/Recall增益、生成分支必要性及跨数据集稳定性未确认的边界。见[审查与补齐记录](../docs/grid-experiments/2026-09-25-copmrec-paper-mechanism-completeness-review.md)。

## 2026-09-25：主方法冻结为仅mass前缀聚合

用户决定CoPMRec正式方法只采用子树总质量`mass`，不把`max`或`mass+max`融合纳入主方法。这里保留的是合法生成分布与mass内容条件分布的逐层概率混合；`max_mixture`仅作为解释mass集合语义的机制消融。深度条件聚合、候选并集、mass来源保护和单max-only槽位均已结题，不再成为主方法组件或调参方向。

最终独特价值定位为：把用户条件商品相关性聚合成合法SID子树的总概率质量，并在不可逆beam剪枝前转为下一token条件分布，从而保留分散在多个后代商品上的内容支持。逐层诊断支持mass margin预测mass/max路径存活，但mass相对max的独立最终NDCG/Recall优势未确认；完整方法相对匹配LIGER的Beauty/seed42整体收益仍保留。新增实验预算0。见[方法梳理](../docs/grid-experiments/2026-09-25-copmrec-mass-only-method-synthesis.md)。

## 2026-09-25：单max-only槽位验证结题

复用既有source-protection evaluation缓存完成固定`q=1`离线验证，0训练、0 prediction、0 decoder search、0 testing。规则以mass20与cold为主体，只允许内容分数最高的一个max-only增量候选参与Top10；未扫描其他quota。22363个用户沿用11182/11181的selection/audit划分，缓存、来源分类和内容分数均通过既有validator。

selection上，q=1相对mass20净增12个命中、NDCG10点差+0.000295398，相对未约束union净增3个命中、NDCG10点差+0.000084443；但相对mass30少1个命中，NDCG10点差-0.000006539。预声明双控制门禁失败，决定为`single_max_slot_selection_gate_failed_stop`，audit保持未计算。

硬性限制max-only竞争比统一mass bonus更接近保留互补收益，但仍没有超过简单mass30控制，不能支持mass/max互补的独立推荐收益。当前实例的候选转化路线结题，不扫描q=2/3、阈值、权重或用户分组。后续若继续，应明确切换到多未来目标场景或计算效率问题，并建立新的独立协议。见[结果报告](../docs/grid-experiments/2026-09-25-liger-single-max-slot-result.md)。


## 2026-09-25：mass来源保护evaluation诊断结题

完成态run `zjsf4eaw`通过split、checkpoint/SID/embedding lineage、schema、候选池、三搜索预算及本地缓存重放核验。22363个evaluation用户按冻结规则分为11182个selection与11181个audit用户；缓存重算与保存JSON完全一致，W&B摘要亦一致。

selection上所有非零beta均不合格，决定为`source_protection_no_stable_selection_plateau_stop`，因此没有选择beta，也未查看audit半区的beta级比较。最弱保护`beta=0.1`相对beta0 union仅有NDCG10 `+0.000005185`点差，同时少2个Recall命中；相对mass30少6个命中且NDCG仍低。其余beta相对两个控制的Recall与NDCG点估计均为负。当前二值mass来源bonus不能稳定把union的候选准入价值转化为推荐收益；不据此否定mass总概率的路径分歧诊断价值。

冻结预算1/1完成，不进入audit，不追加更细beta、负beta、quota、分组、学习排序器或testing复验。该转化路线结题；若后续继续论文主效果，应回到独立seed/数据集确认已有整体方法，而不是继续在当前实例调来源保护。见[结果报告](../docs/grid-experiments/2026-09-25-liger-source-protection-result.md)。

## 2026-09-25：mass来源保护evaluation诊断已获授权

用户授权沿候选并集路线继续，但范围冻结为一次evaluation-only来源保护诊断。依据是union的96个新增Top10命中全部来自max-only目标，而81个损失中80个是mass生成目标被纯内容终排挤出；可检验预测是给mass20来源候选增加单一保护bonus后，union的准入收益能保留且不再牺牲mass候选。

实现固定共享一次encoder/内容projection并执行mass20、max20、mass30三次搜索；只缓存本地候选池、内容分数与来源标记。按用户key与seed42奇偶固定切分selection/audit，beta网格固定为`[0,0.1,0.25,0.5,1.0]`。selection须出现相邻两个beta同时优于beta0 union和mass30，才冻结较小beta；audit只检验该beta，要求相对两个控制的NDCG10配对95%CI下界均大于0且Recall10点不降。预算为0训练、1次evaluation cache prediction、每用户3次decoder search、0次新testing；完整运行由用户手动启动。即使通过也仅是需要独立seed或数据集确认的开发候选。105项LIGER回归、18项聚焦测试、Ruff与OpenSpec strict通过；Mutagen flush后四会话均为`Watching for changes`且无冲突，运行文件本地/node1 SHA-256一致。见[冻结协议](../docs/grid-experiments/2026-09-25-liger-source-protection-protocol.md)。

首次人工运行完成GPU预测后，在writer逐批校验阶段失败：metadata中的`cold_rows`被默认构造为CPU tensor，而候选trace仍在CUDA，导致`torch.cat`设备不一致。该尝试未形成合并缓存、beta曲线或选择结果，不消耗有效科学预算。修复为让metadata派生tensor继承`union_rows.device`，新增异构设备回归；修复文件本地/node1 SHA-256均为`c0b665e2c4337a6c4dfd25332aba2b7fd677224732d04bfe495274e985d352ee`，可按同一冻结命令重跑。

## 2026-09-25：候选并集转化验证结题

两个有效prediction `flk6lrqj`/`cjtljoe1`完成，22363用户的checkpoint/SID/embedding lineage、运行时alpha、标准trace、来源trace及推荐输出逐项核验通过。union候选精确等于冻结mass20与max20的逐用户并集，平均29.369个；目标覆盖从mass20的0.111702提高到0.137012。最终NDCG10/Recall10为0.035557564/0.070831284，相对mass20点估计+0.000311404/+0.000670751、净增15个命中，但NDCG10配对95%CI[−0.000116594,+0.000797610]跨0，第一道冻结门失败，决定为`candidate_union_not_better_than_mass20_stop`。

单次mass30覆盖0.141931，显著高于两次搜索的union；union相对mass30的NDCG/Recall点估计为正但区间均跨0。保留“分散内容支持预测mass/max路径分歧”的诊断价值，不宣称候选并集已转化为推荐收益或具有超过扩beam的独特互补性。0训练2有效prediction预算耗尽，不搜索beam、alpha、融合权重、quota、排序器或subgroup。见[结果报告](../docs/grid-experiments/2026-09-25-liger-candidate-union-result.md)。

完成后的只读损失分解显示，96个新增Top10命中全部来自max-only目标；81个损失中80个是mass生成目标被挤出，另有111个共同命中发生rank下降。NDCG均值由新准入贡献+0.001690、丢失贡献−0.001108、rank稀释−0.000270合成为+0.000311。新增候选数量与净收益不单调，不能据此在testing上选择阈值。若用户明确重开路线，唯一建议的有限后续是在evaluation上校准单参数mass来源保护，成本上限0训练、1个包含mass20/max20/mass30的候选缓存run、0次新testing；须在相邻多个保护强度上同时超过原union和mass30才进入独立确认。当前不自动授权。

## 2026-09-25：候选并集转化验证启动前记录

保留“mass能刻画分散内容支持并预测完整路径分歧”的独特价值，但不再尝试标签路由、深度拼接或学习排序器。本轮只验证候选集合层面的转化：`mass_max_union20`精确保留mass20与max20的逐用户候选并集，再沿用cold union和内容终排；`mass30`作为单源候选预算控制，冻结`5wfpsg9a` mass20为基线。正面依据是两源平均并集约29.37且标签oracle仍有195/200个额外Top10目标；反面依据是mass/max最终差值跨0、深度条件臂失败、无标签路由分组与learned reranker均无稳定NDCG支持。

预算固定0训练、2 prediction、约3次beam搜索。union必须先相对mass20达到NDCG10配对95%CI下界>0且Recall10点不降，再以同一门槛超过mass30；否则分别判为转化失败或仅候选预算效应。testing已被用于形成方案，通过也只能作为需独立确认的探索候选。不调beam、alpha、融合权重、quota、seed或subgroup。实现、命令与门禁见[冻结协议](../docs/grid-experiments/2026-09-25-liger-candidate-union-protocol.md)。本地测试、Ruff和OpenSpec已通过；Mutagen flush后四会话均为`Watching for changes`且无冲突，8个运行文件本地/node1 SHA-256一致。完整prediction仍由用户手动开始。

首次union启动在第二次max beam前因实现错误失败：Transformers原地扩展共享`encoder_outputs`，使第二次搜索的encoder batch与原attention mask不一致并触发CUDA assert。该尝试未形成有效候选产物或指标，不消耗科学预算。修复为每次beam使用独立浅拷贝容器，保持底层encoder张量共享；真实双beam回归及完整93项相关测试通过。修复后Mutagen flush成功、四会话均为`Watching for changes`且无冲突，`module.py`本地/node1 SHA-256一致，可以重跑同一冻结命令。

## 2026-09-25：深度条件聚合转化验证结题

唯一新臂`gwfdimng`完成并通过来源、Artifact、运行时alpha、用户/标签/dense参照及W&B summary核验。max-root/mass-deep覆盖率0.106828，低于mass的0.111702和max的0.107812；相对max覆盖差-0.000984，95%CI[-0.002907,0.000894]，第一道价值保留门槛失败。相对mass覆盖显著下降，NDCG10/Recall10差值区间也跨0，净损11个命中。旧trace的分深度关联不能因果拼接：根层改变会重写后续beam父前缀集合。保留“分散内容支持预测完整mass/max路径分歧”的独特价值，否定固定max-root/mass-deep转化方案；0训练1预测预算耗尽，不搜索其他深度、alpha或subgroup。见[结果报告](../docs/grid-experiments/2026-09-25-liger-depth-conditioned-result.md)。

## 2026-09-25：偏好分散诊断结题

两次人工prediction `5wfpsg9a`/`sjf8qcgs`完成并通过来源、Artifact、运行时alpha、静态内容字段及候选非干预性核验。首次内容rank分歧处，mass-only与max-only的margin gain组间差为+1.637591，95%CI[1.555237,1.721903]；实际分歧方向准确率86.486%，CI[84.479%,88.492%]，支持“分散内容支持预测哪种聚合保住目标路径”。但mass−max的Recall10与NDCG10区间均跨0，冻结决定为`path_condition_only_downstream_advantage_unconfirmed`。论文可把子树总质量定位为层级生成推荐中分散内容支持的前缀建模，不宣称已观测多兴趣、独立下游增益或击败PAG。0训练2预测预算耗尽，不授权alpha或subgroup阈值搜索。见[结果报告](../docs/grid-experiments/2026-09-25-liger-preference-dispersion-result.md)。

## 2026-09-24：2×2交叉解码结题

新run l6hkjuoh/hs3ex5dl完成并通过来源、trace、用户和同训练dense一致性核验。联合训练相对原训练在alpha0.5与alpha1下的NDCG/Recall区间均跨零；预注册difference-in-differences交互也跨零。联合checkpoint内alpha1显著优于固定0.5，而此前learned alpha0.819与alpha1近似。保留完整方法总体效果，收缩联合训练独立增益、训练×解码协同及生成概率必要性主张。0训练2预测预算耗尽，不自动扩展。见[结果报告](../docs/grid-experiments/2026-09-24-liger-training-decoding-factorial-result.md)。下节为启动前历史状态。

## 2026-09-24：两次训练的2×2交叉解码已获授权

复用原训练×0.5（84d7wkgd）与联合训练×1（oxnmyhqy），只新增原训练×1、联合训练×0.5两次人工预测，0训练。主估计量为逐用户配对difference-in-differences NDCG10，用于判断训练来源增量是否依赖混合解码；不定位单个loss的因果贡献，不授权alpha扫描。实现与命令见[冻结协议](../docs/grid-experiments/2026-09-24-liger-training-decoding-factorial.md)。

## 2026-09-24：机制控制已结题

两次预测nvrx2rkd/pc9mo9ld完成并通过配对及来源核验。固定联合checkpoint、合法支持后，内容引导相对合法生成有明确收益；总质量相比最大后代覆盖率显著增加，但NDCG/Recall区间跨零，新增200/损失195/净增5命中。收缩总质量下游优势主张，保留整体效果；不以覆盖改善代替最终推荐增益。0训练2预测预算耗尽，不自动扩展。见[结果报告](../docs/grid-experiments/2026-09-24-liger-mechanism-result.md)。下节为启动前历史状态。

## 2026-09-24：同得分、同支持机制控制已准备

用户授权完成机制控制。新增范围限于同一zl9gv56p checkpoint的legal_generation与max_mixture两次人工预测，0训练；复用c01w22tw与oxnmyhqy。实现、31项测试、OpenSpec及node1同步均完成，正式运行尚未开始。max_mixture仅替换后代聚合，非完整PAG复现。协议和可复制命令见[机制控制](../docs/grid-experiments/2026-09-24-liger-mechanism-control.md)。下文“未授权、预算0”描述此前审查时状态；本次只授权该有界控制，不授权2×2或跨seed扩展。

## 2026-09-24：自然变体测试

[八项完整审查](../docs/2026-09-24-copmrec-natural-variant-audit.md)判定：在研究思想抽象层面，可自然概括为PAG式相关性引导的推荐变体；不等于实现等价或路线否决。当前最扎实为单实例候选可达性与整体效果，待证贡献为总质量引导价值及融合监督交互。继续当前方法，不追求完全无先例；验证建议仅限机制控制、2×2组合与择一独立复验，未获新实验授权，预算保持0。RGD同batch但梯度隔离、V-STAR默认测试普通beam的比较细节已更正。

## 2026-09-24：扩展文献复评

[16项原文机制比较](../literature/2026-09-24-copmrec-content-generative-review.md)补充PAG、RGD、V-STAR等近邻。提前引入对象相关性指导prefix及联合训练已有先例；CoPMRec的增量应限定为全目录内容子树质量条件化、逐层算术混合与对应NLL联合训练。当前方法增量中等偏弱，单实例效果积极，论文证据仍不完整；尚未发现已读方法与完整方案相同，不据此宣称首创。保留主方法与LIGER基线，新增实验预算0。

## 2026-09-24：主方法命名与论文定位

按用户决定，整体混合训练正式命名为 **CoPMRec — Content-conditioned Prefix Mixture Recommendation（内容条件前缀概率混合推荐）**，以 LIGER (GRID adapted v1, 50k) 为主baseline。研究问题收敛为：固定SID、内容、骨干、训练步数与名义候选预算下，内容条件前缀概率的训练和解码能否减少候选遗漏并提升最终推荐。

主效果在Beauty/seed42成立；联合训练独立NDCG增量、生成概率推理必要性与跨设置泛化未确认。工作量不作为贡献证据，当前采用“候选可达性与整体效果”的论文主张。已建立[审查报告](../docs/2026-09-24-copmrec-paper-assessment.md)、[英文工作稿](../src/copmrec.tex)及[当前执行看板](../docs/dasfaa-2027-execution-board.md)。论文尚缺独立重复、跨数据集及完整比较协议；本次新增实验预算0，已完成阶段保持关闭。

## 整体训练三臂 testing 已完成

预算1/1训练、3/3预测完成，剩余0。c01w22tw/84d7wkgd/oxnmyhqy来源、22363用户trace及全部summary复算通过。joint NDCG10/Recall10=0.035246160/0.070160533；匹配50k原LIGER=0.028628485/0.054151947，fixed05=0.034603055/0.067879980，joint content-only=0.035246656/0.070071100。

主门槛通过：joint−original的NDCG10差值+0.006617675，95%CI[+0.005350625,+0.007927356]，Recall10区间亦为正，净增358命中。相对fixed05，Recall10净增51命中且区间为正，但NDCG10区间[−0.000294508,+0.001524043]跨零；相对自身alpha1，@5/@10区间均跨零、净增2命中。

保留整体训练版本作为当前方法，单Beauty/seed42上相对原LIGER的整体收益已得到支持；不宣称联合训练相对简单解码具有已确认NDCG增量，也不宣称推理生成分支不可替代。alpha1消融共享联合训练的表示，不是独立纯内容训练对照。testing已查看，仅作探索证据；本实例结题，不自动追加训练、调参或预算。[完整结果、区间与核验](../docs/grid-experiments/2026-09-24-liger-joint-testing-result.md)。

## 2026-09-24：整体训练完成时的历史快照

训练已由用户完成：`zl9gv56p`，50k步骤、100个dense验证点，best48500的NDCG10/Recall10为0.046371371/0.089388721，alpha=0.819378912；数据/模型共有参数/优化器/预算与50k基线匹配，checkpoint文件摘要和有限参数核验通过。开发集dense有正向点估计，不作为hybrid testing通过证据。预算1/1训练、0/3预测，剩余3预测。三条实际checkpoint命令已通过Bash→Hydra preflight，等待用户手动testing；[训练核验和测试命令](../docs/grid-experiments/2026-09-24-liger-joint-training-result.md)。

用户要求做一次混合概率整体训练，并以超过LIGER作为效果门槛。已按[OpenSpec设计](../../GRID/openspec/changes/train-liger-joint-probability-mixture/design.md)实现，62项不同聚焦测试、Ruff及strict通过；CPU/GPU合成单步训练及GPU未训练合成checkpoint恢复推理通过。交付时Mutagen flush后四会话Watching、无冲突，8运行文件SHA-256匹配node1；[实现与命令](../docs/grid-experiments/2026-09-24-liger-joint-mixture-implementation.md)。推荐主模型从零训练，固定既有SID和内容向量，采用原SID CE+内容CE+混合NLL，系数固定为1；全局alpha与主模型共同学习。技术动机是训练目标与混合解码对齐，不能用训练参数量作为贡献证据。

提议匹配已有50k基线3mntjejz，预算1次训练+3次prediction，完整运行仍由用户手动开始。主门槛为相对原始LIGER hybrid的NDCG10配对95% CI下界>0且Recall10点不降；辅助检查相对同基座固定0.5解码的增量与自身alpha1消融。超过基线是验收条件，不承诺结果；不通过则结题，不无上限调参。testing已查看，只作后续探索。先前常数testing及动态/排序阶段结论保持，旧预算不重开。

以下为已完成阶段的完整证据与边界。

> 2026-09-24独立基线更新：用户完成50k LIGER best49k的testing（yyzwh5xl），22363用户trace复算通过；原始hybrid NDCG10/Recall10为0.028628485/0.054151947，dense参考为0.035095976/0.069310915。testing现已查看，下文“testing未使用”是此前阶段快照。既有四臂协议在此之前已冻结，仍保留30k基座与原预算；本次不计入其完成数，不依据该测试结果调参。详见[50k testing核验](../docs/grid-experiments/2026-09-24-liger-50k-testing-result.md)。

## 四臂 testing 已完成：保留整体效果，收缩机制主张

预算4/4完成、剩余0，未新增训练。22363用户trace、来源与全部summary复算通过。constant NDCG10/Recall10=0.030677817/0.060635872；匹配best26k原LIGER=0.017961057/0.030139069；fixed0.5=0.029728443/0.057774002；content-only alpha1=0.030582482/0.060546438。

constant−original的ΔNDCG10=+0.012716760，95% CI [0.011217238,0.014154077]，净增682命中，主比较通过。constant−fixed0.5的ΔNDCG10=+0.000949374，CI [0.000356391,0.001549634]，Recall10区间也为正、净增64命中，权重校准增量得到支持；@5增量尚不明确。

constant−content_only仅净增2命中，ΔNDCG10区间[−0.000069201,+0.000273205]跨零，生成概率相对纯内容beam的额外价值未确认。保留已选可学习常数，不根据testing换臂或调参；论文可叙述内容概率参与候选生成及全局权重校准，不宣称两路互补不可替代。原始候选有效数不同，整体提升也不可全归于标量学习。

四run为m71bmgtd/kezd01ud/j2g9jbeo/0ywsfwvt，runtime共339秒。testing已使用；独立50k testing此前已查看，本四臂协议在该查看前冻结，不能称新鲜未查看数据。这里baseline为best26k，不能把结果冒充50k基座匹配比较。证据限Beauty/seed42；没有已发现实现疑点需要追加实验，阶段结题，不自动扩预算。sampling仍暂存，独立排序继续暂停。

[完整结果与证据边界](../docs/grid-experiments/2026-09-24-liger-constant-testing-result.md)

## 已完成的方法选择（历史快照）

单次扩预算4/4完成，同一问题累计9/9，剩余预算0；保留链路累计7训练/17prediction。动态未通过预先规定的NDCG10配对区间门槛，采用更简单的可学习常数作为方法形式；不宣称动态显著更差或两者等价，不再追加本问题预算。

最终constant（2qicx7jb）/dynamic（ci6bg4ka）的NDCG10为0.039044603/0.039019952，Recall10为0.076018423/0.076063140。dynamic−constant的ΔNDCG10为−0.000024650，95% CI [−0.000250785,+0.000183725]；动态净多1命中，但没有建立额外推荐收益。

两臂均按内部val/nll选择扩预算checkpoint，来源与22363用户trace复算通过。constant保留e27hkflu的epoch14/step1185，alpha=0.9107562899589539；Pmix=(1−alpha)Pg+alpha Pc。训练时学习全局标量，推理时固定，beam20补cold后沿用原内容排序，不注入dense20或新增独立排序网络。

相对各自3epoch结果，两臂NDCG10和Recall10点估计均下降、区间跨零；不将NLL改善写成推荐收益，不根据evaluation事后切换旧checkpoint。相对固定0.5，常数净增47命中，Recall10区间为正，NDCG10区间跨零，@5点估计下降。

保留概率混合相对原LIGER hybrid的已有正向主线。当前仅决定权重形式，证据限Beauty/seed42探索；testing未使用。sampling暂存，独立排序模块暂停，不自动启动新实验。

[完整评价与配对区间](../docs/grid-experiments/2026-09-24-liger-gate-extended-evaluation-result.md)

## 前一阶段结论（保留）

5/5个run完成，剩余预算0；累计5次训练、15次prediction。动态权重相对固定0.5改善Recall10，但未建立NDCG10优势，且没有超越constant对照。固定/constant/dynamic的NDCG10为.038466514/.039129543/.039087145，Recall10为.073916737/.076376157/.076197290。

constant和dynamic相对固定0.5分别净增55/51命中，Recall差值区间为正，NDCG差值区间跨零，@5点估计下降；dynamic相对constant净少4命中，差值区间跨零，不能称显著退化或等价。候选覆盖改善和内部NLL优势没有证明动态适配的额外推荐收益。

保留候选概率混合相对原LIGER hybrid的正向主线；本动态实现按冻结门槛结题，不自动改特征、增加epoch/网络或启动sampling。constant是有效的简单对照和局部正向结果，不宣称全局最优。独立排序模块继续暂停。testing未使用，当前为Beauty单seed探索。

两run来源与完整trace核验通过，全部指标独立复算；采用本阶段协议的NumPy PCG64 seed42用户配对bootstrap1000次。[完整评价报告](../docs/grid-experiments/2026-09-24-liger-dynamic-evaluation-result.md)。

## 本阶段实施记录（以下为此前阶段快照）

保留候选概率混合，优先尝试前缀级动态权重。sampling 仅保存想法；独立排序模块暂停。本版本只修改beam逐步扩展的混合权重，保留原LIGER的生成20 ∪ cold和内容终排，不注入dense20、不加学习排序网络。

冻结26k基础模型；动态线性sigmoid门控8参数、constant对照1参数，零初始化均为0.5。7维特征为层级、两路熵/间隔、JS与一致性；teacher forcing只用training最后目标，按用户留出10%。每臂3epoch、batch256、lr.01，选择内部NLL最小checkpoint。推理特征不含标签。

## 本轮有限预算与状态

上限5run：1个本地training缓存、2个门控训练、通过内部门槛后2个实际beam评价。当前3/5，两臂训练已完成并核验；累计5次训练、13次prediction。动态内部NLL不优于constant和固定0.5则停止；通过后实际NDCG10须优于两者且Recall点不降，否则结题，不自动调参。全库dense仅参考；原始LIGER为主要baseline；testing保留。

44项聚焦测试、Ruff、OpenSpec strict通过；Mutagen四会话Watching，无冲突，18个运行文件哈希匹配。training缓存j9hq7jcq通过：22363用户，fit20106/val2257，3.31MB，35.15秒，未上传缓存。constant/dynamic各完成3epoch/237步；最佳内部NLL为1.971417/1.967401，均低于固定0.5的2.007651。内部阶段门槛通过，下一步仅手动运行计划内两次evaluation beam评价；默认目录已固定evaluation，testing未使用。效果未知，不能由合成测试推断真实收益。

[完整协议、预算与命令](../docs/grid-experiments/2026-09-24-liger-dynamic-mixture-protocol.md)

## 前一阶段决定（排序研究，已被上文优先级取代）

保留概率混合路线，以原始 LIGER hybrid 为主要 baseline；全库 dense 是补充参考，不是严格指标上界或候选机制成立的必要门槛。研究排序模块：当前 joint 同池残差排序尚无正向增益，dense 对照池只有 Recall@10 正向信号。不含 dense20 补充的纯混合候选池上，排序增量尚未被现有比较隔离。新实验预算未冻结，不自动启动实验或加入动态权重。旧阶段结果与预算保持原样。

[排序研究边界](../docs/grid-experiments/2026-09-24-liger-ranking-scope.md)

## 已结束固定实例的结论

正式范围保持“利用商品内容的生成式推荐方法”，以 LIGER 等同类方法为主要 baseline。Beauty / seed42 的最终 evaluation 已完成并核验；joint 未通过同时超过原 dense 和匹配学习版 dense 的预设门槛，主方法优势尚未建立。该固定实例结题，不自动追加模块、调参或实验预算；固定 checkpoint 无训练组合探索继续保持关闭，CGBS 仅作参考。归档范围外实验不作为当前决策依据。

## 本阶段回答的问题

概率混合候选在冻结 LIGER 特征上经可学习残差排序后，是否比原 dense 及等候选规模的学习对照带来净收益？两臂使用相同 263→64→1 网络及 s=d+tanh(r)，相同共同覆盖训练样本、标准化和训练预算；控制臂同样包含生成评分特征，比较隔离候选组成的额外价值。

训练 3 epoch/69 步，checkpoint 仅由 training split 内部留出 NDCG@10 选择。dense 为 step46，joint 为 step69；最终 evaluation 的全部 22363 用户均保留，testing 未使用。

## 已完成结果

| 方法 | NDCG@10 | Recall@10 |
|---|---:|---:|
| 原 dense | 0.039586921 | 0.077270491 |
| 学习版 dense（jmx6ryws） | 0.039955070 | 0.078656710 |
| joint（er0sejlm） | 0.039507744 | 0.077494075 |

joint−原 dense 的 ΔNDCG@10 为 −0.000079177，配对 95% CI [−0.000445699, +0.000299787]；joint−学习版 dense 为 −0.000447326，CI [−0.000973408, +0.000108714]。均不支持优越性，也不能认定显著退化或等价。joint 相对原 dense 新增79/损失74/净增5个命中，相对学习版 dense 新增122/损失148/净减26个命中。

学习版 dense 相对原 dense 的 Recall@10 区间为正，但 NDCG@10 区间跨零，@5 指标点估计下降，不升级为成功主方法。全部 summary 和配对 bootstrap 已复算，用户/标签、原 dense Top10、来源和最佳 checkpoint 核验通过；没有需要追加实验才能排除的已发现实现问题。

## 预算与后续边界

本阶段 6/6 个 run 全部完成（两次缓存、两次训练、两次评价），剩余0；当前保留链路累计3次训练、12次 prediction 执行。停止本固定实例，不追加 alpha、残差界、网络、特征、学习率、epoch 或 seed 搜索。准确失败原因允许保持未知；新增研究投入须明确正面依据及会改变的决策，不能以尚未穷尽替代解释重开预算。

本结论不否定整个商品内容生成推荐范围。当前 evaluation 已用于探索，不能作为后续设计的独立确认集。后续缓存仅保存在执行主机本地，不上传 W&B；继续保留指标、checkpoint 和最终结果，保留来源身份和 manifest 哈希，不自动删除既有 artifacts。

[最终结果与有效性](../docs/grid-experiments/2026-09-24-liger-learned-reranker-result.md) · [冻结协议](../docs/grid-experiments/2026-09-23-liger-learned-reranker-protocol.md) · [训练结果](../docs/grid-experiments/2026-09-23-liger-learned-training-result.md) · [evaluation 缓存核验](../docs/grid-experiments/2026-09-24-liger-evaluation-cache-result.md)

[动态门控缓存核验](../docs/grid-experiments/2026-09-24-liger-dynamic-cache-result.md)

[门控训练结果与下一步](../docs/grid-experiments/2026-09-24-liger-dynamic-training-result.md)

[扩预算训练核验](../docs/grid-experiments/2026-09-24-liger-gate-extended-training-result.md)
