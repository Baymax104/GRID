# CoPMRec v4.4：仅训练content CE的有效历史支持续训

## 当前状态与五项路线门禁

本阶段已完成并结题。预承诺规则选择的 **control 固定pool** 在完整Testing中 R@10=0.08554308455931672、NDCG@10=0.04742966406441521，相对实际同history政策LIGER为 **+10.770121598147053% / +10.776636054536315%**；两个paired差值CI下界均大于0，旧baseline辅助绝对阈值也通过。完整raw/输出/来源审计确认本线程预登记的整体Testing目标，结果不是W&B summary单独推断。具体证据见[最终Testing审计](evidence/copmrec-v4-autonomous-20261005/eligible-winner-testing-audit.json)及[实际stage closure](evidence/copmrec-v4-autonomous-20261005/eligible-content-continuation-stage-closure.json)。

这一结论适用于固定source nj9elah1 best6000、continued hbgj80bh best2000、独立query后的0.5/0.5原dense logits及保留的history资格层。**训练history CE mask的额外收益未获支持**，最终control保持原CE支持集；小幅续训增量不单独声称达到物质组件效果。原v4.3历史排除层继续RETAIN，不要求每层都独立达到3%或10%。

实现49项模型测试、52项配置脚本测试（新101项）、4项相关旧风险检查和交叉审查通过；实际Mutagen flush三会话Watching、374文件SHA、CPU原生恢复/新优化器/权重与RNG、双卡和单卡统一入口smokes均通过。本阶段2训练/4000步、2完整pool Validation、2matched Testing已全部完成；旧5训练/30000步不重置，累计 **7训练/34000步、全线程Testing 3/3**。此结题不新增预算或实验。以下五门禁保留为正式运行前的预登记设计，末尾记录实际结果及边界。

### 1. 问题、可检验预测与相关工作定位

v4.3已确认“有效history商品不应占据该split推荐资格”的正向层：对自身旧pool R10+8.568824065633551% / N10+17.367341655669733%，188新命中/0丢失、1095共同命中上移/0下移、双CI正，已RETAIN。推理当前应用相同资格规则，但实际 `joint_mixture.py` 的训练content CE只对cold设-100，仍包含当前有效history竞争项。我们检验**仅这一项训练支持与已保留推理资格的局部对齐**；不声称原CE有害，也不把历史mask本身称为原创评分或整mixture对齐。

可反驳预测：在相同warm权重、训练成本和history-pool部署下，treated相对control改善合法目标R/N或位置，且对旧v4.3形成可保留物质增量。负向或CI不确定时收缩mask收益主张；若control本身明显改善，可独立保留续训贡献，不误归因于history CE。

相关工作仅定位成熟的支持集处理：[SASRec 作者固定代码](https://github.com/kang205/SASRec/blob/e3738967fddab206d6eeb4fda433e7a7034dd8b1/sampler.py#L5-L28)通过 `set(user_train)` 排除采样负例中的完整训练历史，其中可包含当前 causal prefix 的未来商品；它使用 sampled BCE，与本阶段有效 last20 / full-catalog CE 的范围不同。[PyTorch 官方 CE 定义](https://docs.pytorch.org/docs/2.9/generated/torch.nn.CrossEntropyLoss.html)明确全类别 softmax 分母；`ignore_index` 忽略目标样本，不排除负类别列。历史列 mask 会改变分母与梯度，是成熟操作；NLL 自动下降不能证明推荐收益，也不构成新的评分方法。

### 2. fresh正面依据及其支持层级

[v4.3实际结题](copmrec-v4-3-history-exclusion-research.md)及[实际pool审计](evidence/copmrec-v4-autonomous-20261005/history-pool-validation-audit.json)（SHA `81cdb8501d0c1d501fc2c9c837aa777d425cdb1e8a33aa248d73338d3b644b73`）确认完整22363用户、175raw文件、零history输出/label-in-history、真实source/checkpoint/合法SID。same-policy pool仍有615新增/401遗漏，整体对公平LIGER R+9.870848708487067% / N+10.28369953005015%；少3hits不是新增预算的唯一理由，已确认有效层及源码支持集差异才是本阶段依据。

[fresh bad case只读描述](evidence/copmrec-v4-autonomous-20261005/history-validation-bad-cases.json)（SHA `9e860c7f5f61c1d4c4ce2541132214e3fa75345f888335304bbb529b2ba6c811`）使用同raw/label与已保存输出，未forward或生成新分数。401遗漏目标训练user频次均值102.446、615新增目标41.790；未投影content相似度也存在关联。这些只描述样本构成与频次/content混杂，不能证明训练CE历史负竞争造成遗漏、表示不够或校准有害。

源码的确定事实是 `_joint_losses` 使用原raw dense logits计算三项：SID teacher-forcing CE、cold=-100的训练content CE、raw logits上的合法mass-prefix mixture NLL。新机制只改content项；两臂 matched continuation才能回答额外mask收益，而loss变小不是推荐证据。

### 3. 全部保留结论与阶段取舍

旧v4 residual正向有限结果、mixed负向关闭、lr20额外收益CI跨0、v4.1 bias不支持增量、v4.2 fixedpool局部tradeoff和本轮CF关系方向不支持均保留，不恢复参数/CF扫描。v4.3是当前有效开发层，新的训练差异不更换history窗口/推理资格/0.5规则或固定source。

| 完整真实结果 | 改变的决定 |
| --- | --- |
| treated对control正向，CI/group支持且自身旧v4.3物质改善 | 可保留history-CE增量，主张限定固定checkpoint/单seed/开发集 |
| control自身改善、treated-control跨0或负向 | 可保留续训贡献，不能确认history CE收益；不扫描或延长 |
| 任一臂对旧v4.3 R/N任一≥3%且另一无点退化，但整体未达 | 可保留bad case/累计贡献，CI/group约束主张；不Testing |
| 两臂都无明显增量/负向或都不合格部署 | 结束本有界阶段，不自动续训/扫描，保留原有效history层 |
| 存在整体qualified臂 | 按预承诺N→R→treated选择唯一部署，才安排2次matched Testing |

### 4. 关键疑点与最小必要验证

训练causal目标必须不在当前有效完整SID history；fresh原始training无重复只支持数据假设，仍需实际训练preprocessing及batch断言。不合格时明确失败，不以label调整mask/回填目标或静默删样本。

treated在原cold=-100之后history=-inf；SID和mixture继续完整原支持。CE eligible行保留autograd、历史行对这项CE直接梯度0，loss/梯度有限；现有推理helper带no_grad，不能直接拿返回分数训练。weights-only来自原native v4.1严格验证，源版本不伪装；保留v0/v4链并新增真实v4.1 continuation链，参数完全相同、新optimizer/scheduler/step0。原三loss间共享表示与clip仍会造成间接轨迹变化，不声称纯最终评分因果。

实现/CPU梯度测试、完整Hydra resolve/脚本、原模型聚焦回归、真实来源/source/Mutagen flush、双卡production-batch smoke足够形成可复核运行准备；不追加与当前决策无关的试验矩阵。

### 5. 明确成本、唯一对照与部署门禁

两训练臂均从l3zyr91b v4.1 best6000 weights-only开始，唯一差异 `exclude_history_from_dense_ce` true/false。bias0、三loss各1、learned alpha、主干LR.0001/item.002/WD.035、FP32/seed42、GPU2,4→local0,1、128/card/global256，warmup300、**scheduler_steps6000**固定、max2000/val1000；各按history-excluded dense N10选自己的best1000或2000，并保留固定2000标量。新scheduler不因max2000改成短horizon。

每臂部署为固定nj9elah1 v4 best6000 source加该臂真实new best continued，独立自己的query/catalog原full logits后0.5/0.5平均，再原exact9 history eligibility。各完整单卡Validation一次，共最多2；复用fair baseline wdms8w77（R10=.09694584805258687/N10=.05404889855718787）及旧v4.3 iy3o3z3q（R10=.10651522604301748/N10=.05960712488411068），不增加baseline Validation。独立raw/输出/来源审计后复算fair、old、互比CI及固定双组/warmcold/newlostshared。

对旧v4.3任一R/N相对≥3%且另一无点退化是组件保留标准，增量归因另由treated-control判断。部署资格同时要求raw/source审计通过、对fair LIGER R10≥1.1倍（2385hits）/N10≥1.1倍、两paired差值CI下界>0、旧ValidationR≥.09877029021151008/N≥.0511173919307933。合格臂按**N10高者、精确同N按R10、再精确同R选treated**预承诺选择唯一winner；不合格不Testing/扫描/续训。

qualified winner才与同policy固定LIGER各做一次Testing（新最多2，全线程已1→最多3）。Testing不选模型；接受需对真实新same-policy LIGER双≥1.1、双CI下界>0，并同时旧042139al点阈值R≥.07855386128873586（1757hits）/N≥.04002978116676896。原Testing开发使用历史、单seed/已选checkpoint/重复开发集和未校正逐点bootstrap边界保留，目标只有实际确认后才完成。

## 固定来源及规格边界

- warm l3zyr91b best6000：`wandb://baymaxam/GRID/l3zyr91b?role=checkpoint&alias=v0&file=checkpoint_epoch=000_step=006000.ckpt`；SHA `4bea4d5771c4f056ad6e0cd69d1204a3220b639868803cbc19aa3332ecf0a3c9`。
- fixed source nj9elah1 best6000：`wandb://baymaxam/GRID/nj9elah1?role=checkpoint&alias=v0&file=checkpoint_epoch=000_step=006000.ckpt`；SHA `7c52228a665b593b50e45e8aba039f3f4aa4a10ca134d9345c59c59fb142f055`。
- new continued best的URI/SHA/digest/step按每臂实际selected Artifact记录，不把warm l3身份冒充新checkpoint；正常新model restore允许0..2000整数step，正式pool只允许selected1000/2000。

类/字段/checkpoint/合同见[OpenSpec design](../openspec/changes/add-copmrec-v4-4-eligible-content-continuation/design.md)；EligibleContentContinuationCoPMRec / HistoryExcludedContinuationPool按该规格完成，保留统一入口、公用loader/lineage/source snapshot/commonwriter。完整实现与实际准备证据见[implementation verification](evidence/copmrec-v4-autonomous-20261005/eligible-implementation-verification.json)。本次文档结题只读取已有实际审计，不修改runtime/research-state/current-plan或启动运行。
## 实际匹配训练记录

两臂均实际训练2000步、exit0、W&B finished；源374文件逐字节、原warm链、两个优化器组、warmup300/horizon6000和参数有限性审计通过。各自best均为2000，未选last.ckpt用于推理。累计真实训练7次/34000步；本阶段2次/4000步，不重置旧5次/30000步。

| 训练臂 | W&B run | best步数 | 成员val R@10 | 成员val NDCG@10 | 独立训练审计 |
| --- | --- | --- | --- | --- | --- |
| eligible-treated | `x4ge0y88` | 2000 | 0.105352587998 | 0.059343136847 | [eligible-treated](evidence/copmrec-v4-autonomous-20261005/training-eligible-treated.json) |
| eligible-control | `hbgj80bh` | 2000 | 0.105129010975 | 0.059199083596 | [eligible-control](evidence/copmrec-v4-autonomous-20261005/training-eligible-control.json) |

treated在1000步的成员R/N均低于control，在固定2000步略高（约+0.213%/+0.243%）；这只是training Validation scalar，不是独立raw配对证据，也没有达到组件保留标准。仍按各自真实best完成预承诺的两次pool单卡Validation。

## 完整单卡 Validation 与冻结部署

| 方案 | 完整run | R@10 | NDCG@10 | 对公平LIGER R增益 | 对公平LIGER N增益 |
| --- | --- | --- | --- | --- | --- |
| 同history LIGER | `wdms8w77` | 0.096945848053 | 0.054048898557 | 基线 | 基线 |
| 保留的v4.3 | `iy3o3z3q` | 0.106515226043 | 0.059607124884 | +9.870849% | +10.283700% |
| eligible-treated-validation | `artnz7m2` | 0.107990877789 | 0.060568542544 | +11.392989% | +12.062492% |
| eligible-control-validation | `rksx8qyh` | 0.107856727630 | 0.060622668610 | +11.254613% | +12.162635% |

两组raw175文件/22363 users、label/key/catalog、合法唯一零history输出、374源字节、实际消费CP和writer metadata均通过。两臂均合格，按预承诺N→R→treated选control，后续固定nj9原source + hbgj80bh best2000，各自query后0.5/0.5与原history资格，CE支持沿用原义。

treated−control：R增量+0.124378%、N增量−0.089284%，两CI跨0；30新增/27丢失、181共享上移/207下移。未证实history CE额外收益，关闭该mask增量主张。control对旧v4.3为+1.259446%/+1.703729%，R的CI跨0、N的CI下界正，未达到独立≥3%组件规则；整体合格部署与独立组件主张分开。已确认的原history层继续保留。

唯一部署冻结记录：[eligible-validation-decision.json](evidence/copmrec-v4-autonomous-20261005/eligible-validation-decision.json)，选定CP `wandb://baymaxam/GRID/hbgj80bh?role=checkpoint&alias=v0&file=checkpoint_epoch=000_step=002000.ckpt`，SHA `1204eeaa3dd7076249ef7b52fe234dc2d4bd50d4b5a4013abc1e8234c50350f8`。此选择发生在两份新Testing之前；只执行该固定部署及同policy固定LIGER各一次Testing，没有用Testing切换臂、checkpoint、权重或history窗口。

## 完整matched Testing与实际结题

| Testing方案 | 实际run | R@10 | NDCG@10 | hits@10 |
| --- | --- | --- | --- | --- |
| 同history LIGER，35ig0tz6 best45000 | `vnhmag7v` | 0.07722577471716675 | 0.04281558436299254 | 1727 |
| 冻结control pool，nj9elah1 best6000 + hbgj80bh best2000 | `ws2fx4oi` | 0.08554308455931672 | 0.04742966406441521 | 1913 |

新同policy主分母要求R≥0.08494835218888344（1900hits）、N≥0.0470971427992918。实际1913hits，比门槛多13；双点提升超过10%，配对差值CI均正。旧未排history的042139al只作辅助阈值：R≥0.07855386128873586（1757hits）、N≥0.04002978116676896，两项均通过，不用它替代公平主分母。

| 独立配对比较 | R@10相对变化 | R绝对差95% CI | NDCG@10相对变化 | N绝对差95% CI | 新增/丢失/净命中 |
| --- | --- | --- | --- | --- | --- |
| Validation treated − 公平LIGER | +11.392988929889292% | [0.008316191924160443, 0.013817466350668516] | +12.062491856151446% | [0.004987844986037772, 0.008013109800254816] | 634 / 387 / +247 |
| Validation control − 公平LIGER | +11.254612546125454% | [0.008183159683405626, 0.01368331619192416] | +12.16263462929399% | [0.005072172991342982, 0.008090286183003257] | 632 / 388 / +244 |
| Validation treated − control | +0.12437810945273853% | [-0.000536600634977418, 0.0007601842328846755] | −0.08928354212927037% | [-0.0003302551357204229, 0.00021274087596760593] | 30 / 27 / +3 |
| Testing control pool − 同policy LIGER | +10.770121598147053% | [0.005767338908017708, 0.010733130617537891] | +10.776636054536315% | [0.003278715644193265, 0.005964255154280129] | 512 / 326 / +186 |

Testing的共同命中521上移/434下移；N差值0.004614079701422671由新增贡献0.009359021551469943、丢失贡献−0.005951843897992782、共同位置贡献0.0012069020479455097相加得到。固定两key组均双点提升超过10%、两个CI下界正；分组仅按预登记规则复核，未据其选择模型。

| 固定Testing子组 | 用户数 | pool/LIGER hits@10 | R相对变化 | N相对变化 | 新增/丢失/净命中 |
| --- | --- | --- | --- | --- | --- |
| key组0 | 11201 | 973 / 877 | +10.946408209806147% | +10.61284779294336% | 243 / 147 / +96 |
| key组1 | 11162 | 940 / 850 | +10.588235294117654% | +10.949368124637381% | 269 / 179 / +90 |
| warm | 22225 | 1901 / 1721 | +10.459035444508991% | +10.47176648697894% | 506 / 326 / +180 |
| cold | 138 | 12 / 6 | +100% | +140.9913724124662% | 6 / 0 / +6 |

两新Testing各自实际exit0/W&B finished、physical GPU2→local0/单进程；独立重建175原始Testing文件/22363用户的末商品标签和有效history，标签SHA `ea9747417e5fec34daba59547f9e12fe14fea08c9c9ff2c512ea8944ae1e7939`，与Evaluation标签分开核验。同keys、catalog、合法唯一Top10、零历史重叠、label-in-history0、实际消费3个CP及Artifact/digest/SHA、两run source archive **374文件**及完整runtime哈希均通过。当前源SHA `31e1f901895ea10a2c29b030b2d8ffa8560730dbd6e712fac5017921b18eb53c`；learned alpha由固定成员checkpoint保留，不新增权重或alpha选择。

实际证据：[baseline Testing审计](evidence/copmrec-v4-autonomous-20261005/eligible-liger-testing-audit.json) SHA `4d649b1d766a4b8c67b435d2e035eb78c4000be3c7ad69fe5feec362618b4b1b`；[winner Testing审计](evidence/copmrec-v4-autonomous-20261005/eligible-winner-testing-audit.json) SHA `50a752572c0c6e4ed4b51e99418b04a112fc1d278221e2dc2307d82a1a02e247`；[Validation比较/选择](evidence/copmrec-v4-autonomous-20261005/eligible-validation-comparison.json) SHA `8ee563bc6db558c748552b5c853d108c1e515cdcc47eac3e52d70e321f6f20f9`。独立literature复核整体/分组/冷热的加权指标与new-lost分解无阻断；closure只读这些实际文件，没有新增推荐分数、forward、实验或goal工具调用。

结题保留已验证的完整control部署及原v4.3 history资格层；关闭本轮训练history CE mask的增量主张，不追调mask、延长训练或扫描成员/权重。两臂相对旧v4.3的点改善约1.3–1.7%，均未达到独立3%物质组件规则；这不否定依预承诺整体资格选择并通过matched Testing的完整方案，也不把整体收益归因于微小续训。

证据边界：Beauty单seed、固定Validation-selected checkpoint、重复开发集和先前Testing开发使用史；2000次paired-user PCG64 seed42 bootstrap是未校正逐点CI。**10%是实际点值门槛，CI只要求差值下界大于0**；本次相对CI下界约R+7.47%、N+7.66%，不能声称总体真实效应的下界也达到10%。cold仅138个目标，不泛化跨seed、跨数据集或所有cold商品。原LIGER训练source archive历史缺口仍保留，当前CP和新推理source核验不会回填历史来源。本阶段已封存累计7训练/34000步和Testing3/3；任何后续成本须另行明确登记。
