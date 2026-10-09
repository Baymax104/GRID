# CoPMRec v4.3：有效历史占位与匹配候选资格评价

## 状态与研究问题

本阶段两次完整Validation及独立审计已完成，**历史排除层保留（RETAIN），本轮不进入Testing，整体目标仍未完成**。相对自己冻结旧pool，R10提升8.568824065633551%、N10提升17.367341655669733%，188个新增命中、0丢失，两paired差值95%CI均为正；同政策公平比较为R10+9.870848708487067%、N10+10.28369953005015%，R10距离门禁少3个命中。未达到整体门禁不丢弃这个有效层。

实现准备154 distinct聚焦用例已通过，7新runtime与367文件源码核验见[实现验证](evidence/copmrec-v4-autonomous-20261005/history-implementation-verification.json)。精确结果见本文“实际Validation结题”；[阶段结题JSON](evidence/copmrec-v4-autonomous-20261005/history-exclusion-stage-closure.json)记录两实际审计、来源SHA、门禁和累计预算。下列五项保留正式运行前的问题证据与预注册标准，旧输出只读统计与本轮正式效果分别记录。

新问题是：**在保留完整目录原分数的条件下，排除用户有效模型输入历史中的已知商品，能否消除无效占位，并在对LIGER应用完全相同政策后仍满足相对双10%目标。** 它是候选资格处理，不声称新增CF信号、新评分或原创过滤机制。root授权当前阶段只有有限推理额度，旧5训练 / 30000 steps已耗尽，不自动恢复或扩大；这不是用户永久禁止后续研究的硬上限。

用户在本阶段正式运行前最新明确：**不要求每层立即达到10%；约3%以上明显增益可保留做bad case或累计贡献继续其他部分。** 因此组件保留与整体晋级独立：v4.3对自己冻结旧pool i4xwruok的R10/N10任一相对提升≥3%，另一项点值无退化，可保留继续只读bad case和累计研究；CI及固定分组限制效果主张。该标准在运行前登记，本轮实际结果符合并保留；未来实验成本须依新正向证据、bad case和新登记明确决定，不自动重置已用预算。

## 五项路线门禁

### 1. 核心可反驳假设与路线变化

真实原始Evaluation的22363个末商品目标均不在本人raw training或有效20商品历史中，已保存Top10却包含历史商品，并将已命中的正确目标压后。固定scores只排除这些已知历史行，在本split上应使历史占位为0，并保留未被政策排除的已有命中及其相对排序；目录外新增恢复和对same-policy LIGER的相对效果仍未知，需要一次匹配完整评价。

本次切换为显式的history eligibility问题。训练关系统计不支持本轮“未用共现/前向CF信号”方向，已停止该排查；不把它改名并恢复graph或训练。

### 2. 已完成的fresh支持与支持层级

[训练关系只读统计](evidence/copmrec-v4-autonomous-20261005/validation-collaborative-relations-analysis.json)（SHA `1fdd7d30a32699fe760383fbd6406028e3c90fe7b034f5029f8454205c5be1c2`）核验175份当前raw training / 22363用户 / 153776事件、0重复训练序列，175份Evaluation raw hashes与独立主审计相同。21428用户raw training恰为Evaluation prelabel；935条长序列已截断，但**全部22363有效history恰等于本人training尾20**。本人training / effective history中的真实目标均为0；这是当前Evaluation事实，不外推其他split。当前训练哈希记录新鲜字节，不回填历史运行数据来源证明。

[既有Top10占位只读统计](evidence/copmrec-v4-autonomous-20261005/validation-history-occupancy-analysis.json)（SHA `6ad7ea3306fa92f9d67f4623388661dd1823fdde0bf3c727e9218f1fd8006988`）直接读取已核验原预测与标签，错误曝光分母为 `10×users−hits10`。下面N位移是**仅已存在真命中**前方历史位置移除后的确定贡献；原miss仍为0，不补Top11、不生成新分数或推荐列表。

| 冻结原模型 | 历史错误曝光 / 全部错误曝光 | 有历史占位用户 | 已命中目标前有历史占位用户 | 已知命中N10位移 / 全用户 |
| --- | ---: | ---: | ---: | ---: |
| LIGER 5azn5vm0 | 15642 / 221622（7.057964%） | 10267 | 905 | 0.0054145133198089225 |
| source mq8hhof5 | 18155 / 221483（8.197017%） | 11780 | 1028 | 0.006031022022549512 |
| control yqsmsdt1 | 20696 / 221421（9.346900%） | 13378 | 1154 | 0.0065209571327474345 |
| 固定pool i4xwruok | 19662 / 221436（8.879315%） | 12715 | 1095 | 0.006217131040123722 |

pool两个固定key组的历史错误曝光为9848 / 9814（8.878231% / 8.880403%），已知命中N位移为0.006458768449737654 / 0.0059746493500067496；LIGER两个组也有同方向占位与位移。支持的是已知history占位和固定评分条件下的局部位置损失，不能把这些下界视为完整政策收益、目录外恢复数或same-policy相对10%保证。

### 3. 全部保留结论、反证与未完成

- [v4残差结题](copmrec-v4-autonomous-research.md)：有限残差正收益保留；固定mixed两CI负已关闭，lr20额外收益两CI跨0，训练序列关闭。旧Testing `1edkvgk5` 对未排历史LIGER为约+5.26%R / +5.64%N，仍是该旧模型已验证结果。
- [v4.1商品截距结题](copmrec-v4-1-score-bias-research.md)：bias对control净−15，两增量CI跨0，未达到原双门槛，没有新Testing；不恢复bias/temperature/LR扫描。
- [v4.2固定pool结题](copmrec-v4-2-fixed-logit-pool-research.md)：i4xwruok对LIGER为+9.262948%R / +9.288610%N、两CI正，但双10%点门槛未过；对control145新 / 160丢失、净−15，两增量CI跨0。固定639恢复+24、其余21724净损39保留；不换pair/权重/归一化。
- fresh CF关系中真实目标cooccur / forward20 / adjacent为48.334% / 38.720% / 11.233%，control false为73.811% / 61.562% / 14.101%；主要频次组同user粗匹配未呈正向优势，频次和关系degree仍混杂。这不支持本轮CF路线，低频小子组不证明unused signal。

上述均为保留的先前结论。历史资格处理的完整same-policy结果见实际结题；此前局部确定位置改善不替代正式结果，旧接近10%也不自动扩大预算。

### 4. 关键疑点与最小必要核验

当前实际 `Liger.retrieve` 和固定pool都在完整dense logits后stable Top10，没有个人历史排除；catalog的全局seen_mask不是该用户历史资格。有效20与实际80 SID token / 4 hierarchy精确对齐，无需重新搜索history定义。会改变决策的未知是目录外恢复及LIGER同样过滤后的公平相对结果。

实现采用一个共享严格 `apply_history_exclusion(scores,input,catalogmodel)`，只读取attention有效完整SID，精确catalog lookup、去重mask负无穷；padding内容忽略，partial/unknown有效SID拒绝，其他分数逐值不变。`HistoryExcludedLiger`正常继承原checkpoint hook/strict state；`HistoryExcludedFixedLogitPool`仅在父完整pool forward后mask。两类同字段 `history_eligibility_contract` exact9keys、单进程、无labels/raw training/user-key政策查找、cold继续可选，原pool_contract不改；完整schema见[design](../openspec/changes/add-copmrec-v4-3-history-exclusion/design.md)。准备检查仅验证实现与来源，不作为推荐效果。

### 5. 有限额度、顺序与结果对应取舍

| 验证单元 | 固定臂 / 实际上限 | 缩小的不确定性 | 决策 |
| --- | --- | --- | --- |
| 匹配完整Evaluation | same-policy LIGER best45000 + 同policy固定pool，各1次，合计最多2次 | 原分数不变下真实目录外恢复、历史零占位、公平相对R/N及pairedCI | 全整体门禁通过才进入Testing；另行判断组件保留，不因未双10丢弃明显正向组件 |
| 条件固定Testing | 上述same-policy LIGER + 同fixedpool，各1次，合计至多2次 | 对新same-policy真实baseline的固定确认 | 不用于选window/pair/参数；全线程先前1次→最多3次，不增加上限 |

Evaluation新same-policy LIGER的实际指标记 `R_LV,N_LV`，pool为 `R_PV,N_PV`，晋级必须同时满足：`R_PV≥1.1×R_LV`、`N_PV≥1.1×N_LV`、两项pool−newLIGER paired绝对差值95%CI下界均>0，以及旧Validation点阈值 `R_PV≥0.09877029021151008`（2209hits）、`N_PV≥0.0511173919307933`。预注册时不预填新baseline结果，实际门槛见结题；旧未过滤分母仅为额外绝对门槛，不能成为单边过滤方法增量的主对照。

条件Testing接受同样要求对新same-policy Testing LIGER两指标≥1.1倍、两pairedCI下界均>0，并同时达到旧042139al点阈值R10≥0.07855386128873586（1757hits）/ N10≥0.04002978116676896。Testing独立实际raw标签审计并显式报告label是否在history，不根据Testing改变规则或假定本split零repeat会普遍成立。保留Testing已有历史开发使用的边界，不称全线程split从未参与开发。

组件保留使用对自己冻结旧pool的完整真实点增量，不以old LIGER作为政策增量分母；≥3%且另一项无点值退化仅是开发保留标准，pairedCI跨0或分组不一致时不能声称整体增量已证明。未通过same-policy双10等整体门禁只停止本轮Testing晋级，不自动否定符合保留标准的正向组件；负向或无明显增量按现有边界收缩。

新增**训练0、optimizer0、CF/train-data构建0**；既有5训练/30000耗尽不重置。本阶段不换history window / pair / weight / temperature / normalization / checkpoint / arm，不自动扩实验额度；未来成本依据新正向证据与bad case另行明确登记，旧上限不是用户永久禁令。目标在实际完整确认之前保持未完成。

## 相关工作与公平性定位

[RecBole官方固定实现](https://github.com/RUCAIBox/RecBole/blob/7b02be5ec80a88310f2d04a27a82adfcbb5dc211/recbole/trainer/trainer.py#L521-L540)在full-sort评价有history_index时将这些分数设为负无穷，说明这是成熟候选资格处理。其sequential分支的history_index语义不同，不能说所有序列推荐都标准排历史，或本20个有效输入SID等同RecBole全部历史。

[LIGER官方固定commit的dense评价](https://github.com/facebookresearch/liger/blob/b6ccc37af5ee623ddc1d1ead3490c31aaeaf4524/src/evaluation.py#L414-L424)对完整logits直接取TopK，没有这一history排除。因此这是显式新政策，不是修复官方LIGER遗漏，也不宣称原创。主要增量只比较两者同policy，分别对原输出的policy增量说明作用；保留未排历史结果作为历史证据，不能单独给CoPMRec过滤来放大对比。

## 固定来源与实施边界

| 固定模型 / 成员 | checkpoint URI | 原始SHA256 |
| --- | --- | --- |
| LIGER best45000 | `wandb://baymaxam/GRID/35ig0tz6?role=checkpoint&alias=v1&file=checkpoint_epoch=000_step=045000.ckpt` | `8508e08e2a2cc9ea5d2bbc902aa4e2b6415c8a45b9c8a0ddc879728a6b66b43b` |
| pool source v4 best6000 | `wandb://baymaxam/GRID/nj9elah1?role=checkpoint&alias=v0&file=checkpoint_epoch=000_step=006000.ckpt` | `7c52228a665b593b50e45e8aba039f3f4aa4a10ca134d9345c59c59fb142f055` |
| pool control v4.1 scale0 best6000 | `wandb://baymaxam/GRID/l3zyr91b?role=checkpoint&alias=v0&file=checkpoint_epoch=000_step=006000.ckpt` | `4bea4d5771c4f056ad6e0cd69d1204a3220b639868803cbc19aa3332ecf0a3c9` |

pool两成员0.5 / 0.5及原温度、residual、scale0、cold、原v0链保持，目录12101商品/33cold沿用同SID和embedding。原class/入口不改，新根推理脚本经 `src.main` / Hydra、publicloader、lineage和commonwriter，GPU2→local0 / NPROC1 / FP32。LIGER顶层真实checkpoint正常恢复；pool顶层ckpt_path=null与两个显式来源不变。root在实际运行前完成Mutagen flush三Watching、实际全runtime源码、输入和checkpoint来源核验，记录唯一job句柄。

正式运行均由root通过登记句柄启动；本报告更新只读取已有审计，不修改研究state/current-plan/runtime，不重新执行forward或实验。

## 实际Validation结题：有效层保留，整体门禁未过

本阶段使用相同367文件源快照 `3923ffcea72925c02f3017b973651c5d49e49f6b2b8851af2e10e4b705a394f5`。两臂均为物理GPU2→local0单进程、FP32、batch32；原固定checkpoint、SID/embedding、catalog和完整输入历史规则不变。真实数据为 `data/beauty/evaluation`；W&B指标字段名仍是 `test/*`，它们对应这次Validation数据，**不计为Testing**。

| 模型 / 真实run | 命中数 / 22363 | R10 | N10 |
| --- | ---: | ---: | ---: |
| 原未排历史LIGER 5azn5vm0 | 2008 | 0.08979117291955462 | 0.04647035630072118 |
| 新same-policy LIGER wdms8w77 | 2168 | 0.09694584805258687 | 0.05404889855718787 |
| 原未排历史pool i4xwruok | 2194 | 0.09810848276170460 | 0.050786806656135254 |
| 新same-policy pool iy3o3z3q | 2382 | 0.10651522604301748 | 0.05960712488411068 |

### 自身政策增量与组件保留

iy3o3z3q相对自己冻结旧pool的R10+8.568824065633551%、N10+17.367341655669733%。绝对差值为R10 `0.008406743281312882`，95%CI `[0.007244108572195144, 0.009658811429593525]`；N10 `0.008820318227975438`，95%CI `[0.00817514022973746, 0.009456376905080534]`。188新命中 / 0丢失；原2194个命中均保留，其中1095个位置上移、0下移。

N10增量恰分解为新增命中贡献 `0.0026031871878517154` 加原命中位置改善 `0.006217131040123722`，后者与正式运行前的只读确定下界一致。原Top10全部非历史候选的相对顺序和压缩后位置均从实际新旧输出验证；历史错误曝光由19662降为0，本split没有因label在history而排除正确目标。该证据支持保留历史资格层，不能单独称为对LIGER的方法增量。

LIGER也从同一政策受益：R10+7.968127490039856%、N10+16.308336883461937%，160新命中 / 0丢失、905个共同命中上移 / 0下移；两项CI均为正。这正是必须使用same-policy baseline的原因。

### 公平相对结果与整体门禁

对新same-policy LIGER，pool的R10+9.870848708487067%、N10+10.28369953005015%。绝对差值R10 `0.009569377990430622`，95%CI `[0.006886374815543532, 0.012386531324062066]`；N10 `0.005558226326922801`，95%CI `[0.004044800506966806, 0.0069816922689833185]`。615新命中 / 401丢失 / 净+214，共同命中635上移 / 524下移。该比较支持本开发集上的正向相对表现，仍不是Testing确认。

预注册新门槛为R10≥ `0.10664043285784557`（2168×1.1需至少2385hits）、N10≥ `0.05945378841290666`。实测2382hits，R门槛少3；N门槛、两正CI及旧Validation两个绝对点阈值均通过，但整体为**false**。因此本阶段Testing=0，不降低门槛、不选择group0、不扫描窗口/成员/权重补这3个命中。组件保留为**true / RETAIN**，整体目标保持未完成。

### 固定分组与边界

两个预先冻结key组仍为11201 / 11162，分组规则和SHA不变。自身政策增量分别为83 / 105新命中、两组0丢失，565 / 530共同命中上移、两组0下移；R10分别+7.635694572217111% / +9.485094850948506%，N10分别+17.794836665518087% / +16.96445414309029%，各组两项CI下界均正。

same-policy公平比较两组净+116 / +98，R10分别+11.005692599620499% / +8.797127468581678%，N10分别+12.345459772768486% / +8.395623427117393%；各组两项CI正，但不能以组0达到双10替代整体。自身政策在warm22312用户上新增188、0丢失；cold51用户原4个命中全部保留、无新丢，只有1个共同命中上移，cold N增量CI下界为0，不泛化冷启动增益。same-policy LIGER在这51例为0命中，相对比例无定义，只报告pool4个命中的绝对事实。

### 独立核验、预算与下一步

[baseline实际审计](evidence/copmrec-v4-autonomous-20261005/history-liger-validation-audit.json) SHA `6160b7fc77a8739b9aa38a0fb6601164b28f3d9dbafa3fef9b8472d7d042d670`；[pool实际审计](evidence/copmrec-v4-autonomous-20261005/history-pool-validation-audit.json) SHA `81cdb8501d0c1d501fc2c9c837aa777d425cdb1e8a33aa248d73338d3b644b73`。两审计均核验真实exit0 / W&B finished、367 source/archive/runtime字节、selected checkpoint Artifact角色/digest/SHA/step、共同catalog、used SID/embedding身份、实际writer exact9契约、175 raw文件/22363用户/末商品标签、SID合法唯一、有效历史零输出；label-in-history实际为0。原训练producer来源沿用明确冻结的已审计锚，没有用新source回填历史证明。

本阶段消耗**2/2完整Validation、0训练、0optimizer、0CF数据构建、0Testing**；全线程旧训练累计仍5次 / 30000 steps，Testing仍1/3。两个条件Testing槽未触发，不自动追加或重置预算；旧Testing部署结果仍是先前1edkvgk5，不把当前Validation冒充其新部署结果。

下一步保留历史资格层，以当前iy3o3z3q实际输出为开发起点继续只读bad case：在无历史占位、原分数和来源冻结后，pool相对same-policy LIGER仍存在615新增与401遗漏的覆盖分歧，401个LIGER命中而pool未命中的合法目标提供另一个待界定机制问题的具体样本。先分析这些分歧及未覆盖目标，形成新证据与可检验预测，再选择机制和登记必要成本；**尚未选择新方法或授权新运行**，不恢复已否定的CF关系方向或旧mixed/bias扫描。原整体目标保持active，保留有效层继续累计改进。
