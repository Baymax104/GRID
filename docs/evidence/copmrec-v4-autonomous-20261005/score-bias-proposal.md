# CoPMRec 商品 score bias 独立问题草案

## 状态与范围

这是研究 proposal 准备，不是本子代理的代码实现或实验启动授权。只写本证据目录，不修改 runtime、研究状态或主报告。自主残差阶段三个训练臂均已完成 6000 步，实际累计 18000 步；第三臂 `nj9elah1` 的 selected best6000 正式单卡 dense Validation `mq8hhof5` 已终态且独立 raw 输出、标签、source 与指标审计通过，原阶段应正式关闭总结。全线程目标仍是固定真实 LIGER dense Testing 的 Recall@10、NDCG@10 均相对提升至少 10%。

本文只用当前明确保留的 v4 阶段证据，不以归档实验作为新方向的晋级或否决依据。新问题须在原阶段正式总结关闭后明确登记路线切换及一次性预算，不用 `v4.1` 命名自动延续或重置额度。

## 现有训练与评价证据

| 实际完成项 | 当前结果 | 支持范围 |
|---|---|---|
| `z9envq1n` residual，6000 步，hybrid-selected best5000 | R10=0.09457585960626602，N10=0.048828549683094025 | 相对下面同预算 control 的 best，R10 +2.4709%、N10 +2.0153%；六个相同步数均正向 |
| `5rlfx2np` zero-residual control，6000 步，best6000 | R10=0.09229531139135361，N10=0.047863930463790894 | 匹配 v0 best48000 初始化、数据、source、optimizer 与 exposure，用于区分新增残差和额外续训 |
| 旧 v4 best5000 全量独立 dense Validation | R10=0.09511246254974735，N10=0.04911456119404004 | 对独立固定 LIGER dense R10=0.08979117291955462、N10=0.04647035630072118，提升 +5.9263%/+5.6901%，两项配对区间为正 |
| 同 checkpoint、同候选的 mixed 对 content 排序 | mixed R10=0.09278719313151187，N10=0.048091188597248556；content R10=0.09457586191476994，N10=0.048828577391167034 | mixed 净损40个命中，两项配对区间为负，固定 mixed 终排不晋级，其竞争训练 loss 未实施 |
| 旧 v4 best5000 第一次独立 dense Testing `1edkvgk5` | R10=0.07516880561641998，N10=0.03844148315032377 | 对固定真实 LIGER dense 提升 +5.2599%/+5.6354%；有净推荐收益，未达两项10%门槛；不以 Testing 错误样本选择新机制或参数 |
| `nj9elah1` residual-lr20，6000 步，dense-selected best6000 的正式单卡独立 Validation `mq8hhof5` | R10=0.09600679694137638，N10=0.049793932567196456 | 对固定 LIGER dense +6.9223%/+7.1520%，两项配对区间为正；仍未达10%。对旧v4增量R10 +0.9403%、N10 +1.3832%，两项配对区间均跨0，不能证明额外LR收益 |
| lr20 预先固定的 step5000 dense 对旧 step5000 dense | 新 R10=0.09560434520244598，N10=0.049646779894828796 | 同步数均正向，约 +0.5172%/+1.0836%；排除仅由 best selection 造成的全部增量解释，但不证明 underfit；LR与每步AdamW decay联合变化 |

来源为同目录 `training-residual.json`、`training-control.json`、`control-comparison.json`、`validation-residual-mixed-validation-audit.json`、`testing-residual-dense-audit.json`、`training-residual-lr20.json`、`training-residual-lr20-supplement.json` 和 `validation-residual-lr20-audit.json`。新减旧配对95%CI：R10 [-0.000536600634977418, 0.00236998613781693]；N10 [-0.000015168409071523037, 0.0013225257441768085]。

## 路线判断五项门禁

1. **核心可反驳假设。** 在 v0 内容表示之上新增 seen 商品协同残差，并在历史输入和目录打分共享，可提高商品区分能力及推荐结果；其收益不应仅由额外续训解释。全线程两项10%是目标门槛，不是核心机制有效性的定义。
2. **支持证据及层次。** residual/zero-residual 匹配续训 best 和六个同步数支持新增自由度的有限增量；独立 dense Validation 与第一次 Testing 支持相对固定 LIGER 的实际正收益。lr20 固定5000为正向点估计，新best6000独立Validation也相对LIGER正向；新对旧的两项增量CI跨0，不能宣称额外LR收益已证明，更不能把它写成更快学习或underfit的单一因果证明。
3. **反证、门槛与缺证区分。** 固定 mixed 终排被同候选配对结果直接否定，保持关闭。原 v4 Testing 和 lr20 独立 Validation 未达10%属于晋级门槛未通过，不是“协同残差无效”的反证。lr20增量区间跨0是证据不足，不是两设置等价或全部残差机制失效。商品 bias 尚未训练；没有其效果或概率校准证据。
4. **是否还有会改变决策的关键疑点。** 更强selected lr20 best6000的新输出已沿同一固定用户key分组只读复核：曝光集中从旧22.45%降到新18.51%，两组top20仍19/20重合、前三名几乎对半一致。它已经缓解但仍存在，支持独立商品截距的一对有限匹配测试；这项检查没有新增forward或实验。不同用户的真实相似偏好、embedding方向和前排选择也可能产生集中曝光，不能排他归因为缺bias。matchedwarmstart可区分新bias的实际增量与继续训练，允许根因保持未知；不追加穷尽性排查。
5. **原阶段剩余预算与取舍。** 训练剩余为0，追加训练实验为0。原18k阶段正式总结关闭：保留有限残差收益、关闭mixed、不扩展LR序列。旧阶段不达10%不会自动转换成新预算。最新Validation仍支持商品层面稳定误排，只有root明确登记下面独立的新问题及预算后才可进入实施与两臂运行。

## 原始频率与跨组审计

`validation-item-exposure-audit.json` 保留旧 `z9envq1n/best5000` 的 `amwchjkc` Validation trace，并加入已独立审计的新 `nj9elah1/best6000` 的 `mq8hhof5` dense输出。固定 LIGER `5azn5vm0` trace和新旧输出严格对齐22363用户；新SID top10只通过catalog SID→row映射，不运行模型。新旧checkpoint的item keys与SID相等。五个checkpoint/trace/输出输入精确SHA、175个training shard及manifest SHA、实际preprocessing/target函数hash均写入证据。

真实训练规则是连续子序列的末项作为 target。长度为n的历史有 T=n(n−1)/2 个子序列；T≤32全部展开，否则 `torch.randint` 有放回抽32次再 `set` 去重。位置j=1…n−1的期望target次数为 j×[1−(1−1/T)^32]，未触发cap时为j。整个训练语料包含22363条历史、153776次raw interaction；一轮target数期望265529.32812107983，实际调用现函数、审计seed813的一轮展开265472。二者均不是6k训练已实现的DDP抽样曝光。

旧 v4 的错误 top10 曝光最多20个商品占22.452969%，这些商品的 training causal expected target占2.896145%。预先固定 `SHA256("copmrec-v4-validation-exposure-v1:<decimal-user-key>")[0] & 1`，两个子组11201/11162人，各自top20 offender重叠19/20、前9名一致，错误曝光top20集中度22.439306%/22.483874%；LIGER为26.700052%/27.027198%。v4两组R10相对LIGER提高6.097561%/5.761719%。这是跨用户稳定的排序现象，不是完整softmax概率校准误差，也不推导任何inverse-prior惩罚或bias初值。

新 lr20 best6000 的错误top10共221483次，top20占40995次/18.509321%，这批商品的training causal expected target占2.937361%。两个同规则子组集中度18.525027%/18.516173%，各自top20仍19/20重合，前6名一致。789/774/510在第0组错误曝光1979/1824/1590，第1组1978/1845/1590。冻结旧top20集合在新输出中曝光39475次，比旧49734次减少20.627740%，与新top20重合16个。新两组R10相对LIGER为6.910569%/6.933594%。现象减弱而未消失，支持有边界的新增自由度测试；不能据这些比率承诺bias会改善或达到10%。

复现只读审计，从本地GRID根目录执行：

```powershell
$exposureProgram = [System.IO.File]::ReadAllText((Join-Path (Get-Location) 'docs/evidence/copmrec-v4-autonomous-20261005/validation-item-exposure-audit-source.txt'))
$exposureProgram | ssh node1 'cd /data3/weizhenyu/projects/GRID && PYTHONDONTWRITEBYTECODE=1 ~/.local/bin/uv run --no-sync python -'
```

## Primary source 与贡献定位

Koren、Bell、Volinsky 的2009年 IEEE Computer 原论文在“Adding biases”中，将用户、商品截距与内积交互分开，并联合学习；Eq3–5已经明确包含 `b_i`。因此普通商品 score bias 是成熟自由度，不能称首创。该工作针对Netflix显式评分及平方误差，本proposal针对序列下一商品、完整目录content CE及合法SID prefix mixture NLL，不能据其评分结果推定本任务收益。[作者自托管原论文](https://chrisvolinsky.com/files/publications/ieeecomputer2009.pdf)。

LIGER把生成与dense推荐结合，官方 `get_target_embed` 将query和商品向量归一化后计算内积/temperature；也存在`ground_truth+item_id`表示设置。这为本项目共享内容/协同表示提供比较定位，不能宣称ID残差首创。当前新增标量截距应作为可检验机制，其论文价值取决于问题证据、匹配收益及整体内容生成式推荐叙事。[LIGER原论文](https://arxiv.org/html/2411.18814v2)、[官方打分实现](https://raw.githubusercontent.com/facebookresearch/liger/main/src/evaluation.py)。

允许表述“现有打分没有独立商品截距，新增这个自由度是否改善稳定误排”；不允许表述“normalized cosine无法表达商品偏好”“popularity bias已证实”或“概率calibration已证实”。

## 下一独立问题及最小对照（待晋级）

问题限定为：**在当前更强的v4共享内容/协同表示上，学习独立商品score截距，能否在同预算续训之外改善稳定的商品层面误排与总体dense推荐？**

唯一干预是 seen-only zero-init可学习标量 `b_item`，在 cosine/temperature 后加入content logit。cold bias严格为0；不放回历史embedding，不更改temperature、alpha策略、SID候选恢复、mixed终排或新增loss。三项loss各1保持；bias直接接受content CE与mixture NLL监督，SID CE本身对纯score bias无直接梯度。商品bias由training loss学习，禁止用Validation target频率拟合或人为inverse-prior设置。

两臂共同从 `nj9elah1` 的 Validation-selected best6000 初始化：

```text
wandb://baymaxam/GRID/nj9elah1?role=checkpoint&alias=v0&file=checkpoint_epoch=000_step=006000.ckpt
sha256=7c52228a665b593b50e45e8aba039f3f4aa4a10ca134d9345c59c59fb142f055
```

1. **score-bias臂：** 共享基础参数继续训练，新seen bias从0学习。
2. **zero-bias匹配臂：** 同结构、同输入和初始化，新bias固定0；基础参数同样继续训练，不能用未续训checkpoint代替该对照。

采用weights-only warmstart、新optimizer/scheduler和一致的训练exposure；两臂均保留主干LR1e−4、残差LR2e−3、weight_decay0.035、已有warmup、global batch、seed、三loss及learnedalpha。root拟预先固定新增bias沿用当前item-specific残差参数组LR0.002/WD0.035，同组且总计仍两个optimizer组；control保留active residual、bias冻结0。这个选择避免新增独立biasLR搜索；负向结果停止本固定设置，不换LR，也不能将步长强弱写成机制结论。原v0 checkpoint路径配置若作为初始化依赖存在，必须防止覆盖当前v4 warmstart；这属于实施前必须通过的装配检查，不是当前模型效果疑点。

每臂最多6000步，双卡训练、单卡推理；两臂在相同dense Validation、相同频次下按NDCG10选点，同时保留预先固定的相同步数结果。最小必要正式评价为两臂各一次全量单卡dense Validation与配对复算；沿已有原始标签/基线，不额外启动重复LIGER实验。未决实施检查包括zero-init输出等价、cold0、合法prefix归一化、bias梯度、严格checkpoint兼容及source lineage；只有准备和检查通过并被root明确晋级后才运行。

可反驳预测是score-bias臂应在匹配续训之外提高总体dense NDCG10且不损害Recall10，并让稳定offender所涉净排序损益更有利。不能只以bias范数、loss下降或错误曝光减少晋级，也不要求所有offender的bias符号相同。若仅减少曝光而指标退化，拒绝本固定机制用途；若配对区间不支持增量，记为证据不足并关闭本stage，不将不显著写成等价。若增量正向但仍不达10%，可以保留有限机制/效果结论，但整个目标未完成且预算不自动增加。新stage至多一次固定winner Testing由root在Validation匹配结果后决定，Testing不用于调参或选点；最终两项10%目标必须以Testing独立审计验收，不能用Validation代替。

## 阶段与全线程成本账

| 预算范围 | 已执行训练臂 | 已执行训练步 | 本次新增上限 | 新增是否已启动 |
|---|---:|---:|---:|---|
| 已完成自主残差stage | 3 | 18000 | 0 | 原stage无剩余额度 |
| 新商品score-bias独立stage（proposal） | 0 | 0 | 2臂×6000=12000 | 否；待明确晋级 |
| 本自主目标线程累计 | 3 | 18000 | 若采用新stage，累计最多5臂/30000步 | 当前实际仍3臂/18000步 |

“全线程累计”指本次自主目标执行以来的正式训练，初始化依赖的既有v0 48000步不伪装成新运行；此前独立阶段的成本账保持原记录、不转移或抹去。新stage是两次真实训练，不把对照藏在单个实验名称中。固定两臂完成后无LR/temperature/bias权重扫描、无默认第三臂、无阶段名称续写式预算重置；硬件失败的实际成本需单独记账。
