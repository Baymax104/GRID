# CoPMRec v5.3：Milestone 3 消融与机制实证计划

协议：`copmrec-m3-v53-20261008-v1`；设计/执行日期：2026-10-08；范围：Linear milestone3 / [BMX-117](<https://linear.app/baymax104/issue/BMX-117>)。状态：Done；用户明确授权该有界范围并在训练完成后要求继续，5/5训练、250k更新、5/5Testing、4/4正式diagnosis及M1零模型forward配对分析已独立核验完成；八个实验issue已Done。未增加训练、数据集、seed或预算。

本计划依据当前代码、组件配置、正式发布契约与主矩阵来源回执重新设计。旧 milestone3、旧 seed42 Legal/Max/Mass 及旧开发版本不是设计或结果来源；旧主计划中 M3 的三臂 seed42 清单由本协议替代。M2/M4 的既有结果与状态不变。后续用户已明确授权补齐必要入口并验证命令；现已完成实现、静态/CPU验证及代码同步；随后用户授权实际启动并在训练完成后要求继续，现已完成全部固定证据交付。

## 1. 任务目的与方法边界

目的：为论文方法提供组件消融和机制观测证据。结果只记录观测值、带符号的数值差、区间、样本数、来源和解释范围，不给实验贴好/坏、积极/消极、成功/失败标签，不据此晋级、否决或自动扩预算。Done 表示预定证据已交付、数据与计算可核验，与差值方向和显著性无关。运行异常及证据缺失仍须记录。

正式 Full 为 v5.3：共享 Encoder query，SID/内容/历史残差输入；目录表示 P(c_i)+r_i；四项等权损失 SID CE、全目录 joint CE、前缀 mixture NLL、native-view CE；alpha 为从0.5起学习的全局标量。训练中 p_mix=(1-alpha)p_gen+alpha\*p_mass；mass 使用后代目录 logits 的 logsumexp，并在合法兄弟分支归一化。正式部署仅用 joint cosine/0.07、排除输入历史、稳定排序的全目录 dense Top10，不执行 beam，也不使用 alpha 或辅助 logits 排序。

native view 的目录侧无残差，query 的历史侧仍有残差；该视图不能称为完整无协同模型。cold 指商品未出现在 training interactions 中，其有效残差恒为零；cold 的 query 仍可能包含 seen 历史信息。

## 2. 从实现发现的测量问题与解释范围

* mixture 项通过共享 Encoder、内容投影和 Decoder 参与优化；dense 部署下要测量其训练监督对应的差异，不能用推理 alpha 开关代替训练消融。
* 共享残差同时进入历史和目录，单个 ResidualOff 无法区分两个作用位置；因此增加固定 checkpoint 的2×2评分干预。
* native CE 与 joint CE 共享同一次 query/内容投影。残差为零时两者相同，移除 native CE 同时减少一个 CE 项；因此必须设置额外 joint CE 替换对照。
* 前缀条件概率可以在真实目标前缀下直接测量；该条件化观测不等于自由生成候选可达性，更不能解释为正式 dense 的 beam 恢复。

LIGER hybrid 是完整方法比较的主基线；同模型训练消融以 Full 为参照。主矩阵回执记录了 CoPMRec 与既有 LIGER 的历史资格差异，跨方法的命中分解只作为描述性比较，不能解释为纯组件因果效应。LIGER dense 的既定内部9次Testing留在原工作范围，不计入本M3、不改主基线。

## 3. 训练对照：三个实验问题、五个新变体

记 L=S+J+M+N；S=SID CE，J=joint catalog CE，M=mass-mixture NLL，N=native-view catalog CE。

| 变体 | 训练定义 | 对照与可测量量 | 限定解释 |
| -- | -- | -- | -- |
| Full | S+J+M+N；共享残差 | 复用Beauty/seed42正式own-best与Testing | 不新增Full训练 |
| A1 NoMixture | S+J+N；alpha冻结0.5、不进优化器 | A1-Full 四指标与逐用户差 | 移除监督项的整体影响；不能单独归因 mass 形式 |
| A2 NoResidual | 历史/目录残差均恒0；S+J+M+N保留 | A2-Full，seen/cold及频次切片 | 参数量/优化器组会变化；共享优于不共享不在此设计的证据范围 |
| A3 NoNative | S+J+M | A3-Full，结合A5解释 | 辅助项移除同时改变CE总权重 |
| A4 LegalGenReplace | S+J+G+N；G为真目标前缀下合法生成概率NLL，按用户×4层平均；alpha冻结0.5 | A4-Full及A4-A1 | 区分额外合法前缀监督与内容条件混合；S保留且与G支持集不同 |
| A5 JointCEReplace | S+2J+M；第二个J复用同一joint logits，无额外投影/dropout | A5-Full及A5-A3 | 区分辅助视图与重复joint CE；只匹配损失项数/名义系数，不保证梯度范数相同 |

每个新变体只执行 Beauty / training seed42，共1个训练单元。选择Beauty作为固定代表数据集，不按消融结果更换数据集或seed。整体多数据集、多seed结果由已完成正式主矩阵承担；本M3只测量这一固定单元的组件和机制。Full 仅作预测/诊断来源，禁止作为新变体初始化。A2中native和joint目录分数相同，N仍保留，因此A2有重复CE，这是预先规定的组件移除含义，不能临时补偿权重。

本计划不增加全排列、参数扫描、无SID骨干或额外baseline训练。不能用这五个对照证明组件独立可加、统计不显著等价或共享表相对未共享表的优势；这些问题保留为解释边界。

## 4. 所有训练单元的固定条件

* scratch随机推荐器；相同dataset/seed的公共参数初值和数据顺序匹配；零残差、零gate logit。输入SID、内容、目录映射和training/evaluation/testing manifests与正式Full匹配，运行前登记digest。
* DDP2，每卡128、global256、accumulate1、FP32；50,000 optimizer updates；AdamW主lr0.0003、有效残差lr0.002、wd0.035；warmup2500/cosine50000，clip1。A2冻结/移除残差组、A1/A4冻结gate的变化需写入variant契约与参数计数。
* 每500步训练内raw full-catalog dense Validation；每臂选自己的首次最高val/ndcg@10；不复制Full best step，不要求所有臂同step，不用last.ckpt替代best。
* 选点审计后一次完整单卡Testing；joint dense、最近20件有效完整SID历史排除、cold eligible、稳定catalog row排序、合法唯一Top10。不追加独立Validation run。
* Recall/NDCG@5/@10全部报告，NDCG@10作主要展示量，无结果方向门槛。
* 差值定义为variant-Full；表保留原始指标。只报告Beauty/seed42，不计算跨seed均值/标准差，不将本消融视为跨数据集或跨初始化的稳定性证据。用户配对bootstrap按相同keys：NumPy PCG64 seed42，2000次，95% pointwise CI；此区间仅反映用户采样差异，不反映training seed不确定性，不以显著性标记结果类别。
* 时间、GPU小时、峰值显存、有效参数量/表规模据实际运行填入；相同updates不等于相同FLOPs或wall time。预算不足需先修改本协议，不能看Testing后选择性取消或扩充对照。

## 5. 机制分析：三个独立证据包

### M1 命中与排序的可加分解

复用Beauty/seed42 Full+5变体的6份Testing bundle；本单元Full-vs-LIGER hybrid的1对可另作描述性背景。无新增模型forward。
对K=5/10，固定paired用户分为仅variant命中、仅Full命中、双方命中、双方未命中；count及全体用户占比同时报告。令D(r)=1/log2(r+1)，未进TopK为0：
delta NDCG = mean\[仅variant的D(r_v)\] - mean\[仅Full的D(r_f)\] + mean\[双方命中的D(r_v)-D(r_f)\]。
分别给各项数值，核对其和等于独立复算的总体差值；Recall差由两个单边命中数/N复核。双方未命中不补造完整rank。
切片由training-only信息与输入历史确定：target seen/cold；seen频次按training目录商品频次33%/67%分位点（保存实际边界，重复阈值如实保留）；有效历史件数0、1–5、6–10、11–20。各slice报告N/占比/四指标与差；空slice=null。附录二维seen/cold×历史长度保留计数，小样本不做方向判断。
全量结构化表优先，案例按固定user key哈希顺序选每命中类别前3例；不足3如实保留，不以案例替代总体分布。

### M2 固定Full checkpoint的残差作用位置与native评分

同一Full own-best、eval模式、相同输入与权重，无优化器更新；干预历史h∈{0,1}、目录c∈{0,1}，四个视图V11/V10/V01/V00。V11复用现有Full结果，其余3视图新增forward；mask、SID、位置、投影、temperature及历史排除保持一致。h=0必须重新encode生成q0，不能复用q1冒充历史去残差。
V10为q1对无目录残差评分，等于正式定义native评分，但只是诊断视图；不与joint融合、不重新选checkpoint。记录四指标、target rank/relative margin、Top10 overlap、残差/投影norm（按seen频次）、分组N与配对差。数值交互项 I=m11-m10-m01+m00；明确它是固定checkpoint干预的描述量，不是重训消融效应。
cold目录项为零的完整性检查：在固定query下移除目录残差时cold原始logit应一致；即使logit不变，seen竞争者变化也可能改变cold rank，不能预设cold指标不变。历史干预会改变query，cold logit也可随之变化。
cold残差有效值为0、V11复算与正式输出一致、V10与native helper一致必须核验。比较A2的重训结果时分别命名“重训消融”与“固定checkpoint评分干预”。
1个Beauty/seed42 Full checkpoint对应1个diagnosis任务，3个新增全量评分视图，共1任务/3个forward等价全量pass。

### M3 真实前缀下的概率与训练监督测量

固定Beauty/seed42 Full、A1、A4各1个own-best，共3个checkpoint；统一Trainer.test diagnosis，不更新权重、不改变正式部署。全部Testing用户，四层真实目标前缀teacher forcing，记录合法归一化p_gen、p_mass及p_mix（仅Full使用其checkpoint alpha；A1/A4 gate未学习，mix栏置null，不伪装成有效混合策略）。
每层记录目标条件概率/NLL、合法兄弟中目标rank、熵、p_gen与p_mass的JS divergence、双方argmax一致比例、Full alpha。JS在相同合法支持计算；entropy/targetrank遵循统一tie规则。报告各depth和seen/cold×depth的N与分布/mean/quantiles；alpha是模型全局值，无用户级gate相关性叙事。可按p_mass(target)-p_gen(target)的带符号连续值做散点和固定\[-1,-0.1,0,0.1,1\]区间统计，区间不按Testing重选。
核对exp(logp)在合法分支和为1，非法概率0，目标前缀合法；Full mean NLL与原四层平均mixture计算一致；G与合法生成NLL一致。该分析使用标签定义诊断条件，标签不得进入dense Top10预测/候选构造。报告“条件概率与表示的观测”，不报告自由beam路径恢复率，不推断dense检索效率或因果中介比例。
新增3个diagnosis任务/3个全量forward pass。loss日志与alpha历史读取已有W&B run，缺失字段为null，不补造曲线。

## 6. 累计预算与执行依赖

| 工作 | 新train | optimizer updates | 额外独立Val | 新Testing | diagnosis任务 | 新全量forward等价pass |
| -- | -- | -- | -- | -- | -- | -- |
| A1/A2/A3 三个移除变体 | 3 | 150,000 | 0 | 3 | 0 | 0 |
| A4/A5 两个解释对照 | 2 | 100,000 | 0 | 2 | 0 | 0 |
| M1 现有bundle分析 | 0 | 0 | 0 | 0 | 0 | 0 |
| M2 三个新增评分视图 | 0 | 0 | 0 | 0 | 1 | 3 |
| M3 前缀概率诊断 | 0 | 0 | 0 | 0 | 3 | 3 |
| 合计 | 5 | 250,000 | 0 | 5 | 4 | 6 |

Beauty/seed42 Full一组复用，不新增Full train/Test；训练内每500步Validation已计入训练成本，共5×100个计划validation时点。diag pass只表示处理本数据集全量用户，M2三视图、M3 teacher forcing的FLOPs不同，实际资源费用另记；局部工程验证另列成本。启动授权=true（用户2026-10-08要求开始<issue id="80fec645-35e3-4634-8123-107777b1a860" href="https://linear.app/baymax104/issue/BMX-117/copmrec-v53-单数据集单seed消融与机制实证">BMX-117</issue>）；training started=5/completed=5，Testing started=5/completed=5；diagnosis started=4/completed=4，工程attempt=5（含1次M2序列化失败、该失败无交付）；M1 bundle分析另记started=1/completed=1、模型forward=0。该授权限定现有固定计划。固定累计计划，三个实验问题合计五次训练，不增加多数据集、多seed消融。

本M3的单数据集单seed观测为该条件下的组件证据；整体主结果已有多seed验证，不能据此声称组件差值也已获得多seed验证。不自动扩大消融范围。

先准备独立variant与diagnosis契约→CPU单元检查/compose/脚本检查→可复制命令→按用户明确授权在独立tmux启动各变体；A4/A5可与对应移除臂并行排期，执行顺序不依赖结果方向。M1可先读取本单元Full和LIGER背景，最终组件表等待5个变体Testing；M2依赖Full来源及诊断入口；M3依赖Full/A1/A4 own-best及诊断入口。本固定依赖链已完整执行，五个Training/Testing、M1/M2/M3全部Done；所有精确来源与完整性核验见末尾完成回执。

## 7. 实现准备与来源核验

用户已授权补齐必要入口并验证命令，并于2026-10-08明确授权开始<issue id="80fec645-35e3-4634-8123-107777b1a860" href="https://linear.app/baymax104/issue/BMX-117/copmrec-v53-单数据集单seed消融与机制实证">BMX-117</issue>；训练启动后报告run ID、不持续监控；来源已具备的非训练实验完整执行。已实现专用 CoPMRecAblation、五种 loss/冻结/optimizer/checkpoint 契约及 hits/residual/prefix 三类 Trainer.test 诊断。正式 Full 入口与发布身份保留；变体不能冒充 Full，跨臂 checkpoint 拒绝。根目录新增 copmrec_ablation_train.sh / copmrec_ablation_inference.sh / copmrec_diagnosis.sh，保留统一 src.main、dry-run/notes/extra override。

8个实验 issue 已补入15个独立命令块。W&B group 采用现有 `paper_<阶段>_<方法>_<数据集>` 格式：paper_ablation_copmrec_beauty（五臂 train/Testing共用），paper_mechanism_copmrec_beauty（全部诊断）；variant/seed/stage 在 config/tags/notes 中区分。训练物理 GPU0,1 → local \[0,1\]，Testing/诊断物理GPU0 → local \[0\]；资源仅为可调整示例。

Full URI/SHA/输出与五臂自己的首次validation-best不可变URI/SHA、五份Testing输出已全部精确登记和核验；各issue实际命令保持quoted assignment与double-quoted变量引用，未以Full、其他臂或last文件替代选点。CPU公共初值/RNG、损失/梯度/冻结、完整恢复/跨臂拒绝、真实 Hydra 实例化、结构化产物、脚本语法/引用/空值/override及原样 issue 命令 compose 检查通过；250项相关回归通过，末次诊断改动后15项聚焦复验通过；Ruff/OpenSpec strict通过。准备阶段未进行真实数据dry-run；执行阶段已完成真实NCCL双rank五臂50k训练、五份单卡Testing及M1/M2/M3全量核验。训练原始source42f0保留，既有两处metadata序列化修复的运行source5e95单独登记；250k/100val每臂与own-best严格恢复/输入字节/输出指标/统计均已独立审计。

实现与命令回执位于 GRID docs/evidence/copmrec-m3-preparation-20261008/；代码通过既有三会话 Mutagen flush，同步状态均 Watching for changes，无 conflict。实际运行source_sha256=42f0d7c7dce0a09961e0c8c23f9ba35982abfb796d13952c3f17eaeba17c9f2e、origin=verified，各run已发布code Artifact；Beauty525分片manifest SHA256=46fbe99589b809ec8b7b9c21a815e4757c7a20d57664342ba8e525ecad8cb0c2；SID/embedding digest匹配Full，Testing175分片与正式Full逐字一致。M2包括V11复现核对的额外评分开销，不将其计为独立新实验。

## 8. Full参考来源登记

本M3固定Beauty/seed42，Full精确引用来自本地正式完成回执，供执行前按Artifact/文件SHA核对；新变体不得从它初始化。Full既有结果不作为M3新实验结果。上游固定引用：Beauty SID dq77e3wo/content3jtt9mpa；使用merged_predictions_tensor.pt，字段role分别semantic_id/semantic_embedding。上游digest与data manifest在每单元运行前填入，不从run名推定相同。

| Dataset | Seed | Full issue | Train run | Testing run | 精确 best URI | SHA256 |
| -- | -- | -- | -- | -- | -- | -- |
| beauty | 42 | <issue id="f6fdd31f-3886-40f3-819d-a715a9fd9daf" href="https://linear.app/baymax104/issue/BMX-120/主结果copmrec-beauty-seed42">BMX-120</issue> | gshpyn49 | vosmuihm | `wandb://baymaxam/GRID/gshpyn49?role=checkpoint&alias=v0&file=checkpoint_epoch=000_step=047500.ckpt` | `a64ca93d49b5bb9ead7b4af803091346ca62af5299cd89266ac5a2acce3894c4` |

## 9. 统一实验 issue 内容模板

所有A/M实验issue使用相同十个标题，具体臂、范围、成本和观测字段填入对应内容：

 1. 协议与状态：protocol_id、类型、variant/analysis、实际执行状态、implementation、launch_authorization；未观测结果为null。
 2. 实证问题与解释边界：要测量的变量与比较对象；不预设方向，不写结果类别或路线决策。
 3. 对照与干预：精确loss/路由/诊断定义及唯一变化；Full与本臂不可混淆。
 4. 数据、seed与来源：Beauty/seed42单一单元；完整keys、labels、manifests、input digests、own-best URI/SHA、runtime snapshot。
 5. 固定协议：预算、optimizer、raw选点、history eligibility、dense scoring、ties及统一指标。
 6. 执行步骤与命令：已实现真实入口、CPU/compose/脚本验证及独立命令；实际own-best/output字段完整，按运行阶段核对来源、字节身份、指标和统计。
 7. 观测指标与统计：绝对值、预先定义差值、CI、N、切片规则和缺失字段。
 8. 成本与依赖：具体运行数量、累计更新数、额外Val、Testing/diagnosis/forward数；runtime记录其原始口径，分配卡数换算与active GPU实测区分；未测峰值或active GPU资源=null。
 9. 结果登记：固定单元的每variant/view/slice一行，执行前未观测的run/artifact/指标/差值/CI/工程核验字段为null，执行后登记实测原值；空组、未测资源和结构上不适用字段继续null，保留运行异常记录，不预填方向性结论。
10. 交付标准：预定范围与结构化结果完整、来源/合法输出/计算可复核、图表可回溯；与方向/显著性无关。

结果表公共键：protocol_id、experiment_id、dataset、training_seed、variant_or_view、training_run、best_uri、best_sha256、prediction_or_diagnosis_run、artifact_or_local_manifest、source_sha256、input_digest、data_manifest、slice、n、observed_value、reference_value、signed_difference、ci_low、ci_high、runtime_seconds、gpu_hours、peak_memory、data_checks、notes。未执行一律null；数据核验描述工程完整性，不能混成方法好坏判断。

## 10. Linear 实验索引

| 实验 | Issue | 新执行量 |
| -- | -- | -- |
| A1 | [BMX-122](<https://linear.app/baymax104/issue/BMX-122/copmrec-v53-%E6%B6%88%E8%9E%8Da1-%E5%8E%BB%E9%99%A4-mixture-nllbeauty-seed42>) | 1 train / 50k / 1 Testing |
| A2 | [BMX-123](<https://linear.app/baymax104/issue/BMX-123/copmrec-v53-%E6%B6%88%E8%9E%8Da2-%E5%8E%BB%E9%99%A4%E5%8E%86%E5%8F%B2%E4%B8%8E%E7%9B%AE%E5%BD%95%E6%AE%8B%E5%B7%AEbeauty-seed42>) | 1 train / 50k / 1 Testing |
| A3 | [BMX-129](<https://linear.app/baymax104/issue/BMX-129/copmrec-v53-%E6%B6%88%E8%9E%8Da3-%E5%8E%BB%E9%99%A4-native-view-cebeauty-seed42>) | 1 train / 50k / 1 Testing |
| A4 | [BMX-142](<https://linear.app/baymax104/issue/BMX-142/copmrec-v53-%E6%B6%88%E8%9E%8Da4-%E7%94%A8%E5%90%88%E6%B3%95%E7%94%9F%E6%88%90-nll-%E6%9B%BF%E6%8D%A2-mixture-nllbeauty-seed42>) | 1 train / 50k / 1 Testing |
| A5 | [BMX-143](<https://linear.app/baymax104/issue/BMX-143/copmrec-v53-%E6%B6%88%E8%9E%8Da5-%E7%94%A8%E9%A2%9D%E5%A4%96-joint-ce-%E6%9B%BF%E6%8D%A2-native-cebeauty-seed42>) | 1 train / 50k / 1 Testing |
| M1 | [BMX-144](<https://linear.app/baymax104/issue/BMX-144/copmrec-v53-%E6%9C%BA%E5%88%B6m1-%E5%91%BD%E4%B8%AD%E6%8E%92%E5%90%8D%E4%B8%8E%E6%95%B0%E6%8D%AE%E5%88%87%E7%89%87%E7%9A%84%E5%8F%AF%E5%8A%A0%E5%88%86%E8%A7%A3>) | 0 forward / 复用6份bundle |
| M2 | [BMX-145](<https://linear.app/baymax104/issue/BMX-145/copmrec-v53-%E6%9C%BA%E5%88%B6m2-%E5%8E%86%E5%8F%B2%E7%9B%AE%E5%BD%95%E6%AE%8B%E5%B7%AE-22-%E8%AF%84%E5%88%86%E5%B9%B2%E9%A2%84>) | 1 diagnosis / 3新增评分pass |
| M3 | [BMX-146](<https://linear.app/baymax104/issue/BMX-146/copmrec-v53-%E6%9C%BA%E5%88%B6m3-%E9%80%90%E5%B1%82%E5%89%8D%E7%BC%80%E6%A6%82%E7%8E%87%E4%B8%8E%E7%9B%91%E7%9D%A3%E5%BD%A2%E5%BC%8F%E6%B5%8B%E9%87%8F>) | 3 diagnosis / 3前缀pass |

## 2026-10-08完整执行与核验回执

五个scratch训练均实际50,000 updates，共250,000；每臂100个val500点，从各臂完整raw val/ndcg@10选首次最高own-best。5/5单卡Testing、4/4正式diagnosis及M1零forward bundle分析已全量完成并独立核验。Full Training/Testing及既有Full prefix复用；未新增训练、seed、数据集或参数扫描。全部八个子issue Done，完成只按来源/预定观测/计算完整性，与数值方向或显著性无关。

| Issue / arm | Training run | Own-best step | Testing run | Test GPU → local |
| -- | -- | -- | -- | -- |
| [BMX-122](<https://linear.app/baymax104/issue/BMX-122>) / no_mixture | [m3frgiim](<https://wandb.ai/baymaxam/GRID/runs/m3frgiim>) | 48500 | [m3veerfx](<https://wandb.ai/baymaxam/GRID/runs/m3veerfx>) | 0 → \[0\] |
| [BMX-123](<https://linear.app/baymax104/issue/BMX-123>) / no_residual | [m364gahb](<https://wandb.ai/baymaxam/GRID/runs/m364gahb>) | 47000 | [m3bk48w2](<https://wandb.ai/baymaxam/GRID/runs/m3bk48w2>) | 2 → \[0\] |
| [BMX-129](<https://linear.app/baymax104/issue/BMX-129>) / no_native | [m3td2jxc](<https://wandb.ai/baymaxam/GRID/runs/m3td2jxc>) | 41000 | [m3xm4hcb](<https://wandb.ai/baymaxam/GRID/runs/m3xm4hcb>) | 3 → \[0\] |
| [BMX-142](<https://linear.app/baymax104/issue/BMX-142>) / legal_generation | [m3odfrrh](<https://wandb.ai/baymaxam/GRID/runs/m3odfrrh>) | 46000 | [m3sfptwc](<https://wandb.ai/baymaxam/GRID/runs/m3sfptwc>) | 4 → \[0\] |
| [BMX-143](<https://linear.app/baymax104/issue/BMX-143>) / joint_ce_replace | [m3kq1tuc](<https://wandb.ai/baymaxam/GRID/runs/m3kq1tuc>) | 43000 | [m3hycech](<https://wandb.ai/baymaxam/GRID/runs/m3hycech>) | 5 → \[0\] |

Beauty/training seed42；N=22363，四指标原值：

| Arm | Recall@5 | NDCG@5 | Recall@10 | NDCG@10 |
| -- | -- | -- | -- | -- |
| Full | 0.05334705 | 0.03659647 | 0.07776238 | 0.04443310 |
| A1 | 0.05249743 | 0.03596258 | 0.07731521 | 0.04394472 |
| A2 | 0.05187139 | 0.03453895 | 0.07981934 | 0.04348076 |
| A3 | 0.05477798 | 0.03755917 | 0.07905916 | 0.04542925 |
| A4 | 0.05146894 | 0.03516278 | 0.07646559 | 0.04324066 |
| A5 | 0.05263158 | 0.03653958 | 0.07825426 | 0.04485631 |

全部用户NDCG@10配对观测（variant−Full；95% pointwise CI，PCG64 seed42/2000rep）：

| Arm | 数值差 | CI |
| -- | -- | -- |
| A1 | \-0.00048838 | \[-0.00122374, 0.00024884\] |
| A2 | \-0.00095234 | \[-0.00306273, 0.00104712\] |
| A3 | +0.00099615 | \[-0.00026333, 0.00237374\] |
| A4 | \-0.00119244 | \[-0.00199778, -0.00036154\] |
| A5 | +0.00042321 | \[-0.00087547, 0.00175918\] |

M1/<issue id="cec62ed8-06e4-4952-ac08-7a61bb9f788a" href="https://linear.app/baymax104/issue/BMX-144/copmrec-v53-机制m1-命中排名与数据切片的可加分解">BMX-144</issue>：[m3hitc8p](<https://wandb.ai/baymaxam/GRID/runs/m3hitc8p>)，6 arms/134178 user-variant/114 slice/608 metric-CI/48 hit counts/132固定哈希案例，K5/10三贡献可加核验通过；包括预定5Full配对、A4−A1/A5−A3和Full-self数值检查。14项最终Artifact文件身份passed；`copmrec_beauty_seed42_hits_full_diagnosis-analysis:v0`，digest=`f50b595fd764b9fb495768d678d20bcb`。模型forward=0，GPU0→local\[0\]，独立tmux。

M2/<issue id="5e26766f-59ac-48d0-b676-5874e98a1de2" href="https://linear.app/baymax104/issue/BMX-145/copmrec-v53-机制m2-历史目录残差-22-评分干预">BMX-145</issue>：[717vgnkn](<https://wandb.ai/baymaxam/GRID/runs/717vgnkn>)，N22363/4 views/76 slice/304 CI/5目录范数组；V11逐key精确复现Full，其余三视图及数值交互独立复算，12项最终文件身份通过。仅复用前轮已完成证据，无本轮重跑。

M3/<issue id="c28c8adb-4c8f-4bd8-b5d1-3585e862d8e2" href="https://linear.app/baymax104/issue/BMX-146/copmrec-v53-机制m3-逐层前缀概率与监督形式测量">BMX-146</issue>：Full [j2rworuj](<https://wandb.ai/baymaxam/GRID/runs/j2rworuj>)复用；A1 [5murlou7](<https://wandb.ai/baymaxam/GRID/runs/5murlou7>)、A4 [pltnvpjf](<https://wandb.ai/baymaxam/GRID/runs/pltnvpjf>)在物理GPU1→local\[0\]顺序、各独立tmux执行。三来源各22363×4=89452记录；A1/A4各92 slice/8最终文件身份核对通过。Full全局alpha=0.9857481122；A1/A4未学习gate，alpha/mixed观测null。全部原值、分位数、N/切片和连续概率差图已交付；teacher forcing不用于自由beam恢复或dense因果中介结论。

源代码/数据/产物：训练runtime304文件SHA=`42f0d7c7dce0a09961e0c8c23f9ba35982abfb796d13952c3f17eaeba17c9f2e`；Testing/诊断SHA=`5e95cf7e87b28b730976b5a6d279acdc7aca3a270a08dd5e4519ce1fc6624867`、origin verified。差异仅两处既有metadata序列化修复，训练/恢复/评分计算源逐字匹配。原训练source archive与新运行身份分别保留，不回填历史来源。525 Beauty分片SHA manifest=`46fbe99589b809ec8b7b9c21a815e4757c7a20d57664342ba8e525ecad8cb0c2`；22363 user/target SHA=`55e2602110f989565aa6c5fc00bef99107a17d3d11222c06c46874b9df50e016`。所有bundle合法唯一Top10/历史排除、4metrics/own-best lineage、cloud manifest与node1文件path/size/MD5/SHA已核验。W&B最终输出为node1文件引用，最终文件保留；code archive发布字节已核对。

资源/异常边界：五训练W&B原始tracked runtime合计119367秒；双卡分配时长换算66.315 GPU小时，口径为run墙钟×分配卡数，active GPU时长和峰值显存未记录=null。50k终态完整checkpoint未留存，last保存各自best；实际预算由完整history/100val/log核验。前轮M2 ls14h0dw序列化失败保留、completion_credit=false；正式diagnosis4完成，工程attempt5含该1异常；M1 bundle分析另计1，工程复现V11/失败开销单列，不改变训练预算。

可追溯证据入口：GRID `docs/evidence/copmrec-m3-completion-20261008/README.md`。`training-audit-summary.json` / `training-verified.json` / `testing-audit-summary.json` / `m1-independent-audit.json` / `prefix-three-source-summary.json`及前轮M2/Fullprefix独立审计；所有实际命令、不可变own-best URI/SHA、输出URI/digest/SHA与tmux已补进各issue统一十章模板。论文表图CSV/LaTeX/PNG/PDF及22项hash manifest在`figures/`，视觉QA passed。仅报告实证原值、带符号差、区间、N、边界；单seed用户bootstrap不表示跨训练seed稳定性，缺失资源保留null。
