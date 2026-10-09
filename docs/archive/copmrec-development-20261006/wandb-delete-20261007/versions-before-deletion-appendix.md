# CoPMRec 历史版本：做法、开发效果与证据边界

## 当前版本与历史结果的关系

2026-10-06 用户指定 **v5.3 为正式 CoPMRec**。本页记录开发演进，不为新正式实验填入已完成 run。v5.3 实现保留，而其开发训练 `hqw189d2`、完整 Validation `6u6g62hk` 与其他开发运行共同归档。LIGER 是主基线，v5.2 等是内部演进对照；辅助 CE 对 v5.2 的增量不确定，不抵消完整 v5.3 对 LIGER 的开发收益。

除另有注明，下面的结果均为 Beauty、seed 42。各表明确标注 split、资格规则及部署方式；跨表指标不能直接比较。报告、结构化数据和 checkpoint 标识保留原事实，旧门槛与阶段建议只表示当时决定。

## v0：内容条件前缀概率混合

v0 对完整目录内容 logits 按 SID 子树聚合 mass，获得合法子节点的条件概率，再与生成概率逐层混合：`p_mix=(1-alpha)*p_gen+alpha*p_content_prefix_mass`。alpha 是全局可学习标量的 sigmoid，通过 mixture NLL 学习。联合目标为 SID CE、content CE、mixture NLL，三项权重均为 1。原训练在完整 Evaluation 上按 dense NDCG@10 选点；部署为一次 beam20 加全部 cold 商品，最后按 content 分数取 Top10。

真实开发训练 `7y54j4m6` 的 best 为 48000；原推理 `6dspa7e3` 的 checkpoint alpha 为 `0.8132556080818176`。该模型的全目录 logits 计算与训练 content CE 的 cold fill 支持集应区分，不能用后来的 v5.2 全目录 CE 回写早期训练协议。

| 原 Testing，无输入历史排除 | run | Recall@10 | NDCG@10 |
|---|---|---:|---:|
| LIGER dense trace | `042139al` | 0.071412601172 | 0.036390710152 |
| v0 hybrid／content 终排 | `6dspa7e3` | 0.071904485087 | 0.036448314264 |
| v0 自身全目录 dense | 同 run 的 dense trace | 0.072217502124 | 0.036526060722 |

v0 对 LIGER dense 的 R10／N10 点增益约为 **+0.689%／+0.158%**，不能称稳定约 1% 增益。v0 有 97 个自身 dense Top10 目标未入候选，同时有 90 个超出 dense Top10 的最终命中，净少 7 个；候选恢复与保留生成侧增量存在取舍。

来源：[v0 配置恢复](history-evidence/copmrec-v0-config-unification.md)、[原配置](history-evidence/evidence/copmrec-v0-config-unification-20261004/historical-v0-config.json)、[原输出指标](history-evidence/evidence/copmrec-v3-testing-zy8946l2-20261004/results.json)。早期 checkpoint／输出曾做字节一致的引用迁移，该迁移不补造历史训练源码。

### 原 Milestone 2 九单元主矩阵

下面是重置前 Linear 中记录的 v0 learned-mass 主矩阵，原协议为双卡50k、own dense Val N10 选点、单卡 hybrid Testing／beam20。**它们曾经完成旧正式矩阵，但新 v5.3 计划不能沿用旧 Done 或 run。** 数值按本次保存的 Linear 原描述精度记录，本归档没有再次下载全部历史输出做独立复算。

| 原 issue | 数据集／seed | v0 训练 run | Testing run | best step | R10 | N10 |
|---|---|---|---|---:|---:|---:|
| BMX-120 | Beauty／42 | `7y54j4m6` | `6dspa7e3` | 48000 | 0.0719045 | 0.0364483 |
| BMX-121 | Beauty／200 | `gdxds9oi` | `8xcw35ag` | 46000 | 0.0719939 | 0.0363544 |
| BMX-118 | Beauty／2026 | `yk7j9k5z` | `19i5khys` | 45000 | 0.0712337 | 0.0359139 |
| BMX-119 | Sports／42 | `0he4of79` | `siiokyhl` | 43000 | 0.0390190460 | 0.0201415585 |
| BMX-15 | Sports／200 | `tamqgytq` | `q2du3vc5` | 48500 | 0.0382324850 | 0.0196746179 |
| BMX-16 | Sports／2026 | `p9qj0ccg` | `z8q74kll` | 46000 | 0.0387943143 | 0.0202285927 |
| BMX-17 | Toys／42 | `9dzi0g5i` | `valgk6yg` | 46500 | 0.0735112302 | 0.0360008125 |
| BMX-18 | Toys／200 | `k1fzr0jz` | `p62too4c` | 45500 | 0.0712961055 | 0.0357504652 |
| BMX-19 | Toys／2026 | `b6t4a1ws` | `ohepscqg` | 48000 | 0.0726869977 | 0.0357861277 |

### 原 Milestone 3 内容引导与聚合机制

原机制对照只改变固定 checkpoint 下的候选规则：Legal 不用内容引导，Max 按后代最高内容值聚合，Mass 按后代概率质量聚合。以下是旧 seed42 Testing；Mass 与主矩阵 run 相同，注册表去重，不计新增训练。

| 原 issue／数据集 | 分支 | 推理 run | R10 | N10 | 目标候选覆盖率 |
|---|---|---|---:|---:|---:|
| BMX-122／Beauty | Legal | `jpqimr2j` | 0.056208917 | 0.030043652 | 7.85226% |
| BMX-122／Beauty | Max | `sgmu9l3l` | 0.072128069 | 0.036475926 | 10.74543% |
| BMX-122／Beauty | Mass | `6dspa7e3` | 0.071904485 | 0.036448314 | 11.00478% |
| BMX-123／Sports | Legal | `sd90tlbj` | 0.031097253 | 0.016761569 | 4.87949% |
| BMX-123／Sports | Max | `z8ymlal9` | 0.038513400 | 0.019969997 | 6.33743% |
| BMX-123／Sports | Mass | `siiokyhl` | 0.039019046 | 0.020141559 | 6.58183% |
| BMX-129／Toys | Legal | `bzgzloh5` | 0.052699361 | 0.027915361 | 8.07748% |
| BMX-129／Toys | Max | `1raiqpvi` | 0.070368844 | 0.034907636 | 11.47229% |
| BMX-129／Toys | Mass | `valgk6yg` | 0.073511230 | 0.036000813 | 12.25531% |

这些观测解释 v0 内容引导的设计来源，不直接证明 dense 部署的 v5.3 收益来自恢复 beam 候选。旧机制 issue 和全部推理 run 的完成状态也随新计划重置。

### 原 LIGER original-hybrid 比较矩阵

下表是旧 Milestone 4 记录的 original-hybrid 输出，**与前文 LIGER dense trace 指标不同**。它们保留作历史基线证据，默认不属于 CoPMRec run 清理目标，不能作为新 dense 核心对照已完成。

| 原 issue | 数据集／seed | LIGER 训练 run | Testing run | R10 | N10 |
|---|---|---|---|---:|---:|
| BMX-133 | Beauty／42 | `35ig0tz6` | `042139al` | 0.0559406 | 0.0298675 |
| BMX-134 | Beauty／200 | `20qs5i9w` | `0b2g9ktg` | 0.0559853 | 0.0297251 |
| BMX-135 | Beauty／2026 | `xeg0lbww` | `eg0morwo` | 0.0551804 | 0.0301941 |
| BMX-136 | Sports／42 | `x69dhrk6` | `p81iez3h` | 0.0282319231 | 0.0155035092 |
| BMX-137 | Sports／200 | `wnpv48oe` | `gppl7usj` | 0.0283161975 | 0.0155089980 |
| BMX-138 | Sports／2026 | `9chlbjns` | `oqdjukup` | 0.0293836732 | 0.0162123531 |
| BMX-139 | Toys／42 | `l99i8w61` | `5vd0g7ak` | 0.0472903359 | 0.0258020953 |
| BMX-140 | Toys／200 | `7lbsl22u` | `b591uoen` | 0.0492994024 | 0.0268886263 |
| BMX-141 | Toys／2026 | `rsz30j8q` | `8hjgn6b5` | 0.0485266845 | 0.0262051905 |

三组表的来源为 [重置前 Linear 26项完整快照](history-evidence/linear-full-before-reset.json)，精确原始值与各 issue 的定位写入 [run 注册表](development-run-registry.json) 的 `former_matrix_units`／`additional_evidence`。当时 issue 的 Done／有效表述原样归档，不转移到新的 v5.3 实验。

### 固定 alpha 的两类尝试

- **v0 仅推理固定 0.813**：准备后被用户取消，没有完整推理 run，也没有效果。见[取消记录](history-evidence/copmrec-v0-fixed-alpha-0813-testing.md)。
- **v0 从头训练固定 0.8**：训练 `ncyol9n0`、dense best44500，Testing `x70hphts`；R10=`0.069713365827`、N10=`0.035501181331`，对原学习 alpha v0 分别 **−3.05%／−2.60%**，两项用户配对区间全负。净少 49 个最终命中，其中自身 dense 命中少 46 个。它改变了完整联合训练轨迹，不能解释成仅推理权重变化。固定训练设置未保留。[结果](history-evidence/copmrec-v0-fixed-alpha08-x70hphts-result.md)、[结构化复算](history-evidence/evidence/copmrec-v0-fixed-alpha08-x70hphts-20261005/results.json)。

## v1／v1.1／v1.2／v2：候选相关性与排序探索

| 版本 | 方法变化 | 实际运行与证据范围 | 当时结论 |
|---|---|---|---|
| v1 | 四路输入 `[h_u,v_i,h_u*v_i,d_ui]` 的相关性 head；content 分数加零初始化残差；基础三 loss 加候选列表 CE，全部推荐参数可训练 | 训练 `uzmkrfoa`；性能诊断／零更新 probe 保存，当前归档未建立可审计完整推荐最终结果 | 实现与性能诊断不能作为效果 |
| v1.1 | 用户／候选跨批评分，chunk256；保持四路 head；随后对排序 loss 作 training 梯度校准 | 原训练 `f91njtjx`；等权／校准各 1000 更新为 `68h3e3eq`／`mg0nrz9a`，固定 2048 selection 用户诊断；校准从头训练 `57hzkcol` 的保存快照是 running、约 30k，未在本归档升级为终态 | 降权改善短程诊断，未证明完整从头模型胜过 v0／LIGER |
| v1.2 | 简化相关性 head 的尝试；完整旧实现与规格已经撤回归档 | 没有 v1.2 效果 run；`57hzkcol` 实际 target 是 v1.1，不因快照在 v1.2 文件夹就改版本 | 撤回不是效果否证 |
| v2 | 独立从 v0 派生，d-only head，LayerNorm，自然候选尺度约束，lambda0.01／beta0.5；融合 hybrid 选点 | 训练 `beawrjef`、best42500，完整 Testing `1gu2dsa1` | 当前相关性修正负向，未晋级 |

v1.1 校准权重为 `0.05726763550972437`。固定 2048 用户短程诊断中，等权／校准的最终 hybrid R10 为 `0.006348／0.012695`，N10 为 `0.003617／0.006439`；R 差值区间正，N 差值区间跨零。这是已有 17.5k 等权 checkpoint 的两臂恢复，不是完整推荐结果。

v2 完整 Testing R10=`0.069087331753`、N10=`0.033285145999`；对 v0 为 **−3.92%／−8.68%**，对 LIGER dense 为 **−3.26%／−8.53%**。同候选相关性重排相对 content-only 的 N10 差为 `−0.002330`，区间全负；主要表现为共同命中目标降位。尺度有界不等于排序受保护。

来源：[原版本说明](history-evidence/copmrec-versions.md)、[v1 性能诊断](history-evidence/copmrec-v1-performance-diagnosis.md)、[v1.1 权重诊断](history-evidence/copmrec-loss-weight-verification.md)、[v2 完整结果](history-evidence/copmrec-v2-testing-1gu2dsa1-result.md)。已退役 v1.2 的原实现见 [旧归档](../copmrec-v1-2/README.md)，更早冻结 conversion 路线见 [旧入口退役归档](../copmrec-conversion-entrypoints-20261003/README.md)，不把它们的结论恢复为当前路线依据。

## v3／v3.1：首层恢复与 root 路径补正

v3 独立从真实 v0 派生，将第 1 层内容聚合改成 max，后续三层仍为 mass；训练 mixture NLL 与候选搜索共用 `[max,mass,mass,mass]`。其余训练配置、dense 选点、一次 beam20 加 cold、content 终排保持。训练 `rbha00vx` 的 best43500，alpha=`0.668786346912384`。

v3.1 不重训，复用同 checkpoint。首层准入仍保留 v3 Max；在后续搜索引入一次 root 包络补正，改变跨 root 累计分数。它不是增大候选预算，也不是将目标商品注入候选。

| 完整 Testing，无输入历史排除 | 推理 run | Recall@10 | NDCG@10 | 自身 dense Top10 遗漏 |
|---|---|---:|---:|---|
| v3 | `zy8946l2` | 0.069892232706 | 0.035701458977 | 114，分层 0／73／29／12 |
| v3.1 | `zznw291t` | 0.070384116621 | 0.035827804065 | 81，分层 0／60／11／10 |
| v3.1 仅推理固定 alpha0.813 | `3lb22b0r` | 0.070384116621 | 0.035855579736 | 52，分层 0／44／6／2 |

v3 将**自身** dense Top10 首层遗漏降为零，但后续遗漏增加，最终对 v0 的 R10／N10 为 −2.80%／−2.05%，区间跨零。不能把“自身首层零遗漏”写成恢复全部旧 v0 案例。

v3.1 对 v3 净增 11 个命中（+0.704% R10／+0.354% N10），高内容遗漏净减少 33，同时丢失生成侧增量，整体区间跨零；用户保留其首层机制用于继续开发，未建立相对 LIGER 的稳定整体收益。

固定推理 alpha0.813 再净恢复 29 个高内容目标，却净丢失 29 个超 dense 增量，R10 不变；N10 仅 +0.0775%、区间跨零。保留学习 alpha 默认，没有证明 0.813 最优。

来源：[v3 方法](history-evidence/copmrec-v3.md)、[v3 结果](history-evidence/copmrec-v3-testing-zy8946l2-result.md)、[v3.1 结果](history-evidence/copmrec-v3-1-testing-zznw291t-result.md)、[固定推理 alpha 结果](history-evidence/copmrec-v3-1-alpha0813-testing-3lb22b0r-result.md)。这组 Testing 已用于开发分析，全部归历史，不作为新正式确认。

## v4 系列：协同残差、评分和资格规则

### v4：共享商品协同残差

从 v0 best48000 做 weights-only 续训，新增零初始化的商品残差 `r_i`，表示为 `P(c_i)+seen_i*r_i`。残差共享给历史编码、目录评分、content CE 和 mixture NLL；cold 有效残差为零，alpha 可学习。每臂 6000 更新，重置 optimizer／scheduler，全球 batch256，主干 LR1e-4；不能当作从头 50k。

| 开发设置 | 训练 run／选点 | 完整评价 run | split 与评分 | R10／N10 |
|---|---|---|---|---|
| 残差 LR1e-3 | `z9envq1n`／best5000 | `amwchjkc` 的 dense trace | Validation，无历史排除 | 0.095112462550／0.049114561194 |
| 同来源零残差续训控制 | `5rlfx2np`／best6000 | 仅训练期 own-best 标量 | hybrid raw Val，非独立输出 | 0.092295311391／0.047863930464 |
| 固定 mixed 终排 | 同 `z9envq1n` | `amwchjkc` | Validation，beam+cold | 0.092787193132／0.048091188597 |
| 残差 LR1e-3 dense | 同 `z9envq1n` | `1edkvgk5` | Testing，无历史排除 | 0.075168805616／0.038441483150 |
| 残差 LR2e-3 | `nj9elah1`／best6000 | `mq8hhof5` | Validation，无历史排除 | 0.096006796941／0.049793932567 |

残差 dense Testing 相对旧 LIGER dense 为 **+5.2599%／+5.6354%**，保留正向结果。固定 mixed 终排相对同候选 content 两项区间为负，停止该评分规则。LR2e-3 Validation 对 LIGER 为 **+6.9223%／+7.1520%**，对原残差的增量区间跨零，未证明独立 LR 收益，未单独进入 Testing。[v4 报告](history-evidence/copmrec-v4-autonomous-research.md)

### v4.1：商品 score bias

从 `nj9elah1` best6000 继续两臂各 6000 更新；学习 bias 为 `5ke7nt98`，固定零 bias 控制为 `l3zyr91b`，各自 best6000。完整 Validation 分别为 `o7ycqky2`／`yqsmsdt1`：

| 设置，无历史排除 | R10 | N10 |
|---|---:|---:|
| 学习 bias | 0.098108482762 | 0.050503051973 |
| 零 bias 续训控制 | 0.098779233555 | 0.050426528912 |

学习 bias 对控制的 R10 下降、N10 微增，两项区间跨零，没有独立增量证据；未启动 Testing，不保留 bias 机制。[v4.1 报告](history-evidence/copmrec-v4-1-score-bias-research.md)

### v4.2：固定双 checkpoint logits pooling

固定 `nj9elah1` best6000 与 `l3zyr91b` best6000，各自独立计算 history query 与目录 logits，再做 0.5／0.5 平均；没有新增训练或权重扫描。完整 Validation `i4xwruok` 的 R10=`0.098108482762`、N10=`0.050786806656`，对旧 LIGER 为 **+9.2629%／+9.2886%**。相对固定零 bias 控制的增量区间跨零，pool 不等于单 query 双目录设计。[v4.2 报告](history-evidence/copmrec-v4-2-fixed-logit-pool-research.md)

### v4.3：有效输入历史排除

评分后把有效输入最多 20 件历史商品设为 `−inf`，在剩余完整目录取 Top10，cold 仍可推荐；不使用真实标签决定资格。对 pool 与 LIGER 同时应用相同规则，没有新训练。

| 相同历史资格的完整 Validation | run | R10 | N10 |
|---|---|---:|---:|
| LIGER native50k | `wdms8w77` | 0.096945848053 | 0.054048898557 |
| 固定 pool | `iy3o3z3q` | 0.106515226043 | 0.059607124884 |

pool 对同资格 LIGER 为 **+9.8708%／+10.2837%**。排历史对自身旧 pool 新增 188 命中、丢失 0，保留作为共同评价规则；不能把只给 CoPMRec 过滤的增量当成公平模型收益。[v4.3 报告](history-evidence/copmrec-v4-3-history-exclusion-research.md)

### v4.4：训练 content CE 的历史支持集与控制续训

从固定 source 两臂各 2000 更新：treated `x4ge0y88` 改 content CE 的有效历史支持集，control `hbgj80bh` 保持原支持集；两者各与 source 做固定等权独立 query pool。Validation `artnz7m2`／`rksx8qyh` 分别为 R10=`0.107990877789／0.107856727630`、N10=`0.060568542544／0.060622668610`。treated 对 control +0.1244% R／−0.0893% N，区间跨零，历史 CE 的独立收益未成立。

按预先冻结规则选择 control pool 后，Testing `ws2fx4oi` 为 R10=`0.085543084559`、N10=`0.047429664064`，对同资格旧 50k 单模型 LIGER `vnhmag7v` 为 **+10.7701%／+10.7766%**。两项区间下界正，但 CoPMRec 生产／选择链已消费 64k 更新并使用两个 checkpoint，不能作为公平 50k 单模型优势。[v4.4 报告](history-evidence/copmrec-v4-4-eligible-content-continuation-research.md)

### 后续 LIGER 预算匹配比较

LIGER 新增 A／B 各6000、C 2000 更新：`6an0pxdy`／`eqlhnpgt`／`034h3uvv`。完整 Validation 比较 C single `d649k83e` 与固定 A／C pool `3or6e3p0`，选择 pool 后做唯一 Testing `a6moio4n`。

| 同历史资格，实际累计64k且固定双成员 pool 的 Testing | run | R10 | N10 |
|---|---|---:|---:|
| 预算匹配 LIGER | `a6moio4n` | 0.079282744 | 0.043867107 |
| 固定 CoPMRec control pool | `ws2fx4oi` | 0.085543085 | 0.047429664 |

预算匹配后的剩余收益是 **+7.8962%／+8.1212%**，两项区间下界正；旧双10.77%是原比较事实，不能继续称预算匹配双10%。更新数和样本呈现预算匹配，不等于 FLOPs、GPU 时间或全部搜索成本相同，也未证明 LIGER 完全收敛。[完整比较](history-evidence/liger-budget-matched-continuation-20261005.md)

## v5 至 v5.4：随机初始化、连续50k、单 checkpoint

这组版本推荐模型从随机初始化开始，所有模块从第0步共同训练，无 teacher、外部推荐 checkpoint、optimizer 重启或 pooling。SID 与固定内容向量沿用相同上游。基础配置为 d128、6层 Encoder／Decoder、4×256 SID、历史20、AdamW、主干 peak LR0.0003、残差0.002、weight decay0.035、warmup2500、cosine50000、全球 batch256、FP32、梯度裁剪1。训练期 raw dense Val N10 own best 选点；完整单卡部署评价使用相同输入历史资格。

| 版本 | 对之前方法的变化 | 训练 run／own best | 完整 Validation run | R10 | N10 | 对同资格 LIGER 的 R／N 相对增益 |
|---|---|---|---|---:|---:|---|
| LIGER native50k | 固定核心基线 | `35ig0tz6`／45000 | `wdms8w77` | 0.096945848053 | 0.054048898557 | — |
| v5 | v0 概率联合目标＋历史／目录共享商品残差，单模型从头50k；训练 content CE 仍把 cold logits 替换为−100 | `m50dan21`／41000 | `8w893ra3` | 0.100433752180 | 0.060150846752 | +3.5978%／+11.2897% |
| v5.1 | 同一含残差 history query 对联合目录和无残差目录各算 cosine logits，部署固定0.5／0.5平均，无新参数 | `cm0i584p`／46000 | `w19eutzj` | 0.095380762867 | 0.055845624680 | −1.6144%／+3.3243% |
| v5.2 | 回到单联合目录；训练 content CE 分母包含全部目录 cold logits，训练目标必须 seen | `rha8mrvs`／47000 | `d7lcftto` | 0.101462236730 | 0.061058281375 | +4.6587%／+12.9686% |
| **v5.3** | 在 v5.2 上增加权重1的无目录残差视图 CE；共享 query 和同次 projection／dropout；部署仍只用联合评分 | `hqw189d2`／45000 | `6u6g62hk` | **0.103787506149** | **0.061550587650** | **+7.0572%／+13.8794%** |
| v5.4 | 从 v5.2 派生，取消历史侧显式残差、保留目录侧残差，不加 v5.3 辅助 CE | `cv5wwhck`／46500 | `y9earebw` | 0.090372490274 | 0.050377637914 | −6.7804%／−6.7925% |

所有五次训练均实际消费50000更新，但保存的 best／last 完整状态只到各自最优步，不能说每个 run 都有50000步终态 checkpoint。v5 首次 Validation `tz88ztdn` 在预测前因 W&B lineage API 超时失败，预测数0；其后重试 `8w893ra3` 完成，失败仍归档，不计完整效果。

### 各版既有判断

- **v5**：N10 的配对差值区间为正，R10 区间跨零；保留部分正向证据。它证明共同从头训练有推荐潜力，没有完成原双8%和第二 seed 目标。[报告](history-evidence/copmrec-unified-50k-research.md)
- **v5.1**：对 v5 两指标下降且区间全负，停止固定共享 query 双目录平均。cold 错误槽增多只是观察，不能作为全部损失的因果解释。[阶段决定](history-evidence/copmrec-v5-1-stage-decision-20261006.md)
- **v5.2**：对 LIGER 两项配对区间下界均正；错误 cold Top10 槽由 v5 的6638降到44，但51个 cold-target 命中由6降到0。对 v5 的 R 增量区间跨零，N 增量证据较弱；不能把整体收益全归给完整目录 CE。[阶段决定](history-evidence/copmrec-v5-2-stage-decision-20261006.md)
- **v5.3**：对 LIGER 两项区间下界均正，已有明显整体开发收益。对 v5.2 的点增量为 +2.2918% R／+0.8063% N，区间均跨零；这限制辅助 CE 的单独归因，不改变 v5.3 整体相对 LIGER 的收益。相对 LIGER 新增923／丢失770／净增153命中，共同命中590升位／437降位。[原阶段结果](history-evidence/copmrec-v5-3-stage-decision-20261006.md)
- **v5.4**：对 LIGER 双负、区间均负，停止这项固定 catalog-only 结构；单次结构对照不证明历史残差在所有条件下必不可少。[结题](history-evidence/copmrec-v5-4-execution-20261006.md)

### 正式 CoPMRec 保留的 v5.3 计算契约

历史商品输入为 SID embedding＋内容投影＋seen 商品残差＋位置表示，经共享 Encoder 得到 query `q`。联合目录表示为 `P(c_i)+r_i`，最终分数是 temperature0.07 的归一化 cosine；cold 有效残差为零。

训练四项权重各1：SID CE、完整联合目录 CE、合法前缀 mass mixture NLL、辅助无目录残差视图 CE。辅助视图仅在商品评分侧去掉 `r_i`，history／query 仍包含共享残差；没有第二 Encoder／query／teacher。残差初始化为零时两项目录 CE 相同，故辅助项也增加内容监督总量，不单凭这个开发单臂结果声称视图保护机制已被隔离证明。

正式 dense 部署只计算联合目录分数，应用相同输入历史排除后取 Top10；Decoder、alpha 混合及辅助 CE 不直接参与推理评分。当前结果支持最终推荐改善，不能全部解释为推理 beam 候选恢复。51个 cold-target 用户的命中为0，不宣称冷启动改善。

**本次用户决定提升的是这个完整实现与固定配方，不是把上述开发 run 变成新正式证据。** 新计划以 LIGER 为核心基线，从新运行开始；既有比较、旧预算与旧晋级门槛原样留在历史，不回填未来结果。

## 机器可读注册与核验

[development-run-registry.json](development-run-registry.json) 去重记录65个 CoPMRec 历史 run、27个 LIGER 比较 run；每条都关联明确原证据及快照 SHA256。包含旧九单元主矩阵、机制对照及多数据集原 LIGER 矩阵，均不能用于完成新的正式计划。三数据集9个 embedding／quantizer／SID 与其他方法的 baseline 不列入自动清理范围。

[history-evidence/manifest.json](history-evidence/manifest.json) 记录本次113个逐字节快照。归档没有新模型 forward、重新训练、重新推理或重新 bootstrap；各项数字来自已经保存的结构化审计与当时报告／旧 Linear 描述。
