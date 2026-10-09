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

下表是旧 Milestone 4 记录的 original-hybrid 输出，**与前文 LIGER dense trace 指标不同**。它们保留作历史基线证据，不属于 CoPMRec run 清理目标，不能作为本轮新 LIGER hybrid 正式基线已完成。

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

## 2026-10-07 删除前完整效果登记

用户明确要求删除 CoPMRec 开发阶段 run 与 Artifact。当前清单覆盖 v0–v5.4 的 65 个登记 ID，以及新增核实的 18 个早期 ranker/decoder 探索 ID：82 个现存运行，`uzmkrfoa` 在删除前已不存在。早期路线只保留历史事实，不恢复其研究状态。正式 v5.3 新实验仍为 0 完成，论文主基线为 LIGER hybrid、dense 仅内部对照。

删除前已保存每个现存 run 的完整 config、summary、notes、状态以及 Artifact ID、版本、digest、aliases、元数据、文件清单和生产/消费关系，见[删除前完整快照](wandb-delete-20261007/deletion-plan.json)。指标采用 W&B API 实际 summary 值；训练 summary 是终态记录，不自动等于 Validation-selected best，选点信息以原版本正文及 Artifact metadata 为准。下面不混合 split、支持集或不同排序器计算增益；缺失数值不补零。

243 个关联 Artifact 版本已逐项核实，均属于本轮开发范围且无范围外消费者。其中 8 个早期 joint inference 输出的原创建者已缺失，但当前开发 run 的 logged 关系、显式 task/local-output 元数据与开发协议相互吻合，所有这些身份依据已单列保存。9 个固定上游 run 和 27 个 LIGER comparator 不在删除范围。

### 开发运行索引

| run | 开发版本／路线 | 阶段 | 原 run 名称 |
|---|---|---|---|
| `0he4of79` | v0 | training | liger_sports_joint_train/2026-09-27_17-21-32 |
| `19i5khys` | v0 | testing | liger_beauty_joint_inference/2026-09-26_15-34-40 |
| `1edkvgk5` | v4-residual-lr10-dense | testing | copmrec_beauty_v4_dense_testing/2026-10-05_04-50-14 |
| `1gu2dsa1` | v2 | testing | copmrec_beauty_v2_inference/2026-10-04_13-38-23 |
| `1raiqpvi` | v0-mechanism-max | testing_mechanism_max | liger_toys_joint_inference/2026-09-29_23-02-47 |
| `29o9dx80` | early-ranking-decoder-exploration | train | copmrec_beauty_decoder_competitive_train/2026-10-02_15-08-50 |
| `3lb22b0r` | v3.1-fixed-inference-alpha0813 | testing | copmrec_beauty_v3_1_inference/2026-10-04_23-09-23 |
| `3r954xoo` | early-ranking-decoder-exploration | train | copmrec_beauty_decoder_ndcg_train/2026-10-02_01-12-12 |
| `57hzkcol` | v1.1-calibrated-scratch | training | copmrec_beauty_v1_1_train/2026-10-04_01-13-20 |
| `5ke7nt98` | v4.1-learned-score-bias | training | copmrec_beauty_v4_1_bias_train/2026-10-05_06-02-28 |
| `5rlfx2np` | v4-zero-residual-control | training | copmrec_beauty_v4_control_train/2026-10-05_03-40-23 |
| `68h3e3eq` | v1.1-loss-weight-equal | diagnosis_training | copmrec_loss_weight_verify/equal |
| `6dspa7e3` | v0 | testing | liger_beauty_joint_inference/2026-09-29_19-06-06 |
| `6u6g62hk` | v5.3 | validation | beauty_unified_native_view_ce50k_val_candidate42/2026-10-06_16-12-33 |
| `7n6n6jta` | early-ranking-decoder-exploration | inference | copmrec_beauty_pairwise_ranker_audit/2026-10-01_18-26-04 |
| `7q2njodh` | early-ranking-decoder-exploration | train | copmrec_beauty_decoder_content_preservation_train/2026-10-02_19-55-36 |
| `7wnteuw0` | early-ranking-decoder-exploration | inference | copmrec_beauty_decoder_head_risk_audit/2026-10-02_23-41-18 |
| `7y54j4m6` | v0 | training | liger_beauty_joint_train/2026-09-26_01-55-16 |
| `8w893ra3` | v5 | validation | beauty_unified50k_val_candidate42/2026-10-05_23-12-45 |
| `8xcw35ag` | v0 | testing | liger_beauty_joint_inference/2026-09-26_23-10-25 |
| `9dzi0g5i` | v0 | training | liger_toys_joint_train/2026-09-29_15-36-32 |
| `amwchjkc` | v4-fixed-mixed-ranking | validation | copmrec_beauty_v4_mixed_validation/2026-10-05_04-24-16 |
| `artnz7m2` | v4.4-history-ce-treated-pool | validation | beauty_eligible_treated_validation/2026-10-05_09-46-14 |
| `b6t4a1ws` | v0 | training | liger_toys_joint_train/2026-09-28_20-43-46 |
| `beawrjef` | v2 | training | copmrec_beauty_v2_train/2026-10-04_05-08-06 |
| `bzgzloh5` | v0-mechanism-legal | testing_mechanism_legal | liger_toys_joint_inference/2026-09-29_23-02-19 |
| `cm0i584p` | v5.1 | training | copmrec_beauty_unified_dualview_50k_seed42_train/2026-10-06_00-17-29 |
| `cv5wwhck` | v5.4 | training | copmrec_beauty_unified_catalog_only_residual_50k_seed42_train/2026-10-06_19-29-52 |
| `d7lcftto` | v5.2 | validation | beauty_unified_full_catalog_ce50k_val_candidate42/2026-10-06_13-38-42 |
| `f91njtjx` | v1.1 | training | copmrec_beauty_v1_1_train/2026-10-03_22-32-12 |
| `gdxds9oi` | v0 | training | liger_beauty_joint_train/2026-09-26_01-44-03 |
| `hbgj80bh` | v4.4-history-ce-control | training | copmrec_beauty_v4_4_eligible_control_train/2026-10-05_09-29-31 |
| `hkokjad0` | early-ranking-decoder-exploration | train | copmrec_beauty_decoder_nll_train/2026-10-02_00-28-16 |
| `hpfflo20` | early-ranking-decoder-exploration | inference | copmrec_beauty_ranking_evaluation/2026-10-01_16-11-40 |
| `hqw189d2` | v5.3 | training | copmrec_beauty_unified_native_view_ce_50k_seed42_train/2026-10-06_14-19-19 |
| `i05boy01` | early-ranking-decoder-exploration | inference | copmrec_beauty_ranker_training_cache/2026-10-01_17-31-01 |
| `i4xwruok` | v4.2-fixed-logit-pool | validation | copmrec_beauty_v4_2_pool_validation/2026-10-05_07-23-19 |
| `iy3o3z3q` | v4.3-history-eligible-pool | validation | beauty_history_pool_validation/2026-10-05_08-20-51 |
| `jpqimr2j` | v0-mechanism-legal | testing_mechanism_legal | liger_beauty_joint_inference/2026-09-29_23-15-00 |
| `k1fzr0jz` | v0 | training | liger_toys_joint_train/2026-09-29_00-10-15 |
| `klqyxo8m` | early-ranking-decoder-exploration | train | copmrec_beauty_decoder_dual_preservation_train/2026-10-02_21-18-13 |
| `l3zyr91b` | v4.1-zero-score-bias-control | training | copmrec_beauty_v4_1_bias_control_train/2026-10-05_06-15-57 |
| `m50dan21` | v5 | training | copmrec_beauty_unified_50k_seed42_train/2026-10-05_20-25-16 |
| `mg0nrz9a` | v1.1-loss-weight-calibrated | diagnosis_training | copmrec_loss_weight_verify/calibrated |
| `mq8hhof5` | v4-residual-lr20-dense | validation | copmrec_beauty_v4_residual_lr20_validation/2026-10-05_05-15-35 |
| `msut1s0r` | early-ranking-decoder-exploration | train | copmrec_beauty_decoder_boundary_train/2026-10-03_00-58-39 |
| `ncyol9n0` | v0-fixed-training-alpha08 | training | copmrec_beauty_v0_fixed_alpha08_train/2026-10-05_00-15-04 |
| `nj9elah1` | v4-residual-lr20 | training | copmrec_beauty_v4_residual_lr20_train/2026-10-05_05-01-06 |
| `nwxzx1qs` | early-ranking-decoder-exploration | inference | copmrec_beauty_decoder_content_preservation_audit/2026-10-02_20-24-09 |
| `o7ycqky2` | v4.1-learned-score-bias | validation | copmrec_beauty_v4_1_bias_validation/2026-10-05_06-29-35 |
| `ohepscqg` | v0 | testing | liger_toys_joint_inference/2026-09-28_23-24-55 |
| `p62too4c` | v0 | testing | liger_toys_joint_inference/2026-09-29_03-19-04 |
| `p9czcxbg` | early-ranking-decoder-exploration | inference | copmrec_beauty_decoder_boundary_audit/2026-10-03_01-30-06 |
| `p9qj0ccg` | v0 | training | liger_sports_joint_train/2026-09-27_03-42-45 |
| `pvhjwp9q` | early-ranking-decoder-exploration | inference | copmrec_beauty_decoder_dual_preservation_audit/2026-10-02_21-42-40 |
| `q2du3vc5` | v0 | testing | liger_sports_joint_inference/2026-09-28_01-24-23 |
| `rbha00vx` | v3 | training | copmrec_beauty_v3_train/2026-10-04_16-56-50 |
| `rha8mrvs` | v5.2 | training | copmrec_beauty_unified_full_catalog_ce_50k_seed42_train/2026-10-06_03-05-10 |
| `ri2813bi` | early-ranking-decoder-exploration | inference | copmrec_beauty_decoder_dual_audit/2026-10-02_02-09-25 |
| `rksx8qyh` | v4.4-history-ce-control-pool | validation | beauty_eligible_control_validation/2026-10-05_09-49-53 |
| `s9nrva4m` | early-ranking-decoder-exploration | train | copmrec_beauty_decoder_competitive_train/2026-10-02_03-26-34 |
| `sd90tlbj` | v0-mechanism-legal | testing_mechanism_legal | liger_sports_joint_inference/2026-09-29_22-40-00 |
| `sgmu9l3l` | v0-mechanism-max | testing_mechanism_max | liger_beauty_joint_inference/2026-09-29_23-15-12 |
| `siiokyhl` | v0 | testing | liger_sports_joint_inference/2026-09-29_19-09-51 |
| `sw8llh9g` | early-ranking-decoder-exploration | train | copmrec_beauty_pairwise_ranker_train/2026-10-01_18-12-00 |
| `szuw834d` | early-ranking-decoder-exploration | train | copmrec_beauty_decoder_head_risk_train/2026-10-02_23-00-00 |
| `tamqgytq` | v0 | training | liger_sports_joint_train/2026-09-27_21-45-48 |
| `tz88ztdn` | v5 | validation_startup_failed | beauty_unified50k_val_candidate42/2026-10-05_22-43-35 |
| `uzmkrfoa` | v1 | already_missing | 删除前已缺失 |
| `valgk6yg` | v0 | testing | liger_toys_joint_inference/2026-09-29_22-01-20 |
| `w19eutzj` | v5.1 | validation | beauty_unified_dualview50k_val_candidate42/2026-10-06_02-15-58 |
| `ws2fx4oi` | v4.4-history-ce-control-pool | testing | beauty_eligible_winner_testing/2026-10-05_10-03-07 |
| `x4ge0y88` | v4.4-history-ce-treated | training | copmrec_beauty_v4_4_eligible_treated_train/2026-10-05_09-24-22 |
| `x70hphts` | v0-fixed-training-alpha08 | testing | copmrec_beauty_v0_fixed_alpha08_inference/2026-10-05_02-42-00 |
| `y9earebw` | v5.4 | validation | beauty_unified_catalog_only_residual50k_val_candidate42/2026-10-06_21-19-34 |
| `yk7j9k5z` | v0 | training | liger_beauty_joint_train/2026-09-26_03-35-14 |
| `yqsmsdt1` | v4.1-zero-score-bias-control | validation | copmrec_beauty_v4_1_bias_control_validation/2026-10-05_06-32-43 |
| `z8q74kll` | v0 | testing | liger_sports_joint_inference/2026-09-27_16-19-28 |
| `z8ymlal9` | v0-mechanism-max | testing_mechanism_max | liger_sports_joint_inference/2026-09-29_22-40-12 |
| `z9envq1n` | v4-residual-lr10 | training | copmrec_beauty_v4_residual_train/2026-10-05_03-23-30 |
| `zy8946l2` | v3 | testing | copmrec_beauty_v3_inference/2026-10-04_19-31-39 |
| `zy940m5u` | early-ranking-decoder-exploration | inference | copmrec_beauty_decoder_competitive_audit/2026-10-02_16-30-06 |
| `zznw291t` | v3.1 | testing | copmrec_beauty_v3_1_inference/2026-10-04_21-26-00 |

### 逐运行 Recall／NDCG、差值与区间原值

保留原 metric key，可区分 `val/*`、`test/*`、`candidate_trace/*` 与各排序器的 `ranking/*`。`test/*` 是统一 Trainer.test 的日志前缀，真实 Validation/Testing 身份仍由上面的阶段以及保存的 predict data folder 判断。没有这些指标的缓存、训练探针或未成功输出仍明确登记，不将其称为有效推荐结果。

#### `0he4of79` — v0 / training

| 原 metric key | 原值 |
|---|---:|
| `val/ndcg@10` | 0.026354342699050903 |
| `val/ndcg@5` | 0.02005494199693203 |
| `val/recall@10` | 0.05079023912549019 |
| `val/recall@5` | 0.03129396587610245 |

#### `19i5khys` — v0 / testing

| 原 metric key | 原值 |
|---|---:|
| `candidate_trace/dense/ndcg@10` | 0.03598390331036605 |
| `candidate_trace/dense/ndcg@5` | 0.026985444698303416 |
| `candidate_trace/dense/recall@10` | 0.07114430085408935 |
| `candidate_trace/dense/recall@5` | 0.0432410678352636 |
| `candidate_trace/hybrid/ndcg@10` | 0.035913907967042066 |
| `candidate_trace/hybrid/ndcg@5` | 0.0267449570231512 |
| `candidate_trace/hybrid/recall@10` | 0.07123373429325225 |
| `candidate_trace/hybrid/recall@5` | 0.042883334078611994 |

#### `1edkvgk5` — v4-residual-lr10-dense / testing

| 原 metric key | 原值 |
|---|---:|
| `test/ndcg@10` | 0.038441483150323794 |
| `test/ndcg@5` | 0.02895971437365374 |
| `test/recall@10` | 0.07516880561641998 |
| `test/recall@5` | 0.04570048741224344 |

#### `1gu2dsa1` — v2 / testing

| 原 metric key | 原值 |
|---|---:|
| `candidate_trace/dense/ndcg@10` | 0.03581753380401717 |
| `candidate_trace/dense/ndcg@5` | 0.027173405681112497 |
| `candidate_trace/dense/recall@10` | 0.07020524974287887 |
| `candidate_trace/dense/recall@5` | 0.04328578455484506 |
| `candidate_trace/hybrid/ndcg@10` | 0.03328514599949916 |
| `candidate_trace/hybrid/ndcg@5` | 0.0235377077153151 |
| `candidate_trace/hybrid/recall@10` | 0.06908733175334257 |
| `candidate_trace/hybrid/recall@5` | 0.03890354603586281 |
| `test/ndcg@10` | 0.033285145999499166 |
| `test/ndcg@5` | 0.023537707715315135 |
| `test/recall@10` | 0.06908733175334257 |
| `test/recall@5` | 0.03890354603586281 |

#### `1raiqpvi` — v0-mechanism-max / testing_mechanism_max

| 原 metric key | 原值 |
|---|---:|
| `candidate_trace/dense/ndcg@10` | 0.036598312369094214 |
| `candidate_trace/dense/ndcg@5` | 0.027389204123284657 |
| `candidate_trace/dense/recall@10` | 0.07500515145270967 |
| `candidate_trace/dense/recall@5` | 0.04636307438697713 |
| `candidate_trace/hybrid/ndcg@10` | 0.03490763589101489 |
| `candidate_trace/hybrid/ndcg@5` | 0.026457702812453323 |
| `candidate_trace/hybrid/recall@10` | 0.07036884401401196 |
| `candidate_trace/hybrid/recall@5` | 0.04414794972182155 |

#### `29o9dx80` — early-ranking-decoder-exploration / train

| 原 metric key | 原值 |
|---|---:|
| `val/ndcg@10` | 0.04699515923857689 |
| `val/recall@10` | 0.09212055802345276 |

#### `3lb22b0r` — v3.1-fixed-inference-alpha0813 / testing

| 原 metric key | 原值 |
|---|---:|
| `candidate_trace/dense/ndcg@10` | 0.03578160777512011 |
| `candidate_trace/dense/ndcg@5` | 0.027653124522355486 |
| `candidate_trace/dense/recall@10` | 0.07038411662120467 |
| `candidate_trace/dense/recall@5` | 0.045029736618521665 |
| `candidate_trace/hybrid/ndcg@10` | 0.03585557973591457 |
| `candidate_trace/hybrid/ndcg@5` | 0.02783069068770338 |
| `candidate_trace/hybrid/recall@10` | 0.07038411662120467 |
| `candidate_trace/hybrid/recall@5` | 0.04543218709475473 |
| `test/ndcg@10` | 0.035855579735914564 |
| `test/ndcg@5` | 0.02783069068770341 |
| `test/recall@10` | 0.07038411662120467 |
| `test/recall@5` | 0.04543218709475473 |

#### `3r954xoo` — early-ranking-decoder-exploration / train

| 原 metric key | 原值 |
|---|---:|
| `val/ndcg@10` | 0.04686911404132843 |
| `val/recall@10` | 0.09185224771499634 |

#### `57hzkcol` — v1.1-calibrated-scratch / training

| 原 metric key | 原值 |
|---|---:|
| `val/ndcg@10` | 0.02568557672202587 |
| `val/ndcg@5` | 0.016302259638905525 |
| `val/recall@10` | 0.05804489925503731 |
| `val/recall@5` | 0.0287094172090292 |

#### `5ke7nt98` — v4.1-learned-score-bias / training

| 原 metric key | 原值 |
|---|---:|
| `val/ndcg@10` | 0.05050303786993027 |
| `val/ndcg@5` | 0.038766149431467056 |
| `val/recall@10` | 0.09810848534107208 |
| `val/recall@5` | 0.06166435778141022 |

#### `5rlfx2np` — v4-zero-residual-control / training

| 原 metric key | 原值 |
|---|---:|
| `val/ndcg@10` | 0.047863930463790894 |
| `val/ndcg@5` | 0.03728262335062027 |
| `val/recall@10` | 0.0922953113913536 |
| `val/recall@5` | 0.05942852050065994 |

#### `68h3e3eq` — v1.1-loss-weight-equal / diagnosis_training

W&B summary 中没有 Recall/NDCG；不登记推荐性能数值。

#### `6dspa7e3` — v0 / testing

| 原 metric key | 原值 |
|---|---:|
| `candidate_trace/dense/ndcg@10` | 0.03652606072154706 |
| `candidate_trace/dense/ndcg@5` | 0.0272079896834676 |
| `candidate_trace/dense/recall@10` | 0.07221750212404418 |
| `candidate_trace/dense/recall@5` | 0.04337521799400796 |
| `candidate_trace/hybrid/ndcg@10` | 0.03644831426431057 |
| `candidate_trace/hybrid/ndcg@5` | 0.027383121857157564 |
| `candidate_trace/hybrid/recall@10` | 0.07190448508697402 |
| `candidate_trace/hybrid/recall@5` | 0.04382238518982248 |

#### `6u6g62hk` — v5.3 / validation

| 原 metric key | 原值 |
|---|---:|
| `test/ndcg@10` | 0.06155058765023169 |
| `test/ndcg@5` | 0.052131534342316585 |
| `test/recall@10` | 0.10378750614854894 |
| `test/recall@5` | 0.07463220498144256 |

#### `7n6n6jta` — early-ranking-decoder-exploration / inference

| 原 metric key | 原值 |
|---|---:|
| `ranking/content/ndcg@10` | 0.04793673807381232 |
| `ranking/content/ndcg@5` | 0.03720592395983405 |
| `ranking/content/recall@10` | 0.0925594705777142 |
| `ranking/content/recall@5` | 0.059202289393668395 |
| `ranking/content_minus_content/common_hit_rank_ndcg@10` | 0 |
| `ranking/content_minus_content/common_hit_rank_ndcg@5` | 0 |
| `ranking/content_minus_content/lost_hit_ndcg@10` | 0 |
| `ranking/content_minus_content/lost_hit_ndcg@5` | 0 |
| `ranking/content_minus_content/new_hit_ndcg@10` | 0 |
| `ranking/content_minus_content/new_hit_ndcg@5` | 0 |
| `ranking/generation/ndcg@10` | 0.03763325189700371 |
| `ranking/generation/ndcg@5` | 0.02691624379270245 |
| `ranking/generation/recall@10` | 0.07851904847075657 |
| `ranking/generation/recall@5` | 0.04471472008585226 |
| `ranking/generation_minus_content/common_hit_rank_ndcg@10` | -0.003596824920374986 |
| `ranking/generation_minus_content/common_hit_rank_ndcg@5` | -0.0019959574935677258 |
| `ranking/generation_minus_content/lost_hit_ndcg@10` | -0.013592869420399764 |
| `ranking/generation_minus_content/lost_hit_ndcg@5` | -0.015698565968530538 |
| `ranking/generation_minus_content/new_hit_ndcg@10` | 0.006886208163966147 |
| `ranking/generation_minus_content/new_hit_ndcg@5` | 0.007404843294966672 |
| `ranking/learned/ndcg@10` | 0.04781648927810209 |
| `ranking/learned/ndcg@5` | 0.03735898761605647 |
| `ranking/learned/recall@10` | 0.09202289393668396 |
| `ranking/learned/recall@5` | 0.059560007154355214 |
| `ranking/learned_minus_content/common_hit_rank_ndcg@10` | 3.485663783540167e-05 |
| `ranking/learned_minus_content/common_hit_rank_ndcg@5` | 1.4679536303073636e-05 |
| `ranking/learned_minus_content/lost_hit_ndcg@10` | -0.00018095633913657805 |
| `ranking/learned_minus_content/lost_hit_ndcg@5` | -6.919205995967477e-05 |
| `ranking/learned_minus_content/ndcg@10` | -0.00012024879571023662 |
| `ranking/learned_minus_content/ndcg@10_ci95_high` | 2.900209870958748e-05 |
| `ranking/learned_minus_content/ndcg@10_ci95_low` | -0.0002921947786919676 |
| `ranking/learned_minus_content/ndcg@5` | 0.00015306365622242318 |
| `ranking/learned_minus_content/ndcg@5_ci95_high` | 0.00035768652782810915 |
| `ranking/learned_minus_content/ndcg@5_ci95_low` | -4.111727782182237e-05 |
| `ranking/learned_minus_content/new_hit_ndcg@10` | 2.5850905590939712e-05 |
| `ranking/learned_minus_content/new_hit_ndcg@5` | 0.0002075761798790243 |
| `ranking/learned_minus_content/recall@10` | -0.0005365766410302271 |
| `ranking/learned_minus_content/recall@10_ci95_high` | -8.942944017170452e-05 |
| `ranking/learned_minus_content/recall@10_ci95_low` | -0.0010731532820604545 |
| `ranking/learned_minus_content/recall@5` | 0.0003577177606868181 |
| `ranking/learned_minus_content/recall@5_ci95_high` | 0.0008942944017170453 |
| `ranking/learned_minus_content/recall@5_ci95_low` | -8.942944017170452e-05 |
| `ranking/mixed/ndcg@10` | 0.04749760278576311 |
| `ranking/mixed/ndcg@5` | 0.0373041198677585 |
| `ranking/mixed/recall@10` | 0.09023430513324988 |
| `ranking/mixed/recall@5` | 0.05866571275263817 |
| `ranking/mixed_minus_content/common_hit_rank_ndcg@10` | 0.0002343076763095105 |
| `ranking/mixed_minus_content/common_hit_rank_ndcg@5` | 0.00024923757417190465 |
| `ranking/mixed_minus_content/lost_hit_ndcg@10` | -0.0022267465938163138 |
| `ranking/mixed_minus_content/lost_hit_ndcg@5` | -0.002193142160489687 |
| `ranking/mixed_minus_content/new_hit_ndcg@10` | 0.0015533036294576037 |
| `ranking/mixed_minus_content/new_hit_ndcg@5` | 0.0020421004942422385 |

#### `7q2njodh` — early-ranking-decoder-exploration / train

| 原 metric key | 原值 |
|---|---:|
| `val/ndcg@10` | 0.046998266130685806 |
| `val/recall@10` | 0.0922994390130043 |

#### `7wnteuw0` — early-ranking-decoder-exploration / inference

| 原 metric key | 原值 |
|---|---:|
| `ranking/content/ndcg@10` | 0.04793673807381232 |
| `ranking/content/ndcg@5` | 0.03720592395983405 |
| `ranking/content/recall@10` | 0.0925594705777142 |
| `ranking/content/recall@5` | 0.059202289393668395 |
| `ranking/content_minus_content/common_hit_rank_ndcg@10` | 0 |
| `ranking/content_minus_content/common_hit_rank_ndcg@5` | 0 |
| `ranking/content_minus_content/lost_hit_ndcg@10` | 0 |
| `ranking/content_minus_content/lost_hit_ndcg@5` | 0 |
| `ranking/content_minus_content/new_hit_ndcg@10` | 0 |
| `ranking/content_minus_content/new_hit_ndcg@5` | 0 |
| `ranking/generation/ndcg@10` | 0.03763325189700371 |
| `ranking/generation/ndcg@5` | 0.02691624379270245 |
| `ranking/generation/recall@10` | 0.07851904847075657 |
| `ranking/generation/recall@5` | 0.04471472008585226 |
| `ranking/generation_minus_content/common_hit_rank_ndcg@10` | -0.003596824920374986 |
| `ranking/generation_minus_content/common_hit_rank_ndcg@5` | -0.0019959574935677258 |
| `ranking/generation_minus_content/lost_hit_ndcg@10` | -0.013592869420399764 |
| `ranking/generation_minus_content/lost_hit_ndcg@5` | -0.015698565968530538 |
| `ranking/generation_minus_content/new_hit_ndcg@10` | 0.006886208163966147 |
| `ranking/generation_minus_content/new_hit_ndcg@5` | 0.007404843294966672 |
| `ranking/mixed/ndcg@10` | 0.04749760278576311 |
| `ranking/mixed/ndcg@5` | 0.0373041198677585 |
| `ranking/mixed/recall@10` | 0.09023430513324988 |
| `ranking/mixed/recall@5` | 0.05866571275263817 |
| `ranking/mixed_minus_content/common_hit_rank_ndcg@10` | 0.0002343076763095105 |
| `ranking/mixed_minus_content/common_hit_rank_ndcg@5` | 0.00024923757417190465 |
| `ranking/mixed_minus_content/lost_hit_ndcg@10` | -0.0022267465938163138 |
| `ranking/mixed_minus_content/lost_hit_ndcg@5` | -0.002193142160489687 |
| `ranking/mixed_minus_content/new_hit_ndcg@10` | 0.0015533036294576037 |
| `ranking/mixed_minus_content/new_hit_ndcg@5` | 0.0020421004942422385 |
| `ranking/ndcg/cold_hits@10` | 0 |
| `ranking/ndcg/cold_hits@5` | 0 |
| `ranking/ndcg/ndcg@10` | 0.04765447397357972 |
| `ranking/ndcg/ndcg@5` | 0.03656300485535277 |
| `ranking/ndcg/recall@10` | 0.09166517617599712 |
| `ranking/ndcg/recall@5` | 0.0572348417098909 |
| `ranking/ndcg_minus_content/common_hit_rank_ndcg@10` | -4.493775191824605e-05 |
| `ranking/ndcg_minus_content/common_hit_rank_ndcg@5` | 5.7836759834846816e-05 |
| `ranking/ndcg_minus_content/lost_hit_ndcg@10` | -0.0022817028936848267 |
| `ranking/ndcg_minus_content/lost_hit_ndcg@5` | -0.0032333864281685987 |
| `ranking/ndcg_minus_content/lost_hits@10` | 81 |
| `ranking/ndcg_minus_content/lost_hits@5` | 84 |
| `ranking/ndcg_minus_content/ndcg@10` | -0.00028226410023259675 |
| `ranking/ndcg_minus_content/ndcg@10_ci95_high` | 0.0008370239027363473 |
| `ranking/ndcg_minus_content/ndcg@10_ci95_low` | -0.0013562526130171018 |
| `ranking/ndcg_minus_content/ndcg@5` | -0.0006429191044812839 |
| `ranking/ndcg_minus_content/ndcg@5_ci95_high` | 0.0006325908912825098 |
| `ranking/ndcg_minus_content/ndcg@5_ci95_low` | -0.0019148679679031244 |
| `ranking/ndcg_minus_content/new_hit_ndcg@10` | 0.002044376545370475 |
| `ranking/ndcg_minus_content/new_hit_ndcg@5` | 0.002532630563852467 |
| `ranking/ndcg_minus_content/new_hits@10` | 71 |
| `ranking/ndcg_minus_content/new_hits@5` | 62 |
| `ranking/ndcg_minus_content/recall@10` | -0.0008942944017170453 |
| `ranking/ndcg_minus_content/recall@10_ci95_high` | 0.0012520121624038634 |
| `ranking/ndcg_minus_content/recall@10_ci95_low` | -0.003040600965837954 |
| `ranking/ndcg_minus_content/recall@5` | -0.0019674476837774997 |
| `ranking/ndcg_minus_content/recall@5_ci95_high` | 9.166517617598492e-05 |
| `ranking/ndcg_minus_content/recall@5_ci95_low` | -0.004203183688070112 |
| `ranking/ndcg_minus_mixed/ndcg@10` | 0.0001568711878166026 |
| `ranking/ndcg_minus_mixed/ndcg@10_ci95_high` | 0.0008653480442829559 |
| `ranking/ndcg_minus_mixed/ndcg@10_ci95_low` | -0.0005507394226510419 |
| `ranking/ndcg_minus_mixed/ndcg@5` | -0.0007411150124057403 |
| `ranking/ndcg_minus_mixed/ndcg@5_ci95_high` | 7.118408726246685e-05 |
| `ranking/ndcg_minus_mixed/ndcg@5_ci95_low` | -0.001546900421317221 |
| `ranking/ndcg_minus_mixed/recall@10` | 0.0014308710427472723 |
| `ranking/ndcg_minus_mixed/recall@10_ci95_high` | 0.0029511715256662495 |
| `ranking/ndcg_minus_mixed/recall@10_ci95_low` | -8.942944017170452e-05 |
| `ranking/ndcg_minus_mixed/recall@5` | -0.0014308710427472723 |
| `ranking/ndcg_minus_mixed/recall@5_ci95_high` | 8.942944017170452e-05 |
| `ranking/ndcg_minus_mixed/recall@5_ci95_low` | -0.003040600965837954 |
| `ranking/ndcg_minus_nll/ndcg@10` | 0.00038041390334128753 |
| `ranking/ndcg_minus_nll/ndcg@10_ci95_high` | 0.0008589812200037141 |
| `ranking/ndcg_minus_nll/ndcg@10_ci95_low` | -7.701001384782622e-05 |
| `ranking/ndcg_minus_nll/ndcg@5` | -5.520026562121059e-05 |
| `ranking/ndcg_minus_nll/ndcg@5_ci95_high` | 0.0005175328433192685 |
| `ranking/ndcg_minus_nll/ndcg@5_ci95_low` | -0.0006382181633778484 |
| `ranking/ndcg_minus_nll/recall@10` | 0.0011625827222321587 |
| `ranking/ndcg_minus_nll/recall@10_ci95_high` | 0.002235736004292613 |
| `ranking/ndcg_minus_nll/recall@10_ci95_low` | 0.00017885888034340904 |
| `ranking/ndcg_minus_nll/recall@5` | -0.00017885888034340904 |
| `ranking/ndcg_minus_nll/recall@5_ci95_high` | 0.0009837238418887498 |
| `ranking/ndcg_minus_nll/recall@5_ci95_low` | -0.001341441602575568 |
| `ranking/nll/ndcg@10` | 0.047274060070238426 |
| `ranking/nll/ndcg@5` | 0.036618205120973975 |
| `ranking/nll/recall@10` | 0.09050259345376498 |
| `ranking/nll/recall@5` | 0.057413700590234304 |
| `ranking/nll_minus_content/common_hit_rank_ndcg@10` | -9.480060303635256e-05 |
| `ranking/nll_minus_content/common_hit_rank_ndcg@5` | 4.639663371972852e-06 |
| `ranking/nll_minus_content/lost_hit_ndcg@10` | -0.002415552921020452 |
| `ranking/nll_minus_content/lost_hit_ndcg@5` | -0.0029021056042960827 |
| `ranking/nll_minus_content/ndcg@10` | -0.0006626780035738843 |
| `ranking/nll_minus_content/ndcg@10_ci95_high` | 0.000437691751415712 |
| `ranking/nll_minus_content/ndcg@10_ci95_low` | -0.0017273740000145947 |
| `ranking/nll_minus_content/ndcg@5` | -0.0005877188388600732 |
| `ranking/nll_minus_content/ndcg@5_ci95_high` | 0.0006027730888313866 |
| `ranking/nll_minus_content/ndcg@5_ci95_low` | -0.001804990280325573 |
| `ranking/nll_minus_content/new_hit_ndcg@10` | 0.00184767552048292 |
| `ranking/nll_minus_content/new_hit_ndcg@5` | 0.002309747102064037 |
| `ranking/nll_minus_content/recall@10` | -0.002056877123949204 |
| `ranking/nll_minus_content/recall@10_ci95_high` | 8.942944017170452e-05 |
| `ranking/nll_minus_content/recall@10_ci95_low` | -0.004113754247898408 |
| `ranking/nll_minus_content/recall@5` | -0.0017885888034340906 |
| `ranking/nll_minus_content/recall@5_ci95_high` | 0.00026828832051511357 |
| `ranking/nll_minus_content/recall@5_ci95_low` | -0.00375603648721159 |

#### `7y54j4m6` — v0 / training

| 原 metric key | 原值 |
|---|---:|
| `val/ndcg@10` | 0.047115258872509 |
| `val/ndcg@5` | 0.0367470309138298 |
| `val/recall@10` | 0.09152044355869292 |
| `val/recall@5` | 0.05919463559985161 |

#### `8w893ra3` — v5 / validation

| 原 metric key | 原值 |
|---|---:|
| `test/ndcg@10` | 0.060150846751954846 |
| `test/ndcg@5` | 0.05113833064515972 |
| `test/recall@10` | 0.10043375217994008 |
| `test/recall@5` | 0.07239636900236998 |

#### `8xcw35ag` — v0 / testing

| 原 metric key | 原值 |
|---|---:|
| `candidate_trace/dense/ndcg@10` | 0.03655077088504207 |
| `candidate_trace/dense/ndcg@5` | 0.027423027203783593 |
| `candidate_trace/dense/recall@10` | 0.07261995260027725 |
| `candidate_trace/dense/recall@5` | 0.04422483566605554 |
| `candidate_trace/hybrid/ndcg@10` | 0.03635436377143752 |
| `candidate_trace/hybrid/ndcg@5` | 0.027426015098367475 |
| `candidate_trace/hybrid/recall@10` | 0.07199391852613692 |
| `candidate_trace/hybrid/recall@5` | 0.04426955238563699 |

#### `9dzi0g5i` — v0 / training

| 原 metric key | 原值 |
|---|---:|
| `val/ndcg@10` | 0.045982323586940765 |
| `val/ndcg@5` | 0.034752532839775085 |
| `val/recall@10` | 0.091279536485672 |
| `val/recall@5` | 0.05640507489442825 |

#### `amwchjkc` — v4-fixed-mixed-ranking / validation

| 原 metric key | 原值 |
|---|---:|
| `candidate_trace/dense/ndcg@10` | 0.04911456119404004 |
| `candidate_trace/dense/ndcg@5` | 0.038245286848631786 |
| `candidate_trace/dense/recall@10` | 0.09511246254974735 |
| `candidate_trace/dense/recall@5` | 0.06126190582658856 |
| `candidate_trace/hybrid/ndcg@10` | 0.04809118859724856 |
| `candidate_trace/hybrid/ndcg@5` | 0.0378992023845953 |
| `candidate_trace/hybrid/recall@10` | 0.09278719313151187 |
| `candidate_trace/hybrid/recall@5` | 0.06108303894826275 |
| `test/ndcg@10` | 0.04809118859724857 |
| `test/ndcg@5` | 0.03789920238459532 |
| `test/recall@10` | 0.09278719313151187 |
| `test/recall@5` | 0.06108303894826275 |

#### `artnz7m2` — v4.4-history-ce-treated-pool / validation

| 原 metric key | 原值 |
|---|---:|
| `test/ndcg@10` | 0.06056854254398821 |
| `test/ndcg@5` | 0.04900058485582896 |
| `test/recall@10` | 0.10799087778920538 |
| `test/recall@5` | 0.07199391852613692 |

#### `b6t4a1ws` — v0 / training

| 原 metric key | 原值 |
|---|---:|
| `val/ndcg@10` | 0.045344531536102295 |
| `val/ndcg@5` | 0.03465671092271805 |
| `val/recall@10` | 0.08942687511444092 |
| `val/recall@5` | 0.05629994720220566 |

#### `beawrjef` — v2 / training

| 原 metric key | 原值 |
|---|---:|
| `val/ndcg@10` | 0.04182206466794014 |
| `val/ndcg@5` | 0.03063962422311306 |
| `val/recall@10` | 0.08625855296850204 |
| `val/recall@5` | 0.051513660699129105 |

#### `bzgzloh5` — v0-mechanism-legal / testing_mechanism_legal

| 原 metric key | 原值 |
|---|---:|
| `candidate_trace/dense/ndcg@10` | 0.036598312369094214 |
| `candidate_trace/dense/ndcg@5` | 0.027389204123284657 |
| `candidate_trace/dense/recall@10` | 0.07500515145270967 |
| `candidate_trace/dense/recall@5` | 0.04636307438697713 |
| `candidate_trace/hybrid/ndcg@10` | 0.027915360528095404 |
| `candidate_trace/hybrid/ndcg@5` | 0.022855056640093817 |
| `candidate_trace/hybrid/recall@10` | 0.052699361219864 |
| `candidate_trace/hybrid/recall@5` | 0.036935915928291776 |

#### `cm0i584p` — v5.1 / training

| 原 metric key | 原值 |
|---|---:|
| `val/ndcg@10` | 0.047847915440797806 |
| `val/ndcg@5` | 0.038831356912851334 |
| `val/recall@10` | 0.08813665062189102 |
| `val/recall@5` | 0.06023342162370682 |

#### `cv5wwhck` — v5.4 / training

| 原 metric key | 原值 |
|---|---:|
| `val/ndcg@10` | 0.04385127127170563 |
| `val/ndcg@5` | 0.0340602770447731 |
| `val/recall@10` | 0.08348611742258072 |
| `val/recall@5` | 0.052944596856832504 |

#### `d7lcftto` — v5.2 / validation

| 原 metric key | 原值 |
|---|---:|
| `test/ndcg@10` | 0.061058281374572726 |
| `test/ndcg@5` | 0.051949186651987715 |
| `test/recall@10` | 0.10146223673031346 |
| `test/recall@5` | 0.07324598667441756 |

#### `f91njtjx` — v1.1 / training

| 原 metric key | 原值 |
|---|---:|
| `val/ndcg@10` | 0.008687474764883518 |
| `val/ndcg@5` | 0.005606374703347683 |
| `val/recall@10` | 0.018602987751364708 |
| `val/recall@5` | 0.008943744003772736 |

#### `gdxds9oi` — v0 / training

| 原 metric key | 原值 |
|---|---:|
| `val/ndcg@10` | 0.04728115350008011 |
| `val/ndcg@5` | 0.03665628284215927 |
| `val/recall@10` | 0.09254512190818788 |
| `val/recall@5` | 0.059504203498363495 |

#### `hbgj80bh` — v4.4-history-ce-control / training

| 原 metric key | 原值 |
|---|---:|
| `val/ndcg@10` | 0.05919908359646797 |
| `val/ndcg@5` | 0.0484619103372097 |
| `val/recall@10` | 0.10512901097536088 |
| `val/recall@5` | 0.07181505113840103 |

#### `hkokjad0` — early-ranking-decoder-exploration / train

| 原 metric key | 原值 |
|---|---:|
| `val/ndcg@10` | 0.046520013362169266 |
| `val/recall@10` | 0.0916733741760254 |

#### `hpfflo20` — early-ranking-decoder-exploration / inference

| 原 metric key | 原值 |
|---|---:|
| `ranking/content/ndcg@10` | 0.047249121734350155 |
| `ranking/content/ndcg@5` | 0.03656577838330635 |
| `ranking/content/recall@10` | 0.09162455842239411 |
| `ranking/content/recall@5` | 0.058400035773375665 |
| `ranking/content_minus_content/common_hit_rank_ndcg@10` | 0 |
| `ranking/content_minus_content/common_hit_rank_ndcg@5` | 0 |
| `ranking/content_minus_content/lost_hit_ndcg@10` | 0 |
| `ranking/content_minus_content/lost_hit_ndcg@5` | 0 |
| `ranking/content_minus_content/new_hit_ndcg@10` | 0 |
| `ranking/content_minus_content/new_hit_ndcg@5` | 0 |
| `ranking/generation/ndcg@10` | 0.03742930881280513 |
| `ranking/generation/ndcg@5` | 0.02658644844799188 |
| `ranking/generation/recall@10` | 0.07803067566963287 |
| `ranking/generation/recall@5` | 0.044090685507311184 |
| `ranking/generation_minus_content/common_hit_rank_ndcg@10` | -0.0034023476245477633 |
| `ranking/generation_minus_content/common_hit_rank_ndcg@5` | -0.0018404696116848744 |
| `ranking/generation_minus_content/lost_hit_ndcg@10` | -0.013159384216408497 |
| `ranking/generation_minus_content/lost_hit_ndcg@5` | -0.016216561751079526 |
| `ranking/generation_minus_content/new_hit_ndcg@10` | 0.0067419189194112385 |
| `ranking/generation_minus_content/new_hit_ndcg@5` | 0.008077701427449922 |
| `ranking/mixed/ndcg@10` | 0.04701131278168725 |
| `ranking/mixed/ndcg@5` | 0.036891565494657816 |
| `ranking/mixed/recall@10` | 0.0907302240307651 |
| `ranking/mixed/recall@5` | 0.05920493672584179 |
| `ranking/mixed_minus_content/common_hit_rank_ndcg@10` | -4.49941458328953e-06 |
| `ranking/mixed_minus_content/common_hit_rank_ndcg@5` | -1.8462096890932788e-05 |
| `ranking/mixed_minus_content/lost_hit_ndcg@10` | -0.0018841886296641717 |
| `ranking/mixed_minus_content/lost_hit_ndcg@5` | -0.0020707188059819756 |
| `ranking/mixed_minus_content/ndcg@10` | -0.00023780895266289933 |
| `ranking/mixed_minus_content/ndcg@10_ci95_high` | 0.00041687855969313697 |
| `ranking/mixed_minus_content/ndcg@10_ci95_low` | -0.0009467051548010684 |
| `ranking/mixed_minus_content/ndcg@5` | 0.00032578711135145943 |
| `ranking/mixed_minus_content/ndcg@5_ci95_high` | 0.0011110259148255175 |
| `ranking/mixed_minus_content/ndcg@5_ci95_low` | -0.00046221093482088954 |
| `ranking/mixed_minus_content/new_hit_ndcg@10` | 0.0016508790915845618 |
| `ranking/mixed_minus_content/new_hit_ndcg@5` | 0.0024149680142243673 |
| `ranking/mixed_minus_content/recall@10` | -0.0008943343916290301 |
| `ranking/mixed_minus_content/recall@10_ci95_high` | 0.0004930018333854968 |
| `ranking/mixed_minus_content/recall@10_ci95_low` | -0.002280552698654027 |
| `ranking/mixed_minus_content/recall@5` | 0.0008049009524661271 |
| `ranking/mixed_minus_content/recall@5_ci95_high` | 0.002146402539909672 |
| `ranking/mixed_minus_content/recall@5_ci95_low` | -0.0005813173545588696 |
| `test/ndcg@10` | 0.04701131278168727 |
| `test/ndcg@5` | 0.03689156549465785 |
| `test/recall@10` | 0.0907302240307651 |
| `test/recall@5` | 0.05920493672584179 |

#### `hqw189d2` — v5.3 / training

| 原 metric key | 原值 |
|---|---:|
| `val/ndcg@10` | 0.05232436582446098 |
| `val/ndcg@5` | 0.04245857521891594 |
| `val/recall@10` | 0.09569378197193146 |
| `val/recall@5` | 0.0651075467467308 |

#### `i05boy01` — early-ranking-decoder-exploration / inference

| 原 metric key | 原值 |
|---|---:|
| `ranking/content/ndcg@10` | 0.05758808274172322 |
| `ranking/content/ndcg@5` | 0.04564777572508667 |
| `ranking/content/recall@10` | 0.1082591781066941 |
| `ranking/content/recall@5` | 0.07105486741492643 |
| `ranking/content_minus_content/common_hit_rank_ndcg@10` | 0 |
| `ranking/content_minus_content/common_hit_rank_ndcg@5` | 0 |
| `ranking/content_minus_content/lost_hit_ndcg@10` | 0 |
| `ranking/content_minus_content/lost_hit_ndcg@5` | 0 |
| `ranking/content_minus_content/new_hit_ndcg@10` | 0 |
| `ranking/content_minus_content/new_hit_ndcg@5` | 0 |
| `ranking/generation/ndcg@10` | 0.04734742605233559 |
| `ranking/generation/ndcg@5` | 0.035197857080113165 |
| `ranking/generation/recall@10` | 0.09475472879309574 |
| `ranking/generation/recall@5` | 0.05683495058802486 |
| `ranking/generation_minus_content/common_hit_rank_ndcg@10` | -0.004113186382229936 |
| `ranking/generation_minus_content/common_hit_rank_ndcg@5` | -0.002111572021247079 |
| `ranking/generation_minus_content/lost_hit_ndcg@10` | -0.014460773066284096 |
| `ranking/generation_minus_content/lost_hit_ndcg@5` | -0.018114960037223946 |
| `ranking/generation_minus_content/new_hit_ndcg@10` | 0.008333302759126394 |
| `ranking/generation_minus_content/new_hit_ndcg@5` | 0.009776613413497512 |
| `ranking/mixed/ndcg@10` | 0.05785187818510919 |
| `ranking/mixed/ndcg@5` | 0.04562483682134016 |
| `ranking/mixed/recall@10` | 0.10901936233957876 |
| `ranking/mixed/recall@5` | 0.07096543397576353 |
| `ranking/mixed_minus_content/common_hit_rank_ndcg@10` | -3.934752247584108e-05 |
| `ranking/mixed_minus_content/common_hit_rank_ndcg@5` | 2.0741393380848607e-05 |
| `ranking/mixed_minus_content/lost_hit_ndcg@10` | -0.0018986619403594512 |
| `ranking/mixed_minus_content/lost_hit_ndcg@5` | -0.002877795154476404 |
| `ranking/mixed_minus_content/ndcg@10` | 0.00026379544338595614 |
| `ranking/mixed_minus_content/ndcg@10_ci95_high` | 0.0009726236789975508 |
| `ranking/mixed_minus_content/ndcg@10_ci95_low` | -0.0004602070865147225 |
| `ranking/mixed_minus_content/ndcg@5` | -2.2938903746516564e-05 |
| `ranking/mixed_minus_content/ndcg@5_ci95_high` | 0.0008159092294796502 |
| `ranking/mixed_minus_content/ndcg@5_ci95_low` | -0.0008395971654976888 |
| `ranking/mixed_minus_content/new_hit_ndcg@10` | 0.002201804906221249 |
| `ranking/mixed_minus_content/new_hit_ndcg@5` | 0.002834114857349039 |
| `ranking/mixed_minus_content/recall@10` | 0.0007601842328846755 |
| `ranking/mixed_minus_content/recall@10_ci95_high` | 0.002235835979072575 |
| `ranking/mixed_minus_content/recall@10_ci95_low` | -0.0006707507937217726 |
| `ranking/mixed_minus_content/recall@5` | -8.9433439162903e-05 |
| `ranking/mixed_minus_content/recall@5_ci95_high` | 0.0013862183070249966 |
| `ranking/mixed_minus_content/recall@5_ci95_low` | -0.0016098019049322542 |
| `test/ndcg@10` | 0.05785187818510923 |
| `test/ndcg@5` | 0.04562483682134011 |
| `test/recall@10` | 0.10901936233957876 |
| `test/recall@5` | 0.07096543397576353 |

#### `i4xwruok` — v4.2-fixed-logit-pool / validation

| 原 metric key | 原值 |
|---|---:|
| `test/ndcg@10` | 0.05078680665613528 |
| `test/ndcg@5` | 0.0393803358898746 |
| `test/recall@10` | 0.0981084827617046 |
| `test/recall@5` | 0.06255869069445065 |

#### `iy3o3z3q` — v4.3-history-eligible-pool / validation

| 原 metric key | 原值 |
|---|---:|
| `test/ndcg@10` | 0.05960712488411073 |
| `test/ndcg@5` | 0.048587839862585955 |
| `test/recall@10` | 0.10651522604301748 |
| `test/recall@5` | 0.07226221884362563 |

#### `jpqimr2j` — v0-mechanism-legal / testing_mechanism_legal

| 原 metric key | 原值 |
|---|---:|
| `candidate_trace/dense/ndcg@10` | 0.03652606072154706 |
| `candidate_trace/dense/ndcg@5` | 0.0272079896834676 |
| `candidate_trace/dense/recall@10` | 0.07221750212404418 |
| `candidate_trace/dense/recall@5` | 0.04337521799400796 |
| `candidate_trace/hybrid/ndcg@10` | 0.03004365239187472 |
| `candidate_trace/hybrid/ndcg@5` | 0.023966687718818633 |
| `candidate_trace/hybrid/recall@10` | 0.056208916513884544 |
| `candidate_trace/hybrid/recall@5` | 0.037338460850512005 |

#### `k1fzr0jz` — v0 / training

| 原 metric key | 原值 |
|---|---:|
| `val/ndcg@10` | 0.04508304223418236 |
| `val/ndcg@5` | 0.03417829051613808 |
| `val/recall@10` | 0.08999458700418472 |
| `val/recall@5` | 0.0560455247759819 |

#### `klqyxo8m` — early-ranking-decoder-exploration / train

| 原 metric key | 原值 |
|---|---:|
| `val/ndcg@10` | 0.04697433486580849 |
| `val/recall@10` | 0.0916733741760254 |

#### `l3zyr91b` — v4.1-zero-score-bias-control / training

| 原 metric key | 原值 |
|---|---:|
| `val/ndcg@10` | 0.05042651668190956 |
| `val/ndcg@5` | 0.0385395772755146 |
| `val/recall@10` | 0.09877923130989076 |
| `val/recall@5` | 0.06175379082560539 |

#### `m50dan21` — v5 / training

| 原 metric key | 原值 |
|---|---:|
| `val/ndcg@10` | 0.05152552947402 |
| `val/ndcg@5` | 0.04219355061650276 |
| `val/recall@10` | 0.09265304356813432 |
| `val/recall@5` | 0.06376603990793228 |

#### `mg0nrz9a` — v1.1-loss-weight-calibrated / diagnosis_training

W&B summary 中没有 Recall/NDCG；不登记推荐性能数值。

#### `mq8hhof5` — v4-residual-lr20-dense / validation

| 原 metric key | 原值 |
|---|---:|
| `test/ndcg@10` | 0.049793932567196414 |
| `test/ndcg@5` | 0.039227289845320125 |
| `test/recall@10` | 0.09600679694137638 |
| `test/recall@5` | 0.06305057460984662 |

#### `msut1s0r` — early-ranking-decoder-exploration / train

| 原 metric key | 原值 |
|---|---:|
| `val/ndcg@10` | 0.04705646634101868 |
| `val/recall@10` | 0.0919416844844818 |

#### `ncyol9n0` — v0-fixed-training-alpha08 / training

| 原 metric key | 原值 |
|---|---:|
| `val/ndcg@10` | 0.046127837151288986 |
| `val/ndcg@5` | 0.03610736131668091 |
| `val/recall@10` | 0.08986171334981918 |
| `val/recall@5` | 0.058699287474155426 |

#### `nj9elah1` — v4-residual-lr20 / training

| 原 metric key | 原值 |
|---|---:|
| `val/ndcg@10` | 0.04979391396045685 |
| `val/ndcg@5` | 0.03922727704048157 |
| `val/recall@10` | 0.09600679576396942 |
| `val/recall@5` | 0.06305057555437088 |

#### `nwxzx1qs` — early-ranking-decoder-exploration / inference

| 原 metric key | 原值 |
|---|---:|
| `ranking/content/ndcg@10` | 0.04793673807381232 |
| `ranking/content/ndcg@5` | 0.03720592395983405 |
| `ranking/content/recall@10` | 0.0925594705777142 |
| `ranking/content/recall@5` | 0.059202289393668395 |
| `ranking/content_minus_content/common_hit_rank_ndcg@10` | 0 |
| `ranking/content_minus_content/common_hit_rank_ndcg@5` | 0 |
| `ranking/content_minus_content/lost_hit_ndcg@10` | 0 |
| `ranking/content_minus_content/lost_hit_ndcg@5` | 0 |
| `ranking/content_minus_content/new_hit_ndcg@10` | 0 |
| `ranking/content_minus_content/new_hit_ndcg@5` | 0 |
| `ranking/generation/ndcg@10` | 0.03763325189700371 |
| `ranking/generation/ndcg@5` | 0.02691624379270245 |
| `ranking/generation/recall@10` | 0.07851904847075657 |
| `ranking/generation/recall@5` | 0.04471472008585226 |
| `ranking/generation_minus_content/common_hit_rank_ndcg@10` | -0.003596824920374986 |
| `ranking/generation_minus_content/common_hit_rank_ndcg@5` | -0.0019959574935677258 |
| `ranking/generation_minus_content/lost_hit_ndcg@10` | -0.013592869420399764 |
| `ranking/generation_minus_content/lost_hit_ndcg@5` | -0.015698565968530538 |
| `ranking/generation_minus_content/new_hit_ndcg@10` | 0.006886208163966147 |
| `ranking/generation_minus_content/new_hit_ndcg@5` | 0.007404843294966672 |
| `ranking/mixed/ndcg@10` | 0.04749760278576311 |
| `ranking/mixed/ndcg@5` | 0.0373041198677585 |
| `ranking/mixed/recall@10` | 0.09023430513324988 |
| `ranking/mixed/recall@5` | 0.05866571275263817 |
| `ranking/mixed_minus_content/common_hit_rank_ndcg@10` | 0.0002343076763095105 |
| `ranking/mixed_minus_content/common_hit_rank_ndcg@5` | 0.00024923757417190465 |
| `ranking/mixed_minus_content/lost_hit_ndcg@10` | -0.0022267465938163138 |
| `ranking/mixed_minus_content/lost_hit_ndcg@5` | -0.002193142160489687 |
| `ranking/mixed_minus_content/new_hit_ndcg@10` | 0.0015533036294576037 |
| `ranking/mixed_minus_content/new_hit_ndcg@5` | 0.0020421004942422385 |
| `ranking/ndcg/cold_hits@10` | 0 |
| `ranking/ndcg/cold_hits@5` | 0 |
| `ranking/ndcg/ndcg@10` | 0.04758613010773037 |
| `ranking/ndcg/ndcg@5` | 0.036769408428827165 |
| `ranking/ndcg/recall@10` | 0.09112859953496692 |
| `ranking/ndcg/recall@5` | 0.05759255947057772 |
| `ranking/ndcg_minus_content/common_hit_rank_ndcg@10` | 7.3797871279915e-05 |
| `ranking/ndcg_minus_content/common_hit_rank_ndcg@5` | 7.955120884634691e-05 |
| `ranking/ndcg_minus_content/lost_hit_ndcg@10` | -0.0023962125821713263 |
| `ranking/ndcg_minus_content/lost_hit_ndcg@5` | -0.0032435051182746135 |
| `ranking/ndcg_minus_content/lost_hits@10` | 85 |
| `ranking/ndcg_minus_content/lost_hits@5` | 84 |
| `ranking/ndcg_minus_content/ndcg@10` | -0.0003506079660819506 |
| `ranking/ndcg_minus_content/ndcg@10_ci95_high` | 0.0007246681625098987 |
| `ranking/ndcg_minus_content/ndcg@10_ci95_low` | -0.0014195183815236862 |
| `ranking/ndcg_minus_content/ndcg@5` | -0.0004365155310068822 |
| `ranking/ndcg_minus_content/ndcg@5_ci95_high` | 0.0008487805260607771 |
| `ranking/ndcg_minus_content/ndcg@5_ci95_low` | -0.0016937490703453509 |
| `ranking/ndcg_minus_content/new_hit_ndcg@10` | 0.001971806744809462 |
| `ranking/ndcg_minus_content/new_hit_ndcg@5` | 0.0027274383784213845 |
| `ranking/ndcg_minus_content/new_hits@10` | 69 |
| `ranking/ndcg_minus_content/new_hits@5` | 66 |
| `ranking/ndcg_minus_content/recall@10` | -0.0014308710427472723 |
| `ranking/ndcg_minus_content/recall@10_ci95_high` | 0.0007176712573779165 |
| `ranking/ndcg_minus_content/recall@10_ci95_low` | -0.003577177606868181 |
| `ranking/ndcg_minus_content/recall@5` | -0.0016097299230906814 |
| `ranking/ndcg_minus_content/recall@5_ci95_high` | 0.0005365766410302271 |
| `ranking/ndcg_minus_content/recall@5_ci95_low` | -0.00375603648721159 |
| `ranking/ndcg_minus_mixed/ndcg@10` | 8.85273219672488e-05 |
| `ranking/ndcg_minus_mixed/ndcg@10_ci95_high` | 0.0008675693088978619 |
| `ranking/ndcg_minus_mixed/ndcg@10_ci95_low` | -0.0006395982445522135 |
| `ranking/ndcg_minus_mixed/ndcg@5` | -0.0005347114389313389 |
| `ranking/ndcg_minus_mixed/ndcg@5_ci95_high` | 0.0003006892296441193 |
| `ranking/ndcg_minus_mixed/ndcg@5_ci95_low` | -0.001403326997349924 |
| `ranking/ndcg_minus_mixed/recall@10` | 0.0008942944017170453 |
| `ranking/ndcg_minus_mixed/recall@10_ci95_high` | 0.0024145948846360224 |
| `ranking/ndcg_minus_mixed/recall@10_ci95_low` | -0.0005365766410302271 |
| `ranking/ndcg_minus_mixed/recall@5` | -0.0010731532820604545 |
| `ranking/ndcg_minus_mixed/recall@5_ci95_high` | 0.0005365766410302271 |
| `ranking/ndcg_minus_mixed/recall@5_ci95_low` | -0.00277231264532284 |
| `ranking/ndcg_minus_nll/ndcg@10` | 0.00031207003749193367 |
| `ranking/ndcg_minus_nll/ndcg@10_ci95_high` | 0.0009032183529904318 |
| `ranking/ndcg_minus_nll/ndcg@10_ci95_low` | -0.00022264683560076665 |
| `ranking/ndcg_minus_nll/ndcg@5` | 0.0001512033078531911 |
| `ranking/ndcg_minus_nll/ndcg@5_ci95_high` | 0.000797497988340631 |
| `ranking/ndcg_minus_nll/ndcg@5_ci95_low` | -0.0005269303020941747 |
| `ranking/ndcg_minus_nll/recall@10` | 0.0006260060812019317 |
| `ranking/ndcg_minus_nll/recall@10_ci95_high` | 0.0017885888034340906 |
| `ranking/ndcg_minus_nll/recall@10_ci95_low` | -0.00044714720085852264 |
| `ranking/ndcg_minus_nll/recall@5` | 0.00017885888034340904 |
| `ranking/ndcg_minus_nll/recall@5_ci95_high` | 0.0013436773385798483 |
| `ranking/ndcg_minus_nll/recall@5_ci95_low` | -0.0010753890180647467 |
| `ranking/nll/ndcg@10` | 0.047274060070238426 |
| `ranking/nll/ndcg@5` | 0.036618205120973975 |
| `ranking/nll/recall@10` | 0.09050259345376498 |
| `ranking/nll/recall@5` | 0.057413700590234304 |
| `ranking/nll_minus_content/common_hit_rank_ndcg@10` | -9.480060303635256e-05 |
| `ranking/nll_minus_content/common_hit_rank_ndcg@5` | 4.639663371972852e-06 |
| `ranking/nll_minus_content/lost_hit_ndcg@10` | -0.002415552921020452 |
| `ranking/nll_minus_content/lost_hit_ndcg@5` | -0.0029021056042960827 |
| `ranking/nll_minus_content/ndcg@10` | -0.0006626780035738843 |
| `ranking/nll_minus_content/ndcg@10_ci95_high` | 0.000437691751415712 |
| `ranking/nll_minus_content/ndcg@10_ci95_low` | -0.0017273740000145947 |
| `ranking/nll_minus_content/ndcg@5` | -0.0005877188388600732 |
| `ranking/nll_minus_content/ndcg@5_ci95_high` | 0.0006027730888313866 |
| `ranking/nll_minus_content/ndcg@5_ci95_low` | -0.001804990280325573 |
| `ranking/nll_minus_content/new_hit_ndcg@10` | 0.00184767552048292 |
| `ranking/nll_minus_content/new_hit_ndcg@5` | 0.002309747102064037 |
| `ranking/nll_minus_content/recall@10` | -0.002056877123949204 |
| `ranking/nll_minus_content/recall@10_ci95_high` | 8.942944017170452e-05 |
| `ranking/nll_minus_content/recall@10_ci95_low` | -0.004113754247898408 |
| `ranking/nll_minus_content/recall@5` | -0.0017885888034340906 |
| `ranking/nll_minus_content/recall@5_ci95_high` | 0.00026828832051511357 |
| `ranking/nll_minus_content/recall@5_ci95_low` | -0.00375603648721159 |

#### `o7ycqky2` — v4.1-learned-score-bias / validation

| 原 metric key | 原值 |
|---|---:|
| `test/ndcg@10` | 0.0505030519730769 |
| `test/ndcg@5` | 0.03876615456431632 |
| `test/recall@10` | 0.0981084827617046 |
| `test/recall@5` | 0.061664356302821625 |

#### `ohepscqg` — v0 / testing

| 原 metric key | 原值 |
|---|---:|
| `candidate_trace/dense/ndcg@10` | 0.03650093115754803 |
| `candidate_trace/dense/ndcg@5` | 0.02746951679315528 |
| `candidate_trace/dense/recall@10` | 0.07433546260045333 |
| `candidate_trace/dense/recall@5` | 0.04615701627859056 |
| `candidate_trace/hybrid/ndcg@10` | 0.03578612765506389 |
| `candidate_trace/hybrid/ndcg@5` | 0.027336946574006867 |
| `candidate_trace/hybrid/recall@10` | 0.0726869977333608 |
| `candidate_trace/hybrid/recall@5` | 0.04626004533278384 |

#### `p62too4c` — v0 / testing

| 原 metric key | 原值 |
|---|---:|
| `candidate_trace/dense/ndcg@10` | 0.03632927042460379 |
| `candidate_trace/dense/ndcg@5` | 0.027633195494147687 |
| `candidate_trace/dense/recall@10` | 0.07294457036884401 |
| `candidate_trace/dense/recall@5` | 0.04600247269730064 |
| `candidate_trace/hybrid/ndcg@10` | 0.03575046521323423 |
| `candidate_trace/hybrid/ndcg@5` | 0.027439035466157707 |
| `candidate_trace/hybrid/recall@10` | 0.0712961055017515 |
| `candidate_trace/hybrid/recall@5` | 0.045538841953430866 |

#### `p9czcxbg` — early-ranking-decoder-exploration / inference

| 原 metric key | 原值 |
|---|---:|
| `ranking/content/ndcg@10` | 0.04793673807381232 |
| `ranking/content/ndcg@5` | 0.03720592395983405 |
| `ranking/content/recall@10` | 0.0925594705777142 |
| `ranking/content/recall@5` | 0.059202289393668395 |
| `ranking/content_minus_content/common_hit_rank_ndcg@10` | 0 |
| `ranking/content_minus_content/common_hit_rank_ndcg@5` | 0 |
| `ranking/content_minus_content/lost_hit_ndcg@10` | 0 |
| `ranking/content_minus_content/lost_hit_ndcg@5` | 0 |
| `ranking/content_minus_content/new_hit_ndcg@10` | 0 |
| `ranking/content_minus_content/new_hit_ndcg@5` | 0 |
| `ranking/generation/ndcg@10` | 0.03763325189700371 |
| `ranking/generation/ndcg@5` | 0.02691624379270245 |
| `ranking/generation/recall@10` | 0.07851904847075657 |
| `ranking/generation/recall@5` | 0.04471472008585226 |
| `ranking/generation_minus_content/common_hit_rank_ndcg@10` | -0.003596824920374986 |
| `ranking/generation_minus_content/common_hit_rank_ndcg@5` | -0.0019959574935677258 |
| `ranking/generation_minus_content/lost_hit_ndcg@10` | -0.013592869420399764 |
| `ranking/generation_minus_content/lost_hit_ndcg@5` | -0.015698565968530538 |
| `ranking/generation_minus_content/new_hit_ndcg@10` | 0.006886208163966147 |
| `ranking/generation_minus_content/new_hit_ndcg@5` | 0.007404843294966672 |
| `ranking/mixed/ndcg@10` | 0.04749760278576311 |
| `ranking/mixed/ndcg@5` | 0.0373041198677585 |
| `ranking/mixed/recall@10` | 0.09023430513324988 |
| `ranking/mixed/recall@5` | 0.05866571275263817 |
| `ranking/mixed_minus_content/common_hit_rank_ndcg@10` | 0.0002343076763095105 |
| `ranking/mixed_minus_content/common_hit_rank_ndcg@5` | 0.00024923757417190465 |
| `ranking/mixed_minus_content/lost_hit_ndcg@10` | -0.0022267465938163138 |
| `ranking/mixed_minus_content/lost_hit_ndcg@5` | -0.002193142160489687 |
| `ranking/mixed_minus_content/new_hit_ndcg@10` | 0.0015533036294576037 |
| `ranking/mixed_minus_content/new_hit_ndcg@5` | 0.0020421004942422385 |
| `ranking/ndcg/cold_hits@10` | 0 |
| `ranking/ndcg/cold_hits@5` | 0 |
| `ranking/ndcg/ndcg@10` | 0.04733685781391438 |
| `ranking/ndcg/ndcg@5` | 0.036640676834636024 |
| `ranking/ndcg/recall@10` | 0.09059202289393668 |
| `ranking/ndcg/recall@5` | 0.05750313003040601 |
| `ranking/ndcg_minus_content/common_hit_rank_ndcg@10` | -1.4867753174275164e-05 |
| `ranking/ndcg_minus_content/common_hit_rank_ndcg@5` | -2.8753808012424528e-06 |
| `ranking/ndcg_minus_content/lost_hit_ndcg@10` | -0.002582171164933878 |
| `ranking/ndcg_minus_content/lost_hit_ndcg@5` | -0.003347293208214126 |
| `ranking/ndcg_minus_content/lost_hits@10` | 91 |
| `ranking/ndcg_minus_content/lost_hits@5` | 87 |
| `ranking/ndcg_minus_content/ndcg@10` | -0.0005998802598979281 |
| `ranking/ndcg_minus_content/ndcg@10_ci95_high` | 0.0005035325728457036 |
| `ranking/ndcg_minus_content/ndcg@10_ci95_low` | -0.0017148418556036488 |
| `ranking/ndcg_minus_content/ndcg@5` | -0.0005652471251980236 |
| `ranking/ndcg_minus_content/ndcg@5_ci95_high` | 0.0006944133814701321 |
| `ranking/ndcg_minus_content/ndcg@5_ci95_low` | -0.0018734401266728951 |
| `ranking/ndcg_minus_content/new_hit_ndcg@10` | 0.0019971586582102257 |
| `ranking/ndcg_minus_content/new_hit_ndcg@5` | 0.0027849214638173444 |
| `ranking/ndcg_minus_content/new_hits@10` | 69 |
| `ranking/ndcg_minus_content/new_hits@5` | 68 |
| `ranking/ndcg_minus_content/recall@10` | -0.0019674476837774997 |
| `ranking/ndcg_minus_content/recall@10_ci95_high` | 0.00026828832051511357 |
| `ranking/ndcg_minus_content/recall@10_ci95_low` | -0.004115989983902701 |
| `ranking/ndcg_minus_content/recall@5` | -0.001699159363262386 |
| `ranking/ndcg_minus_content/recall@5_ci95_high` | 0.00044714720085852264 |
| `ranking/ndcg_minus_content/recall@5_ci95_low` | -0.003937131103559292 |
| `ranking/ndcg_minus_mixed/ndcg@10` | -0.00016074497184872852 |
| `ranking/ndcg_minus_mixed/ndcg@10_ci95_high` | 0.0006321851902438897 |
| `ranking/ndcg_minus_mixed/ndcg@10_ci95_low` | -0.0008829973667107009 |
| `ranking/ndcg_minus_mixed/ndcg@5` | -0.0006634430331224801 |
| `ranking/ndcg_minus_mixed/ndcg@5_ci95_high` | 0.00021207108080550577 |
| `ranking/ndcg_minus_mixed/ndcg@5_ci95_low` | -0.0015009560000431618 |
| `ranking/ndcg_minus_mixed/recall@10` | 0.0003577177606868181 |
| `ranking/ndcg_minus_mixed/recall@10_ci95_high` | 0.0019674476837774997 |
| `ranking/ndcg_minus_mixed/recall@10_ci95_low` | -0.0012520121624038634 |
| `ranking/ndcg_minus_mixed/recall@5` | -0.0011625827222321587 |
| `ranking/ndcg_minus_mixed/recall@5_ci95_high` | 0.00044714720085852264 |
| `ranking/ndcg_minus_mixed/recall@5_ci95_low` | -0.00277231264532284 |
| `ranking/ndcg_minus_nll/ndcg@10` | 6.279774367595639e-05 |
| `ranking/ndcg_minus_nll/ndcg@10_ci95_high` | 0.0006449084746689069 |
| `ranking/ndcg_minus_nll/ndcg@10_ci95_low` | -0.0005056006854635954 |
| `ranking/ndcg_minus_nll/ndcg@5` | 2.2471713662049675e-05 |
| `ranking/ndcg_minus_nll/ndcg@5_ci95_high` | 0.0006719510659916793 |
| `ranking/ndcg_minus_nll/ndcg@5_ci95_low` | -0.0006507255631287717 |
| `ranking/ndcg_minus_nll/recall@10` | 8.942944017170452e-05 |
| `ranking/ndcg_minus_nll/recall@10_ci95_high` | 0.0014308710427472723 |
| `ranking/ndcg_minus_nll/recall@10_ci95_low` | -0.0011625827222321587 |
| `ranking/ndcg_minus_nll/recall@5` | 8.942944017170452e-05 |
| `ranking/ndcg_minus_nll/recall@5_ci95_high` | 0.001341441602575568 |
| `ranking/ndcg_minus_nll/recall@5_ci95_low` | -0.0012520121624038634 |
| `ranking/nll/ndcg@10` | 0.047274060070238426 |
| `ranking/nll/ndcg@5` | 0.036618205120973975 |
| `ranking/nll/recall@10` | 0.09050259345376498 |
| `ranking/nll/recall@5` | 0.057413700590234304 |
| `ranking/nll_minus_content/common_hit_rank_ndcg@10` | -9.480060303635256e-05 |
| `ranking/nll_minus_content/common_hit_rank_ndcg@5` | 4.639663371972852e-06 |
| `ranking/nll_minus_content/lost_hit_ndcg@10` | -0.002415552921020452 |
| `ranking/nll_minus_content/lost_hit_ndcg@5` | -0.0029021056042960827 |
| `ranking/nll_minus_content/ndcg@10` | -0.0006626780035738843 |
| `ranking/nll_minus_content/ndcg@10_ci95_high` | 0.000437691751415712 |
| `ranking/nll_minus_content/ndcg@10_ci95_low` | -0.0017273740000145947 |
| `ranking/nll_minus_content/ndcg@5` | -0.0005877188388600732 |
| `ranking/nll_minus_content/ndcg@5_ci95_high` | 0.0006027730888313866 |
| `ranking/nll_minus_content/ndcg@5_ci95_low` | -0.001804990280325573 |
| `ranking/nll_minus_content/new_hit_ndcg@10` | 0.00184767552048292 |
| `ranking/nll_minus_content/new_hit_ndcg@5` | 0.002309747102064037 |
| `ranking/nll_minus_content/recall@10` | -0.002056877123949204 |
| `ranking/nll_minus_content/recall@10_ci95_high` | 8.942944017170452e-05 |
| `ranking/nll_minus_content/recall@10_ci95_low` | -0.004113754247898408 |
| `ranking/nll_minus_content/recall@5` | -0.0017885888034340906 |
| `ranking/nll_minus_content/recall@5_ci95_high` | 0.00026828832051511357 |
| `ranking/nll_minus_content/recall@5_ci95_low` | -0.00375603648721159 |

#### `p9qj0ccg` — v0 / training

| 原 metric key | 原值 |
|---|---:|
| `val/ndcg@10` | 0.026096943765878677 |
| `val/ndcg@5` | 0.019729452207684517 |
| `val/recall@10` | 0.05048133432865143 |
| `val/recall@5` | 0.03070438653230667 |

#### `pvhjwp9q` — early-ranking-decoder-exploration / inference

| 原 metric key | 原值 |
|---|---:|
| `ranking/content/ndcg@10` | 0.04793673807381232 |
| `ranking/content/ndcg@5` | 0.03720592395983405 |
| `ranking/content/recall@10` | 0.0925594705777142 |
| `ranking/content/recall@5` | 0.059202289393668395 |
| `ranking/content_minus_content/common_hit_rank_ndcg@10` | 0 |
| `ranking/content_minus_content/common_hit_rank_ndcg@5` | 0 |
| `ranking/content_minus_content/lost_hit_ndcg@10` | 0 |
| `ranking/content_minus_content/lost_hit_ndcg@5` | 0 |
| `ranking/content_minus_content/new_hit_ndcg@10` | 0 |
| `ranking/content_minus_content/new_hit_ndcg@5` | 0 |
| `ranking/generation/ndcg@10` | 0.03763325189700371 |
| `ranking/generation/ndcg@5` | 0.02691624379270245 |
| `ranking/generation/recall@10` | 0.07851904847075657 |
| `ranking/generation/recall@5` | 0.04471472008585226 |
| `ranking/generation_minus_content/common_hit_rank_ndcg@10` | -0.003596824920374986 |
| `ranking/generation_minus_content/common_hit_rank_ndcg@5` | -0.0019959574935677258 |
| `ranking/generation_minus_content/lost_hit_ndcg@10` | -0.013592869420399764 |
| `ranking/generation_minus_content/lost_hit_ndcg@5` | -0.015698565968530538 |
| `ranking/generation_minus_content/new_hit_ndcg@10` | 0.006886208163966147 |
| `ranking/generation_minus_content/new_hit_ndcg@5` | 0.007404843294966672 |
| `ranking/mixed/ndcg@10` | 0.04749760278576311 |
| `ranking/mixed/ndcg@5` | 0.0373041198677585 |
| `ranking/mixed/recall@10` | 0.09023430513324988 |
| `ranking/mixed/recall@5` | 0.05866571275263817 |
| `ranking/mixed_minus_content/common_hit_rank_ndcg@10` | 0.0002343076763095105 |
| `ranking/mixed_minus_content/common_hit_rank_ndcg@5` | 0.00024923757417190465 |
| `ranking/mixed_minus_content/lost_hit_ndcg@10` | -0.0022267465938163138 |
| `ranking/mixed_minus_content/lost_hit_ndcg@5` | -0.002193142160489687 |
| `ranking/mixed_minus_content/new_hit_ndcg@10` | 0.0015533036294576037 |
| `ranking/mixed_minus_content/new_hit_ndcg@5` | 0.0020421004942422385 |
| `ranking/ndcg/cold_hits@10` | 0 |
| `ranking/ndcg/cold_hits@5` | 0 |
| `ranking/ndcg/ndcg@10` | 0.04753854620459063 |
| `ranking/ndcg/ndcg@5` | 0.03652702854809126 |
| `ranking/ndcg/recall@10` | 0.09148631729565374 |
| `ranking/ndcg/recall@5` | 0.0573242711500626 |
| `ranking/ndcg_minus_content/common_hit_rank_ndcg@10` | -0.00011201280483301756 |
| `ranking/ndcg_minus_content/common_hit_rank_ndcg@5` | -4.966045643236747e-05 |
| `ranking/ndcg_minus_content/lost_hit_ndcg@10` | -0.0023355447816590675 |
| `ranking/ndcg_minus_content/lost_hit_ndcg@5` | -0.0032333864281685987 |
| `ranking/ndcg_minus_content/lost_hits@10` | 83 |
| `ranking/ndcg_minus_content/lost_hits@5` | 84 |
| `ranking/ndcg_minus_content/ndcg@10` | -0.00039819186922168584 |
| `ranking/ndcg_minus_content/ndcg@10_ci95_high` | 0.0006879936487701545 |
| `ranking/ndcg_minus_content/ndcg@10_ci95_low` | -0.0014555053440870303 |
| `ranking/ndcg_minus_content/ndcg@5` | -0.0006788954117427803 |
| `ranking/ndcg_minus_content/ndcg@5_ci95_high` | 0.0005653796109208611 |
| `ranking/ndcg_minus_content/ndcg@5_ci95_low` | -0.001949675697184621 |
| `ranking/ndcg_minus_content/new_hit_ndcg@10` | 0.0020493657172703994 |
| `ranking/ndcg_minus_content/new_hit_ndcg@5` | 0.002604151472858185 |
| `ranking/ndcg_minus_content/new_hits@10` | 71 |
| `ranking/ndcg_minus_content/new_hits@5` | 63 |
| `ranking/ndcg_minus_content/recall@10` | -0.0010731532820604545 |
| `ranking/ndcg_minus_content/recall@10_ci95_high` | 0.0011625827222321587 |
| `ranking/ndcg_minus_content/recall@10_ci95_low` | -0.003219459846181363 |
| `ranking/ndcg_minus_content/recall@5` | -0.001878018243605795 |
| `ranking/ndcg_minus_content/recall@5_ci95_high` | 0.00018109461634768944 |
| `ranking/ndcg_minus_content/recall@5_ci95_low` | -0.004113754247898408 |
| `ranking/ndcg_minus_mixed/ndcg@10` | 4.094341882751353e-05 |
| `ranking/ndcg_minus_mixed/ndcg@10_ci95_high` | 0.0007309589808245802 |
| `ranking/ndcg_minus_mixed/ndcg@10_ci95_low` | -0.000675589443551001 |
| `ranking/ndcg_minus_mixed/ndcg@5` | -0.0007770913196672371 |
| `ranking/ndcg_minus_mixed/ndcg@5_ci95_high` | 1.83960238362774e-05 |
| `ranking/ndcg_minus_mixed/ndcg@5_ci95_low` | -0.0015719193449063665 |
| `ranking/ndcg_minus_mixed/recall@10` | 0.0012520121624038634 |
| `ranking/ndcg_minus_mixed/recall@10_ci95_high` | 0.00277231264532284 |
| `ranking/ndcg_minus_mixed/recall@10_ci95_low` | -0.00017885888034340904 |
| `ranking/ndcg_minus_mixed/recall@5` | -0.001341441602575568 |
| `ranking/ndcg_minus_mixed/recall@5_ci95_high` | 0.00017885888034340904 |
| `ranking/ndcg_minus_mixed/recall@5_ci95_low` | -0.0029511715256662495 |
| `ranking/ndcg_minus_nll/ndcg@10` | 0.0002644861343521985 |
| `ranking/ndcg_minus_nll/ndcg@10_ci95_high` | 0.0007342060705766286 |
| `ranking/ndcg_minus_nll/ndcg@10_ci95_low` | -0.0002080335315521559 |
| `ranking/ndcg_minus_nll/ndcg@5` | -9.117657288270716e-05 |
| `ranking/ndcg_minus_nll/ndcg@5_ci95_high` | 0.0004557362982348438 |
| `ranking/ndcg_minus_nll/ndcg@5_ci95_low` | -0.0006830734467901627 |
| `ranking/ndcg_minus_nll/recall@10` | 0.0009837238418887498 |
| `ranking/ndcg_minus_nll/recall@10_ci95_high` | 0.002056877123949204 |
| `ranking/ndcg_minus_nll/recall@10_ci95_low` | 0 |
| `ranking/ndcg_minus_nll/recall@5` | -8.942944017170452e-05 |
| `ranking/ndcg_minus_nll/recall@5_ci95_high` | 0.0009837238418887498 |
| `ranking/ndcg_minus_nll/recall@5_ci95_low` | -0.0012520121624038634 |
| `ranking/nll/ndcg@10` | 0.047274060070238426 |
| `ranking/nll/ndcg@5` | 0.036618205120973975 |
| `ranking/nll/recall@10` | 0.09050259345376498 |
| `ranking/nll/recall@5` | 0.057413700590234304 |
| `ranking/nll_minus_content/common_hit_rank_ndcg@10` | -9.480060303635256e-05 |
| `ranking/nll_minus_content/common_hit_rank_ndcg@5` | 4.639663371972852e-06 |
| `ranking/nll_minus_content/lost_hit_ndcg@10` | -0.002415552921020452 |
| `ranking/nll_minus_content/lost_hit_ndcg@5` | -0.0029021056042960827 |
| `ranking/nll_minus_content/ndcg@10` | -0.0006626780035738843 |
| `ranking/nll_minus_content/ndcg@10_ci95_high` | 0.000437691751415712 |
| `ranking/nll_minus_content/ndcg@10_ci95_low` | -0.0017273740000145947 |
| `ranking/nll_minus_content/ndcg@5` | -0.0005877188388600732 |
| `ranking/nll_minus_content/ndcg@5_ci95_high` | 0.0006027730888313866 |
| `ranking/nll_minus_content/ndcg@5_ci95_low` | -0.001804990280325573 |
| `ranking/nll_minus_content/new_hit_ndcg@10` | 0.00184767552048292 |
| `ranking/nll_minus_content/new_hit_ndcg@5` | 0.002309747102064037 |
| `ranking/nll_minus_content/recall@10` | -0.002056877123949204 |
| `ranking/nll_minus_content/recall@10_ci95_high` | 8.942944017170452e-05 |
| `ranking/nll_minus_content/recall@10_ci95_low` | -0.004113754247898408 |
| `ranking/nll_minus_content/recall@5` | -0.0017885888034340906 |
| `ranking/nll_minus_content/recall@5_ci95_high` | 0.00026828832051511357 |
| `ranking/nll_minus_content/recall@5_ci95_low` | -0.00375603648721159 |

#### `q2du3vc5` — v0 / testing

| 原 metric key | 原值 |
|---|---:|
| `candidate_trace/dense/ndcg@10` | 0.019825103345177578 |
| `candidate_trace/dense/ndcg@5` | 0.015056844178630274 |
| `candidate_trace/dense/recall@10` | 0.03837294230012922 |
| `candidate_trace/dense/recall@5` | 0.023484465419405583 |
| `candidate_trace/hybrid/ndcg@10` | 0.019674617939506013 |
| `candidate_trace/hybrid/ndcg@5` | 0.014886560403798163 |
| `candidate_trace/hybrid/recall@10` | 0.03823248497106579 |
| `candidate_trace/hybrid/recall@5` | 0.02328782515871678 |

#### `rbha00vx` — v3 / training

| 原 metric key | 原值 |
|---|---:|
| `val/ndcg@10` | 0.045266613364219666 |
| `val/ndcg@5` | 0.03517000377178192 |
| `val/recall@10` | 0.08888047933578491 |
| `val/recall@5` | 0.057531170547008514 |

#### `rha8mrvs` — v5.2 / training

| 原 metric key | 原值 |
|---|---:|
| `val/ndcg@10` | 0.05236941576004028 |
| `val/ndcg@5` | 0.042724378407001495 |
| `val/recall@10` | 0.09403926134109496 |
| `val/recall@5` | 0.06416849046945572 |

#### `ri2813bi` — early-ranking-decoder-exploration / inference

| 原 metric key | 原值 |
|---|---:|
| `ranking/content/ndcg@10` | 0.04793673807381232 |
| `ranking/content/ndcg@5` | 0.03720592395983405 |
| `ranking/content/recall@10` | 0.0925594705777142 |
| `ranking/content/recall@5` | 0.059202289393668395 |
| `ranking/content_minus_content/common_hit_rank_ndcg@10` | 0 |
| `ranking/content_minus_content/common_hit_rank_ndcg@5` | 0 |
| `ranking/content_minus_content/lost_hit_ndcg@10` | 0 |
| `ranking/content_minus_content/lost_hit_ndcg@5` | 0 |
| `ranking/content_minus_content/new_hit_ndcg@10` | 0 |
| `ranking/content_minus_content/new_hit_ndcg@5` | 0 |
| `ranking/generation/ndcg@10` | 0.03763325189700371 |
| `ranking/generation/ndcg@5` | 0.02691624379270245 |
| `ranking/generation/recall@10` | 0.07851904847075657 |
| `ranking/generation/recall@5` | 0.04471472008585226 |
| `ranking/generation_minus_content/common_hit_rank_ndcg@10` | -0.003596824920374986 |
| `ranking/generation_minus_content/common_hit_rank_ndcg@5` | -0.0019959574935677258 |
| `ranking/generation_minus_content/lost_hit_ndcg@10` | -0.013592869420399764 |
| `ranking/generation_minus_content/lost_hit_ndcg@5` | -0.015698565968530538 |
| `ranking/generation_minus_content/new_hit_ndcg@10` | 0.006886208163966147 |
| `ranking/generation_minus_content/new_hit_ndcg@5` | 0.007404843294966672 |
| `ranking/mixed/ndcg@10` | 0.04749760278576311 |
| `ranking/mixed/ndcg@5` | 0.0373041198677585 |
| `ranking/mixed/recall@10` | 0.09023430513324988 |
| `ranking/mixed/recall@5` | 0.05866571275263817 |
| `ranking/mixed_minus_content/common_hit_rank_ndcg@10` | 0.0002343076763095105 |
| `ranking/mixed_minus_content/common_hit_rank_ndcg@5` | 0.00024923757417190465 |
| `ranking/mixed_minus_content/lost_hit_ndcg@10` | -0.0022267465938163138 |
| `ranking/mixed_minus_content/lost_hit_ndcg@5` | -0.002193142160489687 |
| `ranking/mixed_minus_content/new_hit_ndcg@10` | 0.0015533036294576037 |
| `ranking/mixed_minus_content/new_hit_ndcg@5` | 0.0020421004942422385 |
| `ranking/ndcg/cold_hits@10` | 0 |
| `ranking/ndcg/cold_hits@5` | 0 |
| `ranking/ndcg/ndcg@10` | 0.0474989265671284 |
| `ranking/ndcg/ndcg@5` | 0.036923049008032185 |
| `ranking/ndcg/recall@10` | 0.09050259345376498 |
| `ranking/ndcg/recall@5` | 0.05768198891074942 |
| `ranking/ndcg_minus_content/common_hit_rank_ndcg@10` | 0.00013860414690224085 |
| `ranking/ndcg_minus_content/common_hit_rank_ndcg@5` | 0.0002415609487701622 |
| `ranking/ndcg_minus_content/lost_hit_ndcg@10` | -0.0024242980454093495 |
| `ranking/ndcg_minus_content/lost_hit_ndcg@5` | -0.002875299415376217 |
| `ranking/ndcg_minus_content/lost_hits@10` | 86 |
| `ranking/ndcg_minus_content/lost_hits@5` | 76 |
| `ranking/ndcg_minus_content/ndcg@10` | -0.00043781150668391745 |
| `ranking/ndcg_minus_content/ndcg@10_ci95_high` | 0.0006598145879045601 |
| `ranking/ndcg_minus_content/ndcg@10_ci95_low` | -0.0015234264694811868 |
| `ranking/ndcg_minus_content/ndcg@5` | -0.0002828749518018589 |
| `ranking/ndcg_minus_content/ndcg@5_ci95_high` | 0.0009002255029969756 |
| `ranking/ndcg_minus_content/ndcg@5_ci95_low` | -0.0014591857790542016 |
| `ranking/ndcg_minus_content/new_hit_ndcg@10` | 0.0018478823918231912 |
| `ranking/ndcg_minus_content/new_hit_ndcg@5` | 0.002350863514804196 |
| `ranking/ndcg_minus_content/new_hits@10` | 63 |
| `ranking/ndcg_minus_content/new_hits@5` | 59 |
| `ranking/ndcg_minus_content/recall@10` | -0.002056877123949204 |
| `ranking/ndcg_minus_content/recall@10_ci95_high` | 8.942944017170452e-05 |
| `ranking/ndcg_minus_content/recall@10_ci95_low` | -0.004203183688070112 |
| `ranking/ndcg_minus_content/recall@5` | -0.001520300482918977 |
| `ranking/ndcg_minus_content/recall@5_ci95_high` | 0.0005365766410302271 |
| `ranking/ndcg_minus_content/recall@5_ci95_low` | -0.0034877481666964767 |
| `ranking/ndcg_minus_mixed/ndcg@10` | 1.3237813652822428e-06 |
| `ranking/ndcg_minus_mixed/ndcg@10_ci95_high` | 0.0005525374066697843 |
| `ranking/ndcg_minus_mixed/ndcg@10_ci95_low` | -0.0005265841072861645 |
| `ranking/ndcg_minus_mixed/ndcg@5` | -0.0003810708597263155 |
| `ranking/ndcg_minus_mixed/ndcg@5_ci95_high` | 0.00018615533754846815 |
| `ranking/ndcg_minus_mixed/ndcg@5_ci95_low` | -0.0009694743435081592 |
| `ranking/ndcg_minus_mixed/recall@10` | 0.00026828832051511357 |
| `ranking/ndcg_minus_mixed/recall@10_ci95_high` | 0.001520300482918977 |
| `ranking/ndcg_minus_mixed/recall@10_ci95_low` | -0.0009837238418887498 |
| `ranking/ndcg_minus_mixed/recall@5` | -0.0009837238418887498 |
| `ranking/ndcg_minus_mixed/recall@5_ci95_high` | 0.00017885888034340904 |
| `ranking/ndcg_minus_mixed/recall@5_ci95_low` | -0.002146306564120909 |
| `ranking/ndcg_minus_nll/ndcg@10` | 0.00022486649688996705 |
| `ranking/ndcg_minus_nll/ndcg@10_ci95_high` | 0.0004362075195378542 |
| `ranking/ndcg_minus_nll/ndcg@10_ci95_low` | 1.0062441943392968e-05 |
| `ranking/ndcg_minus_nll/ndcg@5` | 0.0003048438870582143 |
| `ranking/ndcg_minus_nll/ndcg@5_ci95_high` | 0.000606966664543127 |
| `ranking/ndcg_minus_nll/ndcg@5_ci95_low` | 6.435484570974176e-06 |
| `ranking/ndcg_minus_nll/recall@10` | 0 |
| `ranking/ndcg_minus_nll/recall@10_ci95_high` | 0.0003577177606868181 |
| `ranking/ndcg_minus_nll/recall@10_ci95_low` | -0.0003577177606868181 |
| `ranking/ndcg_minus_nll/recall@5` | 0.00026828832051511357 |
| `ranking/ndcg_minus_nll/recall@5_ci95_high` | 0.0008942944017170453 |
| `ranking/ndcg_minus_nll/recall@5_ci95_low` | -0.00027052405651940603 |
| `ranking/nll/ndcg@10` | 0.047274060070238426 |
| `ranking/nll/ndcg@5` | 0.036618205120973975 |
| `ranking/nll/recall@10` | 0.09050259345376498 |
| `ranking/nll/recall@5` | 0.057413700590234304 |
| `ranking/nll_minus_content/common_hit_rank_ndcg@10` | -9.480060303635256e-05 |
| `ranking/nll_minus_content/common_hit_rank_ndcg@5` | 4.639663371972852e-06 |
| `ranking/nll_minus_content/lost_hit_ndcg@10` | -0.002415552921020452 |
| `ranking/nll_minus_content/lost_hit_ndcg@5` | -0.0029021056042960827 |
| `ranking/nll_minus_content/ndcg@10` | -0.0006626780035738843 |
| `ranking/nll_minus_content/ndcg@10_ci95_high` | 0.000437691751415712 |
| `ranking/nll_minus_content/ndcg@10_ci95_low` | -0.0017273740000145947 |
| `ranking/nll_minus_content/ndcg@5` | -0.0005877188388600732 |
| `ranking/nll_minus_content/ndcg@5_ci95_high` | 0.0006027730888313866 |
| `ranking/nll_minus_content/ndcg@5_ci95_low` | -0.001804990280325573 |
| `ranking/nll_minus_content/new_hit_ndcg@10` | 0.00184767552048292 |
| `ranking/nll_minus_content/new_hit_ndcg@5` | 0.002309747102064037 |
| `ranking/nll_minus_content/recall@10` | -0.002056877123949204 |
| `ranking/nll_minus_content/recall@10_ci95_high` | 8.942944017170452e-05 |
| `ranking/nll_minus_content/recall@10_ci95_low` | -0.004113754247898408 |
| `ranking/nll_minus_content/recall@5` | -0.0017885888034340906 |
| `ranking/nll_minus_content/recall@5_ci95_high` | 0.00026828832051511357 |
| `ranking/nll_minus_content/recall@5_ci95_low` | -0.00375603648721159 |

#### `rksx8qyh` — v4.4-history-ce-control-pool / validation

| 原 metric key | 原值 |
|---|---:|
| `test/ndcg@10` | 0.060622668609856434 |
| `test/ndcg@5` | 0.04893725263315011 |
| `test/recall@10` | 0.10785672763046104 |
| `test/recall@5` | 0.07159146804990386 |

#### `s9nrva4m` — early-ranking-decoder-exploration / train

| 原 metric key | 原值 |
|---|---:|
| `val/ndcg@10` | 0.04699907824397087 |
| `val/recall@10` | 0.09212055802345276 |

#### `sd90tlbj` — v0-mechanism-legal / testing_mechanism_legal

| 原 metric key | 原值 |
|---|---:|
| `candidate_trace/dense/ndcg@10` | 0.020149172152384317 |
| `candidate_trace/dense/ndcg@5` | 0.015081243196054145 |
| `candidate_trace/dense/recall@10` | 0.03904713747963369 |
| `candidate_trace/dense/recall@5` | 0.02323164222709141 |
| `candidate_trace/hybrid/ndcg@10` | 0.01676156940901572 |
| `candidate_trace/hybrid/ndcg@5` | 0.013427204976121624 |
| `candidate_trace/hybrid/recall@10` | 0.03109725265464352 |
| `candidate_trace/hybrid/recall@5` | 0.02075959323557503 |

#### `sgmu9l3l` — v0-mechanism-max / testing_mechanism_max

| 原 metric key | 原值 |
|---|---:|
| `candidate_trace/dense/ndcg@10` | 0.03652606072154706 |
| `candidate_trace/dense/ndcg@5` | 0.0272079896834676 |
| `candidate_trace/dense/recall@10` | 0.07221750212404418 |
| `candidate_trace/dense/recall@5` | 0.04337521799400796 |
| `candidate_trace/hybrid/ndcg@10` | 0.03647592579904166 |
| `candidate_trace/hybrid/ndcg@5` | 0.027238088606149777 |
| `candidate_trace/hybrid/recall@10` | 0.07212806868488128 |
| `candidate_trace/hybrid/recall@5` | 0.04350936815275232 |

#### `siiokyhl` — v0 / testing

| 原 metric key | 原值 |
|---|---:|
| `candidate_trace/dense/ndcg@10` | 0.020149172152384317 |
| `candidate_trace/dense/ndcg@5` | 0.015081243196054145 |
| `candidate_trace/dense/recall@10` | 0.03904713747963369 |
| `candidate_trace/dense/recall@5` | 0.02323164222709141 |
| `candidate_trace/hybrid/ndcg@10` | 0.020141558511121305 |
| `candidate_trace/hybrid/ndcg@5` | 0.015049019798877896 |
| `candidate_trace/hybrid/recall@10` | 0.039019046013821 |
| `candidate_trace/hybrid/recall@5` | 0.023175459295466036 |

#### `sw8llh9g` — early-ranking-decoder-exploration / train

| 原 metric key | 原值 |
|---|---:|
| `val/ndcg@10` | 0.044955216348171234 |
| `val/recall@10` | 0.08630713075399399 |

#### `szuw834d` — early-ranking-decoder-exploration / train

| 原 metric key | 原值 |
|---|---:|
| `val/ndcg@10` | 0.047054894268512726 |
| `val/recall@10` | 0.09203112125396729 |

#### `tamqgytq` — v0 / training

| 原 metric key | 原值 |
|---|---:|
| `val/ndcg@10` | 0.026278138160705566 |
| `val/ndcg@5` | 0.020230745896697044 |
| `val/recall@10` | 0.0505651980638504 |
| `val/recall@5` | 0.03182786703109741 |

#### `tz88ztdn` — v5 / validation_startup_failed

W&B summary 中没有 Recall/NDCG；不登记推荐性能数值。

#### `uzmkrfoa` — v1 / already_missing

W&B summary 中没有 Recall/NDCG；不登记推荐性能数值。

#### `valgk6yg` — v0 / testing

| 原 metric key | 原值 |
|---|---:|
| `candidate_trace/dense/ndcg@10` | 0.036598312369094214 |
| `candidate_trace/dense/ndcg@5` | 0.027389204123284657 |
| `candidate_trace/dense/recall@10` | 0.07500515145270967 |
| `candidate_trace/dense/recall@5` | 0.04636307438697713 |
| `candidate_trace/hybrid/ndcg@10` | 0.036000812545409065 |
| `candidate_trace/hybrid/ndcg@5` | 0.026863075590679377 |
| `candidate_trace/hybrid/recall@10` | 0.07351123016690707 |
| `candidate_trace/hybrid/recall@5` | 0.045075211209561095 |

#### `w19eutzj` — v5.1 / validation

| 原 metric key | 原值 |
|---|---:|
| `test/ndcg@10` | 0.05584562468008189 |
| `test/ndcg@5` | 0.046960045676186245 |
| `test/recall@10` | 0.09538076286723604 |
| `test/recall@5` | 0.06779054688548049 |

#### `ws2fx4oi` — v4.4-history-ce-control-pool / testing

| 原 metric key | 原值 |
|---|---:|
| `test/ndcg@10` | 0.04742966406441522 |
| `test/ndcg@5` | 0.038636380947001586 |
| `test/recall@10` | 0.08554308455931672 |
| `test/recall@5` | 0.05826588561463131 |

#### `x4ge0y88` — v4.4-history-ce-treated / training

| 原 metric key | 原值 |
|---|---:|
| `val/ndcg@10` | 0.0593431368470192 |
| `val/ndcg@5` | 0.048632796853780746 |
| `val/recall@10` | 0.10535258799791336 |
| `val/recall@5` | 0.07208335399627686 |

#### `x70hphts` — v0-fixed-training-alpha08 / testing

| 原 metric key | 原值 |
|---|---:|
| `candidate_trace/dense/ndcg@10` | 0.035554398719561696 |
| `candidate_trace/dense/ndcg@5` | 0.027135051642582675 |
| `candidate_trace/dense/recall@10` | 0.07016053302329742 |
| `candidate_trace/dense/recall@5` | 0.044090685507311184 |
| `candidate_trace/hybrid/ndcg@10` | 0.03550118133058993 |
| `candidate_trace/hybrid/ndcg@5` | 0.02713953309689055 |
| `candidate_trace/hybrid/recall@10` | 0.0697133658274829 |
| `candidate_trace/hybrid/recall@5` | 0.04382238518982248 |
| `test/ndcg@10` | 0.03550118133058993 |
| `test/ndcg@5` | 0.027139533096890604 |
| `test/recall@10` | 0.0697133658274829 |
| `test/recall@5` | 0.04382238518982248 |

#### `y9earebw` — v5.4 / validation

| 原 metric key | 原值 |
|---|---:|
| `test/ndcg@10` | 0.05037763791396122 |
| `test/ndcg@5` | 0.04058312319191654 |
| `test/recall@10` | 0.0903724902741135 |
| `test/recall@5` | 0.05992040423914502 |

#### `yk7j9k5z` — v0 / training

| 原 metric key | 原值 |
|---|---:|
| `val/ndcg@10` | 0.04738222062587738 |
| `val/ndcg@5` | 0.03641679883003235 |
| `val/recall@10` | 0.09312918037176132 |
| `val/recall@5` | 0.05901613086462021 |

#### `yqsmsdt1` — v4.1-zero-score-bias-control / validation

| 原 metric key | 原值 |
|---|---:|
| `test/ndcg@10` | 0.05042652891207884 |
| `test/ndcg@5` | 0.03853957602758952 |
| `test/recall@10` | 0.09877923355542638 |
| `test/recall@5` | 0.061753789741984526 |

#### `z8q74kll` — v0 / testing

| 原 metric key | 原值 |
|---|---:|
| `candidate_trace/dense/ndcg@10` | 0.019965945471718795 |
| `candidate_trace/dense/ndcg@5` | 0.015177863137799338 |
| `candidate_trace/dense/recall@10` | 0.038260576436878475 |
| `candidate_trace/dense/recall@5` | 0.02328782515871678 |
| `candidate_trace/hybrid/ndcg@10` | 0.020228592691231964 |
| `candidate_trace/hybrid/ndcg@5` | 0.015381974652057267 |
| `candidate_trace/hybrid/recall@10` | 0.03879431428731951 |
| `candidate_trace/hybrid/recall@5` | 0.023596831282656328 |

#### `z8ymlal9` — v0-mechanism-max / testing_mechanism_max

| 原 metric key | 原值 |
|---|---:|
| `candidate_trace/dense/ndcg@10` | 0.020149172152384317 |
| `candidate_trace/dense/ndcg@5` | 0.015081243196054145 |
| `candidate_trace/dense/recall@10` | 0.03904713747963369 |
| `candidate_trace/dense/recall@5` | 0.02323164222709141 |
| `candidate_trace/hybrid/ndcg@10` | 0.019969997312914937 |
| `candidate_trace/hybrid/ndcg@5` | 0.015101025936448995 |
| `candidate_trace/hybrid/recall@10` | 0.03851339962919265 |
| `candidate_trace/hybrid/recall@5` | 0.023259733692904096 |

#### `z9envq1n` — v4-residual-lr10 / training

| 原 metric key | 原值 |
|---|---:|
| `val/ndcg@10` | 0.04874040186405182 |
| `val/ndcg@5` | 0.0382160022854805 |
| `val/recall@10` | 0.0939498245716095 |
| `val/recall@5` | 0.06126190721988678 |

#### `zy8946l2` — v3 / testing

| 原 metric key | 原值 |
|---|---:|
| `candidate_trace/dense/ndcg@10` | 0.03578160777512011 |
| `candidate_trace/dense/ndcg@5` | 0.027653124522355486 |
| `candidate_trace/dense/recall@10` | 0.07038411662120467 |
| `candidate_trace/dense/recall@5` | 0.045029736618521665 |
| `candidate_trace/hybrid/ndcg@10` | 0.035701458976860896 |
| `candidate_trace/hybrid/ndcg@5` | 0.02788592003253926 |
| `candidate_trace/hybrid/recall@10` | 0.0698922327058087 |
| `candidate_trace/hybrid/recall@5` | 0.04556633725349908 |
| `test/ndcg@10` | 0.035701458976860875 |
| `test/ndcg@5` | 0.027885920032539296 |
| `test/recall@10` | 0.0698922327058087 |
| `test/recall@5` | 0.04556633725349908 |

#### `zy940m5u` — early-ranking-decoder-exploration / inference

| 原 metric key | 原值 |
|---|---:|
| `ranking/content/ndcg@10` | 0.04793673807381232 |
| `ranking/content/ndcg@5` | 0.03720592395983405 |
| `ranking/content/recall@10` | 0.0925594705777142 |
| `ranking/content/recall@5` | 0.059202289393668395 |
| `ranking/content_minus_content/common_hit_rank_ndcg@10` | 0 |
| `ranking/content_minus_content/common_hit_rank_ndcg@5` | 0 |
| `ranking/content_minus_content/lost_hit_ndcg@10` | 0 |
| `ranking/content_minus_content/lost_hit_ndcg@5` | 0 |
| `ranking/content_minus_content/new_hit_ndcg@10` | 0 |
| `ranking/content_minus_content/new_hit_ndcg@5` | 0 |
| `ranking/generation/ndcg@10` | 0.03763325189700371 |
| `ranking/generation/ndcg@5` | 0.02691624379270245 |
| `ranking/generation/recall@10` | 0.07851904847075657 |
| `ranking/generation/recall@5` | 0.04471472008585226 |
| `ranking/generation_minus_content/common_hit_rank_ndcg@10` | -0.003596824920374986 |
| `ranking/generation_minus_content/common_hit_rank_ndcg@5` | -0.0019959574935677258 |
| `ranking/generation_minus_content/lost_hit_ndcg@10` | -0.013592869420399764 |
| `ranking/generation_minus_content/lost_hit_ndcg@5` | -0.015698565968530538 |
| `ranking/generation_minus_content/new_hit_ndcg@10` | 0.006886208163966147 |
| `ranking/generation_minus_content/new_hit_ndcg@5` | 0.007404843294966672 |
| `ranking/mixed/ndcg@10` | 0.04749760278576311 |
| `ranking/mixed/ndcg@5` | 0.0373041198677585 |
| `ranking/mixed/recall@10` | 0.09023430513324988 |
| `ranking/mixed/recall@5` | 0.05866571275263817 |
| `ranking/mixed_minus_content/common_hit_rank_ndcg@10` | 0.0002343076763095105 |
| `ranking/mixed_minus_content/common_hit_rank_ndcg@5` | 0.00024923757417190465 |
| `ranking/mixed_minus_content/lost_hit_ndcg@10` | -0.0022267465938163138 |
| `ranking/mixed_minus_content/lost_hit_ndcg@5` | -0.002193142160489687 |
| `ranking/mixed_minus_content/new_hit_ndcg@10` | 0.0015533036294576037 |
| `ranking/mixed_minus_content/new_hit_ndcg@5` | 0.0020421004942422385 |
| `ranking/ndcg/cold_hits@10` | 0 |
| `ranking/ndcg/cold_hits@5` | 0 |
| `ranking/ndcg/ndcg@10` | 0.04736663774187049 |
| `ranking/ndcg/ndcg@5` | 0.03653581299018692 |
| `ranking/ndcg/recall@10` | 0.0908603112144518 |
| `ranking/ndcg/recall@5` | 0.0573242711500626 |
| `ranking/ndcg_minus_content/common_hit_rank_ndcg@10` | -8.014457070373809e-05 |
| `ranking/ndcg_minus_content/common_hit_rank_ndcg@5` | 1.9305232588064228e-05 |
| `ranking/ndcg_minus_content/lost_hit_ndcg@10` | -0.002523006849632682 |
| `ranking/ndcg_minus_content/lost_hit_ndcg@5` | -0.003540189912167292 |
| `ranking/ndcg_minus_content/lost_hits@10` | 89 |
| `ranking/ndcg_minus_content/lost_hits@5` | 90 |
| `ranking/ndcg_minus_content/ndcg@10` | -0.0005701003319418201 |
| `ranking/ndcg_minus_content/ndcg@10_ci95_high` | 0.000544685683846538 |
| `ranking/ndcg_minus_content/ndcg@10_ci95_low` | -0.0016689748055909329 |
| `ranking/ndcg_minus_content/ndcg@5` | -0.0006701109696471313 |
| `ranking/ndcg_minus_content/ndcg@5_ci95_high` | 0.0006201082431993026 |
| `ranking/ndcg_minus_content/ndcg@5_ci95_low` | -0.0019446497003427355 |
| `ranking/ndcg_minus_content/new_hit_ndcg@10` | 0.0020330510883946 |
| `ranking/ndcg_minus_content/new_hit_ndcg@5` | 0.002850773709932097 |
| `ranking/ndcg_minus_content/new_hits@10` | 70 |
| `ranking/ndcg_minus_content/new_hits@5` | 69 |
| `ranking/ndcg_minus_content/recall@10` | -0.001699159363262386 |
| `ranking/ndcg_minus_content/recall@10_ci95_high` | 0.0005365766410302271 |
| `ranking/ndcg_minus_content/recall@10_ci95_low` | -0.0038454659273832945 |
| `ranking/ndcg_minus_content/recall@5` | -0.001878018243605795 |
| `ranking/ndcg_minus_content/recall@5_ci95_high` | 0.000270524056519394 |
| `ranking/ndcg_minus_content/recall@5_ci95_low` | -0.004024324807726703 |
| `ranking/ndcg_minus_mixed/ndcg@10` | -0.0001309650438926205 |
| `ranking/ndcg_minus_mixed/ndcg@10_ci95_high` | 0.0006637291156779917 |
| `ranking/ndcg_minus_mixed/ndcg@10_ci95_low` | -0.0008971425832899401 |
| `ranking/ndcg_minus_mixed/ndcg@5` | -0.000768306877571588 |
| `ranking/ndcg_minus_mixed/ndcg@5_ci95_high` | 0.00011980013731877272 |
| `ranking/ndcg_minus_mixed/ndcg@5_ci95_low` | -0.0016976994605848103 |
| `ranking/ndcg_minus_mixed/recall@10` | 0.0006260060812019317 |
| `ranking/ndcg_minus_mixed/recall@10_ci95_high` | 0.002146306564120909 |
| `ranking/ndcg_minus_mixed/recall@10_ci95_low` | -0.0008942944017170453 |
| `ranking/ndcg_minus_mixed/recall@5` | -0.001341441602575568 |
| `ranking/ndcg_minus_mixed/recall@5_ci95_high` | 0.0003577177606868181 |
| `ranking/ndcg_minus_mixed/recall@5_ci95_low` | -0.003040600965837954 |
| `ranking/ndcg_minus_nll/ndcg@10` | 9.257767163206448e-05 |
| `ranking/ndcg_minus_nll/ndcg@10_ci95_high` | 0.0006585718587247666 |
| `ranking/ndcg_minus_nll/ndcg@10_ci95_low` | -0.0004693607737998428 |
| `ranking/ndcg_minus_nll/ndcg@5` | -8.239213078705793e-05 |
| `ranking/ndcg_minus_nll/ndcg@5_ci95_high` | 0.0006451567127850515 |
| `ranking/ndcg_minus_nll/ndcg@5_ci95_low` | -0.0008217383333507667 |
| `ranking/ndcg_minus_nll/recall@10` | 0.0003577177606868181 |
| `ranking/ndcg_minus_nll/recall@10_ci95_high` | 0.0016097299230906814 |
| `ranking/ndcg_minus_nll/recall@10_ci95_low` | -0.0008048649615453407 |
| `ranking/ndcg_minus_nll/recall@5` | -8.942944017170452e-05 |
| `ranking/ndcg_minus_nll/recall@5_ci95_high` | 0.001341441602575568 |
| `ranking/ndcg_minus_nll/recall@5_ci95_low` | -0.001520300482918977 |
| `ranking/nll/ndcg@10` | 0.047274060070238426 |
| `ranking/nll/ndcg@5` | 0.036618205120973975 |
| `ranking/nll/recall@10` | 0.09050259345376498 |
| `ranking/nll/recall@5` | 0.057413700590234304 |
| `ranking/nll_minus_content/common_hit_rank_ndcg@10` | -9.480060303635256e-05 |
| `ranking/nll_minus_content/common_hit_rank_ndcg@5` | 4.639663371972852e-06 |
| `ranking/nll_minus_content/lost_hit_ndcg@10` | -0.002415552921020452 |
| `ranking/nll_minus_content/lost_hit_ndcg@5` | -0.0029021056042960827 |
| `ranking/nll_minus_content/ndcg@10` | -0.0006626780035738843 |
| `ranking/nll_minus_content/ndcg@10_ci95_high` | 0.000437691751415712 |
| `ranking/nll_minus_content/ndcg@10_ci95_low` | -0.0017273740000145947 |
| `ranking/nll_minus_content/ndcg@5` | -0.0005877188388600732 |
| `ranking/nll_minus_content/ndcg@5_ci95_high` | 0.0006027730888313866 |
| `ranking/nll_minus_content/ndcg@5_ci95_low` | -0.001804990280325573 |
| `ranking/nll_minus_content/new_hit_ndcg@10` | 0.00184767552048292 |
| `ranking/nll_minus_content/new_hit_ndcg@5` | 0.002309747102064037 |
| `ranking/nll_minus_content/recall@10` | -0.002056877123949204 |
| `ranking/nll_minus_content/recall@10_ci95_high` | 8.942944017170452e-05 |
| `ranking/nll_minus_content/recall@10_ci95_low` | -0.004113754247898408 |
| `ranking/nll_minus_content/recall@5` | -0.0017885888034340906 |
| `ranking/nll_minus_content/recall@5_ci95_high` | 0.00026828832051511357 |
| `ranking/nll_minus_content/recall@5_ci95_low` | -0.00375603648721159 |

#### `zznw291t` — v3.1 / testing

| 原 metric key | 原值 |
|---|---:|
| `candidate_trace/dense/ndcg@10` | 0.03578160777512011 |
| `candidate_trace/dense/ndcg@5` | 0.027653124522355486 |
| `candidate_trace/dense/recall@10` | 0.07038411662120467 |
| `candidate_trace/dense/recall@5` | 0.045029736618521665 |
| `candidate_trace/hybrid/ndcg@10` | 0.03582780406520125 |
| `candidate_trace/hybrid/ndcg@5` | 0.02780684698062608 |
| `candidate_trace/hybrid/recall@10` | 0.07038411662120467 |
| `candidate_trace/hybrid/recall@5` | 0.04543218709475473 |
| `test/ndcg@10` | 0.03582780406520121 |
| `test/ndcg@5` | 0.02780684698062612 |
| `test/recall@10` | 0.07038411662120467 |
| `test/recall@5` | 0.04543218709475473 |

删除状态与最终缺失核验见[删除回执](wandb-delete-20261007/deletion-receipt.json)和[删除后独立核验](wandb-delete-20261007/verification.json)。删除不会使旧开发结果成为新正式证据，也不重置既有 Testing 使用史。

### 更早阶段遗留的孤立 Artifact

另对 207 个相关 Artifact 集合逐版本检查，发现 10 个原创建 run 已不存在的 CoPMRec joint checkpoint、alpha=0.5/content-only 输出及候选/路径 trace；它们的显式 task、联合概率协议和元数据表明是早期开发产物，无现存消费者，故一并删除。删除前所有 metadata/digest/file 索引保存在上述完整快照与[集合审计](wandb-delete-20261007/collection-orphan-audit.json)。两份迁移 checkpoint 的原 producer 及登记 consumer 也已逐 ID 核实不存在。当前共计待删除 **82 个 run、253 个 Artifact 版本**，加 1 个此前已缺失的 run ID。

| Artifact | 原来源／机制 | 可取得的历史效果 |
|---|---|---|
| `baymaxam/GRID/liger_beauty_joint_train-checkpoint-local-reference:v3` | zl9gv56p | 历史 Validation-selected best NDCG@10=0.04637137055397034 |
| `baymaxam/GRID/liger_beauty_joint_train_seed200-checkpoint-local-reference:v0` | 83djht0e | 历史 Validation-selected best NDCG@10=0.04575324058532715 |
| `baymaxam/GRID/liger_beauty_factorial_joint_alpha05-recommendation-output:v0` | liger_beauty_factorial_joint_alpha05 | 原 run 已缺失，无法恢复 summary Recall/NDCG；不补造数值 |
| `baymaxam/GRID/liger_beauty_joint_content_only_testing-recommendation-output:v0` | liger_beauty_joint_content_only_testing | 原 run 已缺失，无法恢复 summary Recall/NDCG；不补造数值 |
| `baymaxam/GRID/liger_beauty_joint_inference-recommendation-output:v0` | liger_beauty_joint_inference | 原 run 已缺失，无法恢复 summary Recall/NDCG；不补造数值 |
| `baymaxam/GRID/liger_sports_joint_inference-candidate-trace:v1` | learned_mass | 原 run 已缺失，无法恢复 summary Recall/NDCG；不补造数值 |
| `baymaxam/GRID/liger_beauty_factorial_joint_alpha05-candidate-trace:v0` | learned_mass | 原 run 已缺失，无法恢复 summary Recall/NDCG；不补造数值 |
| `baymaxam/GRID/liger_beauty_joint_content_only_testing-candidate-trace:v0` | liger-joint-mixture-v1 | 原 run 已缺失，无法恢复 summary Recall/NDCG；不补造数值 |
| `baymaxam/GRID/liger_beauty_joint_inference-candidate-trace:v0` | liger-joint-mixture-v1 | 原 run 已缺失，无法恢复 summary Recall/NDCG；不补造数值 |
| `baymaxam/GRID/liger_beauty_joint_inference-path-trace:v1` | legal_generation | 原 run 已缺失，无法恢复 summary Recall/NDCG；不补造数值 |

历史训练 Validation NDCG 轨迹另存于[35 个运行的曲线快照](wandb-delete-20261007/validation-ndcg-history.json)，共 18,627 个日志点；这些点是原日志记录，不重新解释为独立验证次数。远端本地日志、checkpoint 文件及重资产不属于本次 W&B 删除范围。

### 2026-10-07 删除结果与系统管理对象

已实际删除 **82 个 CoPMRec 开发 run、196 个普通用户 Artifact 版本**。独立读回确认83个登记ID均不存在（其中1个原已缺失），196个普通版本均不可访问，27个LIGER comparator和9个固定上游run仍在。

另有 **57 个 W&B 系统管理版本：30个wandb-history、27个wandb-events**，单独删除返回 `cannot delete system managed artifact`；所属run删除后，独立API查询仍返回COMMITTED，故它们列为服务侧待清理，不计入已删除数量。不存在普通checkpoint、预测、trace、源码等用户Artifact的剩余项。清单中253为删除前总计，实际完成196普通版本，剩余57系统管理对象。

[完整删除回执](wandb-delete-20261007/deletion-receipt.json)与[独立核验](wandb-delete-20261007/verification.json)保留这一区别。W&B线上历史链接已失效，效果追溯使用本页以及保存的config/summary/metadata/曲线，不以线上删除改变历史事实。
