# CoPMRec v4.2 固定完整目录 logit pooling 验证

**阶段状态（2026-10-05）：已完成唯一完整 Evaluation 并结题。pool 对 LIGER 为 R10 +9.262948% / N10 +9.288610%，两个点门槛未达10%，不运行本阶段 Testing；仅关闭此次固定pool问题，whole goal未达且保持active。下文前半保留运行前预注册依据，实际结果见末尾结题段。**

## 决策与范围

原 v4 协同残差阶段与 v4.1 商品截距阶段均已结题，累计 **5 次正式训练 / 30000 steps** 已完成，训练剩余为0。本阶段登记一个推理问题：固定融合 source v4 best6000 与 frozen-bias control best6000 的完整目录 logits，能否保留两种现有排序的优势。

用户已有自主方法和 SSH 授权覆盖本次实现与运行。root 已明确只允许 **新增训练0、一次单卡完整 Evaluation、条件成立后最多一次固定 Testing**；Testing 全线程上限仍3次、此前已用1次。本阶段不重置任何累计额度，不扫描 weight、temperature、normalization、pair、checkpoint、bias 或 LR。当前依据是 fresh Validation 开发证据，目标仍为相对真实 LIGER dense Testing 的 Recall10 / NDCG10 均≥10%。

## 五项路线门禁

### 1. 核心假设与路线变更

原残差假设是商品身份协同自由度能在匹配续训之外改善推荐；已保留有限正收益。v4.1 检验独立 seen 商品截距的匹配增量，该阶段未确认增量，已关闭。

新可反驳假设是：**一次固定完整目录等权 logit pool，能相对冻结 pair 中最强的 control 提高 NDCG10，并使 Recall10 不下降。** source/control 在同一用户上有大量双向命中差异，source 保留更多 Top5 和固定 offender 真目标命中，control 增加其余用户的 Top10 覆盖。融合是否能保留这些优势仍未知，成员并集命中不能当作融合性能或其全目录上界。

本次明确切换为既有两模型的推理融合问题；不将它称为恢复 bias 机制或延续 LR 优化。

### 2. 已完成支持及主张边界

v4 residual 与 zero-residual 同6000步对照的六个等步数点均正向，各自 best 的 R10 / N10 增量约 +2.47% / +2.02%。旧 v4 best5000 的真实 dense Testing 相对 LIGER 为 +5.26% / +5.64%，两项 paired CI 为正，但仍不足10%。该结果支持有限残差收益，不证明当前新 best 或 pooling 的 Testing 效果。

第三 v4 lr20 臂 `nj9elah1` 已完成6000步，完整 Validation 选择的 best6000 成为当前 source。v4.1 的 learned-bias `5ke7nt98` 和 frozen-bias control `l3zyr91b` 均已完成6000步并选出 best6000；两次单卡完整 Validation 分别为 `o7ycqky2` / `yqsmsdt1`，均已独立通过 raw 175 shards / 22363 用户标签、输入、checkpoint、catalog、source 与输出审计。

| 已核验模型 | Validation run | hits10 | Recall10 | NDCG10 | 相对同 split LIGER R10 / N10 |
| --- | --- | ---: | ---: | ---: | --- |
| source v4 best6000 | mq8hhof5 | 2147 | 0.09600679694137638 | 0.04979393256719645 | +6.9223% / +7.1520% |
| learned-bias best6000 | o7ycqky2 | 2194 | 0.09810848276170460 | 0.05050305197307684 | +9.2629% / +8.6780% |
| frozen-bias control best6000 | yqsmsdt1 | 2209 | 0.09877923355542638 | 0.05042652891207879 | +10.0100% / +8.5133% |

fresh `control − source` 的320新命中、258丢失，净增62；R10 +2.8878%，差值CI `[0.0007154675, 0.0048741224]` 为正，N10 +1.2704%，差值CI `[-0.0003886775, 0.0016495466]` 跨0。control 的 Top5 R / N 相对 source 为 −2.0567% / −1.7532%，Top5 命中1410→1381。共同命中中557个位置上升、559个下降，N10 的共同位置贡献为 −0.0001511140。

以下 key 分组在观察本阶段前固定，使用 `SHA256(copmrec-v4-validation-exposure-v1:<decimal-user-key>)[0] & 1`；分组列表按 UTF-8 canonical JSON 序列化，SHA为 `a1f97bdc86ad305578c53a4cee4398b7a4f1f6216812cb8523009b6462b8dea5`。这些组是同一开发集的描述性分组，不能称独立复制。

| 固定组 | 用户数 | source hits10 | control hits10 | control新增 / 丢失 / 净值 | R10相对变化 | N10相对变化 |
| --- | ---: | ---: | ---: | --- | ---: | ---: |
| group0 | 11201 | 1052 | 1095 | 163 / 120 / +43 | +4.0875% | +2.2929% |
| group1 | 11162 | 1095 | 1114 | 157 / 138 / +19 | +1.7352% | +0.3148% |

两组 R/N 点估计方向相同，但两个 N 区间均跨0，group1 的 R 区间也跨0。支持的是存在可检验的排序差异，不是已确认 ensemble 收益。

### 3. 反证、门槛未通过与未运行

固定 mixed 同候选终排净损40命中、两CI负，mixed 及其竞争 loss 前提已关闭。v4 lr20 best6000 对旧 v4 的 +0.94%R / +1.38%N 两CI跨0，不能确认额外 LR 收益，LR 序列已关闭。

v4.1 `bias − control` 为27新 / 42丢失、净损15，R10 −0.6790% / N10 +0.1518%，两CI跨0；点预测中 R不下降也未满足。固定 offender 真目标639用户的命中均为304，bias 对 control 没有净命中改善。两臂均未通过整体双10%门槛，因此 v4.1 没有 Testing；关闭 bias 迭代，不扫描 bias / temperature / LR。不把不显著称等价，也不泛化为所有商品截距方法无效。

固定20商品名单取自 **source best6000 原错误曝光排行**，不在新结果中重新选：catalog rows 为 `789,774,510,104,861,1536,300,1871,443,301,656,278,1355,342,292,295,2079,12,493,446`。以下639用户是原始真实目标属于这20商品的用户，区别于 Top10 曾错曝这些商品的13426用户。

| 冻结 target cohort | 用户数 | source hits5 / hits10 | control hits5 / hits10 | Top10净值 | source N10 | control N10 |
| --- | ---: | --- | --- | ---: | ---: | ---: |
| 真实目标属固定20商品 | 639 | 280 / 352 | 236 / 304 | −48 | 0.3294793339 | 0.2824685195 |
| 其余真实目标 | 21724 | 1130 / 1795 | 1145 / 1905 | +110 | 0.0415671340 | 0.0436011362 |

固定639中 control 为6新增 / 54丢失，R10 −13.6364% / N10 −14.2682%，两项差值CI均负。group0 的该类用户300人，命中154→131（−23）；group1 的339人，198→173（−25）。其余目标用户的两组命中分别898→964（+66）、897→941（+44），组成整体净增62的明确 trade-off。

与此同时，固定20错误曝光计数40995→25575，其占全部错误Top10曝光比例18.5093%→11.5504%。错曝下降与真目标损失并存，曝光不能替代推荐效果或作为校准因果证据。source/control 尚未进行本次 pooling，也都没有对应的新 Testing；不将现有差异改写为已恢复的收益。

### 4. 剩余关键疑点与最小鉴别

既有两次完整 Validation 的实现和原始产物审计通过，无需追加模型运行重查已确认来源。会改变决策的未知只有：平均完整 logits 后，保留的正确排序收益能否超过丢失与共同位置损失。Top10 bundle 没有目录外分数，不能准确离线复原这种融合，最小必要验证为一次实际完整目录 pooling 推理。

实现准备必须核验两个公共 loader 的原始字节和 lineage、同一 catalog 全 buffers、原 v0 来源、control 精确指向 source 的 v4 warmstart、各自原类 normal restore、cold residual / control bias0 / finite-state，以及各自产生 query。该准备不是效果实验，不能用 smoke 或候选并集代替完整 Evaluation。

### 5. 累计额度与各结果取舍

新增训练0，旧累计5 / 30000已用尽；仅一次完整 Evaluation，固定 pair / 0.5 / 0.5 / 原评分设置，未达目标也不更换组合。point prediction 的比较基准为所选 **source/control 中最强的 control**：R10≥0.09877923355542638、N10>0.05042652891207879。未进入 pair 的 learned-bias N点估计只作背景，不能偷换增量基准。

整体晋级和 pooling 增量分别判断：Validation 使用固定 LIGER5azn5vm0，R10≥0.09877029021151008（2209命中）、N10≥0.0511173919307933，且两项相对 LIGER 的 paired delta CI 下界为正，才能进行固定 pool 的一次 Testing。pool-control 增量CI跨0会限制增量主张，不增加整体晋级门槛；整体达标时仍可执行预承诺确认。

Testing 保持固定042139al：R10≥0.07855386128873586（1757命中）、N10≥0.04002978116676896。全线程最多3次、此前1次、本阶段最多消耗1次，Testing不选择参数/规则/成员。结果负向或不确定如实报告；整体不足双门槛即关闭本问题，保留有限增量或否证，不自动扩预算或重开任何扫描。

## 相关工作与机制定位

[Heskes，NIPS 1997，Selecting Weighting Factors in Logarithmic Opinion Pools](https://proceedings.neurips.cc/paper_files/paper/1997/file/59f51fd6937412b7e56ded1ea2470c25-Paper.pdf) Eq1给出概率的加权几何池，分类 canonical logits 可线性平均。对本定义直接推导：`s_pool=(s_source+s_control)/2`，如果 `p_member=softmax(s_member)`，则 `softmax(s_pool)_i ∝ sqrt(p_source_i*p_control_i)`。该机制是成熟的 logarithmic opinion pool，本文不宣称首创；论文的 KL / NLL 分析不能给本任务 Recall / NDCG 保证。

[Deep Ensembles 原论文](https://arxiv.org/html/1612.01474v3)使用均匀概率混合，[Google官方实现](https://github.com/google/uncertainty-baselines/blob/main/baselines/cifar/ensemble.py#L193-L204)先softmax再平均概率。本阶段固定的是 raw dense logits 平均，不增加概率平均的第二验证单元。某一成员的低分仍可能压掉另一成员的正确结果，同源模型也可能共同误排；等权不保证分数尺度对结果的影响相同。这些风险由唯一 Evaluation 的整体及配对损益直接判断。

本次评估的是 CoPMRec 的 dense 推理规则，对生成候选贡献没有新证据。论文主张须结合有限残差效果、真实对照、排序trade-off与最终确认，不将成熟融合组件单独包装为新颖贡献。

## 固定工程契约与来源

| 成员 | 固定 checkpoint URI | 原始 SHA256 | 原类 / 恢复模式 |
| --- | --- | --- | --- |
| source | `wandb://baymaxam/GRID/nj9elah1?role=checkpoint&alias=v0&file=checkpoint_epoch=000_step=006000.ckpt` | `7c52228a665b593b50e45e8aba039f3f4aa4a10ca134d9345c59c59fb142f055` | v4 CollaborativeResidualCoPMRec，normal restore |
| control | `wandb://baymaxam/GRID/l3zyr91b?role=checkpoint&alias=v0&file=checkpoint_epoch=000_step=006000.ckpt` | `4bea4d5771c4f056ad6e0cd69d1204a3220b639868803cbc19aa3332ecf0a3c9` | v4.1 ItemScoreBiasCoPMRec scale0，normal restore |

两者原 v0 为 `wandb://baymaxam/GRID/7y54j4m6?role=checkpoint&alias=v2&file=checkpoint_epoch=000_step=048000.ckpt`，SHA `567aed4610a2cfe6671a1eddd300470184d402853269559a3e719da95ba9a322`。两个 catalog、历史输入和来源链严格核验；不桥接checkpoint版本或恢复optimizer / scheduler / Trainer步数。

工程采用只推理的 `FixedLogitPoolCoPMRec`，接口、factory、九个keyword、四expectedidentity、pool_contract与拒绝边界见[design](../openspec/changes/add-copmrec-v4-2-fixed-logit-pool/design.md)。单卡GPU2→Trainer `[0]` / FP32，经统一 `src.main` / Hydra / 根推理脚本，复用公共loader、lineage、metrics及commonwriter，主bundle恰 `keys` / `predictions`；无wrapper独立checkpoint，不依赖已有Top10分数。生产pair由配置、ready/launch和独立审计冻结，模块不新增W&B逻辑。

## 证据与当前状态

fresh候选、配对和cohort数值来源为[fresh配对分析](evidence/copmrec-v4-autonomous-20261005/bias-validation-pair-analysis.json)（SHA `5da9842c71515e6f39127214fec9a9d8052bdce44d0ccca8d54be5282d4d526e`）、[source主审计](evidence/copmrec-v4-autonomous-20261005/validation-residual-lr20-audit.json)、[bias主审计](evidence/copmrec-v4-autonomous-20261005/validation-bias-validation-audit.json)、[control主审计](evidence/copmrec-v4-autonomous-20261005/validation-bias-control-validation-audit.json)及[固定名单/分组审计](evidence/copmrec-v4-autonomous-20261005/validation-item-exposure-audit.json)，均在同一Evaluation用户和标签上独立复算。旧阶段结论仅沿用已保留的阶段审计，未使用Testing bad cases。

OpenSpec `add-copmrec-v4-2-fixed-logit-pool` 的proposal/design/spec/tasks已complete、strict通过，两个implementation owner已收到规格完成通知。此文规格完成时尚无v4.2正式Evaluation或Testing效果；实际运行句柄、准备/输出审计和结题由root后续按该唯一预算记录。本文件不修改research-state或current-plan。

## 实际结题：一次固定 pool 的 Validation 门禁未通过

唯一正式完整 Evaluation 为 [`i4xwruok`](https://wandb.ai/baymaxam/GRID/runs/i4xwruok)，固定 job PID `4105582` / `logs/autonomous/copmrec_v4_2_pool_validation_20261004T232233Z`，实际 `exit0` 且 W&B `finished`。没有新增训练，未更换两个 best6000 成员、0.5 / 0.5 权重、温度或归一化，也没有 wrapper 独立 checkpoint。前面的五项门禁、路线依据和阈值为运行前登记的历史，保留原文；本段给出完成后的实际结论。

### 整体结果与预注册取舍

| 实际模型 | Evaluation run | hits5 / hits10 | Recall10 | NDCG10 |
| --- | --- | ---: | ---: | ---: |
| 真实 LIGER dense | 5azn5vm0 | 1292 / 2008 | 0.08979117291955462 | 0.04647035630072118 |
| source v4 best6000 | mq8hhof5 | 1410 / 2147 | 0.09600679694137638 | 0.04979393256719645 |
| frozen-bias control best6000 | yqsmsdt1 | 1381 / 2209 | 0.09877923355542638 | 0.05042652891207879 |
| 唯一固定 pool | i4xwruok | 1399 / 2194 | 0.09810848276170460 | 0.05078680665613525 |

| 固定比较 | 新命中 / 丢失 / 净值 | R10 相对变化 | R10 绝对差值 95% CI | N10 相对变化 | N10 绝对差值 95% CI |
| --- | --- | ---: | --- | ---: | --- |
| pool − LIGER | 576 / 390 / +186 | +9.262948% | [+0.005679023387, +0.011000313017] | +9.288610% | [+0.002903361058, +0.005709650368] |
| pool − source | 177 / 130 / +47 | +2.189101% | [+0.000536600635, +0.003711487725] | +1.993966% | [+0.000289035182, +0.001685365100] |
| pool − control | 145 / 160 / -15 | -0.679040% | [-0.002235835979, +0.000894334392] | +0.714461% | [-0.000311015986, +0.001040703913] |

pool 相对同 split 真实 LIGER dense 的 R10 / N10 为 **+9.262948% / +9.288610%**，两项绝对差值 CI 为正，保留这个有限正向 Validation 结果。整体晋级要求 R10≥`0.09877029021151008`（2209命中）、N10≥`0.0511173919307933`；实际仍少 **15命中**，N10 尚差 `0.00033058527465805`。两个点门槛均未通过，主审计及独立统计都得到 `eligible=false`，因此**不运行本阶段 Testing**。不能把单独一个 key 组达到10%代替整体门槛。

原增量比较对象固定为更强的 control。pool 的145新增 / 160丢失为净−15，R10 −0.679040% / N10 +0.714461%，两项 CI 跨0，预承诺的“R不下降且N提高”点预测未满足。共同命中中486位置上升、431下降，N10 的共同位置贡献 `+0.0005587573706988441`，抵消新增与丢失合计 `−0.0001984796266423844` 后得到微小正N点估计；这不足以确认整体增量。CI跨0也不证明等价。pool 对 source 的两CI虽正，仍不能改用较弱成员来满足已登记的增量主张；learned-bias 仅是背景比较，未加入 pair 或参与参数选择。

### 固定 cohort 的真实 trade-off

下表沿用原 source 固定20商品、原真实目标标签及原 SHA256 key 分组，未根据 pool 重新选择名单或子组。`639` 是真实目标属于固定20商品的用户，`13426` 是旧 source 的 Top10 错误曝光这些商品的用户，两种定义不同。

| 冻结 cohort | 用户数 | source / control / pool hits10 | pool 对 control 新 / 丢失 / 净值 | pool 对 control R10 / N10 相对变化 |
| --- | ---: | --- | --- | ---: |
| 固定20商品真目标 | 639 | 352 / 304 / 328 | 25 / 1 / +24 | +7.894737% / +9.509319% |
| 其余真实目标 | 21724 | 1795 / 1905 / 1866 | 120 / 159 / -39 | -2.047244% / -0.961493% |
| warm 真实目标 | 22312 | 2144 / 2205 / 2190 | 145 / 160 / -15 | -0.680272% / +0.730935% |
| cold 真实目标 | 51 | 3 / 4 / 4 | 0 / 0 / +0 | +0.000000% / -11.965123% |
| 固定 key group0 | 11201 | 1052 / 1095 / 1087 | 73 / 81 / -8 | -0.730594% / +0.138627% |
| 固定 key group1 | 11162 | 1095 / 1114 / 1107 | 72 / 79 / -7 | -0.628366% / +1.263243% |
| 旧 source 曾错误曝光固定20商品 | 13426 | 861 / 886 / 876 | 71 / 81 / -10 | -1.128668% / +1.011771% |
| 旧 source 未错误曝光固定20商品 | 8937 | 1286 / 1323 / 1318 | 74 / 79 / -5 | -0.377929% / +0.515305% |

固定639真目标用户，control→pool 从304恢复至328，25新增 / 1丢失、净+24；R10 +7.894737% / N10 +9.509319%，两CI为正，R绝对差值CI `[+0.023474178404, +0.053208137715]`，N为 `[+0.018082006081, +0.036528643558]`。但 source 原352仍比pool多24；pool相对source该cohort两CI为负。两固定 key 组的639子组分别恢复+10 / +14，而其余目标分别损失−18 / −21。

其余21724真目标用户，control→pool 从1905降至1866，120新增 / 159丢失、净−39；R10 −2.047244%且CI全负，N10 −0.961493%但CI跨0。`+24 −39 = −15` 与整体恒等式一致。两key组整体分别−8 / −7，两项增量CI均跨0；它们是同一开发集的固定描述性分组，不能称为独立重复实验。

warm 的22312用户承担全部净−15；cold 的51用户仍为4个共同命中，无新增或丢失，但4个位置全部下降，N10 −11.965123%，绝对差值CI `[-0.007230107900, -0.000600894512]`。cold样本较少，零R差值和 `[0,0]` 区间不证明普遍等价。原错曝光 / 未错曝光cohort分别净−10 / −5，N点估计均正但两项整体增量CI跨0。

固定20商品错误Top10曝光为 source `40995`（全部错误曝光的18.509321%）、control `25575`（11.550395%）、pool `32539`（14.694539%）。pool较control增加6964次错误曝光，两key组分别+3508 / +3456；pool在曝光和真目标命中上均处于成员之间。恢复了部分真目标排序，也付出其他用户覆盖损失，不能将这组描述性 trade-off 写成已证实的校准因果或整体 pooling 收益。

### 原始产物、来源与统计边界

[主审计](evidence/copmrec-v4-autonomous-20261005/pool-validation-audit.json)，SHA `d86df11253ae609e8a8d3af639b0895fba5277efcf152d32d40e302935bb0c95`，独立核验完整 `data/beauty/evaluation` **175个TFRecord / 22363用户**的末商品标签、用户严格值对齐、SID Top10合法且唯一、四个catalog buffers、SID / embedding Artifact、两个 best-selected checkpoint 实际 bytes / role / digest / step及其 producer 来源链。单卡物理GPU2→local0，运行源及prelaunch **360文件 / 360 SHA**全部通过，源SHA `1a27bce437c12ccbd21b02e96dcfab30909300aa4619c3d3763031385c3b285d`。两个checkpoint继续使用本报告工程表中原始SHA，各自在原类中 normal restore；control的v4 warmstart精确等于source，原v0 weights-only48000链保留，cold residual与control bias均为0。

输出 Artifact `copmrec_beauty_v4_2_pool_validation-recommendation-output:v0`，digest `75a4538bc1bb94d108cef3f3b3a766f5`；标准 `keys` / `predictions` bundle SHA `5c2d7751204b7776d6c7768b7c4dcb5075edd00237ef594b2244e1da30bcfa00`。实际 writer 中 `pool_contract` 与冻结配置及CPU原类strict恢复契约相同，使用两条公共loader lineage，不以当前配置自证历史来源。

[独立 paired / cohort / 曝光统计](evidence/copmrec-v4-autonomous-20261005/pool-validation-pair-analysis.json)，SHA `adb9ca0945c5ee4cb7c8581e3f42409a0f0a4b9de17d3079637ebb7151cdf11f`，重新读取真实预测、标签及catalog，对整体指标、三组主pairedCI和晋级结果与主审计独立交叉核对。canonical keys SHA `5c14775a3aec5907a79e94c4cc0b2a00be6344c0e72cb2e40cca75e9c34744d9`、labels SHA `efe624caa09a26812661a9930c29724a5c0d23670c304fe5c2e48de56471a72f` 与既有独立Evaluation记录一致；旧LIGER keys为int32、新输出为int64，按整数值对齐，不接受重复或非整数key。固定分组canonical JSON SHA `a1f97bdc86ad305578c53a4cee4398b7a4f1f6216812cb8523009b6462b8dea5` 沿用不变。

CI 使用2000次 paired users bootstrap / NumPy PCG64 seed42，是固定所选checkpoint与固定cohort条件下的未校正逐点区间，不修正训练seed不确定性或checkpoint选择，不把子组当独立确认。此处只有开发Evaluation结果，未读取Testing样本进行本次选择；旧Testing曾参与历史开发的边界仍保留，不声称全线程Testing完全未经开发使用。旧部署 `1edkvgk5` / `z9envq1n best5000` 的R/N +5.26% / +5.64%仍是此前已验证Testing结果，不能替换为当前pool或新best6000的Testing效果。

### 阶段关闭与累计预算

按运行前取舍，**仅关闭这一次固定pair、固定完整logits等权pool问题**，状态为 `closed_single_fixed_pool_component_increment_unproven_below_goal`。保留局部恢复及相对source / LIGER的有限Validation正向结果，不确认超过更强control的整体增量，不泛化为所有pool方法无效。没有扫描weight / temperature / normalization / pair / checkpoint / bias / LR，不自动扩预算或恢复已关闭mixed、LR、bias路线。

全线程 **5次正式训练 / 30000 steps已耗尽，剩余训练0**；本阶段新增训练0、完整Evaluation **1/1**、Testing0。全线程Testing保持 **1/3已用、2槽剩余**，剩余槽不自动授权新实验。本阶段未达双门槛，不消耗第二Testing槽；原10%目标未达，whole goal保持 `active`。实现与评估任务结题不等于目标完成。

[固定pool阶段结题记录](evidence/copmrec-v4-autonomous-20261005/fixed-logit-pool-stage-closure.json)汇总13份实际证据文件SHA、两个成员checkpoint、整体与cohort结果、准确门禁与预算。结题仅更新此报告、该JSON和OpenSpec任务状态，没有修改runtime、研究state或current-plan，也没有启动forward、训练或推理。

