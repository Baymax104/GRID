# CoPMRec v3：推荐收益转化卡点分析

最新决定（2026-10-04）：用户随后授权从真实v0派生首层Max、后续Mass的v3候选机制，训练/推理采用相同聚合，训练配置与v0保持一致；当前实现和命令见 [v3说明](copmrec-v3.md)。以下方向2选择和“模型未实施”是前一阶段快照，诊断与旧关闭预算保留。

后续决定：用户选择方向2，在现有content排序上分析与优化；已完成 [真实v0具体bad case报告](copmrec-v3-content-bad-cases.md)。本文件保留前一阶段的总体诊断，方向1不作为当前阶段方案，模型/loss仍未实施。

日期：2026-10-04。用户决定停止 v2 迭代、开始 v3，先结合失败尝试与 bad case 定位问题。v3 当前为分析阶段，具体模型与 loss 尚未定案；本轮没有修改推荐运行代码、执行模型前向或启动训练/推理。

## 分析信息

- Origin Skill：academic-research-suite / experiment-agent。
- Origin Mode：validate（统计解释与证据综合）。
- Origin Date：2026-10-04。
- Verification Status：ANALYZED；新增排名分解由既有产物独立复算，不是重跑训练的可复现性验证。
- Version Label：copmrec_v3_bottlenecks_v1。

## 1. 结论与版本边界

收益转化存在两个层次的卡点：**v0 已经把内容可识别性较好地传递到生成候选，但没有建立超出内容排序的稳定增量判别能力；后续排序组件虽然恢复了一部分漏排目标，也破坏了已有命中及前排位置，净收益未落地。**

v3 的问题起点是：保持基础内容检索质量和候选机制，判断候选条件的生成表示能否提供可靠的增量排序信息，减少真实竞争关系的误判。不能直接归为“缺一个 head”“loss 等权”“缺 teacher”或“冻结导致失败”。

- v2 固定实现停止迭代，保留 `beawrjef → 1gu2dsa1` 的负向证据，不追加 v2.x、权重扫描或续训。
- v3 与 v1/v2 平级，以真实 v0 为基础；允许预生成 embedding/SID，推荐参数全部持续可训练，不使用冻结推荐 teacher。
- v2 的 d-only、lambda0.01、beta0.5 不自动成为 v3 的已确认设计。
- 新增完整实验预算0；既有关闭阶段和2000更新校准预算不转移、不重置。本轮结束研发阶段，未杀进程或删除入口/产物。

## 2. 累计尝试与判断

| 尝试 | 保留的有效证据 | 当前判断 |
|---|---|---|
| v0 / BMX-116 | 三数据集×三seed相对LIGER hybrid改善；同checkpoint Mass−Legal有净收益；相对LIGER dense汇总区间跨零 | 候选引导有效，超过强dense的收益未建立 |
| 直接mixed / beta校准 / 冻结小头 | 相对content新增/损失54/80、25/39、1/7，均未晋级 | 路径概率与幅度调整不自动构成相关性 |
| 冻结基座后训练decoder NLL / NDCG | 相对content均新增63/损失86；NDCG对NLL有位置改善，仍未超过content | 提高目标概率或换指标代理不足以保证净收益 |
| competitive NDCG | 新增70/损失89；相对旧NDCG净多4命中但NDCG点值下降 | 竞争负例与高位代价都要计入，不能只看恢复 |
| 单content / 双分支保护 | 前者相对competitive净多3；后者相对单保护净多4，但位置退化抵消NDCG；双保护相对content净少12 | 原正确关系有局部价值，teacher并集未完成闭环 |
| 高位cap | 相对双保护净多2，NDCG增量区间跨零；原9个高位退化只修复1个 | 整体放大用户梯度不等于学对竞争关系；保留历史source origin mismatch限制 |
| 边界辅助 | 相对cap新增5/损失17，Recall区间全负 | 修复局部案例不能代替整体评价，该实例关闭 |
| v1/v1.1等权联合训练 | f91njtjx中间checkpoint基础dense及自然覆盖很弱；6批中4批ranking梯度压过基础目标且共享主干方向冲突 | 全参数训练也会失衡；不是完整testing结论 |
| v1.1 loss校准 | 两臂各1000更新，低权重改善固定2048 selection用户的dense/coverage/Recall；最终NDCG差区间跨零 | 支持该实例等权失衡，不证明完整从零收益 |
| 校准从零v1.1 / v1.2 | 保存run57hzkcol中间曲线，没有本轮独立完整testing结论；v1.2被撤回 | 用户观察保留为线索，不混用评价协议、不把撤回记为效果否证 |
| v2全参数、d-only、有界分数、低权重 | 同testing低于v0/LIGER dense；同checkpoint同候选head的NDCG为负 | 取消冻结、简化head和限制幅度没有解决当前净收益 |

旧冻结阶段是11182 evaluation audit用户的开发证据；v1/v1.1是selection/audit；v0/v2正式比较是22363 testing用户。绝对指标不能直接混比。旧audit经多轮开发使用，不是v3独立确认集；未运行的scratch不记为负向实验。

依据：[旧阶段台账](../../research/docs/2026-10-03-copmrec-conversion-redesign.md)、[v1.1中间诊断](copmrec-f91njtjx-design-diagnosis.md)、[校准验证](copmrec-loss-weight-verification.md)、[v2 testing](copmrec-v2-testing-1gu2dsa1-result.md)。

## 3. 卡点一：候选修复主要恢复 dense 已有能力

v0 用内容mass引导前缀搜索，最终仍按同一content评分排序。它改变候选可达性，没有新增终排判别函数。正式seed42同checkpoint诊断中，warm且dense排名≤10的目标准入率由Legal显著提高到Beauty94.0%、Sports86.2%、Toys90.2%。这一瓶颈已经部分修复。

本轮重新读取已有testing trace、核验MD5/用户/标签，得到：

| Beauty/seed42，22363用户 | 真实v0 | v2 |
|---|---:|---:|
| 自然候选覆盖目标 | 2461 | 2458 |
| 最终Top10命中 | 1608 | 1545 |
| 已覆盖但未进最终Top10 | 853 | 913 |
| dense Top10目标未进候选 | 97 | 98 |
| dense Top10目标已入候选、被终排丢失 | 0 | 211 |
| 最终命中但自身dense未命中 | 90 | 284 |

v0相对自身dense的Recall差为`(90−97)/22363`：恢复dense命中与候选子集晋升近乎抵消。更多高内容分商品入候选，也会挤掉部分原hybrid-only目标，coverage增益不能直接升级为最终收益。

853/913个covered未命中证明存在排序空间，不证明模型可识别所有这些目标。约89%的目标不在候选，也不能全部称为搜索失败：其中绝大多数连同模型dense Top10都不识别。应区分内容识别与搜索准入。

依据：[正式矩阵候选/路径分析](../../research/docs/grid-experiments/2026-10-01-copmrec-problem-scope-and-contribution.md)、[本轮排名分解](evidence/copmrec-v3-bottlenecks-20261004/rank-analysis.json)。

## 4. 卡点二：有恢复信号，缺少可靠的增量竞争判断

同v2 checkpoint和自然候选，content命中1549，融合命中1545；新增239、损失243。完整NDCG10账目为：

| 加head后的变化 | 全部用户平均NDCG10贡献 |
|---|---:|
| 239个新命中 | +0.004819042 |
| 243个丢失命中 | −0.004117317 |
| 共同命中的256个位置改善 | +0.002435504 |
| 共同命中的651个位置恶化 | −0.005467077 |
| 总差 | **−0.002329848** |

命中交换净贡献+0.000701725，随后被位置损失−0.003031573抵消。因此不能说head完全没有信息，也不能只看恢复数量。单正例下，未命中→10收益约0.2891，1→2损失约0.3691，1→3损失0.5。

原候选content第1名262人：141人仍命中但位置变差，11人跌出Top10，共152人降位，组贡献−0.003763271。第2–3、4–5、6–10名组净贡献也均负。原dense Top10组的head贡献−0.006851848，dense Top10之外组+0.004521999，相加仍负。分组使用真实目标，推理时不知道目标dense rank，不能直接部署为安全gate。

旧bad case与此相容：双保护共同退化中9个原第1名贡献60.8%的负向位置损失，70%的新越过关系却已有teacher支持；边界辅助修复6个既有高位案例，同时新增17个Top10丢失。不能由此继续假设“加teacher/加大力度”会解决整体收益。

候选条件表征需要区分正确下一商品与高相似竞争者。旧用户16409的目标蝴蝶发夹被孔雀发夹超过；用户848的目标洁面乳分数提高，但口红竞争者涨得更多；用户22024的爱心发夹由第1跌到第5，仍命中却损失大量NDCG。历史相似、同类别、目标绝对分数上涨都不保证正确排序。部分竞争者更热门，另一些很低频，不能统一归为热门度偏差。

依据：[decoder具体案例](../../research/docs/grid-experiments/2026-10-02-copmrec-decoder-bad-case-analysis.md)、[competitive案例](../../research/docs/grid-experiments/2026-10-02-copmrec-competitive-bad-case-analysis.md)、[双保护关系分析](../../research/docs/grid-experiments/2026-10-02-copmrec-dual-preservation-bad-case-analysis.md)、[后续高位复查](../../research/docs/grid-experiments/2026-10-03-copmrec-head-revisit-analysis.md)。这些旧案例不与本轮testing用户混算。

## 5. 卡点三：训练目标与收益链路不完全对齐

### 排序监督与自然候选准入

v2在自然`beam+cold`之外补充content Top20和training正例，优化补充列表上的融合CE。注入正例是合理监督，不是测试泄漏；但loss下降不要求正例进入推理自然候选。

`generate_candidates`位于`torch.no_grad()`，ranking不能直接对离散选择反传，也没有经该调用直接更新融合alpha；它通过共享encoder/decoder/content参数间接影响以后候选。基础目标仍可训练。所有推荐参数可训练，不等于最终召回链路直接受这项loss约束。

该断点在v1覆盖极低时很重要；v2覆盖已恢复到接近v0，不能用它解释当前主要退化。v3应避免基础检索未学好，却只在正例补入的列表上学会排序。

### d_ui 的用途变化

基础SID/mixed监督`[start,s1,s2,s3]`各位置预测四层SID；v2读取`[start,s1,s2,s3,s4]`之后的末状态做相关性。后者是新增表征用途，基础目标没有直接为它建立完整商品排序监督。参数共享已训练，不能称末状态完全未训练。

cross-attention引入历史，不保证head实际使用有用的历史差异，也不保证同类候选的细粒度区分。当前尚未分离“信息不足、表征用途、参数化、监督、优化”这些原因；不能直接推出扩大或简化head，更不能把它们作为已证实根因。

### 数值尺度约束没有保护正确关系

v1校准支持降低新增目标梯度强度，但不能搬用其比例证明不同head/评分的v2 lambda0.01已充分校准。v2 dense−v0 dense的NDCG区间跨零，也不能确认ranking loss已显著损伤content。

v2约束为`r_i=beta*sigma_u*tanh(a_i)`。若原正确分差`m=c_y−c_j>0`，融合分差为`m+r_y−r_j`；只有`m>2*beta*sigma_u`时，该幅度界足以保证关系不反序。sigma描述全部自然候选的离散程度，未描述这条正确关系的分差，所以有界不等于安全。

beta0.5、sigma平均3.7404，最终Top10的46.34%绝对tanh超过0.95；界的实现有效。不能据此确认训练梯度饱和或cold主导sigma为根因，当前未保存完整候选分数来支持后一判断。

### 分类目标与排名风险

列表CE提高目标相对负例的分数，没有显式按当前Top10进出和高位交换代价分配风险。CE可以学到有效排序，不能说它天然不适合推荐；旧NDCG代理同样未建立收益，不能换回旧loss就认为解决。

容易负例归一化、高位逐用户缩放、竞争梯度份额均有可复核数学现象，但后续干预未完成净收益闭环。v3要检验共享更新后正确竞争关系是否更可靠，不只检验梯度或loss大小。

代码依据：[基础目标](../src/recommendation/liger/joint_mixture.py)、[历史编码与候选生成](../src/recommendation/liger/module.py)、[v2评分与ranking](../src/recommendation/liger/scaled_relevance.py)。

## 6. 证据等级、相关工作与下一步边界

| 判断 | 证据强度与限制 |
|---|---|
| v0修复内容可识别目标的生成遗漏 | 同checkpoint干预/三数据集路径证据支持，不保证超过dense |
| v2当前head净负且高位损失突出 | 同候选testing直接复算，总NDCG配对区间负；单seed固定checkpoint，非训练因果消融 |
| 全参数联合目标可能失衡 | v1中途及有界两臂支持，不能自动移植为v2主因 |
| 有界幅度不保证关系稳定 | 公式确定性结论，不证明哪一幅度会有效 |
| d_ui增量信息未可靠转为终排 | 行为提示待鉴别问题，具体原因未分离 |
| 缺teacher、训练时间、cold占位或单一热门偏差 | 当前不足以作为统一解释，不据此恢复旧实例或扩预算 |

本轮重新读取原始来源：[LIGER§3](https://arxiv.org/html/2411.18814v2)已有联合SID/内容训练与生成后内容排序；[LambdaLoss](https://research.google/pubs/the-lambdaloss-framework-for-ranking-metric-optimization/)定位指标敏感排序代理；[RankDistil](https://proceedings.mlr.press/v130/reddi21a.html)定位teacher头部次序保持。它们用于区分现有基座、排名风险和保护思想，不证明v3效果，也不要求冻结teacher。

v3继续同一个收益转化问题：以已有效的v0候选机制和实际局部恢复为正面依据，将“加head并校准数值”收缩为“学习可靠增量排序，评价全部已有正确关系的损失”。当前不固定新组件，但明确后续验收：

1. 基础content质量与自然coverage不能再次大幅退化，不能用总loss代替。
2. 若引入增量评分，同checkpoint同自然候选应改善整体NDCG，完整报告命中交换与共同命中位置变化，局部修复不够。
3. 最终对真实v0及LIGER dense做匹配Recall/NDCG评价；同候选改善只支持组件，不代替整体方法门槛。
4. 机制须说明要改善哪类竞争关系及各结果对应的保留/停止决定，不在本轮testing坏例上寻找阈值，不部署真值分组gate。

本轮最小必要检查已完成：覆盖与漏排分解、前排退化、dense可识别组/漏排组收益账目。没有发现使v2评价失效的实现疑点，不追加v2重跑；具体v3机制/参数探针/训练对照待设计决定，本轮新run及更新预算0。

## 7. 统计边界与复核

新增分解是固定checkpoint事后描述，不新增bootstrap区间；既有总差区间沿用前轮2000次未校正逐用户配对bootstrap，不代表训练seed总体。testing已进入开发分析，未来独立确认须记录这项边界。

完成11/11统计解释风险检查：聚合反转检查仅限现有分层，不声称穷尽；不把均值推广到每位用户（生态谬误）；真值分层的选择/碰撞偏差不变成部署规则；conditional命中同时给coverage基数；极端bad case不给回归均值后的干预证明；完整报告用户和正负交换，避免幸存者偏差；多轮/多项比较保留未校正标签；确定性选例不挑阈值，保留探索性路径；差值分解不称训练因果比例；关联与反向因果不用于认定历史偏好/优化根因。未检查的假设保留为未知，不宣称全部风险排除。

证据：[只读排名分解源码记录](evidence/copmrec-v3-bottlenecks-20261004/rank-analysis-source.txt)、[结果及输入hash](evidence/copmrec-v3-bottlenecks-20261004/rank-analysis.json)、[前轮完整testing核验](evidence/copmrec-v2-testing-1gu2dsa1-20261004/results.json)。源码记录通过SSH stdin统计既有产物，不是新增pipeline分析入口。
