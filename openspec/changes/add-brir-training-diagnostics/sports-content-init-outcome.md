# Sports 内容初始化：全量推理、分组收益与下一步判断

日期：2026-09-16。新推理run：[vwlwo9wy](https://wandb.ai/baymaxam/GRID/runs/vwlwo9wy)，finished，95秒。使用训练run `6x3rdo7k` 的18k best checkpoint，evaluation、beam10、seed42、GPU0。以下指标由已发布的推荐输出与prefix trace重算，不是训练期峰值。本次未启动新的训练、推理或diagnosis run，未修改远端W&B状态。

## 1. 核心结论

内容初始化的总体收益在全量独立推理中保留：NDCG@10=0.02142543，Hit@10=3.97775%。它相对Mask CE有较明确的配对总体增益；相对Full和Hybrid点估计较高，但按目标SID前两位聚类的95% bootstrap区间包含零，不能宣称已稳定胜出。

长尾假设仍未得到支持。Tail+Cold共有4,778个用户，只命中1个；相对Full净增0，相对Hybrid少2个。总体进步主要是Head/Mid收益。

这轮评价闭环已完成，无需为相同问题再启动一次完整diagnosis。保留简单内容初始化作为后续强基线，停止扩展BRIR v1；下一步设计应优先针对前两层低频目标的条件评分，而不能仅凭末端失败比例再次投入后缀重排或扩大beam。

## 2. 产物与对齐核验

- 新run发布 `recommendation_output:v0` 与 `prefix_trace:v0`，分别包含35,598行推荐结果与35,598行标签/搜索证据。推荐形状为[35598,10,4]，SID目录为18,357个item。
- lineage明确指向 `tiger_catalog_grounded_token_content_init_sports_train-checkpoint:v0`（18k）、`rkmeans_inference-semantic-id:v3`、`sem_embeds_inference-semantic-embedding:v6`。
- Prefix trace通过项目schema校验；推荐和trace的用户key集合完全一致。两份文件的原始行序不同，已先按user_id对齐，未按文件行号直接比较。
- 每条预测均可映射到同一SID目录，每用户10个item无重复；trace最终层survival与按完整SID匹配得到的Hit@10完全一致。
- 新标签与旧original、mask_ce、hybrid、full四组的label_item_id逐用户完全一致。旧证据中的model SID逐item与本次W&B下载的SID v3一致，四份旧证据的训练频次和group映射一致。
- 旧对照使用此前已核验的W&B diagnosis下载文件：original `i7lqvn7a`、mask_ce `7jfv1urw`、hybrid `d68sjtoi`、full `n8i4g6a3`。本次重新读取CSV并按rank公式核对旧指标。
- 新产物与SID文件的SHA-256保存在分析目录，源Artifact及metadata保存在artifacts.json。

evaluation仍是训练时选择best checkpoint的同一划分。这里的“独立推理”指独立重算，不是独立留出测试集。

## 3. 全量指标

| 方法 | NDCG@10 | Hit@10 | 命中用户 | 命中不同标签item |
|---|---:|---:|---:|---:|
| original | 0.01671415 | 3.22490% | 1148 | 260 |
| mask_ce | 0.01855091 | 3.43278% | 1222 | 289 |
| hybrid | 0.02016936 | 3.76145% | 1339 | 363 |
| full | 0.02052866 | 3.85134% | 1371 | 357 |
| token_content_init | **0.02142543** | **3.97775%** | **1416** | **376** |

新条件的NDCG@5=0.01695680，Hit@5=2.59284%。相对mask_ce的NDCG@10提升约15.50%，相对full约4.37%，相对hybrid约6.23%。这些是点估计，不代替不确定性判断。

## 4. 提升来自哪里

| 分组 | 用户数 | Mask CE命中 | Hybrid命中 | Full命中 | 内容初始化命中 |
|---|---:|---:|---:|---:|---:|
| Head | 18117 | 1163 | 1270 | 1324 | 1338 |
| Mid | 12703 | 59 | 66 | 46 | 77 |
| Tail | 4660 | 0 | 3 | 1 | 1 |
| Tail-Cold | 118 | 0 | 0 | 0 | 0 |

配对命中交换如下，新增/丢失均按同一用户计算：

| 对照 | 总新增 / 丢失 | 总净增 | Head净增 | Mid净增 | Tail+Cold净增 |
|---|---:|---:|---:|---:|---:|
| Mask CE | 496 / 302 | +194 | +175 | +18 | +1 |
| Hybrid | 471 / 394 | +77 | +68 | +11 | −2 |
| Full | 433 / 388 | +45 | +14 | +31 | 0 |

相对Mask CE约90.2%的净增命中来自Head。相对Full额外收益主要来自Mid；Tail则是新增1个、丢失原有1个，并未保留同一长尾命中。

内容初始化唯一Tail命中对应训练频次3的item17641、用户11681、排名6。Tail+Cold Hit@10仅0.02093%，不能把Mask CE的0→1写成有意义的相对提升。

按有评价标签的item等权计算，总体macro NDCG@10为0.00503405（Full为0.00482563）；Tail+Cold macro则为0.00006212，低于Full的0.00010082。这里只涉及各一个命中item，排序极不稳定，不能通过选择平均口径支持长尾主张。

## 5. 配对不确定性

固定当前checkpoint和evaluation样本，使用2,000次bootstrap、seed20260916。除用户重采样外，按真实目标raw SID前两位划分7,184个簇，整簇重采样并按抽中用户数计算均值，保留同前缀用户的相关性。

| 新条件减对照 | NDCG@10差值 | 用户bootstrap 95%区间 | 前缀簇bootstrap 95%区间 |
|---|---:|---|---|
| Mask CE | +0.00287452 | [0.00200537, 0.00367937] | **[0.00152718, 0.00426451]** |
| Hybrid | +0.00125607 | [0.00039280, 0.00215493] | **[−0.00016303, 0.00270131]** |
| Full | +0.00089677 | [0.00004112, 0.00175435] | **[−0.00039527, 0.00231345]** |

因此，相对Mask CE的总体增益对这两种重采样均为正；相对Full/Hybrid尚不足以声称稳定优势。Full的Hit@10差值连用户bootstrap区间也包含零。

这些区间只描述固定模型、当前数据划分下的配对不确定性，不包含不同训练seed或基于evaluation选checkpoint的选择不确定性，也不是最终test结果。

## 6. Prefix trace将下一步问题定位到哪里

| 分组 | 第1层存活 | 第2层存活 | 第3层存活 | 第4层存活 |
|---|---:|---:|---:|---:|
| Head | 7130 | 2274 | 1558 | 1338 |
| Mid | 4433 | 702 | 236 | 77 |
| Tail | 1694 | 193 | 31 | 1 |
| Tail-Cold | 30 | 4 | 0 | 0 |

Tail的首次淘汰数依次是2966、1501、162、30。4,660个Tail用户中，4,467个（95.86%）在前两层已经丢失。第3层31→第4层1的条件淘汰率很高，但只能影响此前存活的这31个用户，不能将它等同于全部长尾失败的主要来源。

为区分局部排序与跨父前缀竞争，进一步检查正确父前缀已存活的首次失败用户：

| Tail首次失败层 | 失败数 | 正确父前缀下目标局部rank>10 | 局部rank≤10但未入全局beam |
|---|---:|---:|---:|
| 1 | 2966 | 2966 | 0 |
| 2 | 1501 | 1047 | 454 |
| 3 | 162 | 25 | 137 |
| 4 | 30 | 1 | 29 |

前两层共有4013个用户在发生首次失败时局部rank>10；后层跨父前缀竞争也存在。当前beam10下，问题同时包含早期条件评分不足与全局竞争，不能归结为单一搜索预算或最后去重位问题。只改变最后一位的重排且保持前3层候选不变，其Tail覆盖上界仅31/4660；这个上界也不保证可达或不损害Head。

这还是定位证据，不证明静态SID损伤、训练频次或内容初始化中的某个组件是唯一因果原因。

## 7. 下一步研究边界

本轮无需再补相同推理/diagnosis。现在已有足够信息结束BRIR收尾与Sports缺失对照这两项任务。

- 强基线：保留内容初始化；Full/Hybrid作为对照。根据当前结果，复杂结构尚无足够优势支撑其必要性，简单基线也未被证明在所有seed上优于它们。
- 主张：将“内容有效”与“长尾改善”分开。前者有支持，后者在Sports上仍失败；不要用总体NDCG增长替代Tail结果。
- 若继续以长尾为论文目标，下一项干预必须优先提升前两层低频目标的条件评分，并同时报告总体/Head损失、Tail绝对命中、item-macro与前缀配对不确定性。不能只把固定候选CE替换一个名称，也不能再次盲目叠加后缀残差或频率配额。
- 尚未确定新的可发表主方法。下一阶段应先针对上述定位形成最小、可证伪的方案，再由用户手动启动有限实验；本轮不直接新增训练矩阵。

## 8. 文件与复核

- [原始run快照](../../../tmp/sports_content_init_outcome/runs.json)、[Artifact与lineage](../../../tmp/sports_content_init_outcome/artifacts.json)、[文件SHA-256](../../../tmp/sports_content_init_outcome/file_hashes.json)。
- [总体/分组/配对统计](../../../tmp/sports_content_init_outcome/analysis.json)、[逐用户重算指标](../../../tmp/sports_content_init_outcome/recommendation_metrics.csv)。
- [不确定性](../../../tmp/sports_content_init_outcome/uncertainty.json)、[逐层竞争分解](../../../tmp/sports_content_init_outcome/failure_competition.json)。
- [区间与prefix存活图](../../../tmp/sports_content_init_outcome/outcome.png)。

未新增正式W&B diagnosis run；上述统计属于对已有W&B产物的本地复算，完整脚本与原始bundle保留在同一分析目录。生产代码、配置和启动脚本均未修改。
