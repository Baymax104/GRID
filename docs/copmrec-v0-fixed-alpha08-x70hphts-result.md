# v0固定alpha0.8训练：Testing结果与bad case

日期：2026-10-05。训练 [ncyol9n0](https://wandb.ai/baymaxam/GRID/runs/ncyol9n0) 的dense验证best44500，推理 [x70hphts](https://wandb.ai/baymaxam/GRID/runs/x70hphts)，对照原v0学习alpha训练7y54j4m6/best48000及推理6dspa7e3。

## 结论

本次Beauty/seed42匹配对照中，固定0.8从头训练没有推荐收益：Recall10相对−3.05%、NDCG10相对−2.60%，用户配对95%区间均低于零。固定训练设置不晋级，保留普通v0学习alpha默认及v3.1开发基座；单次训练和Testing已消费，不自动新增alpha扫描或重训。

## 独立Testing指标

| 指标 | 原v0学习alpha | v0固定0.8训练 | 相对变化 |
|---|---:|---:|---:|
| Recall@5 | 0.043822385 | 0.043822385 | 0.00% |
| NDCG@5 | 0.027383122 | 0.027139533 | −0.89% |
| Recall@10 | 0.071904485 | 0.069713366 | −3.05% |
| NDCG@10 | 0.036448314 | 0.035501181 | −2.60% |

完整22363个相同Testing用户，固定checkpoint的用户配对bootstrap2000次、seed42：Recall10绝对差−0.002191119，95%区间[−0.003979788,−0.000402450]；NDCG10绝对差−0.000947133，区间[−0.001848731,−0.000083025]。这是开发数据上的未调整、固定checkpoint区间，不覆盖训练seed随机性，也不推出其他固定值或固定训练普遍无效。

## 损益与候选bad case

最终Top10原v0命中1608个，固定训练命中1559个；新增184、损失233，净−49。共同命中406升位、411降位，其余同位。NDCG分解为新增+0.003013723、损失−0.003710587、共同命中位置变化−0.000250268，合计−0.000947133。

233个最终损失中，100个目标不在新候选、133个目标仍在候选但未进终排Top10；218个在新模型全目录dense排名已超过10。原dense命中但候选遗漏的用户有12个在新模型最终命中；原dense Top10外的增量命中损失40个，新模型dense Top10外且新增最终命中38个。这些是训练、评分及候选同时改变后的损益，不能视作仅改变推理alpha的效果。

| 各自模型的候选表现 | 原v0 | 固定0.8训练 |
|---|---:|---:|
| 自身dense Top10命中 | 1615 | 1569 |
| 自身dense Top10目标候选遗漏 | 97 | 102 |
| 自身dense Top10外的hybrid命中 | 90 | 92 |
| hybrid相对自身dense净命中差 | −7 | −10 |
| 目标候选覆盖率 | 0.110047847 | 0.109242946 |

高内容目标的候选遗漏按首次失败层统计，原v0为[67,25,3,2]，固定训练为[71,22,3,6]。两个模型的dense Top10目标集合已变化，97→102不是同一固定目标集合的逐例准入对照。固定训练首层仍有71例自身dense Top10目标遗漏，前两层合计93/102；未实现v3/v3.1首层保护，本次仍为全层Mass。目标候选覆盖新增236、损失254、净−18。

## 指标下降主要体现在哪里

精确会计恒等式为：hybrid指标变化=dense指标变化+“hybrid−dense”差值的变化。

- Recall10：−0.002191119=−0.002056969−0.000134150。即dense命中少46个，候选相对自身dense的净差再少3个，总计少49个。
- NDCG10：−0.000947133=−0.000971662+0.000024529。dense评价下降解释了大部分差额，候选相对dense的NDCG差值反而轻微改善。

这支持本次固定训练的整体效果下降主要体现在dense评价，候选增量没有抵消下降；这是指标分解，不能将它升级为唯一训练根因。固定训练初始即用0.8，原v0从0.5学习到约0.813，整个联合训练轨迹不同，不能用最终alpha接近来断言应当等价。本轮不实施content排序优化或追加调参。

## 核验与边界

actual config的model/data与准备命令一致，单卡、predict batch32、FP32、seed42、beam20、全层Mass、fixed_mixture_alpha0.8，推理覆盖null。实际消耗ncyol9n0验证best44500，checkpoint Artifact digest `5e490d391fafee0d086d3275eacd9f94`、文件MD5 `aEa/LzeefPLMh33BZFtTEg==`一致。candidate/path trace固定策略与所有可达目标的混合概率核验通过。

输出bundle与candidate trace逐项一致，22363用户/最后目标由原始Testing TFRecord独立读取，SID合法、每用户10个不重复输出、目录/内容输入相同，指标与W&B完全一致。源码归档342个文件、tar/manifest/aggregate hash及12个准备推理文件全部通过，source_sha256=dc71443d4740403c1bfa0493974f8288e3d08ed55bb74bb5fbb57015b7f25b7b。旧v0 checkpoint字节一致引用迁移及历史provenance边界保持，本次不补造早期训练源码。

分析只读取已有产物，没有模型前向、候选搜索、新训练/推理或外部W&B/Linear写入；不修改运行代码、组件配置或脚本。证据见 [results.json](evidence/copmrec-v0-fixed-alpha08-x70hphts-20261005/results.json) 和 [verification.json](evidence/copmrec-v0-fixed-alpha08-x70hphts-20261005/verification.json)。
