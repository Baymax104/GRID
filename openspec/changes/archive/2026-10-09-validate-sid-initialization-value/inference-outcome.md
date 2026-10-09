# 冻结推理配对验证结果

## 结论

A在同一Beauty evaluation集、同一seed42下，优于仅首层和深层残差初始化两个对照；两组NDCG@10和Hit@10差异的名义95%配对前缀bootstrap区间均为正。可以固定A为当前主方法候选，将下一步重点转为有限的跨seed复核，不再新增复杂内容分支。

## 完整性

- A：`cgbs-eval-5g3wpbg7-1789305926`；first_only：`klexi92b`；deep_residual：`2j6nowrc`，均finished。
- 三者同22363个唯一用户、相同完整SID标签、目录fingerprint、SID和内容artifact digest；逐用户A指标复现已有缓存。
- 数据配置在等价W&B URI归一后相同，seed、序列长度、层数、码本大小、beam和arm相同，evaluation、beam10。
- 预测均为每用户10个不同的合法目录物品，rank与trace末层存活/beam rank一致；前缀存活单调。
- 三个推荐checkpoint digest分别为`aaba5b47dc9a5e94c0d23b75cf697238`、`4f2fe5145446eee3bbdeb6089b00f043`、`a5fad97f3a5ae19ce8adb9d49de15457`，与预定最佳模型一致。
- first_only输出预测digest=`150809bd2e538f2f82357b061a4701cc`，trace=`2f392f893d058f6b58dcc2cf8ee3385a`。
- deep_residual输出预测digest=`0e63e57c36cc78adec86b002868a0fd3`，trace=`a03be4a9b12ce8a243a319f76f47f937`；残差码本lineage正确，初始化模式/码本hash已写入trace契约。

## 效果

| 条件 | NDCG@10 | Hit/Recall@10 | 命中人数 | NDCG@5 |
| --- | ---: | ---: | ---: | ---: |
| A | 0.04257613 | 0.08004293 | 1790 | 0.03387891 |
| first_only | 0.03769950 | 0.07199392 | 1610 | 0.03011252 |
| deep_residual | 0.04062445 | 0.07682332 | 1718 | 0.03207148 |

单目标推荐中Hit和Recall相同。所有指标由去重后逐用户预测重算，不能把与训练validation聚合值的微小数值差异解释为新收益。

以目标SID前两位为簇，共5401簇，配对抽样2000次、seed42；簇内保留全部用户，统计按用户总数归一。下表差异方向均为A减对照。

| 对比 | NDCG绝对差 | 相对提升 | NDCG名义95%区间 | Hit绝对差 | Hit名义95%区间 |
| --- | ---: | ---: | --- | ---: | --- |
| A−first_only | 0.00487662 | 12.94% | [0.00109202, 0.00844308] | 0.00804901 | [0.00271656, 0.01333710] |
| A−deep_residual | 0.00195167 | 4.80% | [0.00034922, 0.00346193] | 0.00321960 | [0.00052311, 0.00610492] |

区间未做多重比较校正，只描述固定模型在该evaluation样本上的配对不确定性，不包含训练seed及checkpoint选择不确定性。

## 命中与排名

- 相对first_only：A新增677个命中、失去497个，净增180；共同命中1113。NDCG差异分解：新命中+0.01417314、失去命中−0.00962196、共同命中排序+0.00032544。
- 相对deep_residual：A新增423个命中、失去351个，净增72；共同命中1367，其中478个排名变好、442个变差、447个不变。NDCG差异分解：新命中+0.00765475、失去命中−0.00625488、共同命中排序+0.00055181。

A对残差方案的优势同时包含净新增命中和共同命中排序改善，不是所有用户都改善，也不能只用净增72掩盖双向变化。

## 前缀与分组

| 条件 | 第1层存活 | 第2层存活 | 第3层存活 | 完整SID命中 |
| --- | ---: | ---: | ---: | ---: |
| A | 8371 | 3085 | 2149 | 1790 |
| first_only | 8276 | 2892 | 1954 | 1610 |
| deep_residual | 8329 | 3006 | 2058 | 1718 |

A−deep_residual逐层净差为+42/+79/+91/+72，说明改善涉及到达后续层和最终排序。但首层初始化相同不代表训练后首层预测相同，不能据此作逐层独立因果归因。

沿用既有固定训练频次分组：A相对残差方案在Head/Mid/Tail+Cold的NDCG点估计均为正，Head/Mid区间跨零，Tail+Cold区间为[0.00051869,0.00344848]。这是探索性分组，未做多重校正，不据此改写为专门的长尾方法。

## 研究判断与下一步

本轮完成了预定的两项排除：首层不足以解释A的全部效果；匹配均值/尺度后的残差质心方案也未达到完整内容均值方案。结合此前几何核查，支持“深层token代表量选择有实际价值”这一候选机制。

仍未证明跨seed、跨数据集、独立测试集泛化，也未完全隔离预处理与码向量范数因素。evaluation用于checkpoint选择，当前推理只是对同一选择集的冻结模型配对检查，不是新的独立效果证据。

建议固定A实现和主线叙事，不新增模块。下一步限定Beauty seed43、44的A与deep_residual成对训练（4个run），保持既有协议：优先检验较小但关键的约4.8%优势能否跨seed保持。first_only本轮已回答首层解释问题，暂不复制完整消融矩阵。该建议尚未实现或启动；用户手动授权启动后再执行。若新seed方向冲突，先报告不稳定，不自动调参换假设。

## 证据

- [first_only推理](https://wandb.ai/baymaxam/GRID/runs/klexi92b)
- [deep_residual推理](https://wandb.ai/baymaxam/GRID/runs/2j6nowrc)
- [A推理](https://wandb.ai/baymaxam/GRID/runs/cgbs-eval-5g3wpbg7-1789305926)
- `tmp/sid_inference_results/runs.json`：在线状态、配置、输入输出身份。
- `analysis.json`及三个`*_users.csv`：配对统计和逐用户证据。
- `tmp/inspect_sid_inference_results.py`只读采集；`tmp/analyze_sid_inference_results.py`本地静态复算。

本轮所有一致性断言通过；未执行模型、未发布run、未修改生产代码或创建Git提交。
