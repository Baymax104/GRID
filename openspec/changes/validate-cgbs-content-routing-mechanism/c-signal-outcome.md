# CGBS C 精确聚合与内容置乱结果

后续设计细化见 [状态门控决策](gate-decision.md)：区分科学未知与可固定的工程选择，并收紧同checkpoint常量对照的解释边界。该候选尚未实现或验证。

## 决策结论

2026-09-16 在线核验并复算已发布产物：精确内容聚合没有改善原型版本；正确内容对应关系相对固定置乱有正向证据。按事先的 signal 判据，下一优先修正应转向 CGBS 内部混合决策，保留原有原型、辅助监督与内容初始化，不扩大原型数或将精确聚合作为新主方法。

这是对已有 CGBS 内部修正位置的收敛，不是方法家族切换。C 仍未超过 A；正确内容优于置乱也不等于复杂结构已经提供了强基线之上的净增益。当前仍不足以宣称完整论文机制成立。

## 运行与产物身份

| 模式 | W&B run | 状态 | prediction artifact ID | prefix trace artifact ID |
| --- | --- | --- | --- | --- |
| exact | [0vis21sg](https://wandb.ai/baymaxam/GRID/runs/0vis21sg) | finished | `QXJ0aWZhY3Q6MzYyODczMjM2Mg==` | `QXJ0aWZhY3Q6MzYyODczMjc3Ng==` |
| shuffled_exact | [olsg1a90](https://wandb.ai/baymaxam/GRID/runs/olsg1a90) | finished | `QXJ0aWZhY3Q6MzYyODc1MTM3Mw==` | `QXJ0aWZhY3Q6MzYyODc1MTkxOQ==` |

与既有 trained=`e0t8l1oa`、off=`sgpj4amu` 使用同一 checkpoint：训练 `k6jvoo2v`，step19000，artifact ID `QXJ0aWZhY3Q6MzYyNjg1Mjc4Mw==`，digest `3ad64455421afed359ce757ea54ffa28`。SID=`rkmeans_inference-semantic-id:v2`，embedding=`sem_embeds_inference-semantic-embedding:v5`，四者实际输入 ID/digest 相同。

完整 resolved config 差异仅为 `model.root.content_scoring` 和运行名称、时间、notes、输出目录等元信息；evaluation、GPU0、beam10、seed42、固定内容置乱 seed42 一致。trace metadata 记录 `alpha_source=checkpoint_per_level`、`labels_used_for_scoring=false`，目录 fingerprint 一致。

四个新 artifact 已下载：exact prediction digest=`0f7f1ef04141f10d1b64efac28ab8583`、trace=`c497d343aa5948fda30ff61189a266b3`；shuffled prediction=`aeb4485669ae25501f546d8fa4a48690`、trace=`23cecb70f4e912ba48372034edce8e12`。四模式均有22363名唯一用户，完整目标 SID、用户顺序对齐；每用户10个唯一合法目录物品；预测命中及rank与最终trace一致，存活序列无消失后复活。使用项目 loader 读取协议。

指标来自已发布产物的本地统计，未创建新的 W&B diagnosis run。A 复用此前已核验的 `n5rv1dgm` 逐用户 evidence，重新核对 user_id 与完整 SID 标签。历史输入序列文件字节未重验。

## 同口径结果

| 模式 | NDCG@10 | Hit@10 | 命中用户 | Tail+Cold 命中 |
| --- | ---: | ---: | ---: | ---: |
| A 简单内容初始化 | 0.042576128 | 0.080042928 | 1790 | 51 |
| C/trained 原型聚合 | 0.041375760 | 0.077449358 | 1732 | 37 |
| C/exact 精确聚合 | 0.041238427 | 0.077315208 | 1729 | 37 |
| C/off | 0.040300556 | 0.076733891 | 1716 | 36 |
| C/shuffled_exact | 0.039033798 | 0.073648437 | 1647 | 36 |

以上均为逐用户 evaluation 推理结果，未混用训练 validation 日志的峰值。

| 配对差值 | NDCG 相对变化 | NDCG 绝对差值的95%区间 | 新增 / 丢失命中 |
| --- | ---: | --- | --- |
| exact − trained | −0.332% | [−0.000508290, +0.000220746] | 26 / 29 |
| exact − shuffled_exact | +5.648% | [+0.000617385, +0.003633513] | 216 / 134 |
| exact − off | +2.327% | [−0.000359486, +0.002037632] | 160 / 147 |
| exact − A | −3.142% | [−0.003131289, +0.000437937] | 354 / 415 |
| shuffled_exact − off | −3.143% | [−0.002128203, −0.000381684] | 108 / 177 |

区间沿用预定协议：按前两层SID前缀聚类，5401簇，2000次配对bootstrap，seed42。不同对照区间未做多重比较修正；重点解释预定的 exact/trained 和 exact/shuffled 比较。区间只反映当前evaluation样本的前缀抽样不确定性，不包含训练种子或置乱种子不确定性，且evaluation已用于checkpoint选择。

## 两项不确定性的收敛

### 原型近似：没有显示出值得优先修复的收益缺口

exact相对trained净少3次命中，NDCG略低且区间跨零。平均每用户共有9.713个top10候选，16628人的候选集合完全相同。各层目标存活如下：

| 模式 | 第1层 | 第2层 | 第3层 | 最终 |
| --- | ---: | ---: | ---: | ---: |
| trained | 8368 | 3010 | 2081 | 1732 |
| exact | 8373 | 3013 | 2075 | 1729 |
| shuffled_exact | 8063 | 2828 | 1979 | 1647 |
| off | 8055 | 2893 | 2059 | 1716 |

精确聚合在前两层仅多5/3条路径，最终反而少3次命中。因此当前没有证据优先增加原型、扩大聚合容量或承担精确聚合成本。不能进一步声称原型在任何训练设置下均无近似误差；该实验固定了用原型共同训练出来的query和alpha。

### 正确内容对应关系：模型确实依赖它，但这还不是强基线上的收益证明

exact优于置乱的NDCG差值为+0.002204629，预定区间高于零，最终净多82次命中。置乱保持SID、分支大小和内容向量集合，只破坏向量与物品的对应关系；本结果支持当前模型依赖正确对应关系，而非仅有分支大小效应。

但置乱也会使共适应模型面对分布偏移；更关键的是置乱结果本身低于off，而exact相对off的区间仍跨零。因此不能把“比错误内容好”直接写成“正确内容相对无内容具有稳健增益”，更不能把它写成优于A。Tail+Cold只有37对36次命中，仍不支持广泛长尾改善。

## 固定下一项修正，而不扩展诊断矩阵

截至本轮，证据链为：内容初始化值得保留；完全辅助梯度隔离退化；原C在线路由点估计有益但救回与误伤相抵；精确聚合未改善；正确对应关系优于固定置乱。下一项值得设计的改动是**按用户和当前前缀调节内容混合强度的轻量门控**，属于此前 `refinement-review.md` 中已列出的CGBS内部候选。

### 明确瓶颈

当前每层只有一个全局alpha，同层的不同用户和前缀共享混合强度。已有screen显示新增163次命中同时丢失147次，第2层前缀数量增益与最终命中价值不一致。现有结构缺少按当前评分状态调整内容介入程度的能力。这是代码事实加观测所支持的修正动机，尚未证明全局alpha是唯一或主要因果瓶颈，也未证明损害可由无标签特征预测。

### 单一针对性机制草案

```text
alpha_l(u,p) = alpha_max * sigmoid(b_l + w_l^T z(u,p))
z = [normalized_entropy(P), normalized_entropy(Q), JS(P,Q)]
```

P、Q均在当前合法子分支上归一化；熵按合法分支数归一化，单分支时定义为0，JS有界且不读取目标标签。第一版仅使用每层三维线性修正，w零初始化，使初始行为与原C一致；b保留原层级参数。对门控输入特征停止梯度，保留P/Q混合概率本身的训练梯度，避免把此次变更再混入新的评分梯度反馈。辅助CE权重0.1及其原C编码器梯度保持；四原型、温度、内容初始化、预算与beam保持。

这个草案仍可能失败：低熵不等于正确，JS也不直接表示哪个分支可靠；门控可能学成另一个常数，或者只减少损害而没有新增收益。零初始化仅保证初始一致，不保证训练后不退步。当前不加入额外loss、teacher模型、梯度手术或原型扩容。

### 验证如何证明机制而非堆参数

工程阶段先检查初始前向等价、梯度路径、mask/单分支数值边界和checkpoint身份，完成后才交付一次Beauty seed42、同20k预算的新训练（GPU0/1，由用户手动启动）。本轮尚未实现或启动该条件。

完成后首先比较原C和A；若未改善，不自动扩seed/sweep。若有改善，在同一新checkpoint上将动态项置零或替换为仅由训练样本估计的每层常数，再与动态门控配对比较，用于区分状态依赖与单纯alpha重调。常数参数不得用evaluation标签选择。评估同时报告推荐NDCG、命中新增/丢失、与最终命中连接的路径变化及成本；仅画门控分布不算机制证据。

该设计是下一候选，不能标为已验证主方法。当前仍未通过完整机制资格；不追加B、D推理、rerank、其他数据集或更多signal队列。

## 可复核文件与执行边界

- `tmp/cgbs_c_signal/runs.json`：在线元数据、config、notes、输入/输出artifact身份。
- `tmp/cgbs_c_signal/analysis.json`：配置差异、指标、配对区间、逐层存活、候选重合。
- 同目录五个 `*_users.csv`：完整对齐逐用户结果。
- `tmp/inspect_cgbs_c_signal.py` 与 `tmp/analyze_cgbs_c_signal.py`：只读采集与现有产物统计；后者通过 `uv run python -m tmp.analyze_cgbs_c_signal` 复算。

本轮未修改生产模型或启动脚本，未创建/发布W&B run，未启动训练、推理或完整diagnosis。数值结论来自已有产物的复算与一致性断言。
