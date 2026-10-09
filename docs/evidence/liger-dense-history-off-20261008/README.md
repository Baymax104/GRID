# CoPMRec与LIGER dense：均关闭最终历史排除的实证比较

用户请求“两者都关闭历史排除的效果比较”。[BMX-149](https://linear.app/baymax104/issue/BMX-149) / [run lu8oct42](https://wandb.ai/baymaxam/GRID/runs/lu8oct42)已完成并独立核验；仅新增Beauty/seed42一次单卡Testing，0训练、0独立Validation。CoPMRec关闭排除hc8oct43直接复用。

## 相同推理规则下的结果

两者均FP32完整目录dense、cold商品保留、不屏蔽输入历史商品，稳定catalog-row排序；同22363 Testing用户及标签SHA `55e2602110f989565aa6c5fc00bef99107a17d3d11222c06c46874b9df50e016`。各自使用原validation-selected own-best：CoPMRec gshpyn49 step47500，LIGER 35ig0tz6 step45000；未重新选点。

| 条件 | Recall@5 | NDCG@5 | Recall@10 | NDCG@10 |
| -- | --: | --: | --: | --: |
| CoPMRec：关闭历史排除 | 0.04538747 | 0.02990916 | 0.07092072 | 0.03814285 |
| LIGER dense：关闭历史排除 | 0.04252560 | 0.02702803 | 0.07141260 | 0.03639071 |

| CoPMRec关闭−LIGER关闭 | 差值 | 95% pointwise CI |
| -- | --: | -- |
| recall@5 | +0.00286187 | [+0.00013415, +0.00558959] |
| ndcg@5 | +0.00288113 | [+0.00095453, +0.00477921] |
| recall@10 | -0.00049188 | [-0.00371149, +0.00272772] |
| ndcg@10 | +0.00175214 | [-0.00006695, +0.00365801] |

区间为2000次PCG64 seed42用户配对bootstrap、95% pointwise，未校正多重比较；不表示跨训练seed稳定性或等价。

## LIGER自身最终排除开关

开启结果ldi54f1o复用，固定同一个LIGER检查点和dense评分：

| LIGER开启−关闭 | 差值 | 95% pointwise CI |
| -- | --: | -- |
| recall@5 | +0.00903278 | [+0.00782543, +0.01028485] |
| ndcg@5 | +0.00748112 | [+0.00672148, +0.00824832] |
| recall@10 | +0.00581317 | [+0.00487412, +0.00684166] |
| ndcg@10 | +0.00642487 | [+0.00583263, +0.00703716] |

关闭后历史重叠16879个位置/10584名用户；测试目标在输入历史中的数量为0。关闭列表移除历史商品后的剩余项均是开启列表的前缀，违例0。

## 来源与有效性

LIGER检查点URI `wandb://baymaxam/GRID/35ig0tz6?role=checkpoint&alias=v1&file=checkpoint_epoch=000_step=045000.ckpt`，SHA `8508e08e2a2cc9ea5d2bbc902aa4e2b6415c8a45b9c8a0ddc879728a6b66b43b`；SID dq77e3wo/v0与content 3jtt9mpa/v5固定，checkpoint item/SID/content/seen buffers逐值匹配。CoPMRec与LIGER开启输出均再次核验原文件SHA及指标原值。新增运行源码SHA `a034b49abea85191f19cebc9ebafcca9a1426a7db4be81ad961a791c22e4ed2c`，archive/cloud manifest及最终输出文件身份通过。输出合法唯一[22363,10,4]，四指标独立复算与W&B最大误差5.9e-17。只读audit forward=0。

物理GPU1→cuda:0，独立tmux `liger_dense_historyoff_bmx149_lu8oct42`，group `paper_internal_liger_beauty`。原LIGER入口既有dense模式直接使用，无模型代码修改。

## 解释边界

本记录是Beauty/seed42两种已有训练模型在相同dense与历史规则下的实证对照，不替代原LIGER hybrid主表。开启/关闭排除的差值分别报告，不将联合部署改动拆成未经测量的贡献比例，不按Testing更改模型或检查点。此次授权仅新增1Testing，其余数据集/seed未启动，原M3五臂250k和内部开启dense1/9事实保留。

[精确命令](testing-command.sh)、[启动规格](launch-spec.json)、[独立核验](audit.json)、[配置展开](hydra-compose.yaml)。
