# CoPMRec v5.3：固定Full的最终历史商品排除实证

[BMX-148](https://linear.app/baymax104/issue/BMX-148)；[W&B hc8oct43](https://wandb.ai/baymaxam/GRID/runs/hc8oct43)。用户授权的一次Beauty/seed42单卡Testing已完成并独立核验，新增训练0、独立Validation0、Testing1。工程attempt共2：首次hc8oct42在checkpoint校验阶段失败，尚未运行Testing forward，不计完成；完整失败日志保留。修正仅将冻结参数移到正式检查点校验之后。

## 固定条件与干预

复用正式Full训练gshpyn49的validation-selected own-best step47500。保持历史输入编码、历史/目录残差、参数、FP32完整目录dense评分及稳定排序，仅取消最终历史商品屏蔽。参照Full Testing vosmuihm直接复用。M2的history off关闭历史侧残差，仍屏蔽历史商品，不是本次条件。

Checkpoint：`wandb://baymaxam/GRID/gshpyn49?role=checkpoint&alias=v0&file=checkpoint_epoch=000_step=047500.ckpt`，SHA256 `a64ca93d49b5bb9ead7b4af803091346ca62af5299cd89266ac5a2acce3894c4`。SID dq77e3wo/v0、content 3jtt9mpa/v5。物理GPU1→cuda:0，tmux `copmrec_history_bmx148_hc8oct43`，group `paper_mechanism_copmrec_beauty`。

## 指标原值与配对差值

| 条件 | Recall@5 | NDCG@5 | Recall@10 | NDCG@10 |
| -- | --: | --: | --: | --: |
| Full：最终历史排除开启 | 0.05334705 | 0.03659647 | 0.07776238 | 0.04443310 |
| 固定Full：仅关闭最终历史排除 | 0.04538747 | 0.02990916 | 0.07092072 | 0.03814285 |

| Full−关闭排除 | 差值 | 95% pointwise CI |
| -- | --: | -- |
| recall@5 | +0.00795958 | [+0.00684166, +0.00921164] |
| ndcg@5 | +0.00668731 | [+0.00602466, +0.00740164] |
| recall@10 | +0.00684166 | [+0.00576846, +0.00796069] |
| ndcg@10 | +0.00629025 | [+0.00574222, +0.00687142] |

PCG64 seed42，2000次用户配对bootstrap，95% pointwise CI，未校正多重比较；不表示跨训练seed稳定性或等价。

## 历史重叠与有效性

相同22363用户，user/target SHA `55e2602110f989565aa6c5fc00bef99107a17d3d11222c06c46874b9df50e016`。关闭排除后，Top10与有效输入历史重叠16243个推荐位置，涉及10119名用户；测试目标在输入历史中的用户数为0。输出合法且唯一。

关闭排除列表去掉历史商品后，剩余项均为Full列表的前缀，违例0；这是同一评分排序下仅改变最终排除的产物一致性检查。冷目标138，原值保存在完整audit。

Checkpoint Artifact文件、输入buffers/digests、运行时源码及archive/cloud manifest、最终输出Artifact文件身份核验通过。独立四指标与W&B最大误差4.16e-17。源码SHA `a034b49abea85191f19cebc9ebafcca9a1426a7db4be81ad961a791c22e4ed2c`。只读audit模型forward=0。

## 解释边界

本结果测量最终历史排除在固定CoPMRec Full模型上的作用。历史排除属于实际推理流程，应在方法和消融中明确披露。该结果不能分离此前LIGER对照中全目录候选与历史排除的各自贡献，也不改变原LIGER hybrid基线。保留原M3五臂/250k完成事实；未追加其他模型、seed、数据集或训练。

## 可复现文件

- [精确启动命令](testing-command.sh)与[规格](launch-spec.json)。
- [独立核验](audit.json)、[源码与检查点预检](preflight.json)、[配置展开](hydra-compose.yaml)。
