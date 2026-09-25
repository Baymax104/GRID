# CGBS C 冻结内容开关：结果与机制边界

后续signal已完成，当前结论及下一候选以 [精确聚合与置乱分析](c-signal-outcome.md) 为准；下文保留screen采集时的结果与预定判据。

## 结论（2026-09-16）

原 C 的内容路由在本次 evaluation 上有正向点估计：相对关闭版本，NDCG@10 +2.668%，新增 163 个命中、丢失 147 个，净增 16。真实前缀存活在第 1/2 层分别净增 313/117 个。但预定前两层 SID 聚类 bootstrap 的总体 NDCG 差值区间跨零，且完整 C 仍低于内容初始化 A。当前支持保留内容分支继续定位，不支持“稳健推荐增益”“复杂结构超过初始化”或“论文机制已确立”。

更具体的观察是：前缀存活数量增多，并未按同样幅度转化为最终命中；救回路径与被挤出路径的最终价值不同。这一现象为 CGBS 内部评分修正提供了位置线索，不直接证明近似误差或某个门控模块是唯一解。

## 来源与完整性

| 条件 | W&B run | prediction artifact ID | prefix trace artifact ID |
| --- | --- | --- | --- |
| C/trained | [e0t8l1oa](https://wandb.ai/baymaxam/GRID/runs/e0t8l1oa) | `QXJ0aWZhY3Q6MzYyODU0MDUzOQ==` | `QXJ0aWZhY3Q6MzYyODU0MTA0Mw==` |
| C/off | [sgpj4amu](https://wandb.ai/baymaxam/GRID/runs/sgpj4amu) | `QXJ0aWZhY3Q6MzYyODU1NzkxMw==` | `QXJ0aWZhY3Q6MzYyODU1ODQ4OQ==` |

两者均 finished，checkpoint 为 `k6jvoo2v` 的 `checkpoint_epoch=000_step=019000.ckpt`；实际 checkpoint artifact ID `QXJ0aWZhY3Q6MzYyNjg1Mjc4Mw==`，digest `3ad64455421afed359ce757ea54ffa28`。SID/embedding 输入 artifact ID、digest 相同。完整 config 比较的差异只有 content_scoring 及运行名称、notes、输出路径等元信息；data_split=evaluation、beam10、GPU0、seed42 一致。

四个 artifact 已下载，ID/digest 与在线快照匹配。使用项目 bundle loader 检查 schema 并按 user_id 排序，22363 个唯一用户、完整 4 位 SID 标签逐行相同；每用户 10 个唯一、合法目录物品。预测命中/rank 与 trace 第 4 层 survival/rank 全部一致。

W&B summary 只有运行耗时，没有推荐指标。本报告指标来自已发布 prediction/trace 产物的本地配对复算，不是新发布的 W&B diagnosis run，也没有执行模型或完整 diagnosis pipeline。

A 复用既有 `n5rv1dgm` diagnosis 的本地 evidence（对应训练 `5g3wpbg7`），manifest.complete=true；本轮逐行验证其 user_id 和由 item catalog 还原的完整标签 SID 与 C 相同。A 的训练频次分组固定复用，未根据本次收益重新分组。历史输入序列字节未重新核验。

## 同口径推荐指标

| 模型 | NDCG@10 | Hit@10 | 命中用户 | Tail+Cold 命中 |
| --- | ---: | ---: | ---: | ---: |
| A 简单内容初始化 | 0.042576128 | 0.080042928 | 1790 | 51 |
| C/off | 0.040300556 | 0.076733891 | 1716 | 36 |
| C/trained | 0.041375760 | 0.077449358 | 1732 | 37 |

单目标任务中 Hit@10 与 Recall@10 等价。NDCG 由命中时的 `1/log2(rank+1)` 计算，未命中为零。表中统一使用逐用户推理复算值，不混用训练 validation summary；C 训练 best 值 0.041390747 与本表稍有差异，未归因为任何未经核验的具体原因。

### 配对结果

- C/trained − C/off：NDCG 差值 +0.001075204（相对 +2.668%）；命中净增 16，Hit 增加 0.07155 个百分点；新增命中 163、丢失 147；按截断到 top10 的目标 rank，604 人改善、411 人变差。
- 95% 配对 bootstrap 区间为 `[-0.000346566, 0.002300865]`，跨零；按预先约定的前两层 SID 前缀聚类，共 5401 簇，2000 次重采样，seed42。
- C/trained − A：NDCG 差值 −0.001200368（相对 −2.819%），区间 `[-0.003011700, 0.000504118]`；新增命中353、丢失411，净少58个命中。
- evaluation 同时用于选择 checkpoint；上述区间不包含训练 seed 不确定性，也不构成独立测试集的显著性或泛化结论。
- 只有91名用户的全部推荐顺序相同；1893名用户的top10集合相同；平均每人共有8.126个候选。说明路由改动广泛影响候选，不能由净增16推断“只影响16个用户”。

## 从局部评分到真实路径

| 层 | off 存活 | trained 存活 | 新救回 | 新丢失 | 净变化 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1 | 8055 | 8368 | 681 | 368 | +313 |
| 2 | 2893 | 3010 | 310 | 193 | +117 |
| 3 | 2059 | 2081 | 179 | 157 | +22 |
| 4 / 最终命中 | 1716 | 1732 | 163 | 147 | +16 |

第 2 层在正确父前缀下的 teacher target rank≤10 用户数从15349增加到15929；真实第2层存活从2893到3010。因此局部评分改善确实伴随真实路径变化，不能只归为 teacher-forcing 指标上的变化。

进一步按用户连接最终结果：

- 第1层新救回的681条目标路径中，40条最终在trained命中；第1层新丢失的368条中，31条在off本可最终命中。
- 第2层新救回310条，其中96条最终在trained命中；新丢失193条，其中121条在off本可最终命中。该层存活集合变化对应的最终命中差为96−121=−25；最终全体净增16还包含两者均通过该层的用户在后续竞争中的变化。
- 不同层的救回集合有重叠，不能跨层累加。层间净差缩小也不是严格的转化率；这里只报告真实用户级交集。

这提示应同时关注“救回多少前缀”与“是否保住最终有用路径”。纯粹优化前缀数量或局部 rank 尚不足以保证推荐收益；标签仅用于事后评估，不得拿上述受益/受损标签构造上线 oracle gate。

## 收益分布

| 组 | 用户数 | 新增命中 | 丢失命中 | 净命中 | NDCG 差值（trained−off） |
| --- | ---: | ---: | ---: | ---: | ---: |
| Head | 11149 | 146 | 135 | +11 | +0.001963317 |
| Mid | 8077 | 16 | 12 | +4 | +0.000202547 |
| Tail+Cold | 3137 | 1 | 0 | +1 | +0.000165697 |

Tail+Cold 的排序变化只有7人改善，最终仅多1次命中，不能据此建立广泛长尾改善主张；该组37次命中仍少于A的51次。NDCG总体改变量主要由Head贡献。

## 对方法路线的更新

1. 保留 CGBS 内容初始化、辅助监督和联合训练的原 C 参考；D 完全梯度隔离不晋级。
2. 将已获支持的事实表述为：固定 C checkpoint 时，内容路由能改变早期候选分配，且本样本推荐点估计向好。不能表述为在线模块普遍有效，C/off 是已共适应模型的消融，不是独立训练的无分支基线。
3. 当前尚未解决的瓶颈是评分修正带来的最终收益/误伤平衡，以及相对强初始化基线的净增益。近似误差、分支校准、共同训练的表示变化尚未被分离；不直接选定新训练模块。

## 唯一建议的下一步：原定 C signal 两次冻结参考

原 C 的在线信号有作用但转化有限，现在用同一 C checkpoint 的 `exact` / `shuffled_exact`，区分真实内容对应关系和原型近似。保持 query、各层 alpha 和 beam 不变；精确聚合不是效用上界，置乱也可能产生分布偏移，因此结果只用于下一步定位。

```bash
bash ./tiger_catalog_grounded_mechanism_inference.sh \
  --data-dir data/beauty \
  --condition c \
  --stage signal \
  --checkpoint-path 'wandb://baymaxam/GRID/k6jvoo2v?role=checkpoint&file=checkpoint_epoch=000_step=019000.ckpt' \
  --notes "CGBS C signal; frozen best step19000; exact versus shuffled exact; diagnose prototype approximation and content correspondence"
```

从node1仓库根目录运行；只执行两次GPU0冻结推理，不训练、不重跑screen。已存在入口支持这两个模式，本轮未改模型或脚本。

- exact 明确改善 trained、同时优于 shuffled_exact：才有依据优先针对原型近似修正，并检查相对A和逐用户净收益。
- exact 仍无改进，但真实内容优于置乱：优先保留内容信号，结合已观察误伤研究混合决策/校准，不继续增加原型容量。
- exact 与置乱无可区分的推荐效果：当前证据无法归因于正确内容对应关系，暂不启动新训练或宣称主机制成立。

这些是用于减少不确定性的分支判据，不是提前给任一方案背书。本轮没有自动执行signal或新增训练。

## 可复核产物

`tmp/cgbs_c_screen/runs.json` 保存在线配置、summary、notes及所有artifact身份；下载目录按run ID划分。`analysis.json` 保存全量统计、配置差异、区间与逐层交集；`trained_users.csv`、`off_users.csv`、`A_users.csv` 保存对齐结果。统计脚本为 `tmp/analyze_cgbs_c_screen.py`，从仓库根目录通过 `uv run python -m tmp.analyze_cgbs_c_screen` 复算。
