# 四 checkpoint Prefix Survival Pilot 分析

## 决策

2026-09-11 通过 W&B API 重新读取四条 paired diagnosis、八条 trace 的 config、summary、notes、Artifact 与 lineage，并下载原始 Artifact 核验。**本轮 H2 为 No-Go：不创建 `add-budget-aware-prefix-decoding`，停止主方法实现。**

数据支持 Tail 的目标路径存活率较低、前两层退出较多、增加 beam 能恢复少量目标；尚不支持“控制 frequency/static risk 后，prefix competition 导致稳定的 Tail 特异性剪枝损失，并足以支撑 Budget-Aware Prefix Calibration”。这是当前方法立项的否决，不是对所有搜索干预可能性的否定。

本轮是 Beauty/Sports × RKMeans/RVQ × seed 42 的 evaluation 探索证据，不是三数据集三 seed 的确认性结论。没有启动新训练或修改 decoder score。

## 数据来源与审计

| 设置 | Diagnosis | K=10 trace | K=50 trace | checkpoint | SID |
|---|---|---|---|---|---|
| Beauty RKMeans | [dqkuahdl](https://wandb.ai/baymaxam/GRID/runs/dqkuahdl) | m4h0geda | 56rrarps | 26qh50do | 4vyi4o6w |
| Beauty RVQ | [a50lmism](https://wandb.ai/baymaxam/GRID/runs/a50lmism) | r94ut5sv | k7mr3xlk | ye9u9yj7 | d2hhqdic |
| Sports RKMeans | [kkh97wtp](https://wandb.ai/baymaxam/GRID/runs/kkh97wtp) | 5024wy48 | 4x3steyq | 129w8p0r | 3narllqy |
| Sports RVQ | [xzfxnzyb](https://wandb.ai/baymaxam/GRID/runs/xzfxnzyb) | lzttakbs | n6svfy9g | 49ote174 | ykntf4ve |

- 12 条 run 均为 `finished`。diagnosis evidence 分别为 `tail-sid-diagnosis-evidence:v13/v14/v15/v16`。
- 八条 trace 均为 `tiger_prefix_trace_v1`、evaluation、seed 42、4 层（含 dedup）、codebook size 256；对应 K=10/K=50。
- 每对 trace 的 keys、labels 完全一致，Beauty 每条 22,363 用户，Sports 每条 35,598 用户。checkpoint/SID metadata 与预注册引用一致，配对 `used_artifacts` 版本一致。
- recommendation bundle 仅包含 `keys`、`predictions`；按 key 排序后与 trace 对齐。每个用户的最终 target survival 与最终推荐候选中的 target membership 完全一致。
- schema、唯一 key、rank sentinel、存活后不能从退出状态重新出现、first-failure depth 检查均通过。
- 6,955,320 条最终生成候选均属于对应 model SID catalog，非法完整 SID 数为 0，因此这些最终候选的各级 prefix 也合法。该检查不声称覆盖未保存的中间 beam。诊断 CSV catalog 另外与原始四份 SID bundle 集合逐一核对一致。
- diagnosis 使用 evaluation labels，`calibration_statistics_ready=true`；embedding 为 Beauty `3jtt9mpa`、Sports `psec3u5i`。
- 原始下载及审计脚本位于仓库忽略目录 `logs/h2_pilot_review_2026-09-11/`，精简可审阅结果见本目录 `pilot-audit.json`。

## H2 门槛

| 设置 | 当前 matched Tail−Head survival（百分点） | competition 偏相关 | 未匹配差距的 prefix-cluster 95% CI（百分点） | W&B H2 verdict |
|---|---:|---:|---|---|
| Beauty RKMeans | +5.053 | +0.1053 | [-13.827, -8.381] | not_supported |
| Beauty RVQ | -1.419 | +0.0863 | [-12.848, -7.768] | not_supported |
| Sports RKMeans | +1.690 | +0.2379 | [-7.629, -4.609] | not_supported |
| Sports RVQ | +2.318 | +0.1677 | [-7.843, -5.072] | not_supported |

这里 matched 与 bootstrap 使用的 Tail 均合并 Tail-Cold，survival 是四层存活指示的用户内均值。matched 差距只有 1/4 设置为负，不满足跨设置方向稳定；competition 控制 `log1p_frequency,raw_damage` 后四组全部为正，不满足预设负关联条件。四条 run 的 `prefix_survival_mechanism` 均为 `not_supported`。

CI 的估计量是**未匹配**的 Tail−Head 四层平均 survival，不是 matched 差距；不能把此 CI 放在 matched 数值旁当作其显著性区间。CI 仅支持总体路径存活劣势。

原有 `equal_risk_tail_vulnerability=supported` 属于不同的 recommendation outcome 判据；`cross_setting_stability=supported` 也不能替代这里四组 H2 的交叉判断。

## 逐层发生了什么

以下各表 Tail 单独统计，不含 Tail-Cold。Beauty Tail 有 3,086 用户，Sports Tail 有 4,660 用户；两数据集 Head 分别有 11,149 和 18,117 用户。

| 设置 | K=10 Tail L1 survival | L2 | L3 | L4 | K=10 Head L4 |
|---|---:|---:|---:|---:|---:|
| Beauty RKMeans | 31.432% | 6.384% | 1.782% | 0.940% | 12.118% |
| Beauty RVQ | 32.437% | 7.485% | 2.852% | 1.231% | 12.234% |
| Sports RKMeans | 34.056% | 3.305% | 0.558% | 0.021% | 6.039% |
| Sports RVQ | 33.326% | 3.562% | 0.579% | 0.043% | 6.138% |

Tail 在第一层已有约 66%–69% 的目标退出；到第二层累计约 92.5%–96.7% 退出。因而最终低命中主要发生在 dedup 之前，不能只归因于最终 SID-to-item 映射。

| 设置 | Head L2 teacher target probability | Tail L2 probability | Head L2 legal rank | Tail L2 legal rank |
|---|---:|---:|---:|---:|
| Beauty RKMeans | 0.2155 | 0.0499 | 4.59 | 18.13 |
| Beauty RVQ | 0.2152 | 0.0653 | 4.55 | 17.07 |
| Sports RKMeans | 0.1289 | 0.0401 | 6.41 | 20.83 |
| Sports RVQ | 0.1317 | 0.0428 | 6.06 | 19.36 |

teacher-forcing 在提供真实 parent 的条件下仍显示明显的第二层概率/rank 劣势。**解释性推断：模型的条件概率分配可能是主要限制之一。**这些观测不能把模型分数质量、搜索剪枝和 frequency 的因果贡献分离，也不足以指定新的校准公式。

## Widened beam 恢复了什么

恢复率定义为 `K10 miss 且 K50 final survive / K10 miss`。完整 target membership 已通过 recommendation bundle 独立核验；K=50 是 50 候选可达性，不是 Hit@10 或 NDCG@10 改善。

| 设置 | Tail K10 命中数 → K50 命中数 | Tail 新恢复 / K10 miss | Tail recovery | Head recovery | Tail 最终可达率 K10 → K50 |
|---|---|---|---:|---:|---|
| Beauty RKMeans | 29 → 64 | 35 / 3,057 | 1.145% | 20.759% | 0.940% → 2.074% |
| Beauty RVQ | 38 → 73 | 35 / 3,048 | 1.148% | 20.777% | 1.231% → 2.366% |
| Sports RKMeans | 1 → 5 | 4 / 4,659 | 0.086% | 12.172% | 0.021% → 0.107% |
| Sports RVQ | 2 → 16 | 14 / 4,658 | 0.301% | 13.020% | 0.043% → 0.343% |

Head 新恢复目标数依次为 2,034、2,033、2,072、2,214；Head 和 Tail 的 K10 已命中目标在本轮均无 K50 丢失。Tail-Cold 在 Beauty 有 51 个目标、Sports 有 118 个目标，四组 K10/K50 最终命中均为 0；样本量不足以推断一般性结论。

Tail 平均 failure-depth shift 依次为 +0.390、+0.384、+0.407、+0.426 层（完整存活编码为第 5 层）。K50 的 Tail L1 survival 已升至 61.2%–68.4%，L2 却仅有 9.36%–13.19%；拓宽 beam 确实延后退出，但多数目标仍无法走完整条路径。

## 运行成本

| 设置 | K10 W&B runtime | K50 W&B runtime | 比值 | 单份 trace 大小 |
|---|---:|---:|---:|---:|
| Beauty RKMeans | 392 s | 497 s | 1.27× | 5,729,755 bytes |
| Beauty RVQ | 439 s | 505 s | 1.15× | 5,729,755 bytes |
| Sports RKMeans | 304 s | 1,139 s | 3.75× | 9,118,107 bytes |
| Sports RVQ | 298 s | 1,156 s | 3.88× | 9,118,107 bytes |

runtime 来自各 run 的 `_runtime`，包含运行与记录流程，且任务存在时间重叠；不是隔离环境下的纯 decoder benchmark。没有同环境 trace-off 配对计时，不能据此量化 instrumentation 自身开销。compact trace 大小不随 K 增长；recommendation bundle 随候选数增长。

## 当前统计实现的限制

1. `_risk_matched_survival` 匹配 raw damage 与 log frequency 的加权距离，但没有 caliper、共同支持域或匹配平衡报告。Head/Tail 本来由 frequency 分组，因此不能宣称频率已被严格匹配消除。
2. 匹配选择一条 Head 用户记录，相同 Head item 的多条用户记录拥有相同匹配特征；`min` 的并列选择依赖记录顺序，可能复用同一用户的 survival。当前正负号不应被当作稳健的“等风险后逆转”结论。
3. `_prefix_cluster_bootstrap` 按前两层 prefix 重采样，计算的是未匹配差距；没有重新匹配或输出逐层 matched CI。
4. competition 使用真实 parent 的合法下一 token 数在层间的均值，是 catalog branching proxy；并非完整 beam 中竞争者分数或 Head 竞争质量。正偏相关表明该 proxy 未支持预期方向，不能证明竞争对 Tail 没有因果影响。
5. `supported` 的实现只检查 matched 差距与 competition 相关方向，未将 recovery、CI 和跨 run 稳定性纳入。因此方法 gate 必须在报告中独立执行。
6. summary 的 `tail_recovery_rate` 取 Tail/Tail-Cold 组恢复率的最大值，非加权合并率。本报告明确使用各组行和整数分子分母。
7. `avg_cutoff_margin` 跳过不可达 parent 产生的 NaN，属于条件均值；不能用正均值否认大量目标已经退出。

以上限制已记录为后续机制分析的修订条件。本次保留原始统计实现与 run 结果，不为了追求 Go 而改变口径或事后调整阈值。

## 后续边界

- 完成 pilot 审阅并按预注册分支执行 No-Go，不创建主方法提案。
- 若重新开启 H2，应先明确估计量，修复匹配的共同支持与用户并列选择问题，输出逐层 matched CI，并区分 catalog branching 与真实 beam score competition；随后再决定是否补充独立 seed，而不是直接扩展主方法。
- 即使重分析取得正向结果，也仍需冻结方法后用独立 testing 和更多 seed 验证；当前 evaluation 不能充当确认集。

## 本次验证

原始 trace schema、八份 recommendation bundle、四份 SID catalog、配对 identity、全部最终候选合法性、target membership 与 first-failure 一致性审计均通过。`openspec validate add-tiger-prefix-survival-instrumentation --strict` 与 `git diff --check` 通过。此次只增加分析文档与精简审计数据、回填任务状态；现有未提交代码改动保留，不作提交或归档。本次没有重跑 pytest；验证针对真实 Artifact 与文档，先前测试勾选不代表本轮重新执行。
