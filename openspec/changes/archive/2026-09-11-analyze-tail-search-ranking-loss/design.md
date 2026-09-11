## Context

前置提案 `add-tiger-prefix-survival-instrumentation` 已实现 trace、Artifact 和基本统计，但尚未归档。2026-09-11 pilot 的四条 diagnosis 为 `dqkuahdl/a50lmism/kkh97wtp/xzfxnzyb`，H2 均为 `not_supported`。本变更依赖这些实现，不重写历史结果。

当前 data config 仅加载一份 `recommendation_output_path`；`prefix_trace.py` 用四层平均 survival 做逐用户近邻匹配，距离混合 raw damage 与 frequency，Head item 的重复用户记录会产生并列选择。bootstrap 实际计算未匹配差距。新增分析必须显式区分估计量并复用统一入口，不能把聊天中的临时计算脚本发展为第二个分析 runner。

## Goals / Non-Goals

**Goals:**

- 量化固定 Top10、扩大候选后 Top10、候选可达性和候选内排序空间。
- 定位累计退出与条件退出，并提供可解释、顺序不敏感的静态风险比较。
- 用已保存 Artifact 完成四组 evaluation 重分析，给出有证据与停止条件的下一步建议。

**Non-Goals:**

- 不实现 score calibration、reranking、beam 预算分配、完整候选打分或新 trace 字段。
- 不训练、不重跑 TIGER inference、不扩展 18-setting；不宣称因果识别、方法有效或测试集确认。
- 不重定义原始 `raw_damage`、`priority_score`、Head/Tail 分组或旧 recommendation outcome verdict。
- 不把候选 oracle 当可执行方法，不从仅含 ID 的 bundle 伪造模型 score。

## Decisions

### 1. 输入装配与职责

新增顶层可选输入 `widened_recommendation_output_path: null`，通过 data config、`DiagnosisDataModule`、dataset 和 `DiagnosisBatch` 传递。保留现有 `recommendation_output_path` 为 fixed 输入。根脚本新增 `--widened-recommendation-output-path`，支持两种 flag 形式与非空验证。

模型 component config 中新增默认关闭的 `search_ranking.enabled` 与 `risk_standardization.enabled`；pilot 显式开启。统计参数属于 component，experiment 仅传输入与高层装配。启用 paired analysis 时，fixed/widened recommendation 与两份 trace 缺一应提前报错；未启用时原有无 trace、单 trace、普通 diagnosis 保持可用。

所有 bundle 通过 `src/data/components/artifacts.py` 加载，W&B 下载仍归 `src/utils/wandb.py`；原 URI 与 resolved path 分别记录，`use_artifact` 仍由 lineage callback 执行。统计在 diagnosis domain，文件写入和发布复用共享 writer。

### 2. 身份与排序契约

- 四份输入按唯一业务 key 排序并精确连接，拒绝缺失、额外或重复 key；标签映射到包含 dedup 的唯一 model SID catalog。
- 两份 trace 的 split/checkpoint/SID/schema identity 必须一致，beam width 与各自候选宽度一致；pilot 固定 10/50。
- W&B recommendation 和 trace 必须来自同一对应 source run，并记录 Artifact 版本/digest。不能仅凭相同 key 就认定 checkpoint 一致。
- 本地输入允许计算，但必须记录来源是否可验证；若缺少足以重建 source identity 的已有 manifest/provenance，标记 `identity_verified=false`，不得进入 probe 资格。不得为了本地便利修改标准 `keys + predictions` 协议。
- 每份 recommendation 的 target 最早命中 rank 必须与 trace 最后一层 `target_beam_rank` 一致，且最终 membership 与 survival 一致。非法 catalog SID 应拒绝该配对。
- 按 bundle 中原有排序计算 rank；重复候选只计首次命中，不能压缩重复项后提升 rank。oracle 只使用目标是否存在，不需要候选 score。

### 3. 完备的配对状态与指标

令 `F` 表示 fixed Top10 命中，widened rank 为 `top10`、`below10`（11 至 Kwide）或 `absent`。输出六种互斥状态，而非只统计恢复样本：

| fixed 状态 | widened 状态 | 解释 |
|---|---|---|
| hit | top10 | 保留命中 |
| hit | below10 | 排序下降导致 Top10 丢失 |
| hit | absent | 扩大 beam 后候选丢失 |
| miss | top10 | 新增 Top10 命中 |
| miss | below10 | 进入候选但未进 Top10 |
| miss | absent | 两侧均未命中且 widened 不可达 |

组别输出 All/Head/Mid/Tail/Tail-Cold；Tail 合并 Tail-Cold 必须另命名，不用最大组恢复率冒充合并率。所有比例附分子、分母；零分母使用 null 与原因。

主表包含用户加权 Hit@10、NDCG@10、widened 全候选可达率、六状态人数、Top10 新增/丢失/净增、K10 miss recovery。单目标且 model SID 唯一映射 item 时 Hit@10 等于该用户 Recall@10。item-macro 指标如输出，必须单独标注，不能与用户加权主表混用。

`oracle_hit10_ceiling = P(target in widened candidates)`；`oracle_headroom = oracle_hit10_ceiling - widened_hit10`。这是每个用户有真实标签的候选内 oracle，只是固定候选集的理论可达上限。它既不保证可学习排序收益，也不限制改变候选生成的未来算法。

### 4. 层级指标与首次失败

每组/层输出 `survived_count / group_support` 和 `failed_at_layer / survived_previous_layer`；第一层的 parent support 是全组。完整存活的 `first_failure_depth=-1` 单列，不当作实际失败深度。

只在首次失败且 parent 可达时汇总 beam cutoff margin/parent rank，并输出有效与缺失数量。teacher legal rank 在真实 parent 条件下计算，不等同于全 beam 候选 rank。可记录 `teacher_rank<=K` 的失败比例作为描述性线索，不能以此独立识别跨 parent 竞争因果。

frequency 以训练集 `log1p(freq)` 的预定义分箱输出描述性曲线，并区分 group、层和 at-risk 分母。当前合法 continuation 数命名为 `catalog_branching_proxy`；保留旧偏相关仅作 legacy 描述，不包装成新的因果检验。

### 5. 静态风险标准化取代用户最近邻

主比较是 **静态风险共同支持人群中，Tail 与 Head 的 item-macro 逐层 survival 差距**，不宣称排除了 frequency 的作用。Tail 主分析不含 Tail-Cold；冷启动组单列，合并结果仅作标明权重的扩展。

先按 target item 聚合其 evaluation 用户的存活/at-risk/失败计数与 outcome。以有标签的唯一 Head/Tail item 的 pooled raw damage 构造 5 个等频箱，边界仅由风险决定，重复边界合并，不根据 outcome 选择切点。每箱至少 20 个 Head item 和 20 个 Tail item 才保留。设箱内 item 数为 nH、nT，共同权重为 `min(nH,nT)` 归一化。逐箱分别求 item 均值，使用同一权重标准化后计算 Tail−Head。

必须报告每箱支持数、边界、共同权重、各组 item/user 保留比例、raw damage 标准化均值差（SMD）、frequency 分布差异及原始与标准化估计。SMD 分母使用合并加权组内标准差；零方差且均值相同记 0，否则记 unavailable。默认每组 item 保留率至少 50%、绝对 raw-damage SMD 不超过 0.1；未通过标记 `insufficient_overlap` 或 `imbalanced`，仍保留描述性数值但不得使用合格风险比较的措辞。

这些是本提案的预声明分析质量默认值，不是由结果证实的门槛。主配置在四组重分析前保存；敏感性仅预设 3/5/10 箱，主结论仍使用 5 箱，不按显著性挑选配置。

条件退出率先将每个 item 的失败数、at-risk 数分别除以其 evaluation 用户总数，得到 item 内失败比例与 at-risk 比例；按上述箱内等 item 权重和共同箱权重分别求和，再取两者之比。不把零 at-risk item 记作零失败率，整体分母为零则 unavailable。与累计 survival 分开标明估计量和有效支持。

### 6. CI 必须对应同一估计量

按 model SID 前两层 prefix 对 item 聚类重采样，1000 次、95% percentile CI、seed 来自运行配置。每个抽中的 cluster 带入所有 item 聚合量及其 multiplicity；保留原分析风险箱边界，按同一规则重新计算支持、共同权重、标准化差距与平衡标记。不能复用旧未匹配差距的 CI。

报告有效重复数、无共同支持/零分母重复数和 cluster 数；有效比例低于 90% 时 CI unavailable。多层区间标注为 pointwise 探索性区间，不能据此宣称多层同时显著。另保留未标准化点估计/区间并命名区分。四层平均不代替逐层主表。

不沿用逐 Tail 用户遍历所有 Head 用户的复杂度；聚合与分箱后在 item/cluster 统计量上计算，避免真实 pilot 的 bootstrap 退化为用户级平方匹配。

### 7. 兼容、输出与版本

新增独立 `analysis_schema_version=tail_search_ranking_v1`、`estimator_version=static_risk_overlap_v1`；旧 summary、CSV 和 verdict 保持历史含义。以 summary metadata 明确标注旧 prefix matched/CI 为 legacy，不能把修正结果写回旧 run 或覆盖旧 Artifact。

新增文件：

- `search_ranking_by_user.csv`：user/item/group、fixed/widened rank、六状态、配对 identity 标识。
- `search_ranking_by_group.csv`：主效果、六状态计数、oracle 与全部分子分母。
- `prefix_attrition_by_layer.csv`：累计 survival、条件退出、首次失败描述及有效支持。
- `static_risk_overlap.csv`：箱边界、item 支持、权重、保留率与平衡。
- `static_risk_standardized_by_layer.csv`：估计量标识、逐层差距、原始差距、CI 与有效重复数。
- `frequency_attrition_descriptives.csv`：连续 frequency 分箱的描述性证据。

以上随现有 diagnosis evidence 一起写入/发布，并进入 manifest；缺失输入时不生成误导性的零表。summary 记录分析开关、估计量、数据单位、输入引用、split、不可用原因与质量状态。

### 8. 有限的下一步决策

四组仅运行一次预声明主配置及预声明分箱敏感性。发现确定的实现错误可修复重跑，但必须保留旧 run、原因与新版本；不能为了 Go 添加新搜索空间。

本变更只能形成审阅记录：

- `probe_candidate`：配对数据审计通过，存在非零且可量化的候选内 headroom 或逐层可达线索，并明确候选探针、主要反证和所需新增数据；不等于 H2 supported 或方法有效。
- `stop`：拟议探针所需的候选空间不存在，或现有结果不支持继续投入；记录停止的是哪条路线及依据，不能泛化为所有解码方法无效。
- `inconclusive`：身份未验证、共同支持/区间不合格或无法区分机制；不能用缺失值当 No-Go 的负证据，也不能自动扩实验。

最终跨设置报告同时呈现三个问题：候选是否可达、原排序是否可用、静态风险比较是否可信。任何重启干预必须单独确定实用收益阈值、overall 可接受下降和等计算预算，然后另提案；本提案不虚构这些尚未由用户确认的研究偏好。

## Risks / Trade-offs

- [风险分箱残余差异] → 公开平衡与支持，不声称消除 frequency，不将描述性比较升级为因果。
- [用户加权与 item-macro 结论不同] → 各自标记单位、分母，不交叉套用 CI。
- [已看过四组 evaluation] → 全部标注探索性；不能将其中三组重新命名为独立确认集。
- [新旧统计并存可能误用] → 独立 estimator version、legacy 标注和新报告；不静默改变旧字段。
- [前置提案未归档] → 实施前核查依赖；规格增量只添加本变更独有 requirement，按先前置、后本变更顺序合并。
- [研究仓库状态滞后] → 最终报告提供交接清单：H2 原 pilot No-Go、新分析结果、H3 暂停、18-setting 暂缓；不在 GRID 工作流中跨项目修改文档。

## Migration Plan

1. 保留前置提案及所有原始 run，新增默认关闭的输入/分析开关。
2. 完成内存契约测试、轻量配置和脚本验证后，复用现有 Artifact 运行四组 diagnosis。
3. 保存新的 run/config/manifest、六表与审阅报告；不得覆盖前置结果。
4. 关闭新开关即可回到旧分析输出；新输入默认 null，不影响旧调用。

## Open Questions

- 是否值得运行任何小规模干预，由本变更产出的空间与限制决定。
- 实用收益、overall 容忍度及真实计算预算在未来干预提案中确定；不阻塞当前分析实现。
