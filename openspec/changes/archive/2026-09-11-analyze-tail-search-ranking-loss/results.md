# Tail 搜索—排序损失重分析结果

## 执行与完整性

四个设置均复用既有 evaluation、seed 42、K10/K50 recommendation 与 Prefix Trace Artifact，没有重跑 inference。分析先以 W&B offline logger 生成本地可审阅结果，获得明确发布授权后再经同一根脚本串行在线执行；正式 run 启用了 W&B logger、Artifact lineage 和完整 evidence Artifact 发布。

| 设置 | 本地 offline run | 正式 W&B run | evidence Artifact | 完整 manifest | 输入身份 | 六状态守恒 | 旧/新 fixed Hit@10 |
|---|---|---|---|---:|---:|---:|---:|
| Beauty RKMeans | `a4nvnol9` | `blo793ek` | `tail-sid-diagnosis-evidence:v17` | 是 | 通过 | 通过 | 完全相等 |
| Beauty RVQ | `ccpa5n0k` | `y8cccrc3` | `tail-sid-diagnosis-evidence:v18` | 是 | 通过 | 通过 | 完全相等 |
| Sports RKMeans | `dxzvz6k7` | `ivnfmyzn` | `tail-sid-diagnosis-evidence:v19` | 是 | 通过 | 通过 | 完全相等 |
| Sports RVQ | `mbtn9cxa` | `7703l4z5` | `tail-sid-diagnosis-evidence:v20` | 是 | 通过 | 通过 | 完全相等 |

Beauty RKMeans 在 evidence 原子写入并产生完整 manifest 后，因 Windows GBK 无法编码 Rich progress bar 的项目符号而在 teardown 返回非零；其余三组关闭 progress bar 后均以 0 结束。该异常不影响首组 evidence 内容，但远端执行时应继续关闭 progress bar。

在线并行尝试 Beauty RVQ 时，W&B service 在 lineage 写入阶段 busy 超时，产生不完整 run `synl8yn2`；该 run 没有 summary 和 evidence Artifact，不能用于分析，已在正式结果核验后删除且未删除任何上游 Artifact。随后采用串行策略重跑为 `y8cccrc3`，六个 lineage、summary 和 evidence 均完整。四个正式 run 经 W&B API 确认均为 `finished`，每个 run 都记录六个 `used_artifacts`，并发布包含 summary、manifest、原有表和六张新增分析表的 diagnosis evidence Artifact。

## 搜索空间与排序分解

### 总体

| 设置 | fixed Hit@10 | widened Hit@10 | K50 可达率 | oracle headroom | Top10 新增 | Top10 丢失 | 净变化 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Beauty RKMeans | 6.631% | 6.971% | 17.292% | 10.321% | 212 | 136 | +76 |
| Beauty RVQ | 6.801% | 7.217% | 17.435% | 10.218% | 251 | 158 | +93 |
| Sports RKMeans | 3.225% | 3.629% | 9.405% | 5.776% | 375 | 231 | +144 |
| Sports RVQ | 3.326% | 3.871% | 9.972% | 6.101% | 437 | 243 | +194 |

总体提升主要由 Head 用户贡献。扩大 beam 会同时新增和丢失 Top10，说明 K50 输出不是 K10 的简单超集；因此必须使用六状态配对分解，不能把 widened Hit@10 差值直接解释为候选召回增益。

### Tail + Tail-Cold

| 设置 | 支持 | fixed Hit@10 | widened Hit@10 | K50 可达率 | oracle headroom | fixed miss 在 K50 恢复 | 新增 Top10 | 净变化 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Beauty RKMeans | 3,137 | 0.924% | 0.924% | 2.040% | 1.116% | 35 | 0 | 0 |
| Beauty RVQ | 3,137 | 1.211% | 1.179% | 2.327% | 1.148% | 35 | 0 | -1 |
| Sports RKMeans | 4,778 | 0.021% | 0.000% | 0.105% | 0.105% | 4 | 0 | -1 |
| Sports RVQ | 4,778 | 0.042% | 0.042% | 0.335% | 0.293% | 14 | 0 | 0 |

四组中 Tail fixed miss 虽有 4–35 个进入 K50，但全部落在 rank 11–50，没有一个成为新增 Top10。更关键的是，即使扩大到 K50，97.7%–99.9% 的 Tail 标签仍完全不可达。由此可区分两个瓶颈：

1. 候选空间不足是主导问题：Tail K50 可达率只有 0.105%–2.327%，远低于对应 Head 的 17.475%–30.469%。
2. 条件排序也有损失：已经恢复到 K50 的 Tail 标签全部停留在 Top10 之外，但其绝对支持仅 4–35，不能单独支撑通用 reranking 方法。
3. 单纯增大 beam 不值得继续：四组 Tail Top10 新增均为 0，并出现两组净损失。

## 静态风险标准化

四组主分析均为 `qualified`，Head/Tail item retention 均为 1.0，raw-damage SMD 分别为 -0.0004、-0.0155、-0.0205、-0.0071，满足预声明的 `|SMD| <= 0.1`。默认请求 5 箱；重复分位边界合并后四组实际均为 4 箱。

| 设置 | layer 1 标准化 Tail−Head survival | 95% CI | 最终层标准化 Tail−Head survival | 95% CI |
|---|---:|---:|---:|---:|
| Beauty RKMeans | -3.899 pp | [-6.323, -1.446] | -6.813 pp | [-7.697, -5.930] |
| Beauty RVQ | -4.804 pp | [-7.342, -2.288] | -6.451 pp | [-7.324, -5.655] |
| Sports RKMeans | -2.030 pp | [-4.117, 0.113] | -2.464 pp | [-2.902, -2.058] |
| Sports RVQ | -2.539 pp | [-4.621, -0.394] | -2.498 pp | [-2.939, -2.097] |

最终层四组区间都完全小于 0；Sports RKMeans 仅第一层区间跨 0，第二层起缺口稳定。条件退出率差值方向与 survival 缺口一致，说明差距沿层累积，而不是只由最后一步偶然产生。

### 分箱敏感性

3/5/10 个请求箱数分别因重复边界得到 2–3、4、7 个有效箱。所有设置、所有层和全部敏感性配置均保持 Tail−Head survival 为负，质量状态均为 `qualified`。最终层范围如下：

| 设置 | 3 箱请求 | 5 箱主分析 | 10 箱请求 |
|---|---:|---:|---:|
| Beauty RKMeans | -6.949 pp | -6.813 pp | -6.822 pp |
| Beauty RVQ | -6.587 pp | -6.451 pp | -6.424 pp |
| Sports RKMeans | -2.491 pp | -2.464 pp | -2.459 pp |
| Sports RVQ | -2.495 pp | -2.498 pp | -2.482 pp |

方向和量级对分箱选择稳定。旧 matched estimate 与旧 cluster CI 估计的是不同对象，仍保留为 legacy 描述，不能与新的 item-macro 静态风险标准化区间混用。

## 决策记录

| 议题 | 决策 | 依据 |
|---|---|---|
| 候选空间机制 | `probe_candidate` | 四组 K50 Tail 可达率都极低，且静态风险标准化后的逐层 survival 缺口跨数据集和量化器稳定。|
| 仅扩大 beam | `stop` | Tail 在四组均无新增 Top10，两组还有净损失；K50 没有解决候选分配。|
| 通用 reranking 方法 | `inconclusive` | K50 内确有低排名 Tail 标签，但每组只有 4–35 个 fixed-miss 恢复样本，绝对支持不足。|
| 风险比较可信度 | `probe_candidate` | 四组质量门禁通过、最终层 CI 排除 0、3/5/10 箱方向一致，足以支持一个有边界的候选分配 probe。|

跨设置结论为 **`probe_candidate`**：下一步只应开展一个面向 candidate allocation / prefix survival 的小规模方法探针，并保留明确退出条件；当前证据不支持直接进入全矩阵，也不支持把问题改写为纯 reranking。

## 研究状态交接

- 保留既有 H2 pilot 的 No-Go，不回写或覆盖旧 run。
- 将本次结果作为独立的新分析证据，明确区分候选不可达、候选内低排名和 Top10 净变化。
- 暂停未经当前证据支持的 H3 扩展；若 H3 指向通用 reranking，则维持 `inconclusive`。
- 暂缓 Beauty/Sports/Toys × RKMeans/RVQ × seeds 全矩阵，先定义并审阅 candidate allocation probe 的最小成功/失败标准。
- 本提案只记录分析与决策，不自动创建或实现方法提案。

## 本地 evidence 位置

- `logs/tail_search_ranking/runs/2026-09-11/beauty_rkmeans/diagnosis_evidence`
- `logs/tail_search_ranking/runs/2026-09-11/beauty_rvq/diagnosis_evidence`
- `logs/tail_search_ranking/runs/2026-09-11/sports_rkmeans/diagnosis_evidence`
- `logs/tail_search_ranking/runs/2026-09-11/sports_rvq/diagnosis_evidence`

这些目录均含完整 `manifest.json`、原有 evidence 文件以及六张新增 CSV。对应远端 W&B run 和 evidence Artifact 已发布并通过 API 核验。
