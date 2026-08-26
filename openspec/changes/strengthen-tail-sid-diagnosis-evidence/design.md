## Context

当前 official Tail-SID diagnosis 通过统一 `src.main`、`DiagnosisDataModule` 与 `Trainer.test` 读取一个完整 item batch，并由独立的 structural、semantic、damage、prefix-risk metrics 向 W&B summary 写入标量。Beauty 上的 RK-Means 与 R-VQ diagnosis 已验证输入解析、keyed bundle lookup 和 W&B lineage，但现有结果只暴露 Tail 结构均值；唯一 Head/Mid/Tail 对比是 composite damage，而 Tail priority gate 会改变该对比。官方 experiment 也曾为简化 metric runtime 暂时移除 CSV/JSON/report writer，因此没有 item/prefix 风险 Artifact，也无法与 TIGER recommendation output 做 keyed outcome correlation。

本 change 横跨 diagnosis metric、data input、统计分析和 structured writer，但必须保持项目边界：分析继续走 `Trainer.test`；W&B URI 解析归 data Artifact resolver；输出写入归 `src/common/writers/`；lineage 继续由现有 callback 负责；不得引入另一个离线 runner。

## Goals / Non-Goals

**Goals:**

- 建立不依赖 Tail gate 的 group-comparable raw evidence。
- 显式测量频次非对称 overlap，而不是用 group multiplier 代替证据。
- 建立确定性、可审计的 semantic compatibility 主口径与敏感性口径。
- 可选接入 keyed TIGER recommendation output，按 user/item key 建立 risk-outcome 关联。
- 生成可供论文审计和后续 repair/reranking 消费的稳定本地/W&B Artifact。
- 用 effect size、bootstrap、frequency-matched comparison 和 go/no-go verdict 固化证据标准。
- 保持无 embedding、无 recommendation output、本地路径与短/完整 W&B URI 兼容。

**Non-Goals:**

- 不修改任何 quantizer 或 TIGER 训练/解码算法。
- 不在本 change 中实现 SID repair、reassignment、regularization 或 candidate reranking。
- 不自动调度多 seed、多 tokenizer 或多数据集训练。
- 不引入 pandas、SciPy、PyArrow、Matplotlib 等依赖。
- 不把启发式 composite score 或单次显著性结果当作因果证明。

## Decisions

### 1. 用统一 `DiagnosisEvidence` 结果对象作为 metric 与 writer 的共同事实源

诊断计算拆为纯函数/无副作用组件，接收完整 `DiagnosisBatch`、analysis config 与可选 recommendation batch，返回一个结构化 `DiagnosisEvidence`：包含 summary、group rows、item rows、prefix rows、bounded pair rows、normalization metadata、sensitivity rows 和 verdict。`TailSIDDiagnosisModule.test_step` 在唯一 test batch 上只计算一次该对象，并同时返回 metric callback 所需的标量视图和 structured writer 所需的 evidence。

选择该方式是为了避免 structural/semantic/damage/prefix metric 各自重复 O(N) bucket 与 cosine 计算，并防止 W&B summary 与 Artifact 表来自不同计算路径。未采用“writer 从 summary 反推表格”，因为 summary 已不可逆地丢失 item/prefix 信息；也不恢复独立 CLI/report orchestration。

### 2. `raw_damage` 与 `priority_score` 使用独立字段和独立用途

`raw_damage` 由无 group gate 的 bounded normalized components 构成；component 原始值始终保留。frequency gate 仅生成 `priority_score`，供未来处理排序和 top-risk 展示。所有 H1 effect size、confidence interval、risk-outcome correlation 和 evidence verdict 只允许使用原始 component 或 `raw_damage`。

保留 gate 而不直接删除，是因为它仍可作为后续方法的显式策略参数；改名和用途隔离可以消除循环论证，同时保持现有探索结果的可追溯性。

### 3. 主要证据优先使用原始 component，composite score 只作汇总

每个 group 使用相同 schema 报告原始 collision、near-collision、density、suffix、semantic 与 asymmetric components。`raw_damage` 继续采用稳定 IQR normalization 与 bounded contribution，但 Artifact 必须记录每个 component 的 median、IQR、degenerate flag、clamp 和权重。跨 tokenizer 不能只比较 composite 绝对值；verdict 基于配置声明的 primary raw components 与方向一致性。

未采用单个 learned score，因为当前没有独立标签训练权重，也会掩盖具体可干预机制。

### 4. Frequency asymmetry 从 typed neighbor/bucket composition 直接计算

prefix bucket 构建继续避免全量 O(N²)。每个 item 在 strict prefix 与 full SID bucket 内按 partner group 聚合 neighbor count，并由 bucket composition 派生 head dominance、Tail isolation deficit 和 Tail-to-Head pressure。full collision 与 strict near-collision 始终分开。

该设计直接测量研究主张，不使用 group membership multiplier 代理 asymmetry。pair-level 表只保留配置允许的 top rows，但 item/group 聚合必须在截断前计算，保证统计不受输出上限影响。

### 5. Semantic compatibility 采用一个全局主阈值和一个 bucket-relative 敏感性阈值

主 `semantic_mismatch_global` 使用全局随机 item pair cosine 分布的配置 quantile（默认 0.75）。随机 pair 由 seed 和 sample count 确定，阈值及样本 metadata 写入 summary/Artifact。现有 strict bucket 内 0.25 quantile 逻辑保留为 `bucket_relative_semantic_outlier`，仅作为 sensitivity field。

选择双口径是因为二者回答不同问题：全局阈值判断 prefix sharing 是否缺乏总体语义支持；bucket-relative 阈值发现同 bucket 内相对异常 pair。未采用只保留旧口径，因为它会按定义在 bucket 内产生低分位 outlier，不能独立证明全局不兼容。

### 6. Recommendation correlation 使用 optional keyed input，在 DataModule setup 前置解析

新增可选 `recommendation_output_path`。`DiagnosisDataModule.setup_stage` 与 semantic ID/embedding 相同，通过共享 resolver 使用 `field_name="recommendation_output_path"` 和 experiment `user/project` 解析，使 W&B input 在 lineage callback setup 前注册。Dataset/analysis input 继续接收 resolved local path。

testing records按 user key提取 label item；recommendation bundle 使用其 keys 与 generated SID predictions。join 必须基于 key，重复、缺失或未知 key 按明确策略统计并在不满足完整性阈值时 fail-fast，不能按 row order 对齐。第一版只依赖 generated SIDs 计算 hit@K、rank 和 NDCG contribution；marginal probability 不是本 change 的前置条件。

### 7. Frequency control 使用可复现分桶比较，bootstrap 使用现有数值依赖

第一版使用配置化 `log1p(freq_train)` frequency bins，在每个有足够支持的 bin 内比较 low/high raw-damage item outcomes，再汇总支持数与差值。Tail-vs-Head raw component 使用 seeded bootstrap 输出 absolute delta、ratio（分母有效时）和 confidence interval。实现使用 Python/PyTorch/NumPy，不新增 SciPy/pandas。

未选择第一版直接引入多变量回归，是因为 item label support 稀疏、统计假设和依赖成本更高；Artifact 保留 item-level数据，后续可离线做更完整模型。

### 8. Sensitivity 是同一 evidence schema 下的标记结果，不产生隐式主结论

primary config 保持单一明确值（默认 head/tail ratio 0.2、global semantic quantile 0.75、声明的 damage weights）。可配置 sensitivity grid 重新计算分组、damage 或 semantic threshold，并将每行完整 setting 写入 `sensitivity_metrics.csv` 或 summary 嵌套 JSON。verdict 只基于 primary setting；sensitivity 只报告方向稳定性。

这样避免用户在多组结果中事后选择最有利阈值。未采用每个 setting 单独创建 W&B run，因为会造成 lineage/结果碎片化。

### 9. 新增通用 structured analysis writer，而不是复用 prediction bundle writer

在 `src/common/writers/` 增加面向 `Trainer.test` 的 structured analysis writer。它接收 `DiagnosisEvidence` 的通用 named-table/JSON 表示，在临时 staging 目录序列化 JSON/CSV，成功后原子移动到 `${paths.output_dir}/diagnosis_evidence`。若配置 W&B publication，它通过 logger-owned run 创建一个包含整个 evidence 目录的 Artifact；不得调用 `wandb.init` 或 `wandb.finish`。

不复用 `WandbArtifactWriter`，因为后者专门合并 `ModelOutput(keys,predictions)` 并产生单文件 model output bundle；diagnosis 是多文件 analysis schema。domain 层负责形成字段语义，common writer 只负责稳定序列化、原子落盘和可选发布。

### 10. Verdict 使用可审计的多维状态而不是单一布尔值

`summary.json` 输出独立状态：`tail_structural_asymmetry`、`equal_risk_tail_vulnerability`、`generation_risk_validity`、`cross_setting_stability`。每项为 `supported`、`not_supported` 或 `unavailable`，并记录触发的 effect、CI、support threshold 和 primary setting。总 verdict 只汇总这些状态，不自动授权进入方法实现。

选择多维 verdict 可以区分“Tail 结构更差”和“同等结构风险下 Tail 更脆弱”两条可能研究叙事，避免一条失败就错误否定全部方向。

### 11. 模块职责与依赖方向是实现硬约束

本 change 保持以下单向依赖与所有权：

```text
configs / root script
        │
        ▼
src.main → launcher（仅通用实例化）
        │
        ├── data：URI 解析、bundle/testing 读取、keyed assembly
        │
        ├── quantization/tail_sid_diagnosis：领域证据计算与 verdict
        │
        ├── common/metrics：通用 scalar 生命周期
        │
        ├── common/writers：通用 structured payload 序列化/发布
        │
        └── common/callbacks：通用 resolved-input lineage
```

具体约束：

- `src/data/` 可以产出 key-aligned 的原始输入/label/outcome assembly，但不得 import diagnosis metric、damage 或 verdict 逻辑。
- `src/quantization/tail_sid_diagnosis/` 可以依赖 data contracts 和 common protocols，但不得解析 `wandb://`、下载 Artifact、调用 `use_artifact` 或管理 logger run。
- common structured writer 只接收通用 `StructuredAnalysisOutput`（named JSON documents、named tabular rows、metadata），不得 import `DiagnosisEvidence` 或包含 Tail/collision/damage 字段判断。diagnosis 域负责显式转换。
- `WandbArtifactLineageCallback` 继续只消费 resolved-reference registry；writer 与 diagnosis 域均不得调用 `use_artifact`。
- `MetricCallback` 继续只处理通用 scalar metric logging；不得为 diagnosis 表格或 Artifact 增加条件分支。
- `src/utils/launcher.py` 不识别 diagnosis 路径、payload 或 writer 类型，只按 Hydra 配置执行既有通用实例化。
- input resolver 与 W&B download 仍分别归 `src/data/components/artifacts.py` 和 `src/utils/wandb.py`；output writer 不复用这些输入职责。
- 输出目录只从 `paths.output_dir` 配置传播；root script 只解析参数并透传统一入口。

这些约束通过 import-boundary、callback lifecycle、logger-owned run、Hydra composition 和 local-only writer 测试锁定。若某项功能只能通过反向依赖或 launcher special case 完成，应先修订设计而不是实现例外。

## Risks / Trade-offs

- [Risk] 完整 item/prefix/pair 输出增大内存和文件体积。→ Mitigation：Beauty 规模下保留 item/prefix 全量；pair 统计先聚合后按 per-item/global deterministic limit 截断，CSV 流式写入。
- [Risk] `test_step` 返回大型 evidence 可能被 Lightning 额外保留。→ Mitigation：单 batch、单 CPU device；writer 在 batch-end 接收后释放 module/output 引用，并增加内存聚焦测试。
- [Risk] 全局随机 pair 阈值受采样影响。→ Mitigation：固定 seed、记录 pair count/quantile/threshold，并提供 sample-count stability 检查。
- [Risk] frequency grouping 与测试 label support 稀疏导致伪差异。→ Mitigation：输出 group/bin support、最低支持阈值和 bootstrap CI；不足时 verdict 为 `unavailable`。
- [Risk] raw damage 的启发式 normalization 仍会跨 run 漂移。→ Mitigation：主证据按 raw component 报告；Artifact记录 normalization metadata；禁止仅凭 composite 绝对值做跨 tokenizer verdict。
- [Risk] 可选 recommendation bundle 与 testing records key 不完整。→ Mitigation：记录 missing/duplicate/unknown key counts，并对超过配置容忍度的情况 fail-fast。
- [Risk] 恢复 writer 重新引入 diagnosis lifecycle seam。→ Mitigation：使用 common structured writer 和明确的 `AnalysisEvidence` 协议，不扩展 metric callback，也不让 writer重算 domain metrics。
- [Risk] 本 change 规模较大。→ Mitigation：tasks 按 raw evidence、Artifact、recommendation correlation、statistics/verdict 四个可独立验证里程碑拆分，先在现有 RK/R-VQ Beauty outputs 上验证。
- [Risk] structured output 会诱导 common writer 依赖 diagnosis 领域类型。→ Mitigation：在 common 层定义最小通用 payload protocol，由 diagnosis 域做显式 adapter，并用 import-boundary 测试禁止反向依赖。

## Migration Plan

1. 引入 `DiagnosisEvidence` 与无 gate group-wise raw calculations，同时保持现有 W&B summary key 可读。
2. 增加 frequency-asymmetric 与双 semantic 口径，使用 toy buckets/embeddings 锁定公式和 deterministic sampling。
3. 增加 common structured analysis writer，在无 W&B logger 下先验证完整本地输出与 atomic failure。
4. 接入 logger-owned W&B Artifact publication，并验证不创建/关闭 run。
5. 增加可选 recommendation input 的 resolver、lineage、keyed testing join 与 item outcomes。
6. 增加 bootstrap、frequency-matched sensitivity 和 verdict。
7. 复用现有 Beauty RK-Means/R-VQ Artifact 做两次全量 diagnosis，审计 summary 与 CSV 一致性，再决定后续方法 change。

Rollback 时可移除新 writer、recommendation input 和 evidence config，恢复 summary-only metrics；上游 semantic ID、embedding、recommendation Artifact 及既有 W&B runs 无需迁移。新 Artifact 使用独立 schema version，旧 run 不被回写。

## Open Questions

无阻塞问题。默认主口径为 Tail ratio 0.2、global random-pair cosine 0.75 quantile、seed 42；实现中所有值保持配置化并写入 metadata。
