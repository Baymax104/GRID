## Context

TIGER 当前在 `TigerDecoder.generate()` 内逐层执行 constrained beam search：先按 catalog semantic IDs 屏蔽非法 prefix，再把单步概率与上一层路径概率相乘，最后对 `beam_width × codebook_size` 候选做全局 top-k。循环内部已经存在生成逐层机制证据所需的 logits、prefix、parent beam 与 cutoff，但当前只返回最终 semantic IDs 和最终路径概率。

Teacher-forcing 路径也已返回逐层 global logits，但 evaluation 只聚合 loss，predict 数据配置还会生成 `target_ids` 后在 collate 处丢弃。现有 recommendation writer 和 diagnosis 只接受最终 keyed prediction bundle，无法表达逐层轨迹。

本变更为 DASFAA 方法决策提供前置证据。四个 seed-42 checkpoint 已通过 W&B lineage 确认可复用；本变更必须保持统一 `src.main` 入口、Lightning predict/test 生命周期、现有 recommendation bundle、合法 SID 约束、logger-owned W&B run 和 Artifact lineage callback 的职责边界。

## Goals / Non-Goals

**Goals:**

- 在同一 labeled inference batch 上采集 teacher-forcing、fixed-beam 和 widened-beam 可比较的 target-centric trace。
- 明确定义 target probability/rank、prefix survival、beam rank、cutoff margin 与 first failure depth。
- 以独立、keyed、版本化的 Prefix Trace Artifact 保存轨迹，同时保持标准 recommendation output 文件完全不变。
- 让 trace 的数据 split 显式可审计，方法统计与调参只能使用 evaluation split。
- 让 Tail-SID diagnosis 可选消费 fixed/widened trace，形成 layer-wise risk-matched survival 与 recovery evidence。
- instrumentation 关闭时保持现有 TIGER 输出逐元素一致，并通过直接 decoder 测试证明。

**Non-Goals:**

- 不实现 Budget-Aware Prefix Calibration、frequency-aware baseline 或任何 beam score 修改。
- 不修改 TIGER loss、checkpoint、encoder、Semantic ID 构造或训练流程。
- 不保存每个用户每层的完整 `K × codebook_size` 候选张量。
- 不把 Prefix Trace 字段加入现有 `merged_predictions_tensor.pt` recommendation bundle。
- 不使用 testing split 生成方法先验、选择超参数或决定 calibration 公式。
- 不在本变更内启动 18-setting 训练矩阵。

## Decisions

### 1. 采用 target-centric compact trace，而不是完整 beam dump

每层只保留论文机制判断和后续校准设计需要的目标路径统计：target token probability、合法候选内 rank、目标 prefix 是否存活、存活时的 beam rank、目标 parent beam rank、目标路径分数、beam cutoff 分数、cutoff margin、合法候选数和 first failure depth。最终 generated SIDs 继续由标准 recommendation output 保存。

完整 beam dump 便于任意离线重算，但在 widened beam 与大用户集上会显著增加显存、主机内存和 Artifact 体积；目标中心 trace 足以回答 H2，并可通过 catalog prefix 表补充 composition/competition 属性。

### 2. Trace 只观察 beam，不参与 beam 决策

`TigerDecoder.generate()` 增加可选 target IDs 与 trace 开关。beam legality mask、softmax、累计路径分数和 top-k 仍沿用原顺序；trace 从中间张量派生，禁止回写 candidate scores 或 masks。instrumentation disabled 时走原有返回与 prediction contract；enabled 时附带 trace。

目标合法 rank 定义为：在真实 target parent prefix 仍可达时，对该 parent 的合法下一 token 候选按模型分数降序排名；严格高于 target score 的候选数加一。prefix survival 定义为剪枝完成后 ground-truth prefix 是否出现在 beam 中。first failure depth 使用 1-based hierarchy depth，完整存活使用 `-1`。

### 3. Teacher forcing 与 free-running 共享同一 target 和层级定义

TIGER inference collate 在 trace 模式下保留 `TigerLabelData.target_ids`。`Tiger.forward()` 产生 teacher-forcing logits；active hierarchy slice 和真实 prefix 的合法 continuation mask 用于计算 target probability、合法 rank、top-1 margin。free-running trace 使用同一 model-side SID（包括 dedup hierarchy）和同一 user key，避免两条路径离线猜测对齐。

### 4. 通过 ModelOutput auxiliary payload 保持 recommendation bundle 兼容

运行时 `ModelOutput` 增加默认空的 named auxiliary payload。标准 Local/W&B prediction writers 仍只合并和持久化 `keys` 与 `predictions`，不得把 auxiliary 写入 `merged_predictions_tensor.pt`。trace-enabled inference 将 Prefix Trace tensor payload 放入约定名称，由独立的通用 auxiliary tensor writer 分片、合并并写入 `prefix_trace.pt`。

这一方案避免改变现有 recommendation consumer，同时允许同一 predict pass 同时发布 recommendation output 和 trace。writer 只理解 named tensors、schema metadata 与 key alignment，不导入 TIGER 或 diagnosis domain。

### 5. Prefix Trace Artifact 使用稳定、版本化 tensor schema

`prefix_trace.pt` 顶层包含 `schema_version`、`keys`、`labels`、`trace` 和 `metadata`。`trace` 中每个逐层 tensor 第一维必须与 keys 对齐，第二维必须等于 model SID hierarchy 数；缺失 beam rank 使用 `-1`，布尔 survival 使用 bool tensor。metadata 至少记录 data split、beam width、num hierarchies、codebook size、checkpoint/semantic-ID 输入引用和 trace mode。

W&B Artifact 使用独立 role/type `prefix_trace`，由 logger-owned run 发布；本地写入在无 W&B 时仍可用。上游 checkpoint 与 semantic ID 的 `use_artifact` 仍由 lineage callback 记录。

### 6. Trace experiment 显式选择 split

新增薄 experiment/data/callback 配置或等价显式 trace mode，不改变标准 `tiger_inference` 的默认行为。trace experiment 的 `data_split` 是 mandatory enum，仅允许 `evaluation` 或 `testing`，并用于构造 dataloader 路径和 Artifact metadata。用于 prefix statistics、方法选择和参数调优的运行必须声明 `evaluation`；`testing` 只用于方法冻结后的确认性分析。

### 7. Widened beam 复用 checkpoint，仅改变 runtime beam width

beam width 继续来自 `model.root.top_k_for_generation`，并记录在 config 与 Artifact metadata。pilot 固定比较 K=10 与 K=50；二者使用相同 checkpoint、Semantic ID、data split、target keys 和 trace schema。diagnosis 按 keys 精确连接两个 Artifact，并拒绝 split、label、checkpoint 或 Semantic ID lineage 不一致的比较。

### 8. Diagnosis 扩展为可选机制证据，不改变静态诊断

Tail-SID diagnosis 增加可选 fixed trace 与 widened trace 输入。只有 fixed trace 时输出 layer/group survival、target probability/rank 和 first-failure 证据；同时提供 widened trace 时再输出 recovery。现有静态 damage、recommendation outcome、sensitivity 和 verdict 保持原义，新的 H2 verdict 独立命名，不能改写 `raw_damage` 或 `generation_risk_validity`。

pilot 的 Go / No-Go 至少检查：risk-matched Tail survival deficit 是否跨四组方向稳定、widened beam 是否恢复或推迟 Tail path failure、prefix competition 是否在控制 frequency/static risk 后仍有增量关联。该 verdict 只决定是否创建后续主方法提案。

## Risks / Trade-offs

- [Trace 计算增加 inference 成本] → 只保存 target-centric tensor，并将 trace 作为显式关闭的诊断模式；分别记录 wall time 与 Artifact 大小。
- [开启 trace 意外改变 baseline 排序] → trace 逻辑只读中间值，增加 disabled/enabled 逐元素 parity test 和固定 tiny-decoder golden test。
- [目标 prefix 已退出后无法计算真实 beam rank] → survival/beam rank 记为 false/-1，但 teacher-forcing rank 和 target score 继续按 ground-truth prefix 单独计算，区分模型概率与搜索剪枝。
- [evaluation/testing 泄漏] → data split 必填并进入 Artifact metadata；diagnosis 拒绝把 testing trace 标记为 calibration statistics source。
- [K=10/K=50 Artifact 错配] → 加载阶段按 user keys、labels、split、checkpoint 与 semantic-ID reference 严格校验。
- [现有 checkpoint config 记录了解析后的本地路径] →以 W&B `used_artifacts()` lineage 和 source run 作为可复用引用，不依赖旧 config 中的机器本地路径。
- [逐层样本共享 prefix，普通 CI 偏窄] →代表性设置使用 prefix-cluster bootstrap；单 item/user 行不被当作独立 prefix 样本。

## Migration Plan

1. 先增加 runtime dataclass、trace calculator 与 decoder parity tests，不接入 writer。
2. 接入 trace-enabled prediction 和独立本地 writer，验证 recommendation bundle byte-level schema不变。
3. 增加 W&B Prefix Trace Artifact writer、resolver 与 lineage tests。
4. 增加 split-safe trace experiment/config/launcher 参数和 Hydra compose、dry-run 验证。
5. 扩展 diagnosis input/evidence/writer，保持 trace 输入可选。
6. 在实验主机复用四个 checkpoint，分别运行 evaluation split 的 K=10/K=50 pilot；记录 Go / No-Go。

回滚时关闭 trace callback/config 即恢复原 TIGER inference；新增字段均为可选，旧 recommendation 和 diagnosis 调用不需要迁移。

## Open Questions

- 在真实数据 pilot 前，不冻结后续 calibration 的具体 score 公式、layer range 或 support threshold。
- 如果 K=50 的资源开销超过可接受范围，可在不改变 schema 的情况下增加 K=20 作为中间敏感性点，但 K=10/K=50 仍是预注册主对照。
