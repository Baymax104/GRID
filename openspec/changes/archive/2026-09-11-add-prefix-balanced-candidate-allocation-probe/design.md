## Context

当前 constrained beam search 在每层合法性过滤后，按累计路径概率直接保留全局 top-k。四组重分析显示 Tail 标签大多在进入 K50 前已经不可达，且单纯扩大 beam 没有新增 Tail Top10；因此 probe 必须改变候选槽位分配，同时保留原模型分数、checkpoint 和合法 prefix 约束。

Tail/Head 身份来自训练频次。任何使用 evaluation/testing 标签决定候选的实现都会造成泄漏。现有 semantic ID bundle 保存 item key 与 SID，training TFRecord 保存历史序列，因此可以在模型构造时由明确的数据输入生成与 semantic ID 行对齐的频次 tensor，无需新离线入口或新依赖。

## Goals / Non-Goals

**Goals:**

- 以训练 split 的 item frequency 和 catalog SID 计算逐层 prefix training mass。
- 只在原始路径分数 shortlist 内保留少量低 mass prefix，限制探索范围和 Head/overall 代价。
- 关闭 probe 时逐元素复现当前生成结果；开启时不修改模型参数、logits 或返回的原始路径分数。
- 产生可配对审计的 trace、recommendation 和 diagnosis evidence。
- 提供人工运行入口以及预声明的 probe 成功/停止门槛。

**Non-Goals:**

- 不训练新 checkpoint，不增加 loss，不修改 SID 或 quantizer。
- 不按 evaluation/testing target、用户 Head/Tail 标签或未来交互分配 beam。
- 不自动启动完整数据集 inference、diagnosis 或 W&B 发布。
- 不在本 change 中启动三数据集、多 seed 完整矩阵。

## Decisions

### 1. 训练频次在 data 层按 semantic ID keys 对齐

新增 data helper，从 `${data_dir}/training` 的 TFRecord 序列统计 item frequency，再按 keyed semantic ID bundle 的 keys 生成一维非负 tensor。未知训练 item 不进入 catalog；catalog 中未出现的 item 频次为 0。helper 必须拒绝空 training split、重复 semantic ID keys 和无效 item id。

选择运行时构造而非另建统计 producer，原因是现有 diagnosis 已使用同一 training 数据定义，且 probe 只有四个受控运行。Hydra/W&B config 记录 data path、semantic ID reference 和 `source_split=training`，prefix trace metadata 记录频次摘要，保证可审计性。若后续扩大矩阵，再将该输入物化为独立 Artifact。

### 2. decoder 使用 sparse encoded-prefix lookup

`TigerDecoder` 接收与 `semantic_ids` 行对齐的 item frequency tensor。每个层级把 SID prefix 以 base-`codebook_size` 编码为 int64 key，按 key 聚合训练频次得到 prefix mass，并保存排序后的 sparse key/value lookup。候选查询使用 `torch.searchsorted`，避免创建 `codebook_size ** depth` 的 dense table。

首层和后续层都在 legal-prefix mask 生效后查询 prefix mass。合法候选必须存在 lookup；不一致时直接失败，避免将缺失统计默认为高优先级。

### 3. allocation 在模型分数 shortlist 内保留固定槽位

对每个用户、每层先按现有累计路径概率取 `min(K * pool_multiplier, legal_candidate_count)` 个候选作为 shortlist。固定宽度 tensor 中用于补齐 shortlist capacity 的 catalog 外位置必须保持不可选，不能进入 reserve 或 score backfill。随后：

1. 按较低 prefix mass、再按较高原始路径概率和稳定 flat index 选择最多 `reserved_slots` 个 reserve 候选；
2. 按原始路径概率补满未被 reserve 占用的槽位；
3. 将最终 union 按原始路径概率重新排序，并返回原始累计路径概率。

约束为 `0 < reserved_slots < K`、`pool_multiplier >= 1`。`enabled=false` 或 `reserved_slots=0` 走现有 top-k 分支，不构造或使用 allocation prior。稳定 tie-break 保证相同输入可复现。

选择 shortlist + quota，而不是给 logits 加 bonus，是为了让 intervention 强度具有直接的槽位含义，并保持返回 score 可与 baseline 比较。选择低 mass prefix，而不是直接选择 Tail item，是为了在不知道目标标签的生成时刻使用训练期可得信息。

### 4. trace 记录 intervention 身份和 target-centric allocation 观测

trace schema 增加：target prefix training mass、target 是否进入 allocation shortlist、target 是否由 reserve 槽位保留、每层实际 reserve 数。metadata 记录策略名、`reserved_slots`、`pool_multiplier`、source split、训练频次摘要和原始输入引用。

baseline 与 intervention 必须使用相同 checkpoint、semantic ID、data split、keys、labels、beam width、seed 和 SID shape。差异只允许出现在 allocation 策略字段。

### 5. diagnosis 增加同宽 baseline/intervention 配对模式

Tail-SID diagnosis 新增 `candidate_allocation_probe` 比较模式，消费 baseline/intervention recommendation 和 prefix trace。它复用现有 key、label、lineage、rank/membership 守恒审计，并额外要求相同 beam width、baseline allocation 关闭、intervention allocation 开启。

evidence 至少包含按 user 和 frequency group 的：baseline/intervention candidate reach、Top10 命中、新增/丢失、净变化、逐层 prefix survival、shortlist/reserve 命中及 overall/Head/Tail 差值。summary 按预声明门槛给出 `advance`、`stop` 或 `inconclusive`，不把中间 survival 改善自动解释为推荐改善。

### 6. 完整实验由用户手动启动

根脚本只负责参数校验和统一 `src.main` Hydra 入口。仓库验证运行单元测试、Hydra compose、脚本 dry-run 和 shell syntax；execution plan 给出四设置 baseline/intervention/diagnosis 的人工命令模板，不由代理自动执行。

## Risks / Trade-offs

- [低 mass prefix 与 Tail item 不完全等价] → evidence 同时报告训练频次组和 prefix mass，不把策略命名为 Tail oracle。
- [reserve 候选概率过低，损害 Head/overall] → 限定在模型 shortlist 内，并用很小的槽位 sweep 与明确损失门槛。
- [运行时扫描 training split 增加启动成本] → 仅构造一次 tensor；四组 probe 可接受，后续全矩阵再物化 Artifact。
- [相同 SID 对应多个 item] → prefix mass 对所有 catalog item 的训练频次求和，保留 dedup hierarchy 的完整 prefix 区分。
- [浮点并列导致 baseline 漂移] → disabled 分支保留当前排序/top-k 路径；intervention 使用稳定 index tie-break 并测试确定性。
- [candidate reach 改善但 Top10 不改善] → verdict 保持 `inconclusive` 或 `stop`，后续才考虑 candidate 内排序或训练目标。

## Migration Plan

1. 以默认关闭字段扩展 TIGER 构造和 decoder，不改 checkpoint state dict。
2. 添加独立 probe config/script，不修改既有 `tiger_inference` 与 `tiger_prefix_trace` 默认配置。
3. 添加 diagnosis comparison mode，保留原 search-ranking 模式和 evidence 文件。
4. 轻量验证通过后，由用户人工运行 pilot；关闭配置即可回滚到现有生成行为。

## Open Questions

- 首轮人工 pilot 的推荐 sweep 为 `reserved_slots in {1, 2}`、`pool_multiplier in {2, 5}`；具体完整运行组合由用户启动前根据 dry-run 命令审阅。
- 若四设置 probe 达到 advance 门槛，下一 change 再定义 Toys 与多 seed 扩展，不在本次实现中预置全矩阵。
