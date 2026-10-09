## Context

已有 audit 对同一 query 执行 reference、dynamic、fixed64/128/256。reference 已计算全目录分数，但只返回 Top-K。dense 的候选来自冻结 reference，残差分支来自冻结 base；候选为目标 + 64 hard + 16 random（小目录截断）。

## Goals / Non-Goals

目标：一次 4×128 evaluation 用户审计能区分评分、候选支持集和搜索问题；另补一个 Sports 内容初始化训练。完整任务仍由用户手动启动。

不改变训练目标、checkpoint 协议、搜索顺序及预算；不扩展 BRIR seeds 或 datasets，不宣称已找到因果根源。

## Decisions

- `model.root.audit_training_diagnostics=false` 默认保留 v1。开启时发布 `brir_audit_v2`，沿用同一 writer、artifact role 和统一预测入口。
- reference SearchResult 额外返回已计算的全目录分数，其他策略不保留。诊断计算在五种策略计时之外，不重复 residual 前向。
- 记录冻结 anchor、当前 base、最终全目录分数及候选 item keys，计算同分按 item key 排序的目标排名、全目录/候选 CE、候选外概率质量与 Top-K 数量、hard-negative 重合、残差饱和度和有界残差可达到的乐观排名。CE 均为 eval 模式每用户原始值，不受梯度累积缩放。
- dense 的 anchor 使用冻结 reference，其他 arm 使用当前 base（残差分支已校验冻结指纹）。base 上的候选 CE 仅是共同候选集的反事实，不称为其训练 loss。
- 零更新测试真实构建三个分支并比较 base 全目录分数/排名/候选；GPU audit 记录的 base 分数是零残差反事实，不等同于恢复训练初始状态。
- 采用新有界套件逐个运行四个 arm，默认最后完成的 checkpoint，单卡 0、evaluation、sampling_seed=42、128 用户；允许 `--arm` 单独重跑，notes、dry-run、print-only、Hydra override 透传。

## Risks / Trade-offs

- 128 用户只用于机制诊断 → 不据此报告总体显著性；先核对跨 arm keys、输入摘要、标签、catalog 和 anchor 一致。
- 额外保存三组全目录 float 分数，Beauty 约 18 MiB/arm → 只在明确开启诊断时发布；保留原始值便于独立复核。
- 不同设备浮点实现可能有微小误差 → 分数比较用容差，Top-K 同分使用稳定 key 顺序。
- 饱和/候选外竞争是观察性证据 → 不将 CE 差异本身作为因果证明；只有修复性对照才能进一步确认。

## Migration Plan

聚焦测试、Hydra compose、shell stub 与 OpenSpec strict 验证后通过既有 Mutagen 边界同步。关闭诊断 flag 即恢复 v1 输出。Sports 沿用现有脚本，不改变其协议。
