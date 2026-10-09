## Context

`ProbabilityMixtureProcessor` 当前在合法 SID 子节点上分别构造生成条件分布与内容条件分布；`mass` 使用后代内容 logits 的 `logsumexp`，`max` 使用后代最大 logit。已有 `learned_mass` 与 `max_mixture` 运行只保存最终候选，无法定位两者在哪一层因有效支持差异而改变目标路径。

诊断沿用联合 checkpoint、learned alpha、beam20、目录、cold union 和最终内容排序。完整 prediction 仍由用户手动开始。

## Goals / Non-Goals

**Goals:**

- 对每个用户和 SID 深度记录目标子分支的最大值、log-mass、二者差值 `dispersion=logmass-max` 与后代数。
- 在目标父前缀下记录目标子分支在 max、mass 及对应混合局部分布中的 rank/margin。
- 分别记录 learned_mass 和 max_mixture 实际搜索中目标父前缀是否仍在 beam，从而得到首次淘汰深度。
- 配对两个诊断 bundle，报告分散度与 mass-only/max-only 路径恢复的关系，并按后代数分层，避免把大分支偏置解释为偏好分散。

**Non-Goals:**

- 不把内容分数分散度等同于真实多兴趣或多个可接受标签。
- 不重新训练、搜索 alpha、改变候选或最终排序。
- 不复现完整 PAG，不用事后 subgroup 结果声称跨 seed 或跨数据集规律。

## Decisions

1. **新增独立 `liger_dispersion_v1` 辅助 payload。** 保留 `liger_candidates_v1` 不变，避免破坏既有证据和比较脚本。专用 writer 合并 bundle 并生成描述性 summary。
2. **诊断只观察，不介入搜索。** processor 在每个调用深度计算并缓存诊断张量，返回的 logits 与未启用诊断时逐元素一致。label 只传给 observer，不参与返回分数。
3. **同时保存内容准则与实际混合准则。** 内容字段回答 `logsumexp=max+dispersion` 的机制预测；仅当目标父前缀实际存活时，额外记录同一生成条件分布下 mass/max 混合的目标局部 rank 和 margin。不可用值使用 NaN，存活使用显式布尔字段。
4. **实际 beam 结果以父前缀出现判定。** 深度 `d` 调用 processor 时，目标长度 `d` 的父前缀出现在该用户 beam 中即为 active；首次不 active 的深度为首次淘汰深度。最终 `target_generated` 继续以候选 trace 为准。
5. **后代数作为预声明混杂变量。** 汇总至少报告总体、同后代数或预声明后代数分层，以及 `dispersion` 在层内的方向性关系。全目录 entropy 只作补充，不作为主判据。
6. **两个实际解码臂配对。** 使用同一 checkpoint 分别运行 `learned_mass` 与 `max_mixture`，每臂一次 prediction；不从单臂 counterfactual 分数推断另一个搜索轨迹。

## Risks / Trade-offs

- **现有 testing 已被使用，subgroup 属于探索性分析** → 明确标记为 consumed testing，不把阈值或分箱结果称为独立确认。
- **单一观测目标不能识别真实偏好分散** → 主变量命名为内容相关性有效支持或 score dispersion。
- **processor 回调中的 beam 布局依赖 Transformers 生成约定** → 用多用户、多 beam 合成测试验证用户映射、深度和前缀存活，并 fail closed。
- **诊断张量增加显存与产物体积** → 只保存 batch×深度标量，不保存全目录 logits 或全部 beam 分数。
- **局部 rank 不等于全局 beam rank** → 字段和文档明确命名 `local_rank`；全局结论只使用实际父前缀存活与最终候选。

## Migration Plan

默认关闭诊断，旧配置、checkpoint 和 trace schema 不变。启用新配置后生成独立本地产物；验证后由用户手动运行两臂。回滚只需关闭诊断配置，不涉及 checkpoint 或数据迁移。

## Open Questions

- 若真实运行发现 target parent 在同一用户 beam 中出现重复路径，validator 将拒绝产物，再根据实际生成契约决定聚合规则；不预先静默选择重复项。
