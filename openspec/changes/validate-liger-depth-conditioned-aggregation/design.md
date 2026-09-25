## Context

固定联合checkpoint的mass/max诊断显示：mass-only与max-only目标路径分别产生171和144个Top-10命中，路径包含本身净增27；但两臂都生成目标的用户中，候选竞争净损23，最终只剩5个净命中。按首次内容rank分歧深度分解，depth0的mass相对max净损17个命中，depth1–2合计净增48个命中。该分解来自已消耗的Beauty testing，只能用于形成探索性控制。

## Goals / Non-Goals

**Goals:**

- 在内容前缀聚合中固定使用depth0=max、depth1及以后=mass。
- 复用联合checkpoint的learned alpha和全部其他推理条件，隔离聚合深度调度。
- 验证新臂是否同时保留相对max的覆盖优势，并相对mass改善最终NDCG且Recall不降。
- 保存候选与分散度trace，使运行来源、实际alpha、逐层行为和非目标候选变化可审计。

**Non-Goals:**

- 不训练、搜索切换深度、alpha、beam、seed、数据集或subgroup阈值。
- 不恢复learned reranker、动态门控或候选union路线。
- 不把同一testing上的通过结果称为独立确认或跨场景规律。

## Decisions

1. **增加显式机制控制`max_root_mass_deep`。** processor接受深度聚合计划；depth0读取原max势能，depth1以后读取logsumexp质量。训练路径继续只允许`learned_mass`，新控制为inference-only。
2. **保持同一混合门。** 新控制使用checkpoint恢复的`JointConstantGate`，生成分布、合法支持、learned alpha和beam累计方式不变，避免把alpha或生成权重变化混入深度效应。
3. **沿用两个独立trace schema。** `liger_candidates_v1`用于最终覆盖与排名，`liger_dispersion_v1`用于逐层父前缀存活；只扩展metadata取值，不改变字段集合。
4. **只新增一次正式prediction。** 现有`5wfpsg9a` mass与`sjf8qcgs` max作为冻结对照。新run必须使用同一checkpoint、SID、embedding、Beauty testing、seed42和beam20。
5. **冻结两道门槛。** 相对max的覆盖差配对95%区间下界必须大于0；相对mass的NDCG@10差配对95%区间下界必须大于0且Recall@10点估计不下降。第一道失败表示未保留总质量价值；第一道通过而第二道失败表示未转化；全部通过只保留为testing-informed候选。
6. **候选等价作为实现门禁。** 合成测试必须证明新processor在depth0逐元素等于max、后续深度逐元素等于mass；默认mass/max输出不得变化。

## Risks / Trade-offs

- **深度分解来自已查看testing，存在选择偏差** → 明确标记为post-hoc探索，成功后仍需独立seed或数据集确认。
- **完整beam轨迹不可由单层局部等价直接推出** → 正式run保存实际父前缀存活和最终候选，不用局部counterfactual代替真实搜索。
- **depth1后的mass可能依赖depth0保留下来的不同父前缀** → 这正是新臂需实测的交互，不从旧两臂trace外推效果。
- **单次结果可能只有点估计改善** → 按预声明区间门槛停止，不继续搜索切换深度。

## Migration Plan

默认`learned_mass`不变；新机制仅由显式实验配置启用。回滚只需移除该override，不涉及checkpoint或数据迁移。完整prediction由用户手动启动。

## Open Questions

无；切换深度固定为根层之后，不在本阶段比较其他计划。
