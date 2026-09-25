## Context

固定联合checkpoint的mass20与max20生成候选平均交集约10.63、并集约29.37；标签oracle显示并集仍有可利用覆盖空间，但无标签的逐用户路由特征没有稳定收益，已实现的学习排序器也未通过NDCG门槛。深度条件聚合进一步失败，说明当前验证应直接保留候选互补性，并用候选预算对照区分“互补来源”与“更多beam”。这些结论来自已查看的Beauty testing，只能支持一次封闭探索。

## Goals / Non-Goals

**Goals:**

- 在一次共享encoder和内容投影后，分别执行mass20与max20 beam，逐用户构造稳定去重并集。
- 使用既有cold union和内容logit终排，隔离候选集合变化。
- 保存源候选与输出候选，机器校验精确并集、目标覆盖OR和标准候选trace一致性。
- 以mass30作为更省搜索的候选预算对照，并按冻结三臂门槛给出停止决定。

**Non-Goals:**

- 不训练路由器、排序器或主模型，不恢复learned reranker。
- 不搜索beam、alpha、融合权重、候选quota、seed、数据集或subgroup。
- 不把同一testing上的通过称为独立确认、普遍规律或显著优于PAG。

## Decisions

1. **新增inference-only `CandidateUnionLiger`。** 基类只抽取一个可传processor和beam数的内部生成helper；默认调用参数不变。新子类固定接受`mass_max_union20`或`mass30`，训练调用直接拒绝。
2. **并集保留确定性源顺序。** 每个用户先保留mass20有效候选的首次出现，再追加max20中尚未出现的候选，尾部以-1填充到40。最终排序仍对“生成并集 ∪ cold”按共享内容logit降序稳定排序，因此源顺序不会成为推荐分数。
3. **mass30只扩大单一mass搜索。** 它用相同learned alpha、相同mass聚合与30条beam，不调用max；用于判断union收益是否仅由接近30的候选规模解释。冻结mass20现有run作为低预算基线。
4. **标准trace与来源trace分离。** `liger_candidates_v1`继续描述实际输出候选，metadata记录臂和搜索预算；新`liger_candidate_union_v1`记录primary、secondary与output行及目标成员关系。union validator强制`output=dedup(primary+secondary)`且`target_output=target_primary OR target_secondary`；mass30 validator强制output等于primary且secondary宽度为0。
5. **冻结三臂判定。** union相对mass20的NDCG@10配对95%CI下界必须大于0且Recall@10点估计不降；随后相对mass30也须满足同一条件。第一项失败即停止；仅第一项通过视为预算效应；两项通过只保留testing-informed候选，需独立确认。
6. **来源一致性先于效果判定。** 比较器要求相同用户、标签、catalog、cold集合、TopK、联合协议、运行时alpha、dense rank与dense Top10；允许generation_candidates不同。union源trace必须与其标准trace的generated rows逐元素一致。

## Risks / Trade-offs

- **union需要两次beam，计算量高于mass30** → 同时报告搜索次数和候选规模，不把它描述为等算力胜出。
- **mass30的候选数只是近似匹配并集均值** → 将它定义为保守的单源预算对照，不宣称严格FLOPs匹配。
- **testing已被用于提出方案** → 结果始终标记为testing-informed；通过后才考虑独立seed或数据集确认。
- **beam宽度变化可能改变搜索质量而不只改变数量** → 这正是mass30控制代表的实际替代方案，比较结论限定为端到端候选策略。

## Migration Plan

默认模型与launcher不变；新实验必须显式选择新配置和臂。回滚只需停用新launcher，不修改checkpoint或数据。完整prediction由用户手动启动。

## Open Questions

无；本阶段的臂、预算和停止条件均冻结。
