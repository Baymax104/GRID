## Context

既有 joint、alpha1 共用 zl9gv56p checkpoint。整体效果支持方法，独立聚合增益未确认。

## Goals / Non-Goals

冻结基础得分、合法目录、beam20、cold union、排序与数据；仅改变决策机制。无需训练、不自动运行完整实验，不复现完整 PAG。

## Decisions

抽取 processor 工厂。joint 默认 learned_mass；合法生成 legal_generation 使用合法归一化 Pg；max_mixture 保留同一 learned alpha 与算术混合，只把 logsumexp 后代聚合换成 max，再在合法子节点归一化。该最大后代控制借鉴 lookahead 思路，但不是 PAG 加性解码的复现。相比直接复用旧 content_guided，此设计不同时改变融合算子、温度、系数和归一化。

控制选项仅推理，训练拒绝非默认选项，checkpoint参数结构不变。trace 记录机制和有效 alpha。相同基础得分不等于不同路径上的生成 logits 必须相同；固定模型保证同一前缀得分相同。

## Risks / Trade-offs

旧测试已用于探索，不能称独立确认。合法支持相同不要求实际候选相同。新预算仅2次人工预测、0训练；失败不自动调权重或增加对照。不得把 max_mixture 结果推广为击败完整 PAG。
