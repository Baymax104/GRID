## Context

已有原训练checkpoint `3mntjejz`和联合训练checkpoint `zl9gv56p`。现有交叉单元为原训练×固定0.5（`84d7wkgd`）和联合训练×纯内容1（`oxnmyhqy`）。需要两个预测补齐矩阵。

## Goals / Non-Goals

**Goals:** 只改变推理解码alpha，严格复用对应训练checkpoint、数据、SID、内容、beam、合法支持、cold union与排序；计算训练来源主效应、解码主效应和difference-in-differences交互。

**Non-Goals:** 不重新训练、不调alpha、不把2×2交互解释成单一损失项的因果贡献，不提供跨seed确认。

## Decisions

联合模型增加可空的`inference_mixture_alpha`。为空时保持checkpoint学习alpha；取[0,1]时，推理processor使用固定alpha且不调用learned gate。训练阶段与content_only同时使用均报错，trace记录有效alpha和覆盖来源。

原训练×1使用现有普通LIGER概率混合入口，无需代码分支。统计以用户配对为单位；交互为 `(joint_alpha05-base_alpha05)-(joint_alpha1-base_alpha1)`，同时报告四单元、简单效应、新增与损失命中。交互只能描述联合训练的相对收益是否随解码权重变化。

## Risks / Trade-offs

[两个训练checkpoint的表示不同] → 2×2可估计训练来源的条件差异，但不能定位SID CE、内容CE或混合NLL中的单项原因。

[测试集已用于探索] → 标记为探索性机制证据，不称独立确认。

[固定0.5不是联合checkpoint学到的0.819] → 此处目的为正交矩阵而非部署最优结果；不追加alpha扫描。
