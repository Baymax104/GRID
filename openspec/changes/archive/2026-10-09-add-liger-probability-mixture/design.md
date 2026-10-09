## Context
当前LIGER干预是max势差加完整词表log概率；CGBS代码提供条件概率混合思路。当前结果支持内容前置有增量，不支持整体优于dense。用户确认继续在LIGER发展主方法。

## Goals / Non-Goals
目标：用固定条件概率混合验证同一候选保留取舍。非目标：训练新query、恢复旧CGBS路线、参数搜索、额外数据集、自动完整实验。

## Decisions
对每个合法子前缀c计算M(c)=logsumexp_{i属于c}s_content(i)，内容条件概率q(c|p)=softmax_c M(c)。生成条件概率p_g为完整词表log概率在合法子分支上重新归一化。局部分数log((1-alpha)*p_g+alpha*q)，按生成步累加；最终仍以原dense分数重排候选与cold并集。
conditional_only使用alpha0，probability_mixture使用alpha0.5固定，不设可学习参数。两臂共享归一化、合法前缀约束、beam20；原valid_only仅遮蔽不归一化，不能替代匹配对照。
使用稳定scatter max/exp/sum计算精确log质量，不复用CGBS原型近似与训练参数。最大势差processor旧行为不变，抽取前缀查找供复用。处理无合法候选的死beam为全负无穷，避免softmax产生NaN；alpha0/1明确分支。
主比较混合vs条件归一化控制。与max势差的比较只解释整套机制差异，不声称logsumexp或概率形式单独因果收益。使用既有trace配对汇总，不改产物结构。

## Risks / Trade-offs
分支logsumexp可能偏向商品更多的分支，是当前固定机制的一部分，不额外增加分支大小校正。alpha0.5是对称预设，不保证最优。全目录内容打分和精确分支归约仍有成本，不宣称加速。通过选择original可恢复旧baseline行为，无checkpoint迁移。
