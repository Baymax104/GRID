## Context
参考实现已具备原始HF生成、全目录dense logits、cold并集及逐用户trace。当前固定问题是候选遗漏，不追求单一根因。

## Goals / Non-Goals
目标：两个解码干预复用同一分数/模型，记录净收益。无训练、调参、其他数据集或自动完整实验。

## Decisions
设c(i)为模型现有cosine/temperature分数，V(p)=max_{i以p开头}c(i)。每步在原始完整词表log概率上加lambda*(V(child)-V(parent))，并遮蔽不在目录中的前缀。合法性对照lambda=0，引导lambda=1固定。累计引导等于lambda*(c(i)-V(root))，不会逐层重复计入相同分数；有限beam仍为近似搜索。禁用二次归一化；原模式完全不注入processor。
使用目录SID预计算前缀编码、scatter_reduce amax和searchsorted查找，不构造每个beam对全目录的三维掩码。两新增臂均beam20，depth4，生成上限相同；实际有效候选数及cold重叠仍须报告，不称严格等FLOPs。
主比较引导vs合法性对照，原hybrid和dense为既有参照。保留全体配对NDCG/Recall、候选覆盖、原1146找回/原278保留与新损害；汇总工具拒绝用户、标签、目录或dense排名不一致。用户配对bootstrap仅表征本开发集采样不确定性，不替代多seed。

## Risks / Trade-offs
内容最大分可能偏向dense，权重1可能没有收益；结果否定该实例则收缩，不自动权重搜索。合法性约束本身可能解释提升；用lambda0匹配对照隔离。全目录打分和前缀归约增加成本，记录runtime不宣称加速。模式原值original可恢复旧行为。
