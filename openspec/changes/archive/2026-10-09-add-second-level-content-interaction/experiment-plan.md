# 有限开发验证计划

状态：实现及轻量验证完成；seed42 interaction已完成，尚未达到晋级门槛，additive/shuffled待补齐。结果见 [training-outcome.md](training-outcome.md)。协议候选名 `level2-content-interaction-v1`。完整训练/推理由用户手动执行；可执行命令与验证记录见 [implementation-results.md](implementation-results.md)。

## 1. 比较对象与预算

固定Beauty、现有SID/内容、A full_content初始化、20k steps、每500步验证、原batch/Adam/精度、beam10和best-val NDCG checkpoint。不从A最佳checkpoint续训，所有候选从相同初始A状态出发。

第一阶段：seed42的interaction、additive、shuffled三个条件，共3次20k训练，复用同seed A作为参考。全部三个条件预先锁定，不因某个消融较差而中途删去。若通过开发门槛，第二阶段只重复seed43三项；新增训练上限6次/120k steps。这是顺序开发筛选，不是无偏多seed确认或已批准执行的矩阵。

历史A只有在resolved config、数据与Artifact身份、初始化以及受影响执行路径可比时才复用。历史源码/环境不可核实则明确列为局限；若差异足以混淆结论，须先提供新的匹配A重跑方案与预算，而不能把历史A当严格对照。新增A重跑不隐藏在上述6次预算中。

## 2. 进入完整训练前

- 验证原A状态、随机流和零门输出；构造性非零梯度、optimizer与DDP注册；无需GPU实验即可做的部分先完成。
- 验证带padding/SEP的item对齐、decoder因果mask、teacher forcing与beam局部评分一致、未知pair错误及inactive beam。
- 验证保存加载，推理不重算尺度；特征/表hash与模式一致。
- Hydra compose、脚本语法/quoting/错误输入/override透传、聚焦单元测试、strict OpenSpec校验。
- 远端实验交付前按Mutagen契约flush且四session无冲突；不在本轮执行同步。

## 3. 预先固定的开发门槛

第一阶段候选相对A的best-val NDCG相对增益至少1%，最佳点Recall不下降，最后五个validation点NDCG均值不下降；interaction还须超过additive与shuffled的best-val NDCG。1%是本轮人为设置的继续投入门槛，不是统计显著性或理论阈值。相邻后五点不是独立重复。

满足才进入seed43重复；两个seed都须满足相对A的上述门槛，两个seed的平均NDCG须超过两个固定输入对照，且不以删除不利seed补救。对照个别seed胜负仍完整报告。若失败，停止当前版本；不自动扫门学习率/尺度或叠加W。

只通过这些条件仍是初步开发正信号。后续需单独制定多seed、第二数据集和A+PrefixMem-style比较，以及独立于已观察Beauty testing的新确认依据。已有随机mask_ce缺少的seed复核是另一项问题，不混入该改进预算。

## 4. 成本与机制记录

按同硬件、同DDP/global batch和稳定区间比较step/s、训练总时长、每用户推理时延、峰值显存、checkpoint字节及实际可训练参数。报告构造成本，不能把3.46MiB固定表当总成本。若step/s退化超过10%，先检查实现是否产生同步或重复计算；10%是工程调查线，不据此篡改训练协议。

记录g/tanh(g)、注入向量/基础embedding的RMS比值及非零比例。prefix survival只作次要诊断；不能用teacher forcing改善替代真实beam的最终NDCG。

固定开发诊断按目录pair单例/非单例分别列出样本数、A新增/丢失命中，但不挑有利分组重新定义主指标。冻结模型关闭注入是依赖性干预，不能替代从头训练A的效果对照。

## 5. 证据归属

W&B config记录mode、protocol、pair_seed、reference_A_run、catalog与buffer身份；notes记录实验意图。保存best checkpoint及selection规则、输入lineage和代码身份。inference/diagnosis继续统一入口及共享writer。本阶段不复用Beauty testing调参或选择版本，也不因开发结果好就自动运行testing。
