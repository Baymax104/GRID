## Context

研究报告已通过用户路线选择。第一门槛只实现global/branch两臂，复用成熟A40k，训练各2000更新。旧CGBS不可改写为新方法，现有prefix_trace的概率字段不能承载能量分数。

## Goals / Non-Goals

Goals：来源严格冻结、同缓存同目标的结构对照、零起点和off复现、可手动运行的缓存/训练/推理链路。
Non-Goals：本轮不运行完整实验，不实现或开启第二/第三门槛，不声称新颖性或实验收益。

## Decisions

- 单卡FP32缓存，训练集固定有界流，复用训练causal augmentation；按批写分片，不存全部hidden。以顺序样本ID为缓存key，历史/目标摘要识别重复窗口；不将窗口数当唯一用户数。
- 缓存每层全部A展开prefix及累计A log分数，补入缺失gold prefix且去重；原A beam用原topk规则。每层以目录prefix索引编码，padding=-1/-inf。缓存manifest含source SHA、catalog fingerprint、beam、样本数与分片SHA；训练验证完整性，checkpoint绑定manifest digest。
- 只训练独立残差头；保留冻结A的eval状态。两个头参数结构相同，global仅将attention query的候选原型替换为固定目录均值（由可训练projection生成全局query）；交互MLP仍接收真实原型。输出层零初始化，池化以有效原型均值为初始明确选择。
- F=累计A log概率+当前节点r，不累加祖先r，不局部重归一化。原A的off直接调用旧beam。新trace单独存能量与存活，不伪装概率。
- cache/train/inference通过src.main与不同experiment，writer归common/writers，cache读取归data/components/artifacts的入口。正式cache必须达到预定样本数且不能覆盖旧输出；dry-run不发布正式cache。
- 训练单卡，global batch256（微批16累积16），固定seed、manifest与样本顺序；2k更新和每500更新验证。候选块分块与gradient checkpoint限制attention激活内存，报告未实测吞吐。

## Risks / Trade-offs

### 2026-09-22 等价吞吐修复

- 同一微批的历史 key/value 按用户投影一次，训练 frontier 与 beam search 各自在全部层间复用；不持久缓存跨更新张量，不 detach 可训练投影。
- global 模式按用户计算一次 attention，再向候选原型广播；branch 保持候选 query，仅共享历史投影。
- history 非空校验移至共享准备阶段。branch 的候选块继续 checkpoint，索引复制留在 checkpoint 内；global 块只含小型评分 MLP，不再重算完整 attention。
- 保持参数键、checkpoint 契约、候选集合、mask、正则项、预算和默认块大小不变；用旧公式独立参考验证非零头输出、loss、全参数梯度和跨更新行为。
- CPU 验证不代表 GPU 加速比；不自动重启正在运行的训练。

- 全量候选较多 → 分片、流式dataset与候选分块，不暗中截候选。
- FP32重评分可改变排序 → 原beam缓存直接保存，off复用原路径，新增能量语义独立审计；不调容差过关。
- 离线A前缀训练与新beam分布不同 → 明确off-policy限制；不加入额外训练矩阵。
- 第一门槛仅结构增量 → 后续必须同分数重排、PAG式先验和等延迟A对照；通过前不晋级。
