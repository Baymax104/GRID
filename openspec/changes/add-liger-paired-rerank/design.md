## Context
现有LIGER只在候选阶段使用生成信号，最终dense排序；此前7次预测均已完成，最后混合NDCG@10 .038467仍低于dense .039587。验证终排信号互补性是同一问题下明确的新阶段。
## Goals / Non-Goals
一次完整预测、两种同池排序，0训练、0权重搜索。非目标：整体方法立即定型、跨seed声明、效率优势、再开展候选池搜索。
## Decisions
池=dense Top20 ∪ original生成20条beam的有效商品 ∪ cold商品，排序去重，相同池同时计算两臂。dense Top20包含原Top10，纯内容臂必须逐用户复现原dense Top10；此性质避免把扩大候选的影响误算为排序增量。
每个池内商品（含非生成商品）均用冻结decoder teacher forcing完整4位SID，求原始完整词表logsoftmax的4位和，不附加EOS，不使用生成beam分数。按128个用户-商品对分块，共用encoder输出；generation传入独立BaseModelOutput容器，避免HF扩beam修改后续teacher forcing的batch映射。
C=softmax_pool(内容cosine/temperature)，G=softmax_pool(完整SID loglikelihood)，联合=.5C+.5G；无温度/alpha搜索。正常bundle输出joint Top10，辅助trace保存全部池row及两种分数、joint分数、两臂排名/Top10和标签；标签只在候选及分数算完后用于诊断。
共享AuxiliaryTensorWriter合并固定宽度tensor，data校验器重算两臂排序，验证dense复现与无label插入。writer输出逐用户证据、汇总与1000次用户bootstrap（numpy PCG64 seed42），日志指标为rerank/*。
## Risks / Trade-offs
融合0.5不一定最优；负向只停止本固定实例/解码组合继续搜索，不宣称所有生成信号无用。生成概率长度均为4无需长度归一化。cold可能生成概率低；报告实测，不设临时补救规则。候选池最多73（20+20+33），额外teacher forcing成本显式报告，不能等同之前beam20计算预算。默认模式不变，旧checkpoint无迁移。
