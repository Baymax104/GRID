# 设计

## Context

v1.1 head读取[h,v,h*v,d]，与content路径职责重叠。d在完整SID输入后经过最终decoder block FFN、残差、最终归一化及dropout，cross-attention读取用户历史。该信息不保证与h/v完全等价；简化是待验证假设。

## Goals / Non-Goals

目标是提供仅改head输入的v1.2，保留可匹配训练与评价条件。此次不启动完整训练、停止已有run、扫描权重或重置已关闭的2000更新预算。

## Decisions

- 抽出head构造与残差计算两个可覆写方法，旧版本保持参数名、初始化和checkpoint契约。
- DecoderRelevanceCoPMRec继承批量评分；head从Linear(4d,128)缩为Linear(d,128)，GELU及输出层不变，输出层零初始化。
- score=cos(h,v)/0.07+MLP(d)，loss=L0+0.05726763550972437*ranking_CE。此系数来自旧结构，不宣称为新head最优；先保持它以隔离结构变化。
- v1.2记录版本与head_input；拒绝直接恢复v1/v1.1，包括开启loss重权例外时。可选v0 weights-only初始化继承既有支持，默认两种checkpoint引用均null。
- 无冻结/缓存推荐表示；保留对encoder、decoder、投影和融合参数的联合训练。

## 风险及最小验证

删除候选连续内容输入会限制残差的细粒度内容表达；无界残差、候选覆盖、共享梯度冲突仍可能存在。单元核验head只读取d、完整SID与用户映射、非零head的cross-attention/encoder梯度、联合更新、合法输出/标签独立、checkpoint/trace以及DDP。配置compose与脚本替身只验证契约，不构成推荐收益。

## 研究决策与运行边界

延续“生成表示能否转为Recall/NDCG收益”的问题，以当前校准v1.1为结构对照。已有短程校准阶段2/2臂已耗尽；本次新增agent训练数/预算0，仅交付用户手动v1.2命令，不自动补对照或audit。用户手动正式运行时以selection NDCG选点，同协议比较及独立audit才支持最终收益；旧v0全量dense曲线仅是有口径差异的参考。若匹配最终收益改善则保留简化候选；若仍更差则不晋级此head；若只有局部曲线波动或口径不匹配则结论未知，不自动增加预算。
