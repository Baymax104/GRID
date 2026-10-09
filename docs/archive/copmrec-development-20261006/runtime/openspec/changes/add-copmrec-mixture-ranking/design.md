## Context

沿用 JointMixtureLiger checkpoint 的合法条件生成与精确 mass 内容分布。原 LIGER 与现有 CoPMRec 输出保留。所有命令从仓库根目录通过 src.main 启动。

## Goals / Non-Goals

目标：只改变 CoPMRec 的候选内最终次序，同时产出三种次序和配对净收益。非目标：训练新网络、扫描 alpha、修改 baseline、扩大候选或启动完整运行。

## Decisions

新增推理子类复用父类一次候选搜索及其内容参考 trace；再次执行确定性的 encoder/content 前向得到评分输入，避免改动共享 baseline retrieval。统一对并集内生成与 cold 商品分块 teacher forcing，完整目录合法归一化，累加每层混合/生成 log probability。二次 encoder 前向是明确的推理额外成本，无效率主张。

新的 copmrec_ranking_v1 trace 保存固定宽度的候选行号、SID、三路分数、三路 TopK 和目标排名。原 liger_candidates_v1 不用作新排序 trace。共享辅助 writer 负责合并和发布，新子类追加三臂汇总、命中损益与固定用户配对 bootstrap 区间。

## Risks / Trade-offs

混合概率可能损害 cold 或既有命中；实际收益未知。所有来源统一打分，不按目标标签调节。分块前向与生成累计分数须数值一致。推理必须处于 eval 模式。

## Migration Plan

新实验 liger_joint_ranking_inference 默认 evaluation、默认主输出 mixed。旧实验与原模型路径不变。由用户在 node1 手动运行。

## Open Questions

混合概率是否带来全体用户最终净收益，需运行后审计；实现验证不代表方法有效。
