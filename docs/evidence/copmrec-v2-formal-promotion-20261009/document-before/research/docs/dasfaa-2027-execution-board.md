# DASFAA 当前执行看板｜CoPMRec v1 开发

> BMX-150及151～156：v1 Beauty42五臂无排除重新评价和M1六bundle分析已核验完成；0训练/5Testing/1analysis/0forward。见 [v1消融结果](copmrec-v1-ablation-evaluation-20261009.md)，原v0 M3与250k预算保留。下文零新增运行描述前一文档阶段，当前追加批次已执行完毕。

2026-10-09：当前正式参考为 v0（原 v5.3），当前开发为 v1（训练、推理均不排除）。版本登记不产生新训练；本轮只更新文档与计划。

| 工作包 | 已完成及当前边界 | 下一安排 |
| -- | -- | -- |
| M1 协议与来源 | 冻结上游与数据协议保留；原来源身份不重写 | 补充新版本映射及无排除协议说明 |
| M2 核心主结果 | v0 主矩阵 9/9 已核验；v1 仅 Beauty42 `hc8oct43` | 保留旧表，v1 不冒充完成 9/9 |
| M3 消融与机制 | BMX-117 五臂 5训练/250k/5Testing、4diagnosis、M1零forward已完成 | 原有排除指标按实际协议报告；不自动重跑或增加 seed/数据集 |
| M4 同类方法对照 | 原 LIGER hybrid 9 配对保留；BMX-147/148/149 单条件对照已完成 | v1 与 LIGER dense 共同无排除表单列；其余条件不自动执行 |
| M4 其他 baseline | TIGER/LETTER/SASRec 既有有效状态保留 | 接入时审核实际协议；不重置完成状态 |
| M5 论文 | 主指标 NDCG@10、方向保护 Recall@10；版本及部署差异需披露 | 后续区分 v0 原正式表、v1 初始表与机制观测，不修改旧来源 |

当前新增执行预算 0 训练 / 0 Testing / 0 diagnosis；下一工作为开发入口/元数据方案与证据表述对齐，尚未实施。未来实验 issue 保持统一模板和完整可复制命令，训练双卡、推理单卡，group 格式沿用其他实验。

历史训练累计为 v0 主矩阵 9/450k + M3 5/250k，不清零。v0 原机制细节、失败 attempt 和源码 identity 保存在切换前看板快照及原实证文件中。

当前权威：[开发计划](copmrec-development-plan-20261009.md)、[版本定义](copmrec-v0-v1-definition-20261009.md)、[机器状态](../research-state.yaml)、[共同无排除实证](copmrec-liger-both-history-off-20261008.md)。

[切换前看板原字节](../../GRID/docs/evidence/copmrec-development-version-reset-20261009/before/research/docs/dasfaa-2027-execution-board.md)、[原 M3 终验](../../GRID/docs/evidence/copmrec-m3-completion-20261008/README.md)。
