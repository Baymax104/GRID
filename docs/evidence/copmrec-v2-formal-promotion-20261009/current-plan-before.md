# CoPMRec 当前计划｜v1 消融重新评价

2026-10-09 用户要求先按 v1 重新进行消融评估。本批次已完成并独立核验，见 [v1消融结果](../docs/copmrec-v1-ablation-evaluation-20261009.md)。本次固定 Beauty / training seed42，复用五臂原 own-best，新增 **0训练 / 0独立Validation / 5单卡Testing**；Full复用 hc8oct43。另重新做 M1 六bundle命中/排名/NDCG分解，model-forward=0。父任务 BMX-150，子任务 BMX-151～156，均在 Linear M3；每个实验独立 node1 tmux。

不重选点，不扩 seed/数据集，不重新运行残差/前缀 diagnosis。只登记四指标原值、带符号差、指定对照、预定切片及用户配对CI，完成与结果方向无关。原 v0 M3 与250k训练事实保留。完整来源与命令见 [当前执行证据](../../GRID/docs/evidence/copmrec-v1-ablation-eval-20261009/README.md)。

版本定义见 [v0/v1定义](../docs/copmrec-v0-v1-definition-20261009.md)，总体开发计划见 [开发计划](../docs/copmrec-development-plan-20261009.md)。此前“零新增运行”描述版本文档整理阶段，本次用户授权是独立的有界评价批次。
