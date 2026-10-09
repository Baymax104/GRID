## 2026-10-08完整执行与核验回执

五个scratch训练均实际50,000 updates，共250,000；每臂100个val500点，从各臂完整raw val/ndcg@10选首次最高own-best。5/5单卡Testing、4/4正式diagnosis及M1零forward bundle分析已全量完成并独立核验。Full Training/Testing及既有Full prefix复用；未新增训练、seed、数据集或参数扫描。全部八个子issue Done，完成只按来源/预定观测/计算完整性，与数值方向或显著性无关。

| Issue / arm | Training run | Own-best step | Testing run | Test GPU → local |
| -- | -- | --: | -- | -- |
| [BMX-122](https://linear.app/baymax104/issue/BMX-122) / no_mixture | [m3frgiim](https://wandb.ai/baymaxam/GRID/runs/m3frgiim) | 48500 | [m3veerfx](https://wandb.ai/baymaxam/GRID/runs/m3veerfx) | 0 → [0] |
| [BMX-123](https://linear.app/baymax104/issue/BMX-123) / no_residual | [m364gahb](https://wandb.ai/baymaxam/GRID/runs/m364gahb) | 47000 | [m3bk48w2](https://wandb.ai/baymaxam/GRID/runs/m3bk48w2) | 2 → [0] |
| [BMX-129](https://linear.app/baymax104/issue/BMX-129) / no_native | [m3td2jxc](https://wandb.ai/baymaxam/GRID/runs/m3td2jxc) | 41000 | [m3xm4hcb](https://wandb.ai/baymaxam/GRID/runs/m3xm4hcb) | 3 → [0] |
| [BMX-142](https://linear.app/baymax104/issue/BMX-142) / legal_generation | [m3odfrrh](https://wandb.ai/baymaxam/GRID/runs/m3odfrrh) | 46000 | [m3sfptwc](https://wandb.ai/baymaxam/GRID/runs/m3sfptwc) | 4 → [0] |
| [BMX-143](https://linear.app/baymax104/issue/BMX-143) / joint_ce_replace | [m3kq1tuc](https://wandb.ai/baymaxam/GRID/runs/m3kq1tuc) | 43000 | [m3hycech](https://wandb.ai/baymaxam/GRID/runs/m3hycech) | 5 → [0] |

Beauty/training seed42；N=22363，四指标原值：

| Arm | Recall@5 | NDCG@5 | Recall@10 | NDCG@10 |
| -- | --: | --: | --: | --: |
| Full | 0.05334705 | 0.03659647 | 0.07776238 | 0.04443310 |
| A1 | 0.05249743 | 0.03596258 | 0.07731521 | 0.04394472 |
| A2 | 0.05187139 | 0.03453895 | 0.07981934 | 0.04348076 |
| A3 | 0.05477798 | 0.03755917 | 0.07905916 | 0.04542925 |
| A4 | 0.05146894 | 0.03516278 | 0.07646559 | 0.04324066 |
| A5 | 0.05263158 | 0.03653958 | 0.07825426 | 0.04485631 |

全部用户NDCG@10配对观测（variant−Full；95% pointwise CI，PCG64 seed42/2000rep）：

| Arm | 数值差 | CI |
| -- | --: | -- |
| A1 | -0.00048838 | [-0.00122374, 0.00024884] |
| A2 | -0.00095234 | [-0.00306273, 0.00104712] |
| A3 | +0.00099615 | [-0.00026333, 0.00237374] |
| A4 | -0.00119244 | [-0.00199778, -0.00036154] |
| A5 | +0.00042321 | [-0.00087547, 0.00175918] |

M1/BMX-144：[m3hitc8p](https://wandb.ai/baymaxam/GRID/runs/m3hitc8p)，6 arms/134178 user-variant/114 slice/608 metric-CI/48 hit counts/132固定哈希案例，K5/10三贡献可加核验通过；包括预定5Full配对、A4−A1/A5−A3和Full-self数值检查。14项最终Artifact文件身份passed；`copmrec_beauty_seed42_hits_full_diagnosis-analysis:v0`，digest=`f50b595fd764b9fb495768d678d20bcb`。模型forward=0，GPU0→local[0]，独立tmux。

M2/BMX-145：[717vgnkn](https://wandb.ai/baymaxam/GRID/runs/717vgnkn)，N22363/4 views/76 slice/304 CI/5目录范数组；V11逐key精确复现Full，其余三视图及数值交互独立复算，12项最终文件身份通过。仅复用前轮已完成证据，无本轮重跑。

M3/BMX-146：Full [j2rworuj](https://wandb.ai/baymaxam/GRID/runs/j2rworuj)复用；A1 [5murlou7](https://wandb.ai/baymaxam/GRID/runs/5murlou7)、A4 [pltnvpjf](https://wandb.ai/baymaxam/GRID/runs/pltnvpjf)在物理GPU1→local[0]顺序、各独立tmux执行。三来源各22363×4=89452记录；A1/A4各92 slice/8最终文件身份核对通过。Full全局alpha=0.9857481122；A1/A4未学习gate，alpha/mixed观测null。全部原值、分位数、N/切片和连续概率差图已交付；teacher forcing不用于自由beam恢复或dense因果中介结论。

源代码/数据/产物：训练runtime304文件SHA=`42f0d7c7dce0a09961e0c8c23f9ba35982abfb796d13952c3f17eaeba17c9f2e`；Testing/诊断SHA=`5e95cf7e87b28b730976b5a6d279acdc7aca3a270a08dd5e4519ce1fc6624867`、origin verified。差异仅两处既有metadata序列化修复，训练/恢复/评分计算源逐字匹配。原训练source archive与新运行身份分别保留，不回填历史来源。525 Beauty分片SHA manifest=`46fbe99589b809ec8b7b9c21a815e4757c7a20d57664342ba8e525ecad8cb0c2`；22363 user/target SHA=`55e2602110f989565aa6c5fc00bef99107a17d3d11222c06c46874b9df50e016`。所有bundle合法唯一Top10/历史排除、4metrics/own-best lineage、cloud manifest与node1文件path/size/MD5/SHA已核验。W&B最终输出为node1文件引用，最终文件保留；code archive发布字节已核对。

资源/异常边界：五训练W&B原始tracked runtime合计119367秒；双卡分配时长换算66.315 GPU小时，口径为run墙钟×分配卡数，active GPU时长和峰值显存未记录=null。50k终态完整checkpoint未留存，last保存各自best；实际预算由完整history/100val/log核验。前轮M2 ls14h0dw序列化失败保留、completion_credit=false；正式diagnosis4完成，工程attempt5含该1异常；M1 bundle分析另计1，工程复现V11/失败开销单列，不改变训练预算。

可追溯证据入口：GRID `docs/evidence/copmrec-m3-completion-20261008/README.md`。`training-audit-summary.json` / `training-verified.json` / `testing-audit-summary.json` / `m1-independent-audit.json` / `prefix-three-source-summary.json`及前轮M2/Fullprefix独立审计；所有实际命令、不可变own-best URI/SHA、输出URI/digest/SHA与tmux已补进各issue统一十章模板。论文表图CSV/LaTeX/PNG/PDF及22项hash manifest在`figures/`，视觉QA passed。仅报告实证原值、带符号差、区间、N、边界；单seed用户bootstrap不表示跨训练seed稳定性，缺失资源保留null。

