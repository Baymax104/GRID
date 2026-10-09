# CoPMRec v5.3 / BMX-117：M3 完成证据

协议 `copmrec-m3-v53-20261008-v1`。本轮按用户“run已完成，继续bmx117”完成五臂 own-best Testing、M1配对分析和A1/A4前缀诊断；仅使用既定Beauty/seed42单元，不追加训练。

[Linear父任务](https://linear.app/baymax104/issue/BMX-117)及八个实验issue已Done；[统一协议](https://linear.app/baymax104/document/copmrec-v53m3-消融与机制实证协议2026-10-08-643847ec8183)和milestone3已同步。Done表示预定证据完整并可独立核验，不用于数值方向或方法优劣判断。

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

## 本目录的可复现证据

- [机器可读全部实证](m3-empirical-summary.json)：四指标原值、配对CI、贡献分解、来源指纹和资源。
- [五训练完成摘要](training-audit-summary.json)、[原始history与选点](training-collected.json)、[严格状态恢复与数据核验](training-verified.json)、[原始过宽源码检查](training-verified-initial-scope.json)：保留检查范围修正，不回填原训练来源。
- [Testing实际启动命令](testing-launch-specs.json)、[准确启动回执](testing-launch-receipt.json)、[逐字同步与GPU资源](testing-readiness.json)、[五Testing终验](testing-audit-summary.json)及`testing-audit-BMX-xxx.json`：每臂不可变checkpoint/输出URI、SHA/MD5/digest、4metrics与配对CI。
- [M1实际命令](m1-launch-spec.json)、[退出回执](m1-check-receipt.json)、[M1独立复算](m1-independent-audit.json)：全114 slice/608 metric-CI、命中分类、可加分解与固定哈希案例，模型forward=0。
- [三来源前缀完整登记](prefix-completion.md)、[原值及数值差](prefix-three-source-summary.json)、[A1独立审计](prefix-no_mixture-independent-audit.json)、[A4独立审计](prefix-legal_generation-independent-audit.json)、[实际命令](prefix-launch-specs.json)；Full独立审计复用[前轮回执](../copmrec-m3-launch-20261008/prefix-full-independent-audit.json)。
- M2复用[前轮独立复算](../copmrec-m3-launch-20261008/m2-residual-independent-audit.json)，异常保留[首次工程失败](../copmrec-m3-launch-20261008/m2-initial-failure.json)。本轮没有重跑这两个已完成来源。
- [论文表图入口](figures/README.md)、[六臂四指标](figures/paper-six-arm-metrics.md)、[五臂−Full CI](figures/paper-five-arm-minus-full-ci.csv)、[指定控制差值](figures/paper-specified-pair-ci.csv)、[三贡献原值](figures/paper-m1-ndcg-contributions.csv)、[22文件hash manifest](figures/paper-ablation-evidence-manifest.json)、[视觉与数据QA](figures/paper-visual-qa.json)。CSV保留原值，LaTeX/PNG/PDF仅作显示；无最优加粗或方向分类。原始曲线未平滑，own-best与50k预算端点分开显示，不以预算完成声称converged。
- [原始资源口径](resource-observations.json)、[研究四文档一致性](research-document-consistency.json)、[原始Linear快照](linear-before.json)、[最终独立Linear回读](final-linear-independent-readback.json)。研究主矩阵9/9与其他formal范围已逐对象核对保持原值。

## 解释与保留边界

所有推荐指标均使用相同22363 Testing用户、自己的validation-selected checkpoint与正式历史排除规则。用户bootstrap使用NumPy PCG64seed42/2000rep/95%pointwise，不用于跨训练seed稳定性或等价性声明。NDCG分解是按用户输出的数值恒等式；固定checkpoint残差干预与scratch A2分别登记；真实前缀条件观测不用于自由生成路径恢复或dense因果中介结论。所有预定切片及空组null保留。

W&B的最终预测/分析Artifact登记node1 `file://`最终文件引用，路径/size/base64-MD5/SHA已与实际文件逐项核验；source code archive的实际上传字节另行核验。运行端checkpoint/最终输出与分析目录保留，不对远端Git推断版本或删除实验产物。缺失的active GPU资源、峰值显存明确null，不用W&B墙钟代替这些实测量。

