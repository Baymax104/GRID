## 目的与唯一有效计划

以正式CoPMRec v5.3为基础，为论文方法提供组件消融与机制观测证据。2026-10-08重新读取正式代码、配置、测试与来源回执后设计；本次用户进一步要求减少训练，不做多数据集、多seed消融。旧M3内容及旧Legal/Max/Mass计划不作为设计依据，旧计划中的M3条目由[本次统一协议](<https://linear.app/baymax104/document/copmrec-v53m3-%E6%B6%88%E8%9E%8D%E4%B8%8E%E6%9C%BA%E5%88%B6%E5%AE%9E%E8%AF%81%E5%8D%8F%E8%AE%AE2026-10-08-643847ec8183>)替代。

固定 **Beauty / training seed42** 作为单一代表单元，不根据结果更换。整体三数据集、三seed验证继续由既有正式主结果承担；本消融只描述这一条件下的组件观测。

协议ID：`copmrec-m3-v53-20261008-v1`。父任务Done，implementation=runtime_validated，full_run_authorization=true（用户2026-10-08明确授权本范围，并在训练完成后要求继续）。五训练/250k、五Testing、四diagnosis和M1零forward配对分析均完整交付并独立核验；全部八个实验子issue已Done。未增加训练、数据集、seed或预算。Done仅表示预定证据完整，与差值方向/显著性无关。

## 三个实验问题、五个训练变体

* 前缀监督：A1去mixture NLL；A4将其替换为同样四层合法生成NLL，区分监督项增量与内容条件混合形式。
* 共享残差：A2同时去历史/目录残差，保留四loss及原名义权重；两个位置的区别用M2固定checkpoint干预测量，不额外重训，不声称共享优于未共享。
* 辅助视图：A3去native CE；A5改为重复同一次joint CE，区分辅助视图与CE重复加权。native query仍含历史残差，不能称完整无协同模型。

Full复用<issue id="f6fdd31f-3886-40f3-819d-a715a9fd9daf" href="https://linear.app/baymax104/issue/BMX-120/主结果copmrec-beauty-seed42">BMX-120</issue>的正式own-best/Testing（gshpyn49/vosmuihm），新变体从头初始化，不加载Full或开发checkpoint。五变体各1个50k/DDP2/global256/FP32训练，各自训练内raw dense val/ndcg@10选首次best，随后单卡joint dense Testing。所有精确loss、固定条件、URI/SHA和模板见协议与子issue。

## 机制证据包

* M1：6份bundle的5配对，按仅一方命中、共同命中rank变化做可加NDCG分解，复核总差；seen/cold、训练频次、历史长度切片。可附本单元LIGER hybrid整体描述，其历史资格差异显式保留。
* M2：同Full checkpoint历史/目录残差2×2评分干预；正式V11复用，V10/V01/V00新增，包含native评分、相对margin、rank、Top10 overlap及数值交互。与A2重训结果分别解释。
* M3：Full/A1/A4 own-best的逐层真目标前缀概率、NLL、熵、JS divergence及全局alpha测量；条件化诊断不等同free-running候选恢复，dense部署不跑beam。

## 子issue与实际成本

| 实验 | Issue | 新执行量 |
| -- | -- | -- |
| A1 | [BMX-122](<https://linear.app/baymax104/issue/BMX-122/copmrec-v53-%E6%B6%88%E8%9E%8Da1-%E5%8E%BB%E9%99%A4-mixture-nllbeauty-seed42>) | 1 train / 50k / 1 Testing |
| A2 | [BMX-123](<https://linear.app/baymax104/issue/BMX-123/copmrec-v53-%E6%B6%88%E8%9E%8Da2-%E5%8E%BB%E9%99%A4%E5%8E%86%E5%8F%B2%E4%B8%8E%E7%9B%AE%E5%BD%95%E6%AE%8B%E5%B7%AEbeauty-seed42>) | 1 train / 50k / 1 Testing |
| A3 | [BMX-129](<https://linear.app/baymax104/issue/BMX-129/copmrec-v53-%E6%B6%88%E8%9E%8Da3-%E5%8E%BB%E9%99%A4-native-view-cebeauty-seed42>) | 1 train / 50k / 1 Testing |
| A4 | [BMX-142](<https://linear.app/baymax104/issue/BMX-142/copmrec-v53-%E6%B6%88%E8%9E%8Da4-%E7%94%A8%E5%90%88%E6%B3%95%E7%94%9F%E6%88%90-nll-%E6%9B%BF%E6%8D%A2-mixture-nllbeauty-seed42>) | 1 train / 50k / 1 Testing |
| A5 | [BMX-143](<https://linear.app/baymax104/issue/BMX-143/copmrec-v53-%E6%B6%88%E8%9E%8Da5-%E7%94%A8%E9%A2%9D%E5%A4%96-joint-ce-%E6%9B%BF%E6%8D%A2-native-cebeauty-seed42>) | 1 train / 50k / 1 Testing |
| M1 | [BMX-144](<https://linear.app/baymax104/issue/BMX-144/copmrec-v53-%E6%9C%BA%E5%88%B6m1-%E5%91%BD%E4%B8%AD%E6%8E%92%E5%90%8D%E4%B8%8E%E6%95%B0%E6%8D%AE%E5%88%87%E7%89%87%E7%9A%84%E5%8F%AF%E5%8A%A0%E5%88%86%E8%A7%A3>) | 0 forward / 复用6份bundle |
| M2 | [BMX-145](<https://linear.app/baymax104/issue/BMX-145/copmrec-v53-%E6%9C%BA%E5%88%B6m2-%E5%8E%86%E5%8F%B2%E7%9B%AE%E5%BD%95%E6%AE%8B%E5%B7%AE-22-%E8%AF%84%E5%88%86%E5%B9%B2%E9%A2%84>) | 1 diagnosis / 3新增评分pass |
| M3 | [BMX-146](<https://linear.app/baymax104/issue/BMX-146/copmrec-v53-%E6%9C%BA%E5%88%B6m3-%E9%80%90%E5%B1%82%E5%89%8D%E7%BC%80%E6%A6%82%E7%8E%87%E4%B8%8E%E7%9B%91%E7%9D%A3%E5%BD%A2%E5%BC%8F%E6%B5%8B%E9%87%8F>) | 3 diagnosis / 3前缀pass |

累计固定预算：**5 train / 250,000 optimizer updates / 0额外独立Validation / 5 Testing**；另 **4 Trainer.test diagnosis任务 / 6全量forward等价pass**。训练内每500步选点不取消；Full不额外训练或Testing。W&B tracked runtime与分配卡数换算已实测登记；active GPU时长/峰值显存未记录=null。LIGER dense原内部9Testing不计入本M3。不得按结果自动增加seed、数据集、变体或扫描。

## 实现准备与完成核验

必要variant与diagnosis入口、公共初始化/RNG、损失/梯度/冻结、完整恢复与跨臂拒绝、Hydra装配、共享结构化writer以及脚本语法/quoted变量/空值/额外override在准备阶段验证通过，Ruff/OpenSpec strict通过。随后实际双rank NCCL五臂50k训练、五份单卡Testing和M1/M2/M3全量实验完成；五臂首次raw validation-best的不可变URI/SHA、Testing bundle及机制输出已完整登记、独立核验。

训练实际预算和选点由完整history/100val500/log证明，CPU严格恢复及optimizer/scheduler/frozen/cold state/data/catalog/source已核对；Testing和机制的合法输出/用户目标身份/metrics/CI/可加分解/固定哈希案例/全部预定slice以及Artifact最终文件身份通过。原始训练source42f0与后续运行source5e95分别保留，metadata修复不追改历史身份。完整运行回执和原值如下。

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

## 统一内容与交付标准

全部8个实验issue使用十个同名章节：协议与状态、实证问题与解释边界、对照与干预、数据/seed/来源、固定协议、执行步骤与命令、观测指标与统计、成本与依赖、结果登记、交付标准。

只提供原始观测、带符号的数值差、样本数、pointwise CI和解释范围；不判别好坏，不预填积极/消极结果，不给出晋级/否决决定。单seed用户bootstrap不代表训练seed稳定性。执行前未观测字段null，执行后登记实测原值；未测资源、空组及结构上不适用字段继续null，异常与缺失如实保留。Done仅表示预定证据、来源、合法输出与独立计算完整，与方向或显著性无关。父任务与八个实验子issue已按以上标准完成证据交付。
