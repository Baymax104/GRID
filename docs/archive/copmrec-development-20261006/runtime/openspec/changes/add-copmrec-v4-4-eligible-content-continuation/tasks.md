## 1. 仅content CE的匹配续训与来源

- [x] 1.1 实现EligibleContentContinuationCoPMRec严格bool exclude_history_from_dense_ce，两臂同原权重/raw logits/SID与mixture，仅treated训练dense CE cold=-100后history=-inf；保留eligible autograd及target∉有效history断言。
- [x] 1.2 实现公共v4.1 weights-only严格初始化、fork_rng不扰动、保留原v0/v4链、新exact9 eligible_content_contract与exact5 continuation_metadata；normal新checkpoint严格step0..2000/flag/state，bias0与cold residual0。
- [x] 1.3 实现HistoryExcludedContinuationPool九kwargs，固定source原v4best6000与真实新continuedbest1000或2000，独立query/.5/.5、原9keyhistory规则、newpool contract/身份、单进程/无wrapper checkpoint，不松旧默认契约。
- [x] 1.4 聚焦CPU验证CE supports/有限梯度/control原loss及梯度、SID/mix不收masked logits、SID/attention/target边界、来源/step/错臂拒绝与旧父回归。

## 2. 配置、脚本与实际准备

- [x] 2.1 新薄model/experiment及根v4.4 train/inference脚本，warm与newcontinued字段分开，公共loader/lineage/source snapshot/commonwriter，真实checkpoint、flag与metadata对齐。
- [x] 2.2 Hydra完整resolve与脚本quoting/notes/dry-run/empty/error/extraoverride及双卡训练/单卡推理guard；显式scheduler_steps6000、warmup300/max2000/val1000、LR.0001/.002/WD.035/globalbatch256不漂移。
- [x] 2.3 root完成实际training causalhistory/target不重叠、输入和warm bytes、动态全source/Mutagen flush三Watching、统一入口双卡production batch smoke及聚焦回归/strict，不将准备称为效果。

## 3. 有界正式验证与决策

- [x] 3.1 root各一次启动treated/control共2×2000训练，审计exit0/终态2000、两次Validation、真实selected bestArtifact/step/SHA/两组LR与scheduler/source/continuation链，固定2000标量对比。
- [x] 3.2 各真实best与固定source组成pool，仅各一次完整单卡Validation，复用wdms8w77/iy3o3z3q已审计输出；独立raw22363/SID/history/source/CP/metadata复算fair、old及互比CI/groups/warmcold/newlostshared。
- [x] 3.3 分开判断对旧pool≥3%组件保留与treated-control mask归因；仅raw审计及same-policy双10/双CI/旧点阈值全部合格者，预commit按N、R、treated选唯一winner，否则本阶段不Testing/不续训/扫描。
- [x] 3.4 qualified唯一winner才与同policy固定LIGER各做一次Testing（全线程已1→最多3），实际审计same-policy双10/双CI/旧Testing点阈值；不以Testing选模型。真实结题保留有效原history层及有依据的续训贡献、整体目标未达则active。

实际结题：模型49项与配置脚本52项（新101项）、4项旧配置风险检查及两名agent交叉review通过；374 runtime源、实际CPU原生恢复、双卡与单卡统一入口smokes通过。两训练各2000步、两完整pool Validation均完成；按预承诺N→R→treated选择control hbgj80bh best2000，再固定与nj9elah1 best6000形成.5/.5 history pool。完整matched Testing ws2fx4oi对vnhmag7v R@10+10.770121598147053%、NDCG@10+10.776636054536315%，两paired差值CI下界正、旧辅助点阈值通过；175raw/22363用户/source374/checkpoint/合法唯一零history的独立审计通过。整体目标证据已验证，但不称CI效应下界达到10%；Testing未选择模型。

treated−control两CI跨0，30新增/27丢失、181共同上移/207下移，不支持训练history CE mask增量；两臂自身微小续训未达独立3%物质组件标准。保留原v4.3有效history层与完整已验证control部署，不要求各层独立达到3%或10%。旧5/30000封存，本阶段2/4000完成、累计7/34000；新Validation2/2、matched Testing2/2、全线程Testing3/3，无自动新预算/扫描或跨seed/数据集泛化。实际审计与结题见 `docs/evidence/copmrec-v4-autonomous-20261005/eligible-content-continuation-stage-closure.json` 和 `docs/copmrec-v4-4-eligible-content-continuation-research.md`；root独占research state/current plan/goal工具更新。
