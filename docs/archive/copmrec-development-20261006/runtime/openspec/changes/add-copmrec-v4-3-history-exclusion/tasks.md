## 1. 共享政策与薄推理子类

- [x] 1.1 实现严格apply_history_exclusion，仅有效完整SID精确lookup、padding忽略、历史去重mask、合法分数不变和eligible数量拒绝。
- [x] 1.2 实现HistoryExcludedLiger原kwargs/on_load_checkpoint/strict state兼容，以及HistoryExcludedFixedLogitPool仅父forward后mask；统一exact9keys history_eligibility_contract，保持父pool_contract和双来源。
- [x] 1.3 添加最小CPU测试覆盖partial/unknown SID、attention/rightpadding/inactive tokens、重复/冷商品/不足TopK、stable ties、labels/keys无关、原分数和statekeys不变，以及训练/optimizer/多进程拒绝。

## 2. 配置、脚本与聚焦验证

- [x] 2.1 新增薄model/experiment及liger_history_exclusion_inference.sh、copmrec_v4_3_inference.sh，冻结真实LIGER与pool来源、同policy/单卡FP32、公共loader/lineage和commonwriter metadata，不修改旧默认入口。
- [x] 2.2 验证完整Hydra resolve、shell语法、quoted URI、notes/dry-run/extraoverride、空值/错误参数/NPROC1、LIGER真实ckpt_path与pool顶层null恢复契约。
- [x] 2.3 运行风险匹配的旧模型兼容回归、Ruff及openspec validate add-copmrec-v4-3-history-exclusion --strict，记录实现准备而不声称效果。

## 3. 有界匹配评价与结题

- [x] 3.1 root完成实际checkpoint/input/catalog、两policy契约、全runtime source字节、Mutagen flush三Watching和统一入口单卡准备核验；不新增optimizer/训练或CF数据构建。
- [x] 3.2 仅启动至多2次完整Evaluation（same-policy固定LIGER与固定pool），独立raw用户/末商品标签、合法唯一SID、历史零输出、Artifact/source审计，复算对新baseline/各自旧输出的R/N、pairedCI与固定key组/warmcold/newlostshared。
- [x] 3.3 执行新same-policy双10%及两pairedCI正、同时旧绝对点阈值的完整Validation门禁；仅全部通过时运行至多2次same-policy固定Testing，累计Testing1→最多3，不以Testing选规则。
- [x] 3.4 分开报告整体目标门禁与组件保留：对自己冻结旧pool任一R/N≥3%且另一项无点值退化可保留bad case/累计贡献，CI与分组限制主张；未双10不自动丢弃正向组件。保留旧结题，不扫描或自动重置预算，未来成本依据新证据另行登记；原目标未达保持未完成。

实现准备已核验：core新23+父pool50共73项、配置/脚本新38+父入口43共81项，合计154 distinct；7新runtime、全部367文件SHA与旧360父字节不变证明见history-implementation-verification.json。完整Validation wdms8w77 / iy3o3z3q及实际两审计已完成，2/2额度用完。自身旧pool R+8.568824065633551% / N+17.367341655669733%、188新/0丢、1095上移/0下移，双CI正，历史资格层RETAIN；same-policy整体R+9.870848708487067% / N+10.28369953005015%，2382hits距2385门槛差3，整体gate false，条件Testing不触发（0次），整体目标未完成。3.3完成表示已执行门禁和条件决策，不表示双10达成或Testing已运行。证据与预算见history-exclusion-stage-closure.json及研究报告；下一步仅保留层bad case与新机制问题界定，尚未选择方法，不自动新增预算。
