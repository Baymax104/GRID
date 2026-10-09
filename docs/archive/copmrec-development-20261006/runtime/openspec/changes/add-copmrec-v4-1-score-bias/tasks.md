## 1. 实现

- [x] 1.1 添加独立seen scalar bias、v4显式weights-only初始化与严格v4.1恢复。
- [x] 1.2 添加两组optimizer/dense部署限制及cold安全，保持旧v4默认行为。
- [x] 1.3 添加薄model/experiment、根训练与单卡推理脚本。

## 2. 准备与验证

- [x] 2.1 CPU零初始化/RNG/三loss梯度/cold/恢复与旧v4组合回归。
- [x] 2.2 Hydra实际shell参数/quoting/dry-run透传、Ruff与OpenSpec strict。
- [x] 2.3 Mutagen/实际best字节及源hash/双卡production batch dry-run。
- [x] 2.4 五项路线门禁和新累计预算登记，实际启动唯一matched pair并记录句柄/source。

## 3. 效果评估

- [x] 3.1 两臂6000步/Validation best和固定step6000对照、单卡全量Evaluation独立配对复算。
- [x] 3.2 按预定正负/不确定门禁结题；若达到原两项门槛，固定winner单卡Testing并完整审计。

结题依据：两臂完整Evaluation为o7ycqky2/yqsmsdt1；bias相对control的R/N配对区间均跨0，两臂均未达到同split双10%门槛，故按原规则关闭本阶段，Testing条件未触发、实际新增Testing为0。任务完成表示实现与评估闭环，不表示全线程10%目标达成；新阶段2次/12000步及全线程5次/30000步训练预算已实际耗尽。见`docs/copmrec-v4-1-score-bias-research.md`与`docs/evidence/copmrec-v4-autonomous-20261005/score-bias-stage-closure.json`。
