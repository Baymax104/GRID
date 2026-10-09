## 1. 推理模块与输入契约

- [x] 1.1 添加 FixedLogitPoolCoPMRec、九个 keyword 输入及双公共 loader identity 校验，分别使用原模型正常恢复与 strict load。
- [x] 1.2 添加 catalog / 来源链 / step / scale0 / cold residual / finite-state 校验，冻结 eval 成员，拒绝训练、wrapper checkpoint 及多进程推理。
- [x] 1.3 实现各自 history query 的完整 dense logits 固定0.5平均、稳定 Top10 和标准无标签 ModelOutput / pool_contract。

## 2. 配置与聚焦验证

- [x] 2.1 添加薄 model / inference experiment / 根推理脚本，固定双生产 URI / SHA、ckpt_path=null、单卡FP32，复用 common writer / lineage / metrics。
- [x] 2.2 CPU 测试完整目录平均而非 Top10截断、独立编码、ties、无标签、两原模型 strict restore、错误来源 / catalog / finite-state / 训练与多卡拒绝。
- [x] 2.3 CPU 通过真实公共 loader 的最小文件验证双 registry records / raw SHA；Hydra compose、shell语法、quoted双URI、notes / dry-run / extraoverride 透传及错误参数检查。
- [x] 2.4 运行聚焦旧模型兼容回归、Ruff 与 openspec validate add-copmrec-v4-2-fixed-logit-pool --strict，记录实现/验证来源。

## 3. 唯一验证与结题

- [x] 3.1 root 核验冻结source/control / 原v0链、实际成员评分配置、Mutagen三Watching及runtime源字节，完成单卡准备检查；准备结果不作为效果证据。
- [x] 3.2 启动唯一单卡完整 Evaluation，独立 raw用户/标签/SID/catalog/Artifact/source 审计，与固定 source/control / LIGER 比较指标、配对CI、warm/cold、固定分组与639/21724 trade-off。
- [x] 3.3 按预承诺整体双10%及LIGER两项CI门禁决定唯一条件Testing；增量CI仅决定pool主张强度。结题保留正负/不确定结果，不扫pair/weight/temperature、不新增训练或重置累计5/30000与Testing3上限。

结题证据：唯一 Evaluation `i4xwruok` 已终态并通过主审计及独立paired/cohort统计。相对 LIGER 的 R10 / N10 为 +9.262948% / +9.288610%，两项CI正但两个10%点门槛均未通过，因此预承诺条件Testing不触发、没有新Testing。pool对更强control净−15命中，两增量CI跨0；固定639恢复+24与其余21724损失−39均保留。仅关闭此次固定pair / 0.5等权完整logits问题，不扫描或追加预算；5训练 / 30000已耗尽，Testing保持1/3，whole goal未达且active。3.2 / 3.3完成表示验证与决策任务完整，不表示10%目标完成。详见 `docs/copmrec-v4-2-fixed-logit-pool-research.md` 与 `docs/evidence/copmrec-v4-autonomous-20261005/fixed-logit-pool-stage-closure.json`。
