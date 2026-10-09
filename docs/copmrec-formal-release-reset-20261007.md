# CoPMRec v5.3 正式化与实验重置回执

## 当前正式身份

用户确定 **v5.3 为 CoPMRec 正式版本**，并于 2026-10-07 明确 **LIGER hybrid 为论文正式核心基线，LIGER dense 仅作内部对照**。当前代码、研究状态、Linear issue 和正式实验计划已按此统一。

CoPMRec 保留 v5.3 的共享历史/目录残差、全目录联合 CE、mixture NLL、权重为 1 的无目录残差视图 CE、learned global alpha 与全目录联合 dense 部署。正式入口为 `copmrec_train.sh` / `copmrec_inference.sh`；正式 checkpoint 加入发布来源契约，不接受开发 checkpoint 完成新正式单元。

2026-10-07按用户纠正，LIGER hybrid复用原9组正式训练/Testing及原best，BMX-132与BMX-133～141恢复原内容和Done/有效。原Testing使用Liger、original gen20、cold并集与内容终排；新增HistoryExcludedHybridLiger仅为已准备代码能力，不冒充旧run的实际实现。

## 归档与清理

- [历史版本文档](archive/copmrec-development-20261006/versions.md) 保留各版本方法、效果、负面证据与解释边界。
- 代码清理前保存 371 个原始文件或阶段快照及 SHA-256，随后退出 297 个活动路径；正式实现必要内部基类、原 LIGER 和上游工具保留。
- 113 个原始文档或主要证据文件保存在 `history-evidence/`；原 Linear issue、实验计划和 Run Registry 另有完整清理前快照。
- 开发 run 登记包含 65 个 CoPMRec run 和 27 个 LIGER comparator，均不能完成本轮主配对。64 个现存 CoPMRec run 已加历史标记；`uzmkrfoa` 在本轮清理前已查无记录。
- W&B 清理采用历史标记与独立视图；没有删除 run 或 Artifact。最终复核所有现存目标的 Recall/NDCG 指标保持不变。

## Linear 与新实验计划

最初24个相关issue重置范围错误地包含M4的10个LIGER父/子任务，现已撤销这部分。M2的10个CoPMRec父/子任务及M3的4个任务保持Todo；M4的BMX-132与BMX-133～141逐字恢复原标题、描述、Done/有效。其他baseline原状态保留；进入新主表前核实际比较协议。[恢复与核验回执](evidence/liger-baseline-restore-20261007/README.md)。

| 工作包 | 新训练 | 完整 Validation | Testing | 当前完成 |
| --- | --- | --- | --- | --- |
| CoPMRec v5.3：三数据集 × seeds 42/200/2026 | 9 | 0，训练内选点保留 | 9 | 0 |
| 正式 LIGER hybrid：相同九条件 | 0，复用原训练 | 0，原训练内选点 | 0，复用原Testing | 9 |
| 同既有 LIGER checkpoint 的 dense 内部对照 | 0 | 0 | 9，内部单列 | 0，命令待准备 |

主配对新增为 **9次CoPMRec从头训练、450k更新、0额外Validation、9次Testing**；训练内每500步validation及best选点保留。原LIGER9组正式结果复用，dense内部另加9Testing。新CoPMRec18个train/Testing引用保持空，开发run不能复用。9个CoPMRec主结果issue统一六段模板、只保留训练与Testing命令；[模板更新回执](evidence/copmrec-issue-template-20261007/README.md)。

AuxOff、ResidualOff、MixtureOff 三项固定消融列为 `pending_preparation`，本次未实现或运行；其建议的 9 次训练和评价不计入已准备主配对。上游技术产物可按计划复用，实际 digest、数据 manifest 与消费身份在正式运行前逐单元核验。

- [Linear 正式实验计划](https://linear.app/baymax104/document/copmrec-v53-正式实验计划liger-hybrid-主基线与内部对照-09c9588d3f47)
- [Linear 正式 Run Registry](https://linear.app/baymax104/document/正式-wandb-run-registrycopmrec-v53-6c30a2524055)
- [新正式 W&B 视图](https://wandb.ai/baymaxam/GRID?nw=bdorsjboze0)
- [历史开发 W&B 视图](https://wandb.ai/baymaxam/GRID?nw=prtp6c5ajpj)
- 第一组：[CoPMRec Beauty/42：BMX-120](https://linear.app/baymax104/issue/BMX-120) 与 [LIGER hybrid Beauty/42：BMX-133](https://linear.app/baymax104/issue/BMX-133)，各 issue 含独立阶段命令。

## 验证与执行边界

[代码验证回执](evidence/copmrec-formal-release-20261006/local-code-verification.json) 记录核心组 327 passed、来源组 22 passed、入口清理后组 93 passed、最终 hybrid/正式入口组 86 passed；各组有重叠，不合计为唯一测试数。全测试收集 1048 项成功，Ruff、脚本契约和 OpenSpec strict 通过。

[最终命令复核](archive/copmrec-development-20261006/final-command-review.json) 验证三段计划命令、三数据集训练分支、单卡 Validation/Testing、9 个内部 dense 命令，以及 SHA-256 的 Hydra 字符串传递。最后仅修订准备状态文字，三个命令块保持逐字不变；[Linear 最终计划复核](archive/copmrec-development-20261006/linear-final-plan-verification.json) 已确认在线文档含最终命令和 hybrid 主基线角色。

正式化代码交付时Mutagen flush/status已成功，三个session Watching且无conflict；此为当时同步回执。本轮基线恢复仅文档与Linear更新，无代码同步需要。新CoPMRec训练/Validation/Testing仍为started 0/completed 0；原LIGER9组正式结果已完成并恢复。无新实验、无Git提交。
