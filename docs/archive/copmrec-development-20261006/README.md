# CoPMRec 开发阶段归档（2026-10-06）

## 归档决定

用户于 2026-10-06 将 **v5.3 的完整方法与训练配方选为正式 CoPMRec**，以 LIGER 为核心基线，重新组织 Milestone 2 的正式实验。v0、v1、v2、v3、v4、v5 及其变体的做法、结果和阶段判断归入开发历史。

这次提升针对实现和固定配方，不针对既有训练成果：**v5.3 的开发训练 `hqw189d2` 和 Validation `6u6g62hk` 同样是历史开发运行，不计入新计划的任何已完成实验。** 新计划的训练、checkpoint 选择、Validation、Testing、基线对照和消融均须按照正式计划重新执行并登记新的 run。历史结果可以解释设计来源，不能填充新正式实验的完成状态或结果表。

此前阶段报告中的“停止 v5.3／保留 v5.2”属于当时预登记门槛下的阶段决定；原文保持不变。本次用户选择 v5.3 为正式实现的决定覆盖旧的活动版本选择，不反向修改旧指标、预算、统计区间或运行事实。

## 内容入口

- [2026-10-07 v4／budget 补充清理](versions.md#2026-10-07-补充清理v4-开发对照与-liger-budget)：用户追加认定的3个v4阶段LIGER开发对照与6个LIGER budget run已删除，44项效果原值已保存；[本轮回执](wandb-delete-v4-budget-20261007/receipt.json)与[核验](wandb-delete-v4-budget-20261007/verification.json)。之前“保留27个LIGER comparator”的范围已由用户追加授权更新，当前保留原hybrid矩阵18个run与固定上游9个run。
- [2026-10-07 删除前完整效果登记](versions.md#2026-10-07-删除前完整效果登记)：82 个现存开发 run 的 1,138 项 Recall/NDCG、贡献差值与区间；更早的孤立 Artifact 记录可取得的指标，缺失不补造。
- [删除前 run/Artifact 完整快照](wandb-delete-20261007/deletion-plan.json)、[训练 Validation NDCG 历史](wandb-delete-20261007/validation-ndcg-history.json)、[删除回执](wandb-delete-20261007/deletion-receipt.json)、[删除后核验](wandb-delete-20261007/verification.json)。
- [版本做法、效果和边界](versions.md)：统一整理 v0 至 v5.4，包括被撤回或未运行的设置。
- [开发 run 注册表](development-run-registry.json)：65 个 CoPMRec 历史 run 和 27 个明确单列的 LIGER 比较 run，记录 entity、project、ID、版本、阶段及证据路径；包含旧 Milestone 2 主矩阵和 Milestone 3 机制 run。
- [原证据快照清单](history-evidence/manifest.json)：113 份原 Markdown、主要结构化审计及旧 Linear 内容的逐字节快照，包含来源路径、大小和 SHA256。
- [原版本文档快照](history-evidence/copmrec-versions.md)：保留当时版本边界与手动入口说明；其中启动命令属于历史复现记录。

实现代码及配置的快照由同目录的运行代码归档另行记录；历史快照中的命令不保证在清理后的活动代码树中仍可执行。正式执行入口、参数和实验顺序以新的活动计划为准。

## run 与清理边界

注册表中 `role=development` 的 65 条是本次明确版本链的历史运行，包括旧 v0 正式矩阵、本次开发、诊断续训和零预测失败启动。旧 run 曾经是当时有效正式结果的事实保留；这次版本重置后它们都标记 `eligible_for_new_formal_completion=false`，不能用于完成新的 v5.3 计划。W&B 的清理执行结果由主任务另外记录；本注册表不是删除或归档已完成的回执。

`role=development_comparator` 的 27 条是用于历史比较的 LIGER 训练／推理运行，包括旧九单元 original hybrid 矩阵，默认不列入自动清理目标。本轮新基线表也不能复用它们作已完成 run。特别是 `35ig0tz6` 和 `042139al` 为其他研究记录共用的基线锚，不能因它们曾参与 CoPMRec 比较就当作 CoPMRec 模型删除。

Beauty／Sports／Toys 三数据集的9个固定 embedding、quantizer、SID 输入依赖保留，包括 `3jtt9mpa`（Beauty 内容向量）和 `dq77e3wo`（Beauty SID）；LETTER、TIGER、SASRec 及其他无关 baseline 的运行与产物不属于本归档的清理范围。旧研究状态快照中引用的更早冻结 decoder／ranker 路线单列为范围外历史引用，不仅凭快照出现 ID 就自动处理。

本归档只保存已有证据并重新标记用途，没有启动训练、推理或评分，没有改写原证据，没有通过新代码或新归档补造早期 run 的源码 provenance。

### 2026-10-07 用户追加的 W&B 删除授权

上述清理边界描述的是 2026-10-06 归档时的范围。2026-10-07 用户明确要求删除 CoPMRec 开发 run 和 Artifact，效果落入本历史版本文档；本轮实际删除范围扩展为原 65 个 run ID 加 18 个已核实的早期 ranker/decoder run。共 82 个现存 run，`uzmkrfoa` 已缺失；逐项核实的开发 Artifact 共 253 个版本，包含 10 个原创建 run 已缺失的 joint 开发产物。早期路线保存为历史事实，不恢复为活动研究依据。

本次先保存完整效果和产物元数据，再按确切 ID/版本执行删除；最终执行数量以删除回执和独立核验为准。27 个 LIGER comparator、9 个固定上游依赖，以及 LETTER/TIGER/SASRec 和其他范围外 run/Artifact 继续保留。远端本地 checkpoint、日志与缓存不在此次 W&B 清理范围。历史 W&B 链接可能已失效，追溯使用本地快照和文档；原归档事实不会因线上删除而改变。

## 阅读结果时的约束

不同 split、不同历史排除规则、不同训练预算及单模型／pool 部署的指标不能直接横向排序。`042139al` 的 LIGER dense 指标取自其 dense trace，不能替换成 hybrid summary。v4 的续训／pool 结果也不能当作 v5.3 从头 50k 单 checkpoint 的效果。

报告中的用户配对 bootstrap 区间条件于固定 checkpoint，未估计训练 seed 波动；重复使用的 Beauty Validation／Testing 是开发证据。没有完成的设置保留“未运行”，smoke、CPU probe、loss／梯度诊断不计作完整推荐效果。
