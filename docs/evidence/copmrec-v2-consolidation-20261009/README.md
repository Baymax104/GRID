# CoPMRec v2 正式运行面与计划收敛回执

执行日期：2026-10-09。授权范围为默认代码、当前 Linear 计划和指定历史 W&B 清理；本次正式训练、Testing、diagnosis 启动数均为 **0**，未创建 Git commit。

## 默认代码与来源

默认模型为 `src.recommendation.copmrec.CoPMRec`；默认训练/推理脚本为 `copmrec_train.sh` / `copmrec_inference.sh`。目标固定为 SID CE、joint catalog CE、mixture NLL 三项等权；共享历史与目录残差、cold residual=0；训练、Validation、Testing 全部无历史排除，无 native view。

旧 native/history 控制、v1 评估、joint-only 临时别名、预算续训与混合终排入口退出运行面；底层旧 alpha 扫描和生成控制支路已移除，旧参数只保留明确拒绝校验。必要的共享逻辑收敛到 `src/recommendation/copmrec/`，原 LIGER baseline 保持原行为。

[源码归档](../../archive/copmrec-v0-v1-runtime-20261009/README.md)含 **74** 个逐字备份文件，SHA256/大小已逐项核验；其中 **51** 个旧文件从活动运行面移除，其余为修改前副本。

已执行 `./mutagen_sync.ps1 flush` 并回读 status：三个 session 均为 Watching for changes，无 conflict。本地与 node1 的 288 个白名单源码文件 aggregate SHA256 一致：

```text
3e486ce5d7a206c13bee21f108d1e806f862712a3cc1afd5a6535bcd06bd8a00
```

来源记录为 verified / local-workspace，工作区为 dirty；没有以远端 Git 判断版本。见 [本地来源](source-local.json)、[node1 来源](source-remote.json)。数据、环境、日志、checkpoint、预测与缓存不在源码同步清理范围内。

## Linear 当前计划

M2：[BMX-116](https://linear.app/baymax104/issue/BMX-116)及九个主单元均 Todo，进度0%；Beauty/Sports/Toys × seeds42/200/2026，未来9次 DDP2 scratch训练/450k与9次单卡 Testing。

旧 M3 的17个 issue 已 Canceled、description清空、labels清空并解除项目/父任务关联；过时历史过滤对照 BMX-147 同样处理。连接器不提供物理删除 issue 接口，因此保留项目外的作废空壳及 Linear 系统变更历史，不再作为当前计划。

新 M3 父任务为 [BMX-157](https://linear.app/baymax104/issue/BMX-157)，全部 Todo，进度0%。

| Issue | 当前实验 | 新增训练 |
| --- | --- | --- |
| [BMX-158](https://linear.app/baymax104/issue/BMX-158) | A1 去除 mixture NLL | Beauty42，50k |
| [BMX-159](https://linear.app/baymax104/issue/BMX-159) | A2 去除共享历史与目录商品残差 | Beauty42，50k |
| [BMX-160](https://linear.app/baymax104/issue/BMX-160) | A3 去除 joint catalog CE，保留 SID CE 与 mixture NLL | Beauty42，50k |
| [BMX-161](https://linear.app/baymax104/issue/BMX-161) | M1 四份 Testing bundle 的命中与排名贡献分解 | 0，零模型 forward |
| [BMX-162](https://linear.app/baymax104/issue/BMX-162) | M2 固定 Full checkpoint 的残差2×2干预 | 0 |
| [BMX-163](https://linear.app/baymax104/issue/BMX-163) | M3 Full/A1 真实前缀概率分析 | 0 |

Full仅复用M2的 BMX-120；M3总预算三次训练/150k、三次单卡Testing和四个单卡机制run，没有额外Full训练、多seed或多数据集消融。新A3定义与历史删除native CE的A3不同，以v2 issue为准。

实验单元使用统一十章模板，包含复制命令、结构化结果空值、实际来源登记要求、成本与交付标准。group沿用 `paper_main_copmrec_<dataset>`、`paper_ablation_copmrec_beauty`、`paper_mechanism_copmrec_beauty`；训练DDP2，推理/机制单卡，各实际run独立tmux。未生成的own-best和bundle保持空值并由shell门禁阻止执行。

原LIGER九有效单元、Beauty42无排除dense对照 lu8oct42 和其他baseline状态保留。项目、三份计划/registry文档及方法协议issue均改为当前v2。见 [issue最终回读](linear-final-readback.json)、[项目成员和状态](linear-project-final.json)、[当前文档回读](linear-documents-final.json)。

## W&B 删除与实际剩余

先保存旧config、summary、validation曲线、Linear内容、源文件及artifact manifest与引用关系，再执行删除。

- 已删除 **42个run**：41个历史CoPMRec主结果、消融和迭代run，加上已作废的历史过滤内部对照 ldi54f1o。
- 已删除 **83个普通artifact版本**；最终回读：目标run剩余0，普通目标artifact剩余0。
- **86个系统history/events artifact仍保留**。W&B服务端拒绝删除，错误为 `cannot delete system managed artifact`；包括14个原run关联对象与72个已无生产者的系统对象，不能报告为已删除。
- 其他 **94个run** 的存储身份以及 **6个固定SID/content输入** 的digest已回读核验。未删除W&B项目，也未删除node1重资产。

见 [删除计划和依赖](wandb-deletion-plan.json)、[删除回执](wandb-deletion-receipt.json)、[最终W&B核验](wandb-final-verification.json)、[系统孤儿盘点](wandb-collection-orphans.json)。

## 历史结果与验证

[v0/v1历史实证文档](../../../../research/docs/copmrec-v0-v1-results-archive-20261009.md)保存旧主矩阵均值/标准差和逐run原始指标；[完整历史状态](../../../../research/docs/archive/copmrec-v0-v1-20261009/research-state.yaml)保存旧定义与累计预算。旧A3结果用于v2选型，不能当作新正式独立确认；删除后的W&B身份只作离线追溯。

聚焦验证 **126 passed**，涵盖三目标梯度、冻结/恢复、cross-variant拒绝、无排除与cold资格、keyed机制分析、配置/脚本、LIGER与artifact加载回归；4个依赖包既有弃用提示。Ruff检查通过。

从Linear最终回读提取 **28段实际命令**，以uv stub验证shell解析、Hydra compose、resolved config、notes/group/来源/单双卡，未知checkpoint和bundle空值门禁通过；不实例化Trainer、不下载真实artifact、不启动正式实验。命中分析命令已修正为 `+prediction_paths=...`。见 [命令核验](issue-command-verification.json)。

验证命令和汇总见 [最终验证回执](verification.json)。OpenSpec `consolidate-copmrec-v2-formal-surface` strict校验通过。
