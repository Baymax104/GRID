# CoPMRec 开发计划（2026-10-09）

> 当前追加评价已按用户授权完成：Beauty42五臂5次单卡Testing及M1零forward分析，0新训练；见 [v1消融实证](copmrec-v1-ablation-evaluation-20261009.md)。下文零运行预算描述版本文档整理阶段，本次新增批次已执行完毕，未追加其他运行。

当前阶段为 **v1 开发**；原正式实现 v5.3 作为 **v0 参考版本**保留。此轮交付是版本定义、研究状态与 Linear 计划对齐，不启动训练、推理或 diagnosis。

## 当前问题与方法边界

沿用“利用商品内容改善推荐概率建模”的研究问题，保留内容条件前缀混合、共享历史/目录残差、完整目录 CE 与 native-view CE。v1 将最终历史排除从默认推荐协议中去掉；训练本来不排除，损失和参数不变。方法收益、评分/历史规则差异与 teacher-forcing 机制观测分别报告，dense 指标不能直接证明自由 beam 候选恢复。

可检验主张为：在固定输入、预算、own-best 和共同 dense 无排除协议下，完整训练方法与 LIGER dense 的推荐指标存在何种差异。原 LIGER hybrid 保留为同类方法参照；其原生候选策略和历史规则明确披露，不能把部署差异全部归为训练机制。

## 已完成证据与累计成本

| 证据 | 已完成事实 | 当前用途 |
| -- | -- | -- |
| v0 主矩阵 / BMX-116 | CoPMRec 9 训练 / 450k updates / 9 Testing；原 LIGER hybrid 9 配对复用，全部核验 | 原正式完整方法比较；不能称为 v1 无排除主矩阵 |
| v0 M3 / BMX-117 | Beauty42 五臂 5 训练 / 250k / 5 Testing；4 diagnosis，M1 零 model-forward | 训练组件及机制实证，保留当时有排除的评价口径 |
| LIGER dense 有排除 / BMX-147 | `ldi54f1o`，1 Testing / 0 新训练 | 评分与历史规则的内部对照；其余 8 条件未运行，不列自动待执行预算 |
| v1 初始推理 / BMX-148 | `hc8oct43`，同一 v0 checkpoint 的 1 Testing / 0 新训练 | 无排除 v1 初始证据；失败的 `hc8oct42` 为 pre-forward 工程 attempt，不计完成 |
| 匹配 LIGER dense 无排除 / BMX-149 | `lu8oct42`，1 Testing / 0 新训练 | 与 v1 相同历史规则的单条件对照 |

已完成两类 CoPMRec 训练共 14 个 / 700k updates，不因版本重新编号清零。原 M3 的独立选点、失败成本与 source metadata 保留；不自动重跑五臂，不扩展消融的 seed 或数据集。无排除证据目前只有 Beauty/training seed42，不将原多 seed 结果写成 v1 已验证的多 seed 性能。

## v1 初始对照：Beauty / seed42

| 指标 | v1 无排除 `hc8oct43` | LIGER dense 无排除 `lu8oct42` | 相对差（以 LIGER 为分母） |
| -- | --: | --: | --: |
| Recall@5 | 0.04538747 | 0.04252560 | +6.73% |
| NDCG@5 | 0.02990916 | 0.02702803 | +10.66% |
| Recall@10 | 0.07092072 | 0.07141260 | -0.69% |
| NDCG@10 | 0.03814285 | 0.03639071 | +4.81% |

N=22363，固定 own-best，0 新训练、0 独立 Validation；用户配对 bootstrap 2000 次、PCG64 seed42、95% pointwise 区间。NDCG@10 差为 +0.00175214，CI [-0.00006695, +0.00365801]；Recall@10 差为 -0.00049188，CI [-0.00371149, +0.00272772]。保留 NDCG@10 为主要指标，Recall@10 为方向保护；区间跨零不能表述为等价或已确认总体提升，不将 @5 事后提升为主要标准。

两种模型自身开启/关闭历史排除的差值均已有实测，说明最终排除影响这组数据上的推荐结果；这是部署规则的实证，不证明 CoPMRec 特有训练机制。v1 采用无排除协议是用户确定的方法边界，不依据 Testing 再选 checkpoint。

## 当前安排与下一步

1. **本轮文档对齐（已完成后核验）**：登记新 v0/v1、更新机器状态、当前计划、版本索引、看板与 Linear 项目/里程碑；保留已完成 issue 状态和原证据。
2. **后续入口与元数据方案（尚未实施）**：将开发版本标签与旧实现/来源契约分开设计；明确训练仍双卡、推理单卡，沿用 `paper_main_<method>_<dataset>`、`paper_ablation_copmrec_<dataset>`、`paper_mechanism_copmrec_<dataset>` 的 group 格式。未来实验 issue 继续沿用现有统一模板，写入可独立复制的完整命令、选定 URI/SHA、seed、split、输出目录和物理 GPU 映射。
3. **后续证据与论文对齐（尚未实施）**：v0 原正式表、v1 无排除表、共同 dense 规则对照和 M3 机制表分别标注实际协议。若需要 v1 对应的消融指标，先判断既有 own-best/bundle 可复用范围；有排除 Top10 不等于无排除结果，不能改标签冒充，禁止为同一训练条件重复训练。

本轮新增运行预算为 **0 训练 / 0 Testing / 0 diagnosis**，未追加正式实验矩阵，不自动启动其他数据集、seed、消融或 LIGER dense 条件。后续若用户要求补充运行，再固定具体条件、总运行数和决策价值；现阶段无需为版本登记追加实验。

已有证据支持报告各指标原值、带符号差和区间；NDCG@10 差异仍有统计不确定性，保留主张边界，不据此新增模块或改变主要指标。若后续匹配证据与主张不一致，则收缩相应性能或机制表述；若仍不确定，明确未决范围，不无限增加 seed/数据集。实验完成依据合法性、来源和独立复算，与结果方向无关。

## 权威记录

- [v0/v1 版本定义](copmrec-v0-v1-definition-20261009.md)、[当前计划](../ideas/current-plan.md)、[机器状态](../research-state.yaml)。
- [v0 原正式版本定义](copmrec-formal-version-20261006.md)、[v0 原正式计划](copmrec-formal-experiment-plan-20261006.md)、[来源登记](copmrec-formal-run-registry-20261007.md)。
- [均关闭历史排除的实证](copmrec-liger-both-history-off-20261008.md)、[CoPMRec 开关实证](copmrec-final-history-exclusion-control-20261008.md)。
- [GRID 版本索引](../../GRID/docs/copmrec-versions.md)、[本次文档变更回执](../../GRID/docs/evidence/copmrec-development-version-reset-20261009/README.md)。

编号与阶段变更仅改变当前研究组织；原 run、Artifact URI、运行时配置、源码快照和各次累计成本保持其真实身份。
