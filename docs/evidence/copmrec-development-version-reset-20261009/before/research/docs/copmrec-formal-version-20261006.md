# CoPMRec 正式版本定义（v5.3）

## 决定与证据边界

2026-10-06，用户明确将开发版本 **v5.3** 固定为 **CoPMRec 正式版本**；2026-10-07 明确论文正式 baseline 为 **LIGER hybrid**，**LIGER dense 仅为内部对照**。论文主比较为 CoPMRec 对 LIGER hybrid；v5.2 等版本仅属于方法演进历史，不作为取代 LIGER 的主基线。

本次完成的是方法冻结、代码入口整理和正式实验重新排期。新CoPMRec训练与Testing完成数均为 **0**；每单元仅训练（含validation选点）与Testing，不额外Validation。原LIGER9组已完成正式结果与原issue保持复用；CoPMRec开发run不得完成新单元。

推荐训练必须从头开始，不从开发 checkpoint 初始化、续训或合并模型。既有固定内容向量、SID 等上游产物可以作为技术输入继续使用，但必须登记其身份、哈希和生产协议；这不等于本轮推荐训练或推理已完成。

## 方法定义

CoPMRec 全称为 **Content-conditioned Prefix Mixture Recommendation（内容条件前缀概率混合推荐）**。保留既定问题：在固定内容、SID、骨干与匹配训练预算下，内容条件前缀概率建模能否改善候选可达性并提升最终推荐。

本轮正式实现冻结为 v5.3 的计算流程：

1. 四层 SID 历史按商品分组，从固定目录查取内容向量；将 SID embedding、内容投影、同一商品的协同残差及位置表示相加，输入共享 Encoder。
2. 取最后一个有效 SID token 的 Encoder 输出作为 query。历史最多 20 件商品、80 个 SID token。
3. 商品侧使用 `P(c_i) + r_i`；历史和目录共享同一协同残差表。残差初始化为零，训练出现商品可学习，cold 商品有效残差为零。
4. 训练使用四项权重均为 1 的损失：SID CE、全目录联合 CE、mass 内容条件前缀与合法生成概率的 mixture NLL、共享 query 的无目录侧残差辅助 CE。
5. mixture alpha 为从 0.5 开始学习的全局标量，不固定复制开发所得 alpha，也不增加用户级动态门控。
6. 辅助视图使用同一个 query 和同一次内容投影，仅在商品评分侧去掉残差。query 的历史侧仍包含残差，不宣称其为完整无协同信息模型。
7. 正式推理使用全目录联合 cosine logits `/ 0.07`，将有效输入历史中的商品分数设为负无穷后稳定取 Top-10；不把辅助分数再混入部署评分。

部署为 **dense**。Decoder、alpha 与 mixture 目标参与训练，正式 dense 推理不执行逐层候选生成。既有“减少 beam 候选遗漏”的机制线索属于问题动机；不得把新 dense 收益直接归因为推理时恢复生成候选，也不据此宣称生成检索效率优势。

## 冻结参数与匹配规则

| 项目 | 正式 CoPMRec 配置 |
|---|---|
| 输入 | 每数据集固定一份 SID、内容向量及数据 manifest |
| SID | 4 层，codebook size 256 |
| 历史 | 最近 20 件商品；有效完整 SID；右侧 padding |
| T5 | d_model 128，Encoder/Decoder 各 6 层，6 heads，d_kv 64，d_ff 1024 |
| 内容投影 | 固定输入维度以产物为准；隐藏层 768、512、256，输出 128 |
| 训练起点 | 推荐器随机初始化；gate 与商品残差零初始化；无上游推荐 checkpoint |
| 训练预算 | 50,000 个 optimizer updates；global batch 256；DDP 两卡，每卡 128 |
| 精度与优化 | FP32；AdamW；主参数 lr 0.0003，残差 lr 0.002，weight decay 0.035 |
| 调度 | warmup 2,500；单次 cosine 日程 50,000；不追加续训阶段 |
| 选点 | 每 500 步 raw full-catalog dense Validation；按 val/ndcg@10 选首次最佳 checkpoint |
| CoPMRec 正式评价 | 单卡、单进程；全目录联合 dense；有效输入历史排除；报告 Recall/NDCG@5/@10 |
| 正式范围 | Beauty / Sports / Toys × seed 42 / 200 / 2026 |
| 核心基线 | 复用既有正式LIGER hybrid；原双卡50k/global256、raw dense Validation-selected best与Testing；接入主表前核对输入与实际评价资格 |
| 内部对照 | 同一既有正式LIGER best的dense评价；无额外训练，不替代hybrid主表 |

LIGER hybrid保留原方法：original生成20、seen候选与全部cold并集、内容终排，原SID/content训练目标。原正式Testing使用Liger实现，不将新增HistoryExcludedHybridLiger写成既有结果的运行协议。该wrapper的CPU验证只是准备记录；接入新主表时核对历史资格差异，必要重评分复用原best、另列成本，不要求重训或回退原完成状态。

相同更新数和训练样本呈现预算不等于相同参数量、FLOPs 或 GPU 时间；新增残差参数及辅助计算须如实报告。辅助 CE 的组件归因与完整方法对 LIGER 的整体比较分别报告，不要求每个组件单独显著才能保留完整方法。

## 正式结果身份

新 CoPMRec run 必须记录 `evidence_phase=formal`、`formal_release_id=copmrec-v5.3`、`copmrec_version=v5.3`、数据集、训练 seed、split、源代码快照、输入身份和 checkpoint 来源。正式入口默认记录 `copmrec-formal-v53` tag；正式单元 ID 与 run 关联登记到研究状态和 issue。

W&B group统一为`paper_main_copmrec_${dataset_name}`，与其他主实验的`paper_main_<method>_<dataset>`一致；训练/Testing及各seed共用数据集group，版本和阶段不写入group，由config、notes及tags记录。

- 新训练 run 不得复用开发 run ID；各 seed 独立训练，不只改变推理 seed。
- 新推理只加载该正式训练按 Validation 选出的 checkpoint，记录引用和 SHA-256；不能用 `last.ckpt` 代替审计后的 best。
- 训练期间每500步validation与best选点保留；best审计通过后直接Testing，不要求独立Validation run。主结果与待准备消融均使用训练→best审计→Testing流程。
- LIGER hybrid的9个配对单元复用原正式训练、own-best、Testing和指标，BMX-132与BMX-133～141恢复原Done/有效；不要求新训练、独立Validation或重复Testing。开发v4/budget对照仍不复用。既有best的dense内部评价单列9次Testing、0训练/0Val，输出标记`evidence_phase=internal`、`copmrec-internal-v53`。所有既有正式baseline接入主表前核实际比较协议；原完成不等于新CoPMRec完成。
- fresh run 仅表示冻结版本后的新执行，不表示此前已消费的 Beauty Testing 重新成为盲测；各数据集 Testing 使用史继续登记。
- 运行效果、选点、合法输出和指标必须由真实产物复核。准备完毕、dry-run、probe、同步成功均不等于正式结果。

完整运行需用户授权。2026-10-07用户另行授权在node1以tmux启动Beauty/Sports/Toys seed42三个双卡训练，已启动3、完成0；未启动Testing或其余seed。启动回执见GRID `docs/evidence/copmrec-tmux-train-launch-20261007/`。

## 文档入口

- [正式实验计划](copmrec-formal-experiment-plan-20261006.md)
- [开发阶段研究状态原文](archive/copmrec-development-20261006/research-state-before-formal-v5.3.yaml)
- [开发阶段计划原文](archive/copmrec-development-20261006/current-plan-before-formal-v5.3.md)
- [方法与开发效果历史](../../GRID/docs/copmrec-versions.md)

历史方案、开发数值和当时的晋级/停止决定用于追溯，不作为本轮已完成实验、剩余预算或新结果。
