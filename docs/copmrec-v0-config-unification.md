# CoPMRec v0真实配置恢复与v2对齐

日期：2026-10-04。用户明确：BMX-116已完成的基础方法就是真正的v0，不能另设一个改变了评价协议的“新v0”。用户同时确认v2统一使用同一evaluation/testing数据链路，保留融合相关性终排。

## 原配置依据与修正

实时读取 [原v0训练run7y54j4m6](https://wandb.ai/baymaxam/GRID/runs/7y54j4m6) 的完整W&B配置并留存 [快照](evidence/copmrec-v0-config-unification-20261004/historical-v0-config.json)。`copmrec_v0_train`和`copmrec_v0_inference`恢复原`liger_joint_train/inference`的数据、模型与trainer装配，不再覆盖为开发selection/audit。

| 配置 | v0真实基础方法 | v2 |
|---|---|---|
| datamodule | FileDataModule | FileDataModule |
| 训练数据 | 完整training流 | 相同 |
| 验证数据 | 全evaluation | 相同 |
| 推理数据 | testing，保留user_id输出key | 相同 |
| 验证评分/选best | dense content，val/ndcg@10 | hybrid content＋相关性，val/ndcg@10 |
| 最终推理 | mass beam20＋全cold，content终排 | 同候选机制，融合终排 |
| 验证/训练日志间隔 | 500/50微批 | 相同 |
| 更新预算/累积 | 50000/1 | 相同 |
| train/val/predict每卡batch | 128/32/32 | 相同 |
| AdamW | lr0.0003、weight_decay0.035 | 相同 |
| scheduler | warmup2500、cosine总长50000 | 相同 |
| 主干 | d128，6层/6头，d_kv64、d_ff1024 | 相同，增加d-only head |
| dropout/temperature | 0.2，input0.5，projection0.2；temperature0.07 | 相同 |
| 排序loss与修正 | 无新增排序目标 | lambda0.01、beta0.5 |

v0恢复原DDP strategy与指标配置；v2保留已实现的DDP关闭buffer广播和distributed sampler，以及Recall/NDCG compute时同步全局总和/用户数，避免rank分片问题。这些属于分布式执行设置，必须记录，不能称为逐字节复刻历史运行。v0实际配置与原run相比，仅输出目录、版本化Artifact/task名称及当前新增但关闭的path_trace字段不同；数据、模型有效参数与训练设置一致。

v1/v1.1已结束迭代，继续使用其原selection/audit与2500间隔。为防止恢复v0后继承变化，已显式配置其data/trainer、评价元信息与val指标同步。四个训练/推理resolved行为配置与修改前完全一致；配置文件字节因显式声明而变化，推荐组件与训练权重未改。v1旧结果不重写，也不与新统一数据链路的v2曲线直接比较。

## 命令与评价边界

v0、v2根脚本与参数形式保持，新增默认行为已通过experiment配置生效；v2从零命令见 [v2文档](copmrec-v2.md)，notes已更新为full evaluation/testing，两个checkpoint引用仍为null。双卡物理GPU2/3对应逻辑[0,1]，500间隔不按GPU数缩短。

v0继续以原dense验证选点，v2以组件实际融合分数选点，这是公开的方法差异。最终应使用各自validation-selected best，在同一testing目录、相同用户/标签及SID指标口径下评价；不把v0的dense验证曲线当作v2 hybrid的同评分机制曲线。此次恢复不改写已有run、Artifact、best选择或历史核验结论。

旧selection/audit协议下已经保存的v2 checkpoint不能直接用于新full协议续训：已有评价history契约与FileDataModule不匹配时会拒绝恢复。正式运行仍由用户手动开始，本次没有启动、停止或续训实验。

## 验证

本地62项聚焦检查通过，覆盖v0/v1/v1.1/v2的脚本装配、配置、quoting、override、评分与恢复；Ruff check/format及两个相关OpenSpec strict通过。原v0完整配置与当前装配逐字段比对，v1/v1.1四个resolved行为配置相同；v0/v2训练与推理data组件完全一致。

node1核验八个训练/推理组合与本地resolved配置相同，八个Bash双进程入口通过语法、LF及参数替身检查，v2文档命令确认全evaluation/testing、500间隔、固定lambda/beta和两个checkpoint引用为null。33个运行文件指纹一致，Mutagen三个session均Watching且无conflict；原liger_joint_inference与版本化v0的data/model/trainer组件也完全一致。

初始与修改后配置分别留存于 [before-configs.json](evidence/copmrec-v0-config-unification-20261004/before-configs.json)、[after-configs.json](evidence/copmrec-v0-config-unification-20261004/after-configs.json)。完整核验记录在 [verification.json](evidence/copmrec-v0-config-unification-20261004/verification.json)，不构成推荐收益实验。
