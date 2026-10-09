# LIGER 正式基线恢复回执

2026-10-07，按用户纠正，撤销将已有 LIGER 正式主矩阵当作 CoPMRec 开发实验重置的改动。

## 恢复结果

- BMX-133～141共9个正式 LIGER hybrid单元及父任务BMX-132恢复原标题、原描述、Done与原标签“有效／基线”。
- 原描述中的训练／Testing命令、run ID、best checkpoint、原指标及历史notes逐字恢复，不新增独立Validation、训练或Testing。
- 10个issue均已在线回读，四项相等检查全部通过，见 `linear-readback-verification.json`。
- 新CoPMRec仍为0/9完成，禁止复用其开发run；LIGER原9/9完成与本轮CoPMRec新增完成分开登记。

## 只读核验

`wandb-live-audit.json`保存18个现存run的config、相关summary、9组checkpoint Artifact metadata/digest与Testing消费lineage。全部run均finished；Testing为原Liger／original hybrid、generation candidates 20，原NDCG@10与Recall@10和现存summary在1e-6内一致，原best文件可访问、Artifact为COMMITTED且selection=best。

原训练配置均为max_steps 50000、双卡、每卡batch128、FP32。历史summary的trainer/global_step为49999，不能把这一日志计数或best step单独写成新的终态full-state审计。本次继承原issue已有有效性审核，补做存在性与配置/指标核对，不重新执行完整数据评价或独立复算。

原Testing使用src.recommendation.liger.Liger，未使用后来新增的HistoryExcludedHybridLiger。接入新CoPMRec主表前仍核实际split、keys与历史资格；必要重评分复用原best、另列评价成本，不据此回退已有issue或要求重新训练。本次没有增加或启动该重评分。

## 计划纠正

主配对新增由18训练／900k更新／18完整Validation／18Testing，改为CoPMRec的9训练／450k更新／9完整Validation／9Testing。原LIGER9训练／9Testing复用。dense内部另列9Testing、0新训练／0Validation，命令待按既有best准备；消融仍待准备。

研究状态、当前计划、正式版本定义、正式实验计划、Linear计划和Run Registry同步纠正。模型代码、训练配置、数据与W&B run均未修改；无新实验、无Git提交。
