# CoPMRec 当前计划｜v1 开发

2026-10-09 用户决定重新进入开发阶段。**v0 = 原正式 v5.3，训练不排除、推理排除；v1 = 训练和推理均不排除**。训练与 raw Validation 未改变，当前变更是最终排序规则。

当前权威安排为 [开发计划](../docs/copmrec-development-plan-20261009.md) 与 [版本定义](../docs/copmrec-v0-v1-definition-20261009.md)，机器状态为 [research-state.yaml](../research-state.yaml)。本轮仅完成文档与 Linear 计划对齐，新增运行预算 0 / 0 / 0。

v0 主矩阵 9/9 与 M3 五臂 250k 保留。v1 初始证据为 Beauty42 `hc8oct43`，匹配无排除 LIGER dense 为 `lu8oct42`，均使用各自原 own-best，未增加训练。v1 不是已完成的三数据集三 seed 主矩阵；M3 有排除结果不改标签作为无排除指标。

下一步为开发版本元数据/入口方案，以及现有证据与论文表述对齐，尚未实施；不自动重训、不重置旧预算、不扩展消融 seed/数据集。实验 issue 沿用统一模板与实际命令格式，训练双卡、推理单卡。

旧当前计划按原字节存于 [本次切换前快照](../../GRID/docs/evidence/copmrec-development-version-reset-20261009/before/research/ideas/current-plan.md)，其中时间快照只描述发生时状态；本页及新开发计划为当前安排。
