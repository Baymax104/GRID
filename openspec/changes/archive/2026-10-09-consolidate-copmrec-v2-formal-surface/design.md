## Context

v2 三目标类已验证，但默认入口和消融仍为 v5.3。用户授权清理旧实验并要求 Linear 只显示当前计划。主结果固定三数据集三 seed；消融固定 Beauty42。

## Goals / Non-Goals

**Goals:** 正式入口收敛、旧运行源码归档、M2 重置、M3 三训练预算和当前模板、W&B 依赖审查后清理。

**Non-Goals:** 本次不运行正式训练/推理，不修改 LIGER 基线和上游数据，不提交 Git。

## Decisions

- 将已验证 v2 类升为 CoPMRec 默认模型；需要的训练协议、共享残差与 mixture 组件归入 CoPMRec，删除旧 native/history/版本别名入口。
- 三个训练控制为 no_mixture（gate 固定 0.5）、no_residual（历史和目录均为零）、no_joint_ce（mixture 仍可监督目录）。保留同一初始化、预算、输入、raw dense own-best 与 Top10 协议。后三目标消融只证明删除目标时的条件变化，不声称 mixture 优于任意替代监督。
- hits 复用四份预测，residual 对 Full 做 2×2 单卡评分，prefix 对 Full/no_mixture 逐层测量；均无训练。结果使用观测值、带符号差与样本配对区间。
- 在线旧 issue 作废并移出项目、清空旧描述；连接器无物理删除 issue 能力，必须记录实际操作，不能宣称硬删除。新计划使用新 issue。
- W&B 先保存配置、summary、validation history 和 artifact 生产/消费关系，保护未删除 run 的依赖，按准确 ID 执行。系统 history/events 若 SDK 不支持单独删除，记录服务端剩余。

## Risks / Trade-offs

- [历史 URL 删除后失效] → 文档保存原 ID、摘要与 manifest，node1 原文件不删；失效链接明确标记。
- [清理影响基线] → 生产者范围和 outside_users 双重门禁；复核基线与上游 digest。
- [删除 loss 后 DDP 未用参数] → gate/residual 按控制冻结，聚焦梯度与参数组测试。
- [旧 dirty 工作区混入] → 先复制每个改动文件并保存哈希，只改 CoPMRec 范围。

## Migration Plan

先归档，再改源码与验证，再更新 Linear，最后删除已审查 W&B 产物并回读；正式实验运行数保持零。
