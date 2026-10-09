# CoPMRec 开发阶段与版本登记回执（2026-10-09）

用户决定将原正式实现 v5.3 登记为 v0，重新进入开发阶段，以训练和推理均不排除历史的 v1 为当前开发版本。本轮仅更新文档、机器状态和 Linear 计划。

v0 训练本来不排除；v1 复用相同训练，关闭最终推理历史排除。Beauty42 hc8oct43 为已有无排除初始证据，lu8oct42 为同规则 LIGER dense 对照；既有 run/Artifact 和来源契约保持原身份。

- `before/`：本次修改前的相关本地文档原字节及研究状态。
- `linear-*-before.json`：Linear 项目与两个文档的修改前快照。
- `local-verification.json`：快照哈希、原研究证据 root 语义不变、代码/配置/脚本哈希不变与零运行预算检查。
- `linear-readback.json`：线上保存与回读一致性核验（完成后生成）。

原 v0 主矩阵 9/9、M3 五臂 250k 及失败成本保持；完成状态不撤回。v1 当前仅 Beauty42 无排除证据，不把 v0 旧表改标签为 v1。归档开发路线不自动恢复。

[开发计划](../../../../research/docs/copmrec-development-plan-20261009.md)、[版本定义](../../../../research/docs/copmrec-v0-v1-definition-20261009.md)。
