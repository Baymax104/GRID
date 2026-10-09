# CoPMRec 无 native-view CE 正式版本

## Why

用户决定在无历史排除版本上删除 native-view CE，采用监督内容与商品残差联合评分的训练目标作为新正式版本。Beauty42 A3 已有实证提供选型依据；共享内容参数接受两种不同目标并非计算错误，本次是明确的设计取舍。

## What Changes

- 增加独立正式模型与薄训练/推理配置，保留 SID CE、完整目录 joint CE、mixture NLL，取消 native 评分及其辅助损失。
- 训练、raw Validation 和最终 dense 推理均不排除历史商品；共享历史/目录残差、cold 规则、50k 日程和 own-best 选点保持。
- 增加双卡训练、单卡推理的根脚本，支持 dry-run、notes 与额外 override。
- 独立版本/恢复身份；保留 v0、原 v1、A3 及其真实 run/Artifact 来源，记录用户选择与非独立确认边界。
- 同步研究文档、机器状态与 Linear 项目/里程碑，新增正式运行预算为零。

## Capabilities

### New Capabilities

- `copmrec-joint-only-formal`: 无 native-view CE 的正式训练、无排除推理、独立来源与版本登记。

### Modified Capabilities

无。旧正式及消融入口保持真实协议。

## Impact

涉及 `src/recommendation/copmrec/`、model/experiment 配置、根脚本、聚焦 CPU 测试和研究/Linear 文档。不新增依赖，不启动训练/推理，不追改 W&B，不改变已完成矩阵或消融的指标与累计成本。
