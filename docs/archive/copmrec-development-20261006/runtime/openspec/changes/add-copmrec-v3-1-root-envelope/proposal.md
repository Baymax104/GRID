## Why

v3 已将自身 dense Top10 的首层遗漏降到 0，但后续仍遗漏 114 个目标，其中第二层 73 个。用户要求保留首层 Max，实施上一轮提出的 v3.1 路径先验修正，并提供单卡推理命令。

## What Changes

- 新增仅推理的 v3.1 decoder，复用 v3 best43500 checkpoint 和学习 alpha；首层输出完全一致。
- 第二 SID 层一次性将累计 root 分数修正为 Max/Mass 混合先验的归一化上包络；后续局部 Mass 概率保持原值。
- 新增有版本区分的路径 trace，分开记录条件概率、搜索增量、累计分数和 root 修正。
- 新增统一入口的单卡脚本与配置，验证后通过项目 Mutagen 同步；完整实验由用户启动。
- 2026-10-04 用户追加授权：在同一 v3.1 decoder 上通过已有 `inference_mixture_alpha` 字段固定推理 alpha=0.813。默认仍读 checkpoint 权重；仅准备一次同 checkpoint 推理对照，不训练或扫描。

## Capabilities

### New Capabilities

- `copmrec-root-envelope-decoding`: 保留首层 Max 的 v3.1 推理与可核验路径观测。

### Modified Capabilities

无。

## Impact

新增 decoder/model、薄配置、根脚本、测试和文档；LIGER 仅增加默认不变的 trace 工厂钩子。无新增依赖、训练目标或 checkpoint 格式变化。相关定位、预测、风险及阶段门禁沿用 docs/copmrec-v3-1-exploration.md；当前仅准备一次新推理，零训练，不重置历史预算。
