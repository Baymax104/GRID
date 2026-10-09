# CoPMRec v1 无历史排除消融评价

## Why

用户要求先按 v1 重新进行消融评估。v1 与 v0 训练相同，但五臂现有 Testing 与 M1 均使用最终历史排除，不能直接作为 v1 指标。

## What Changes

- 增加五臂仅推理的无历史排除适配，保留原消融 checkpoint 恢复契约。
- 为 M1 增加明确的排名历史规则，默认保持旧行为；新增薄 experiment 和根脚本。
- 固定 Beauty/seed42，复用五臂 own-best，5 次单卡 Testing；Full 复用 hc8oct43，M1 六份 bundle 零 forward 分析。
- 独立核验来源、合法输出、用户/标签、四指标与配对统计，结果落入 Linear M3 和研究文档。

## Impact

新增训练 0、独立 Validation 0、Testing 5、M1 analysis 1（model-forward 0）。不改变 v0 默认协议，不重选点，不增加 seed/数据集，不运行残差/前缀诊断。
