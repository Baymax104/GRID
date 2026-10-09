## Why

六条完成训练中 MIR 在两数据集均落后固定第二层，48个未完成条件已停止推进。现有曲线不能区分 item 评分不足与有限搜索预算漏失，需要复用既有 checkpoint 的有界决策证据。

## What Changes

- 新增独立的 checkpoint 推理审计模块，固定用户样本，对全部目录 item 计算精确边缘概率和目标排名。
- 同一模型、用户比较 Q=64/128/256、S=4096 的搜索结果、剩余质量和解析深度贡献。
- 新增确定性无标签抽样、keyed evidence writer 配置、四个 checkpoint 串行单卡入口。
- 不训练、不修改 MIR 或 baseline 目标，不启动GPU实验；结果不能替代正式全量多seed测试。

## Capabilities

### New Capabilities

- `item-resolution-score-search-audit`: 有界的评分与搜索误差分离证据。

### Modified Capabilities

无。

## Impact

新增 recommendation 子模块、data 抽样与证据校验模块、Hydra配置和脚本。复用统一 src.main inference 入口与共享辅助tensor writer；不修改既有训练数据流。
