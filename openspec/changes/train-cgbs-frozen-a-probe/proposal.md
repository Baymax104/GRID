## Why

C128 单 seed 在线评分存在小幅收益，但 40k 未超过 A，增宽、门控和分阶段训练未建立稳定优势。需要固定已训练 A，区分主干退化与附加内容评分本身缺少增量。

## What Changes

- 从显式 A checkpoint 初始化，严格核验目录、参数和来源。
- 冻结所有 A 参数并关闭其 dropout，仅训练 mean pooling 后的 128 维 MLP 和逐层 mixture logits，最多 2k 更新。
- 保留独立 content loss 验证记录，提供统一入口的配置和手动命令。
- 用有限内存测试验证 C-off 与 A、参数冻结、checkpoint 和实际 Bash/Hydra 透传。

## Capabilities

### New Capabilities
- `cgbs-frozen-a-probe`: 已训练 A 上的有界内容增量验证。

### Modified Capabilities

无。

## Impact

新增推荐模型子类、模型/训练器配置、Artifact loader helper、聚焦测试与实验协议。不修改既有实验默认行为、不自动启动完整实验、不增加依赖。
