## Why

现有两次训练已经覆盖原 LIGER 与联合混合训练，但现有解码臂没有形成完整的训练来源×解码权重2×2设计，无法判断训练增量是否依赖混合解码。用两个新预测补齐矩阵，无需重新训练。

## What Changes

- 为联合模型增加仅推理的固定混合系数覆盖，并写入候选trace。
- 复用原训练×0.5与联合训练×1，新增原训练×1及联合训练×0.5。
- 冻结2×2配对统计与停止规则，完整实验仍由用户手动启动。

## Capabilities

### New Capabilities
- `liger-training-decoding-factorial`: 支持和审计两种训练来源与两个固定解码权重的完整交叉。

### Modified Capabilities
无。

## Impact

联合LIGER推理配置、候选processor选择、trace元数据、测试、研究状态和人工运行协议；不新增训练或依赖。
