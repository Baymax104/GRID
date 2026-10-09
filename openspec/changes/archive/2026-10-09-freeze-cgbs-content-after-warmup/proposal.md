## Why

用户报告attention512、10k+20k组合效果不佳，明确授权预训练后冻结内容分支的一次尝试。检验联合阶段的内容表示漂移是否妨碍Decoder利用该表示；验证content CE上升不证明此机制，仅作为假设。

## What Changes

- 独立训练子类，保留10000内容预训练更新，之后20000步固定内容评分路径。
- 冻结共享Encoder、共享SID embedding、attention与query MLP，关闭其dropout；只更新Decoder独有参数和mixture logits。
- 保留Adam并清除冻结参数梯度，防止共享embedding路径和历史动量继续更新。
- 新schedule身份防止错误恢复；不改推理模型结构。

## Capabilities

### New Capabilities
- `cgbs-frozen-content`: 预训练后确定性固定内容评分与共享表示。

### Modified Capabilities
无。

## Impact

新增模型及配置、测试与协议。0自动训练，用户手动从头启动1次30k。旧attention512联合方案保留。冻结也限制Decoder的共享Encoder，故不是只冻结MLP的单因素实验；不单独主张新颖性或已确认梯度冲突。
