## Why

整体效果已优于匹配 LIGER，但合法约束、内容聚合与混合贡献尚未分离。固定同一 checkpoint 与目录支持，检验总质量相对最大后代证据的增量。

## What Changes

- 增加仅推理的合法生成与最大后代概率混合控制，默认训练与推理保持不变。
- 冻结同得分、同支持协议，提供人工启动命令及聚焦验证。

## Capabilities

### New Capabilities
- `liger-mechanism-control`: 同 checkpoint、同合法支持的解码机制控制。

### Modified Capabilities
无。

## Impact

候选 processor、联合模型推理选项、配置与测试；不新增训练和依赖。
