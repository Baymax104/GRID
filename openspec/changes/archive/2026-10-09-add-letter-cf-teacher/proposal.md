## Why

BMX-58 缺少可复现的32维 CF 生产入口，无法交付完整 LETTER 命令。作者仅提供 CF 导出片段，不能复用项目其他模型或把随机向量作为正式 CF。

## What Changes

- 新增独立 LETTER CF teacher：按官方 SASRec 公式实现32维商品表、training-only BCE，evaluation选优。
- 接入统一 Hydra 训练/导出入口与共享 keyed writer；固定来源、目录和checkpoint身份。
- 补齐九个实验槽位的分阶段命令及有限验证，不启动完整实验。

## Capabilities

### New Capabilities

- `letter-cf-teacher`: 独立训练及导出32维协同向量。

### Modified Capabilities

无。

## Impact

新增 LETTER 域模型、数据适配、配置、根脚本、测试及交付说明。复用公共框架和本方法指标，不依赖项目 SASRec/RQ-VAE/TIGER 模型，不新增依赖。
