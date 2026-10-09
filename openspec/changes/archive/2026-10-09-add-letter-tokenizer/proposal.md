## Why

正式 LETTER 基线需要作者方法的独立 tokenizer。当前 RQ-VAE 未验证复现可信度，不能继承其算法实现。

## What Changes

- 对照 HonghuiBao2000/LETTER@8d0154e 实现独立 MLP、四层残差 VQ、STE、Sinkhorn、CF CE 与 constrained-cluster diversity。
- 分离纯编码和训练 loss，支持初始化、分组状态恢复与有界碰撞修复。
- 本提案仅实现 tokenizer 算法；数据、推荐骨干及 pipeline 分别后续提案。

## Capabilities

### New Capabilities
- `letter-tokenizer`: 作者 tokenizer 数值、梯度、分组与编码契约。

### Modified Capabilities
无。

## Impact

新增 src/quantization/letter 与内存测试；新增 k-means-constrained 依赖。禁止 import 当前 rqvae、rvq、tiger、liger、sasrec 模型。
