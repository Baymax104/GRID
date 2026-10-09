## Why

正式基线矩阵需要 SASRec。当前框架只有 SID 推荐模型，直接使用标准 Transformer 或近期第三方 SASRec 会改变作者算法；先建立可独立验证的官方算法骨干。

## What Changes

- 新增 PyTorch SASRec backbone、逐位置正负 logits 与官方 BCE/embedding L2。
- 固定作者源码 `kang205/SASRec@e3738967fddab206d6eeb4fda433e7a7034dd8b1` 并验证公式对应关系。
- 本提案仅涉及算法模块；数据适配、Lightning、配置脚本分别由后续模块提案实现。

## Capabilities

### New Capabilities

- `sasrec-backbone`: 与作者算法对应的因果自注意力、逐位置监督和共享 embedding 打分。

### Modified Capabilities

无。

## Impact

新增 `src/recommendation/sasrec/` 和 CPU 内存单元测试；不增加依赖，不改变现有模型。
