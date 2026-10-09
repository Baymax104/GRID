## Why

LETTER 验证中逐 beam 的 CUDA prefix.tolist() 产生大量同步。实际 checkpoint 的 32 用户、20 beam、5 步生成触发 3200 次前缀回调，验证占训练墙钟约 98%；诊断中的批量原型已经得到逐位相同的 IDs 和 scores。

## What Changes

- 在 LETTER 模块内批量执行全目录前缀约束，每个解码步只读取一次完整前缀矩阵。
- 保持 HF beam search、EOS、完整词表概率、长度评分及 checkpoint 契约。
- 补充 HF 差分测试和真实 checkpoint 的有限性能核验。

## Capabilities

### New Capabilities

- `letter-batched-prefix-constraints`: LETTER 批量前缀约束的语义及同步成本契约。

### Modified Capabilities

无。

## Impact

仅影响 LETTER backbone 的生成路径、对应测试与修复记录；无新依赖、配置或训练参数变更。同步 node1 后进行有限验证，不启动正式训练。
