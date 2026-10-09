## Why
两次内容候选干预均改善hybrid但仍低于dense。用户授权验证生成信号在最终排序中是否提供额外价值，避免继续仅追近dense。
## What Changes
- 一次统一入口预测共享候选池，输出dense与等权联合排序配对记录。
- 对池内全部商品teacher forcing完整SID评分，不依赖是否被beam选中，不用目标标签筛候选。
- 共享writer发布证据与自动配对bootstrap汇总，保留旧模型默认行为。
## Capabilities
### New Capabilities
- `liger-paired-rerank`: 同候选池联合排序的有界验证。
### Modified Capabilities
无。
## Impact
LIGER推理模式、reranking领域helper、data证据契约、writer与callback配置、测试。无新依赖、不训练。
