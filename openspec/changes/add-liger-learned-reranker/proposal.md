## Why
候选概率混合具有匹配控制增量，但尚未超越dense。用户批准以冻结LIGER上的受限残差排序训练验证候选互补能否兑现；已结束的无训练解码搜索不恢复。
## What Changes
- 标签无关的双臂候选与特征缓存，分片产物和训练用户隔离。
- 263→64→1残差排序器、共同可训练样本、统一训练统计和配对评价。
- Hydra入口、手动脚本、证据与有限预算。
## Capabilities
### New Capabilities
- `liger-learned-reranker`: 冻结模型候选缓存、匹配残差排序训练与评价。
### Modified Capabilities
无。
## Impact
LIGER prediction模式、data缓存组件、共享writer、配置、脚本与测试；不引入依赖，不修改已有训练和预测默认行为。
