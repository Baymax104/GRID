## Why
直接混合终排及冻结beta校准均未晋级。用户明确要求校准失败后尝试排序学习，以训练侧监督学习CoPMRec候选内修正，不改baseline或研究问题。
## What Changes
- 本地training候选缓存及来源manifest，禁止默认发布缓存。
- 冻结主模型，以内容分数为基础训练一个小型pairwise残差排序器。
- 统一入口的缓存、训练、evaluation推理配置与脚本。
## Capabilities
### New Capabilities
- `copmrec-pairwise-ranking`: 训练侧候选监督的有界残差排序与来源验证。
### Modified Capabilities
无。
## Impact
CoPMRec专用组件、共享排序trace的向后兼容新schema、缓存数据模块、配置与聚焦测试。无新依赖、无baseline训练或搜索变化。
