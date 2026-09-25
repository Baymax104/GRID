## Why

40k下CGBS与A最终接近，内容CE缓慢改善；已有加宽负结果和正向梯度快照不支持容量不足或梯度冲突作为已确认瓶颈。用户授权一次固定预算的分阶段训练，检验内容表示的训练起点是否影响后续推荐。

## What Changes

- 新增可选训练子类：5000步内容CE预训练，再35000步原联合目标，总计40000步。
- 保留128维MLP、共享Encoder、SID初始化、目录与评分公式，不新增网络。
- 明确阶段、DDP未使用参数、Adam状态延续和仅联合阶段选择best checkpoint。
- 复用根训练脚本，通过模型/回调/Trainer配置选择，不新增runner。

## Capabilities

### New Capabilities
- `cgbs-content-warmup`: 有界内容预训练和自动阶段切换。

### Modified Capabilities
无。

## Impact

新增模型、回调及组件配置和回归测试；旧CGBS默认行为不变。一次seed42训练由用户手动启动；不自动推理或testing。分阶段训练本身不作为区别LIGER/COBRA等混合检索工作的创新主张，仍须验证原CGBS剪枝前机制。
