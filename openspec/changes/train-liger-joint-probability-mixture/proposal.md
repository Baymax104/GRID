## Why

现有方法在冻结 LIGER 上学习常数，只能证明解码阶段的内容引导和校准效果，不能回答围绕混合概率整体训练是否更有效。用户要求做一次从头整体训练，以超过匹配 LIGER 为成功标准；这是训练目标对齐的新变更，不重开旧动态门控或排序预算。

## What Changes

- 新增主模型从随机初始化训练的混合概率目标，联合更新生成模型、内容投影与全局混合系数；复用固定 SID 和预计算内容向量。
- 训练与 beam 解码共享合法前缀条件概率定义，保留原内容终排。
- 冻结一次 Beauty/seed42、50k 更新的候选协议，与现有 50k LIGER 做匹配比较；超过基线是验收门槛，不是结果保证。
- 先做可微分性、概率一致性和轻量运行验证，完整实验仅由用户启动。

## Capabilities

### New Capabilities
- `liger-joint-mixture-training`: 可微分混合概率整体训练、checkpoint 恢复和有限实验协议。

### Modified Capabilities
无。原 LIGER 默认训练与已有 checkpoint 行为保持兼容。

## Impact

涉及 src/recommendation/liger、model/experiment 配置、根训练和推理脚本、单元测试与研究状态。不新增依赖，不重新训练语义编码器或量化器，不启动完整实验。2026-09-24实现与聚焦验证已完成，已同步node1，等待用户手动训练；效果未知。
