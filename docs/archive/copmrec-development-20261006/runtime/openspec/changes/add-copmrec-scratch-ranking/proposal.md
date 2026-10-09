## Why
用户采纳全模型随机初始化训练，检验CoPMRec能否在训练中适应高位混合排序。旧decoder微调最佳早于预算末端，未证明单纯延长有用；双保护和cap有局部正向，边界辅助固定实例负向。此变更不修改baseline，不宣称唯一根因。
## What Changes
- 新增随机初始化基础联合训练、冻结本次teacher并training校准、全模型高位排序联合训练。
- 在线短候选训练、真实混合候选推理，固定selection/audit，无旧模型初始化依赖。
- 专用配置/入口、checkpoint阶段契约、测试和手动运行交付。
## Capabilities
### New Capabilities
- `copmrec-scratch-ranking`: CoPMRec从头联合排序训练。
### Modified Capabilities
无。
## Impact
仅新增CoPMRec路径并向既有候选评分helper增加兼容可选返回；原baseline和decoder微调默认行为保持。
