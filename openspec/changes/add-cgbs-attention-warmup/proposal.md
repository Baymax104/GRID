## Why

用户明确指定一次attention pooling、MLP512、10k内容预训练+20k联合训练的组合尝试。此前512加宽和5k+35k未优于原C128，验证CE上升不证明容量不足；本实验只检验该组合是否改善验证推荐，不宣称解决已确认的因果瓶颈。

## What Changes

- 新增可学习掩码attention pooling，默认mean保持兼容。
- 独立配置512隐藏层、10000预训练步、30000总步数，保留验证content CE。
- 保存pooling checkpoint契约，提供匹配推理配置。

## Capabilities

### New Capabilities
- `cgbs-attention-pooling`: 内容query的可选注意力读取与结构身份校验。

### Modified Capabilities
无，阶段更新和优化器连续性沿用既有能力。

## Impact

原模型可选参数、训练/推理组件配置及测试。无新依赖，无自动完整实验。更复杂pooling与分阶段本身不构成相对LIGER/COBRA等混合检索方法的新颖性；核心仍需C-on/off机制证据。此次同时改变pooling、宽度、阶段和总预算，不能做单因素归因。
