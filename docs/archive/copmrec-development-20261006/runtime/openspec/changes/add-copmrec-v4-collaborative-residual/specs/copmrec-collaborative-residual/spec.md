## ADDED Requirements

### Requirement: 冷商品安全的共享协同残差

系统 SHALL 为seen商品提供零初始化可训练残差，并在历史与目录投影后共享使用；cold residual SHALL 强制为零。v0未启用hook时 SHALL 保持初始化、RNG及数值行为。

#### Scenario: 初始化等价与训练信号

- **WHEN** 从匹配学习alpha v0初始化v4且残差为零
- **THEN** 编码query、目录logits、三个loss及预测与v0相等，seen残差获得有限非零训练梯度，cold残差梯度为零

#### Scenario: 匹配续训对照

- **WHEN** residual_scale设为0
- **THEN** 残差参数冻结且其他v0参数仍可训练，无额外随机初始化变化

### Requirement: 严格初始化和checkpoint恢复

系统 SHALL 校验v0来源、学习alpha策略、catalog身份及完整state结构；weights-only初始化 SHALL 不恢复旧优化器/步数，来源 SHALL 被记录。v4恢复 SHALL 校验版本和残差scale。

#### Scenario: 拒绝不匹配来源

- **WHEN** 来源是固定alpha、其他版本、目录不一致或state缺损
- **THEN** 初始化明确失败

#### Scenario: 推理恢复

- **WHEN** 以v4训练best checkpoint执行单卡推理
- **THEN** 严格加载全部参数，使用相同残差策略、输入Artifact和合法SID bundle输出

### Requirement: 统一运行和效果证明

训练与推理 SHALL 经src.main/Hydra，训练脚本支持dry-run/notes/override，推理 SHALL 单卡。checkpoint SHALL 依据evaluation选择；达成10% SHALL 由完整Testing原始标签/合法输出独立复算并与真实LIGER dense比较证明。

#### Scenario: 未验证收益

- **WHEN** 仅CPU测试、dry-run或训练存活通过
- **THEN** 记录为准备或运行中，不能标记推荐效果达标
