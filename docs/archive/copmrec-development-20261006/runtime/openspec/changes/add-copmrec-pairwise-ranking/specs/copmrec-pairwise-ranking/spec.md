## ADDED Requirements
### Requirement: 训练侧缓存来源约束
系统 SHALL 只允许training缓存用于拟合，记录payload、checkpoint、目录与协议指纹，缓存默认本地不发布。
#### Scenario: evaluation数据不可用于训练
- **WHEN** 用户提供evaluation trace作为训练缓存
- **THEN** 系统拒绝训练
### Requirement: 有界候选内排序学习
系统 SHALL 冻结原CoPMRec并用训练正目标与同池竞争者学习有界残差，初始等于内容次序。
#### Scenario: 目标未入候选
- **WHEN** training目标未覆盖
- **THEN** 不补入目标，记录覆盖并不参与pairwise拟合
### Requirement: 可复算开发验证
系统 SHALL 保留内容、混合、生成和learned同候选四臂，选择/复验用户互斥，使用统一src.main入口。
#### Scenario: 不匹配来源
- **WHEN** head checkpoint与候选缓存契约不一致
- **THEN** 系统拒绝加载或评价
