## ADDED Requirements

### Requirement: Native fixed 256 content
系统 SHALL 采用固定PCA256向量，且新网络不含商品adapter参数，训练时仅学习用户内容序列模型。

#### Scenario: Backpropagation
- **WHEN** 运行内容预测CE反向传播
- **THEN** 用户编码器梯度有效，商品bank不变且不包含up/down参数

### Requirement: Bounded capacity comparison
系统 SHALL 保持10k/320000曝光和相同数据流，与既有128固定终点比较。

#### Scenario: Invalid comparison
- **WHEN** 步数、历史、SID、数据流或模型结构与协议不符
- **THEN** 拒绝发布正式效果结论

### Requirement: Legacy compatibility
系统 SHALL 保持已有128固定/可学习模型加载及训练默认行为。

#### Scenario: Original configuration
- **WHEN** 使用既有128配置
- **THEN** 原checkpoint参数结构与初始化行为保持一致
