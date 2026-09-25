## ADDED Requirements

### Requirement: 零初始化状态门控

系统 SHALL 提供 `content_init_state_gate`，每层新增三个全零线性权重，输入为合法P/Q归一化熵与归一化JS，保留原C初始化、损失、辅助编码器梯度和预算。

#### Scenario: 对齐原C起点
- **WHEN** C与E以相同seed、输入和随机状态初始化并计算训练
- **THEN** E仅多3L参数，原参数、前向值、RNG状态和原参数梯度与C一致，新增gate参数可学习，辅助CE仍更新编码器

### Requirement: 合法且稳定的状态特征

特征 MUST 仅使用当前合法子分支并停止梯度，训练和beam SHALL 复用相同条件评分公式。

#### Scenario: 极端与单分支状态
- **WHEN** 存在非法logits、零概率、单合法子分支或极端门控权重
- **THEN** 特征有限且位于[0,1]，单分支特征为零，输出合法概率归一化且训练梯度有限

#### Scenario: 真正按状态评分
- **WHEN** 非零gate权重遇到同层不同P/Q分布
- **THEN** 混合强度可不同且不依赖目标标签或batch划分，beam分数与同路径teacher评分一致

### Requirement: 门控身份与推理干预

系统 SHALL 以独立版本化契约保存E；base/fixed干预 MUST 仅推理使用且记录metadata。

#### Scenario: 恢复和来源
- **WHEN** E checkpoint用于dynamic/base/fixed推理
- **THEN** 恢复同一训练参数，fixed要求逐层有限合法常量及非空来源，C与E混用失败，旧arm契约不变

#### Scenario: 非法干预组合
- **WHEN** 非E使用门控干预、训练使用base/fixed，或E使用exact/shuffled_exact/rerank
- **THEN** 给出明确错误，避免隐式改写实验含义

### Requirement: 有效状态统计与手动启动

系统 SHALL 复用现有MetricCallback按teacher状态记录各层alpha均值/最小/最大，并提供独立E单次双卡命令。

#### Scenario: 非等长batch统计
- **WHEN** 不同batch或rank贡献不同数量状态
- **THEN** 均值由总和/总状态数定义，极值使用全体状态，不平均各batch均值，不额外运行decoder

#### Scenario: 显式E启动
- **WHEN** 用户执行mechanism train的`--condition e`
- **THEN** 只选E，物理GPU0/1、两进程、原20k预算；notes/dry-run/用户override透传，默认both仍B/C
