## ADDED Requirements

### Requirement: 联合模型固定alpha推理覆盖
系统 SHALL 接受可空的固定推理alpha；设置时仅用于候选解码，训练 SHALL 拒绝该覆盖，旧checkpoint SHALL 严格兼容。

#### Scenario: 固定0.5解码
- **WHEN** 联合checkpoint以`inference_mixture_alpha=0.5`预测
- **THEN** processor使用固定0.5而不使用checkpoint gate，并在trace记录0.5。

#### Scenario: 默认推理
- **WHEN** 未提供固定推理alpha
- **THEN** 保持现有learned gate或content_only行为。

### Requirement: 有界2×2协议
系统 SHALL 复用两个已有单元并仅新增原训练×1和联合训练×0.5两个预测，固定共同数据与检索协议。

#### Scenario: 交叉完成
- **WHEN** 两个新预测完成且来源校验通过
- **THEN** 输出四单元指标、配对简单效应与difference-in-differences交互，不将其扩展为新训练或alpha搜索。
