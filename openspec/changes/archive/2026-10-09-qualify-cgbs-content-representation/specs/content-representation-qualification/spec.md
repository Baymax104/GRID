## ADDED Requirements

### Requirement: Matched content training
系统 SHALL 对固定/可学习商品bank采用相同历史输入、共有参数初始化、窗口流及固定预算，仅允许adapter训练状态不同。

#### Scenario: Fixed endpoint
- **WHEN** 正式训练完成
- **THEN** checkpoint包含2000步/64000窗口及初始化、数据、目录指纹，不经验证选点

#### Scenario: Explicit 10k budget extension
- **WHEN** 用户显式指定training_steps=10000
- **THEN** 数据流、模型、Trainer与checkpoint同步采用10000步/320000窗口；两臂从头匹配训练，评价显式检查10000终点并拒绝混用2000终点

### Requirement: History and split integrity
系统 SHALL 拒绝非法部分SID、未知商品、空历史和冲突标签，并排除所有校准用户窗口。

#### Scenario: Terminal prediction placeholder
- **WHEN** 历史末尾包含合法预测占位符
- **THEN** 将其剔除而非当作商品，且保持真实历史顺序

### Requirement: Paired qualification
系统 SHALL 在512匹配开发用户上比较固定终点的全目录NLL和Top10，先检查两臂契约再输出结论。

#### Scenario: Failed effect gate
- **WHEN** NLL改善95%CI下界不大于零或Recall下降
- **THEN** 报告停止该配置且不自动扩预算

### Requirement: Manual unified launch
启动脚本 SHALL 使用src.main、支持dry-run/notes/额外override，不自动运行完整实验。

#### Scenario: Dry run
- **WHEN** 显式传入--dry-run
- **THEN** 统一入口缩小运行且禁用正式产物发布
