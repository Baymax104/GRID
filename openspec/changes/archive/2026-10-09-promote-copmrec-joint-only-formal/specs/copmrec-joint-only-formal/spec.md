## ADDED Requirements

### Requirement: 正式联合视图训练目标

系统 SHALL 在新正式入口仅优化 SID CE、完整目录 joint CE、mixture NLL，各权重为1；SHALL 保留共享历史与目录商品残差、内容投影、训练cold规则和固定50k双卡日程，不构建native目录评分或native CE。

#### Scenario: 与已有A3计算对齐

- **WHEN** 固定同一参数、输入与dropout随机数
- **THEN** 三项损失、总损失和参数梯度与A3一致；joint CE 对seen商品残差有有限非零梯度，cold残差梯度为零

### Requirement: 全阶段不排除历史商品

系统 SHALL 在训练目录支持集、raw Validation及最终单卡dense推理中保留历史商品，使用稳定catalog-row排名，并继续使用完整历史输入和残差。

#### Scenario: 历史商品分数最高

- **WHEN** 合法历史商品位于完整目录原始分数TopK
- **THEN** Validation与推理均保留该商品，排序不读取目标标签

### Requirement: 独立可核验来源

系统 SHALL 保存严格的新正式版本契约和完整模型/optimizer/scheduler状态，拒绝将原Full或A3checkpoint改标签当作新正式训练；SHALL 保留其已有实证供明确标注的选型分析。

#### Scenario: 恢复来源不同

- **WHEN** 新入口收到旧Full、A3或不同版本/损失契约的checkpoint
- **THEN** 拒绝普通正式恢复；新版本自己的完整checkpoint可恢复且预测一致

### Requirement: 入口及研究状态一致

系统 SHALL 提供统一src.main的双卡训练/单卡推理根脚本，透传两种notes、dry-run和额外override，记录方法/损失/历史协议，并同步研究和Linear最新定义；本次新增运行预算SHALL为零。

#### Scenario: 版本冻结完成

- **WHEN** 新代码通过CPU测试、Hydra compose、shell验证及OpenSpec strict验证
- **THEN** 报告实现完成与已有选型结果，不把未启动正式实验标记完成、不重置旧任务/成本、不自动启动新运行
