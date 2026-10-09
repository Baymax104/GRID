## ADDED Requirements

### Requirement: v0固定全程混合权重

系统SHALL支持v0的fixed_mixture_alpha=0.8，从初始化开始冻结全局gate，保留全层Mass、SID CE/content CE/混合NLL三项等权与两路可训练表示。

#### Scenario: 固定权重反向传播
- **WHEN** 在内存合成目录计算混合NLL并更新优化器
- **THEN** 概率与独立0.8混合枚举相同，encoder/decoder/content投影梯度有限且非零，gate无梯度且值不变，初始化随机序列与v0相同

### Requirement: 训练推理与恢复策略一致

系统SHALL在teacher forcing、验证和推理使用相同固定值，checkpoint及trace记录固定策略，并拒绝不匹配的恢复。

#### Scenario: 保存和恢复固定模型
- **WHEN** 固定模式保存并恢复相同固定值checkpoint
- **THEN** state和预测一致，trace alpha为0.8且来源为fixed_training，逐步目标路径NLL与训练一致

#### Scenario: 策略或状态不匹配
- **WHEN** 固定模型读取旧学习checkpoint、其他固定值或被改变的gate状态
- **THEN** 明确拒绝恢复；默认学习模式仍接受历史v0 checkpoint且学习alpha

### Requirement: 匹配v0训练协议

新入口SHALL继承v0数据、backbone、优化器/调度器、双卡trainer和dense验证选点，随机初始化不加载checkpoint，正式运行由用户手动开始。

#### Scenario: 双卡根脚本
- **WHEN** 物理GPU2/3、NPROC_PER_NODE=2、devices=[0,1]传入notes和末尾override
- **THEN** 通过uv run torchrun调用src.main，默认每卡batch128、累积1、50k更新、间隔500验证，dry-run仅显式开启，quote/空值/非法参数按既有契约处理
