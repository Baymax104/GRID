## ADDED Requirements

### Requirement: 训练统计 SHALL 遵守原子序列展开的期望目标口径

系统 MUST 只读取 training 数据，按有放回抽样后去重的入选概率计算期望目标次数，记录展开参数、文件摘要和 SID 身份，并拒绝空数据、非法 keys 或未知 item。

#### Scenario: 超过抽样上限
- **WHEN** 一行有 T 个候选子序列且 T>m
- **THEN** 位置 j 的目标质量为 (j-1)(1-(1-1/T)^m)，计算不改变随机数状态

### Requirement: 独立训练模块 SHALL 保持 baseline 数据与评估兼容

系统 MUST 复用原 dataset、collate、forward、全词表 CE 和生成方法，仅在新类 training_step 对指定层加权；验证 loss 保持普通 CE。

#### Scenario: 干预关闭
- **WHEN** 新模块使用 CE 模式或权重全为 1
- **THEN** 固定 batch 下 loss、梯度、参数更新与 baseline 一致，新 checkpoint 可由原模型严格加载

### Requirement: 条件分支权重 SHALL 有界且可审计

系统 MUST 按父前缀条件概率计算有上限的逆频率权重，在训练期望分布内逐层归一化，拒绝非法参数，保留权重与统计 metadata 到 checkpoint。

#### Scenario: 不同父前缀使用相同 token
- **WHEN** 相同 token 出现在不同父前缀
- **THEN** 查询使用完整前缀，并保持未干预层权重为 1

### Requirement: 探针初始化 SHALL 明确区分权重与训练状态

系统 MUST 通过共享 Artifact resolver 加载初始化权重，严格校验 state_dict；新实验使用新优化器与零起始步数，拒绝 Trainer ckpt_path 恢复。

#### Scenario: 初始化与恢复同时指定
- **WHEN** 新实验传入非空 Trainer ckpt_path
- **THEN** 在 fit 开始前报错，不混合两种状态

### Requirement: 验证执行 SHALL 保持用户手动启动与固定预算

系统 MUST 提供根脚本，经 src.main 启动，支持必填数据/SID/初始化/group/devices/arm/notes、seed 默认42、dry-run、两种 flag 语法与末尾 overrides；默认关闭自动 testing，发布固定步数 last checkpoint。

#### Scenario: 只完成实现验证
- **WHEN** agent 完成单元测试和 compose
- **THEN** 不自动启动完整训练；运行手册明确配对条件、效用门槛和继续/停止/不确定结论
