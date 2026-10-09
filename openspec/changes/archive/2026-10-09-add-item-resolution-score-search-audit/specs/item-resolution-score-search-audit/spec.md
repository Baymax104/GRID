## ADDED Requirements

### Requirement: 无标签确定性有界取样
系统 SHALL 在单进程 evaluation 中按用户key的固定hash选择有限样本，拒绝重复key，且不根据标签或模型结果选择用户。

#### Scenario: 文件顺序变化
- **WHEN** 相同用户数据以不同遍历顺序输入
- **THEN** 相同seed选出相同用户，修改标签不影响入选key

### Requirement: 精确目录分布与预算搜索分离
系统 SHALL 对每个样本枚举全部目录的精确边缘概率，并独立运行预声明预算搜索，不能把目标加入搜索候选。

#### Scenario: 完整分布验证
- **WHEN** 在小目录上执行审计
- **THEN** 目录总概率近似一，与穷举搜索一致，分块大小不改变精确分布

### Requirement: 配对证据与兼容加载
系统 SHALL 保留已有checkpoint契约，并通过共享writer输出keys、labels、输入hash、分布、预算指标和来源指纹。

#### Scenario: 两模型配对
- **WHEN** 比较MIR和depth2
- **THEN** 可验证相同用户/输入/标签/目录；非计时结果与checkpoint和预算绑定

### Requirement: 人工单卡启动
系统 SHALL 提供四个checkpoint串行单GPU命令，支持notes、dry-run、print-only和额外Hydra参数，禁止多卡推断。

#### Scenario: 预览与执行边界
- **WHEN** 用户传入print-only或非法GPU配置
- **THEN** 前者只打印四个统一入口命令，后者报错而不启动实验

### Requirement: 审计证书必须与精确集合一致
系统 SHALL 在 v2 审计产物中分别保存搜索内部界证书、精确 Top-K 集合一致性和二者合取的最终证书。集合一致性 SHALL 按 item key 比较，不依赖浮点重合率阈值；精确同分 SHALL 按 item key 升序选择。最终证书仅保证集合，不保证集合内顺序。

#### Scenario: 近边界分数产生计算路径分歧
- **WHEN** 搜索内部界证书为真，但返回集合与全目录枚举的 Top-K 不同
- **THEN** 最终审计证书为假，原始界证书保持为真，推荐及预算不变

#### Scenario: 校验与旧产物兼容
- **WHEN** 读取审计产物
- **THEN** v2 校验器从完整分布和 item keys 重算集合一致性及证书合取，并拒绝矛盾字段；v1 保持原始字段含义和可读性，不能追溯升级为 v2 认证
