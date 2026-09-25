## ADDED Requirements

### Requirement: 训练选样与标签隔离
系统 SHALL 从已核验training缓存选择2048不同用户各一个确定窗口，在任何evaluation标签参与之前拟合三个线性模型。

#### Scenario: 标签不能影响输入特征
- **WHEN** 仅修改evaluation目标
- **THEN** 用户内容特征、训练选样和线性系数保持不变

#### Scenario: 历史末尾包含预测占位符
- **WHEN** TIGER预处理产生末尾`[0,-1,...]`与mask`[1,0,...]`
- **THEN** 内容聚合排除此非商品位置且保持A输入不变，其他不完整SID仍拒绝

### Requirement: 匹配真实与置乱内容
系统 SHALL 使用同一A offset、训练窗口、6维结构与200更新预算比较真实/置乱内容，并提供2维频次控制。

#### Scenario: 固定零初始化与完整候选
- **WHEN** 三个校准器尚未拟合
- **THEN** 全部合法root的分数与A精确一致，选样不按A成功与否过滤

### Requirement: 有界评价与停止判定
系统 SHALL 只对固定512用户做第一层前向，汇总NLL、root存活与2000次配对bootstrap，并按预定规则决定是否有资格进一步研究。

#### Scenario: 数据不完整或dry run
- **WHEN** 用户数不完整、来源不匹配或dry-run
- **THEN** 不发布正式资格证据
