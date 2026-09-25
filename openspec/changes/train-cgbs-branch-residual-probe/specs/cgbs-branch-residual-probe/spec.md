## ADDED Requirements

### Requirement: 等价复用历史计算
系统 SHALL 在同一微批内跨候选与层复用历史key/value，global模式 SHALL 复用与候选无关的attention结果；复用 MUST 保留参数梯度且不得跨优化更新使用旧激活。
#### Scenario: 非零评分头的等价重构
- **WHEN** 使用不同候选块大小执行global或branch模式的前向及反向
- **THEN** 输出、frontier目标与全部参数梯度在浮点误差范围内匹配旧公式，历史投影不随候选数重复

### Requirement: 顺序门槛与人工执行
系统 SHALL 记录三个顺序门槛、实际作业数和停止条件；完整实验 MUST 由用户手动启动。
#### Scenario: 第一门槛未通过
- **WHEN** 结构对照未达到预设增量条件
- **THEN** 不自动启动搜索机制或跨seed实验

### Requirement: 冻结来源与匹配两臂
系统 MUST 严格载入A的来源契约与权重，仅优化新增头；global和branch SHALL 使用相同参数结构、数据和frontier目标。
#### Scenario: 训练后关闭残差
- **WHEN** 已更新内容头后执行off
- **THEN** 主干张量不变且原A排序完全复现

### Requirement: 有界可信竞争缓存
缓存 SHALL 仅来自training且包含全部合法A beam展开候选和去重后的gold前缀；manifest MUST 绑定来源与每个分片哈希。
#### Scenario: 缓存损坏或来源不匹配
- **WHEN** 分片哈希、样本数、目录或A SHA不匹配
- **THEN** 训练失败且不静默使用不匹配数据

### Requirement: 残差能量与mask正确
系统 SHALL 使用累计A分数加当前prefix残差，初始残差为零，mask后的history/prototype不影响评分；非概率trace MUST 使用独立语义。
#### Scenario: 不同候选读取历史
- **WHEN** branch模式的候选原型不同
- **THEN** attention可随候选变化；global模式history聚合保持不依赖候选

### Requirement: 统一入口与可复现启动
脚本 MUST 支持dry-run、两种notes语法、URI安全quoting和额外Hydra override，数据/输出路径SHALL由配置提供。
#### Scenario: 手动准备第一门槛
- **WHEN** 用户运行cache/train/inference入口
- **THEN** 经src.main装配对应组件并记录来源、arm、预算和缓存身份
