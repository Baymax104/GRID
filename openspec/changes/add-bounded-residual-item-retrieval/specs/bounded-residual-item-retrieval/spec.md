## ADDED Requirements

### Requirement: 独立基础评分与有界修正
系统 SHALL 使用 masked-mean 历史查询及原始内容投影的 normalized dense 分数；BRIR 的最终分数 SHALL 为基础分数加 delta*tanh 残差，并支持相同容量的无前缀对照。

#### Scenario: 零初始化与梯度边界
- **WHEN** 从合法基础 checkpoint 创建两个残差组
- **THEN** 初始最终分数等于基础分数；基础模型不产生梯度，独立残差组件可学习。

### Requirement: 确定的共同候选与初始化
三分支 SHALL 使用相同冻结基础模型挖掘 top64 非正例和其余目录16个随机非正例；随机选择 SHALL 仅依赖历史与固定seed。系统 MUST 拒绝不兼容 checkpoint、目录、温度、基础步数和标定来源。

#### Scenario: Dense 分支发生更新
- **WHEN** dense 继续训练改变当前权重
- **THEN** 其挖掘候选仍与两个残差组相同；优化器重新开始，基础来源不变。

### Requirement: 训练历史标定
标定 SHALL 通过统一入口在 training split 按固定key hash选取1024条历史，计算第10与128分数间距，并通过共享writer发布带来源的产物。加载 SHALL 检查行数、有限正delta和基础/目录身份。

#### Scenario: 错误来源或退化间距
- **WHEN** 使用evaluation标定、错checkpoint或中位间距为零
- **THEN** 系统拒绝分支训练而不是静默替换delta。

### Requirement: 可核验的有界检索
系统 SHALL 提供 reference、fixed 和 dynamic 策略，稳定处理同分；未评分上界严格低于当前第K分数时才可声明bound成立。数值审计 SHALL 分别记录bound、精确集合一致、精确顺序一致和最终verified标记。

#### Scenario: 预算耗尽或近边界
- **WHEN** 剩余候选仍可能进入TopK或reference集合不一致
- **THEN** 不得返回已验证完成标记；保留候选与预算计数用于诊断。

### Requirement: 手动入口和产物边界
所有任务 SHALL 通过src.main及Hydra运行，数据/模型字段通过共享artifact helper解析，writer负责发布，lineage callback负责上游引用。根脚本 SHALL 支持非空参数、两种notes形式、dry-run、print-only和末尾Hydra覆盖。

#### Scenario: 用户准备启动
- **WHEN** 执行print-only或完成本地实现验收
- **THEN** 不启动训练，不访问外部Artifact；提供基础训练及后续阶段明确命令。

#### Scenario: 双卡训练保留更新尺度
- **WHEN** 用户指定训练使用GPU 0、1及两个进程
- **THEN** 使用DDP、每卡batch8和累积8次，保持有效batch128及每1000优化步验证；标定与审计继续拒绝多卡。
